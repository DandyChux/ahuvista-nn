use tracing::info;

use crate::{
    core::{fusion::LateFusionNet, Module, PrivacyEngine},
    data::{DataLoader, MultiModalBatch},
    training::{
        balancing::{weighted_binary_cross_entropy, BayesianCalibrator},
        loss::binary_cross_entropy,
        metrics::{calculate_auroc, ConfusionMatrix, EvaluationMetrics},
    },
};
use std::collections::HashMap;

/// Configuration for model training
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,

    /// Batch size
    pub batch_size: usize,

    /// Learning rate
    pub learning_rate: f32,

    /// Validation frequency (every N epochs)
    pub val_frequency: usize,
}

/// Train the model with sample weights and calibration
pub fn train_model_with_weights(
    model: &mut LateFusionNet,
    train_loader: &mut impl DataLoader<MultiModalBatch>,
    val_loader: &mut impl DataLoader<MultiModalBatch>,
    config: &TrainingConfig,
    train_weights: &[f32],
    val_weights: &[f32],
    calibrator: &BayesianCalibrator,
) {
    let privacy_engine = PrivacyEngine::new(
        1.0,
        1e-5,
        train_loader.len(),
        config.batch_size,
        config.epochs,
    );

    for epoch in 0..config.epochs {
        // Training phase
        let mut epoch_loss = 0.0;
        let mut correct_predictions = 0;
        let mut total_predictions = 0;

        train_loader.reset();
        let mut sample_idx = 0;

        // Process batches
        while let Some(batch) = train_loader.next_batch(config.batch_size) {
            let batch_size = batch.tabular.len();

            // Process each sample in the batch
            for i in 0..batch_size {
                // Get sample weight
                let sample_weight = if sample_idx < train_weights.len() {
                    train_weights[sample_idx]
                } else {
                    1.0 // Default weight
                };

                sample_idx += 1;

                // Zero gradients before forward pass
                model.zero_grad();

                // Forward pass
                let prediction = model.forward((
                    batch.tabular[i].clone(),
                    batch.temporal[i].clone(),
                    batch.text[i].clone(),
                    batch.images[i].clone(),
                ))[0];

                // Compute weighted loss
                let target = batch.targets[i];
                let (loss, grad) =
                    weighted_binary_cross_entropy(&[prediction], &[target], &[sample_weight]);

                epoch_loss += loss;

                // Calibrate prediction for accuracy tracking
                let calibrated_pred = calibrator.calibrate(prediction);
                let predicted_class = if calibrated_pred > 0.5 { 1.0 } else { 0.0 };

                if (predicted_class - target).abs() < 1e-6 {
                    correct_predictions += 1;
                }
                total_predictions += 1;

                // Backward pass with weighted gradient
                model.backward(vec![grad[0]]);

                // Differential Privacy (DP) occurs here
                let mut model_grads = model.gradients(); // Get all gradients
                privacy_engine.privatize_gradients(&mut model_grads, 1.0); // Clip norm to 1.0
                model.apply_gradients(&model_grads, config.learning_rate); // Apply privatized gradients
            }
        }

        // Calculate training metrics
        let train_loss = epoch_loss / total_predictions as f32;
        let train_accuracy = correct_predictions as f32 / total_predictions as f32;

        // Validation phase (if it's time)
        if (epoch + 1) % config.val_frequency == 0 || epoch == config.epochs - 1 {
            let (val_metrics, _calibration_data) =
                evaluate_model_with_calibration(model, val_loader, val_weights, calibrator);

            info!(
                "Epoch {}/{}: Train loss={:.4}, accuracy={:.2}% | Val loss={:.4}, accuracy={:.2}%",
                epoch + 1,
                config.epochs,
                train_loss,
                train_accuracy * 100.0,
                val_metrics.loss,
                val_metrics.accuracy * 100.0
            );
        } else {
            info!(
                "Epoch {}/{}: Train loss={:.4}, accuracy={:.2}%",
                epoch + 1,
                config.epochs,
                train_loss,
                train_accuracy * 100.0
            );
        }
    }
}

/// Evaluate the model with sample weights and calibration
/// Evaluate the model with sample weights and calibration, returning a full set of metrics.
pub fn evaluate_model_with_calibration(
    model: &mut LateFusionNet,
    data_loader: &mut impl DataLoader<MultiModalBatch>,
    sample_weights: &[f32],
    calibrator: &BayesianCalibrator,
) -> (EvaluationMetrics, Vec<(f32, f32)>) {
    // Return metrics AND calibration data
    let mut total_loss = 0.0;

    // We need to collect all predictions and targets to calculate metrics like AUROC
    let mut all_raw_predictions = Vec::new();
    let mut all_calibrated_predictions = Vec::new();
    let mut all_targets = Vec::new();

    data_loader.reset();
    let mut sample_idx = 0;

    while let Some(batch) = data_loader.next_batch(32) {
        let batch_size = batch.tabular.len();

        for i in 0..batch_size {
            let sample_weight = if sample_idx < sample_weights.len() {
                sample_weights[sample_idx]
            } else {
                1.0
            };
            sample_idx += 1;

            let raw_prediction = model.forward((
                batch.tabular[i].clone(),
                batch.temporal[i].clone(),
                batch.text[i].clone(),
                batch.images[i].clone(),
            ))[0];

            let target = batch.targets[i];

            // Store everything for later calculation
            all_raw_predictions.push(raw_prediction);
            all_calibrated_predictions.push(calibrator.calibrate(raw_prediction));
            all_targets.push(target);

            // Loss is calculated on the raw model output
            let (loss, _) =
                weighted_binary_cross_entropy(&[raw_prediction], &[target], &[sample_weight]);
            total_loss += loss;
        }
    }

    let total_samples = all_targets.len();
    if total_samples == 0 {
        // Return default metrics if there's no data
        return (EvaluationMetrics::default(), Vec::new());
    }

    // --- METRIC CALCULATIONS ---
    // Use a threshold of 0.5 on the *calibrated* predictions for classification metrics
    let conf_matrix =
        ConfusionMatrix::from_predictions(&all_calibrated_predictions, &all_targets, 0.5);

    let accuracy = (conf_matrix.tp + conf_matrix.tn) as f32 / total_samples as f32;
    let sensitivity = if conf_matrix.tp + conf_matrix.fn_ == 0 {
        0.0
    } else {
        conf_matrix.tp as f32 / (conf_matrix.tp + conf_matrix.fn_) as f32
    };
    let specificity = if conf_matrix.tn + conf_matrix.fp == 0 {
        0.0
    } else {
        conf_matrix.tn as f32 / (conf_matrix.tn + conf_matrix.fp) as f32
    };

    // AUROC should be calculated on the raw, uncalibrated scores to measure
    // the model's intrinsic ability to rank predictions.
    let auroc = calculate_auroc(&all_raw_predictions, &all_targets);

    let metrics = EvaluationMetrics::new(
        total_loss / total_samples as f32,
        accuracy,
        conf_matrix.ppv(),
        conf_matrix.npv(),
        sensitivity,
        specificity,
        auroc,
    );

    // --- CALIBRATION PLOT DATA GENERATION ---
    // Create 10 bins for probabilities [0.0-0.1), [0.1-0.2), ...
    // Each bin stores (sum_of_predicted_probs, count_of_true_positives, total_samples_in_bin)
    let mut bins = vec![(0.0, 0, 0); 10];
    for (calibrated_pred, target) in all_calibrated_predictions.iter().zip(all_targets.iter()) {
        let bin_index = (calibrated_pred * 10.0).floor() as usize;
        if bin_index < 10 {
            bins[bin_index].0 += calibrated_pred;
            if *target == 1.0 {
                bins[bin_index].1 += 1;
            }
            bins[bin_index].2 += 1;
        }
    }

    let calibration_data: Vec<(f32, f32)> = bins
        .iter()
        .filter_map(|(prob_sum, true_pos, total)| {
            if *total > 0 {
                let avg_predicted = prob_sum / *total as f32; // Mean predicted value
                let fraction_of_positives = *true_pos as f32 / *total as f32; // Actual fraction
                Some((avg_predicted, fraction_of_positives))
            } else {
                None
            }
        })
        .collect();

    // You can now print or plot this calibration_data. A perfect calibration
    // would have pairs like (0.1, 0.1), (0.2, 0.2), etc.

    (metrics, calibration_data)
}

/// Evaluate performance for each cause of mortality
pub fn evaluate_by_cause(
    model: &mut LateFusionNet,
    data_loader: &mut impl DataLoader<MultiModalBatch>,
    causes: &[usize],
    calibrator: &BayesianCalibrator,
) -> std::collections::HashMap<usize, (f32, f32)> {
    let mut cause_losses: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
    let mut cause_correct: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    let mut cause_total: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

    data_loader.reset();
    let mut sample_idx = 0;

    while let Some(batch) = data_loader.next_batch(32) {
        let batch_size = batch.tabular.len();

        for i in 0..batch_size {
            // Get the cause of mortality
            let cause = if sample_idx < causes.len() {
                causes[sample_idx]
            } else {
                4 // Default to \"other causes\"
            };

            sample_idx += 1;

            // Forward pass
            let prediction = model.forward((
                batch.tabular[i].clone(),
                batch.temporal[i].clone(),
                batch.text[i].clone(),
                batch.images[i].clone(),
            ))[0];

            // Compute loss
            let target = batch.targets[i];
            let loss = binary_cross_entropy(prediction, target);

            // Calibrate prediction
            let calibrated_pred = calibrator.calibrate(prediction);
            let predicted_class = if calibrated_pred > 0.5 { 1.0 } else { 0.0 };

            // Update metrics for this cause
            *cause_losses.entry(cause).or_insert(0.0) += loss;
            *cause_total.entry(cause).or_insert(0) += 1;

            if (predicted_class - target).abs() < 1e-6 {
                *cause_correct.entry(cause).or_insert(0) += 1;
            }
        }
    }

    // Calculate average metrics for each cause
    let mut cause_metrics = std::collections::HashMap::new();

    for cause in cause_total.keys() {
        let total = *cause_total.get(cause).unwrap_or(&1) as f32;
        let correct = *cause_correct.get(cause).unwrap_or(&0) as f32;
        let loss = *cause_losses.get(cause).unwrap_or(&0.0) / total;
        let accuracy = correct / total;

        cause_metrics.insert(*cause, (loss, accuracy));
    }

    cause_metrics
}
