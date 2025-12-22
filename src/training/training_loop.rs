use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::{
    config::TrainingParams,
    core::{fusion::LateFusionNet, Module, PrivacyEngine},
    data::{DataLoader, MultiModalBatch},
    training::{
        balancing::{weighted_binary_cross_entropy, BayesianCalibrator},
        loss::binary_cross_entropy,
        metrics::{calculate_auroc, ConfusionMatrix, EvaluationMetrics},
    },
};

/// Train the model with sample weights and calibration
pub fn train_model_with_weights(
    model: &mut LateFusionNet,
    train_loader: &mut impl DataLoader<MultiModalBatch>,
    val_loader: &mut impl DataLoader<MultiModalBatch>,
    config: &TrainingParams,
    train_weights: &[f32],
    val_weights: &[f32],
    calibrator: &BayesianCalibrator,
) {
    let privacy_engine = if config.use_privacy {
        info!(
            "Privacy enabled: epsilon={}, delta={}, clip_norm={}",
            config.privacy_epsilon, config.privacy_delta, config.gradient_clip_norm
        );
        Some(PrivacyEngine::new(
            config.privacy_epsilon,
            config.privacy_delta,
            train_loader.len(),
            config.batch_size,
            config.epochs,
        ))
    } else {
        warn!("Privacy DISABLED for debugging - do not use in production!");
        None
    };

    let mut current_lr = config.learning_rate;
    let mut total_batches = 0;

    for epoch in 0..config.epochs {
        // Training phase
        train_loader.reset();
        let mut sample_idx = 0;
        let mut epoch_predictions = Vec::new();
        let mut epoch_targets = Vec::new();
        let mut epoch_losses = Vec::new();

        // Process batches
        while let Some(batch) = train_loader.next_batch(config.batch_size) {
            let batch_size = batch.targets.len();
            total_batches += 1;

            // Adaptive learning rate decay
            if config.use_adaptive_lr && total_batches % config.lr_decay_steps == 0 {
                current_lr = (current_lr * config.lr_decay_rate).max(config.min_learning_rate);
                info!("Learning rate adjusted to: {:.6}", current_lr);
            }

            // Collect batch data
            let mut batch_logits = Vec::with_capacity(batch_size);
            let mut batch_targets = Vec::with_capacity(batch_size);
            let mut batch_weights = Vec::with_capacity(batch_size);
            let mut batch_inputs = Vec::with_capacity(batch_size);

            // Process each sample in the batch
            for i in 0..batch_size {
                // Get sample weight
                let sample_weight = if sample_idx < train_weights.len() {
                    train_weights[sample_idx]
                } else {
                    1.0
                };
                sample_idx += 1;

                // Forward pass
                let tabular_input = if i < batch.tabular.len() {
                    batch.tabular[i].clone()
                } else {
                    vec![]
                };

                let temporal_input = if i < batch.temporal.len() {
                    batch.temporal[i].clone()
                } else {
                    vec![]
                };

                let text_input = if i < batch.text.len() {
                    batch.text[i].clone()
                } else {
                    vec![]
                };

                let image_input = if i < batch.images.len() {
                    batch.images[i].clone()
                } else {
                    ndarray::Array3::<f32>::zeros((0, 0, 0))
                };

                batch_inputs.push((tabular_input, temporal_input, text_input, image_input));
                batch_targets.push(batch.targets[i]);
                batch_weights.push(sample_weight);
            }

            // Zero gradients before batch processing
            model.zero_grad();

            // Process entire batch
            for i in 0..batch_size {
                // Forward pass - returns logits
                let logit_output = model.forward(batch_inputs[i].clone());
                let logit = logit_output[0]; // Single output for binary classification

                batch_logits.push(logit);
            }

            // Compute batch loss and gradients using logits
            let (batch_loss, loss_gradients) =
                weighted_binary_cross_entropy(&batch_logits, &batch_targets, &batch_weights);

            // Backward pass for each sample in batch
            // We need to accumulate gradients across the batch
            for i in 0..batch_size {
                // Re-forward to set up computational graph for this sample
                let _ = model.forward(batch_inputs[i].clone());

                // Backward pass with the gradient for this sample
                model.backward(vec![loss_gradients[i]]);
            }

            // Convert logits to probabilities for metrics
            let batch_predictions: Vec<f32> = batch_logits
                .iter()
                .map(|&z| 1.0 / (1.0 + (-z).exp()))
                .collect();

            // Store for epoch metrics
            epoch_predictions.extend(&batch_predictions);
            epoch_targets.extend(&batch_targets);
            epoch_losses.push(batch_loss);

            // Debug logging for first batch of first epoch
            if epoch == 0 && total_batches == 1 {
                for i in 0..3.min(batch_size) {
                    info!(
                        "Sample {}: logit={:.6}, prob={:.6}, target={}, loss_grad={:.6}",
                        i,
                        batch_logits[i],
                        batch_predictions[i],
                        batch_targets[i],
                        loss_gradients[i]
                    );
                }
            }

            // Check prediction variance (important for debugging)
            let pred_variance = if batch_predictions.len() > 1 {
                let mean = batch_predictions.iter().sum::<f32>() / batch_predictions.len() as f32;
                let variance = batch_predictions
                    .iter()
                    .map(|p| (p - mean).powi(2))
                    .sum::<f32>()
                    / batch_predictions.len() as f32;
                variance
            } else {
                0.0
            };

            if epoch == 0 || (epoch + 1) % 10 == 0 {
                info!(
                    "Epoch {} batch {}: loss={:.4}, pred_variance={:.8}",
                    epoch + 1,
                    total_batches,
                    batch_loss,
                    pred_variance
                );
            }

            // Get accumulated gradients
            let mut model_grads = model.gradients();

            // Average gradients over batch size
            for grad in &mut model_grads {
                *grad /= batch_size as f32;
            }

            // Gradient clipping for stability
            let grad_norm: f32 = model_grads.iter().map(|g| g * g).sum::<f32>().sqrt();

            if grad_norm < 1e-8 {
                warn!("WARNING: Gradient norm too small: {:.2e}", grad_norm);
            } else if grad_norm > config.gradient_clip_norm {
                // Clip gradients by norm
                let scale = config.gradient_clip_norm / grad_norm;
                for grad in &mut model_grads {
                    *grad *= scale;
                }
                info!(
                    "Clipped gradients: norm was {:.4}, scaled to {:.4}",
                    grad_norm, config.gradient_clip_norm
                );
            }

            // Apply privacy if enabled
            if let Some(ref engine) = privacy_engine {
                engine.privatize_gradients(&mut model_grads, config.gradient_clip_norm);

                // Check if privacy noise is too large
                let noisy_grad_norm: f32 = model_grads.iter().map(|g| g * g).sum::<f32>().sqrt();
                if noisy_grad_norm / grad_norm > 10.0 {
                    warn!(
                        "Privacy noise may be too large: original_norm={:.4}, noisy_norm={:.4}",
                        grad_norm, noisy_grad_norm
                    );
                }
            }

            // Apply gradients with current learning rate
            model.apply_gradients(&model_grads, current_lr);

            // Zero gradients after applying
            model.zero_grad();
        }

        // Calculate epoch metrics
        let epoch_auroc = calculate_auroc(&epoch_predictions, &epoch_targets);
        let mean_loss = epoch_losses.iter().sum::<f32>() / epoch_losses.len() as f32;

        // Log epoch statistics
        if epoch == 0 || (epoch + 1) % 5 == 0 {
            let mean_pred = epoch_predictions.iter().sum::<f32>() / epoch_predictions.len() as f32;
            let std_pred = {
                let variance = epoch_predictions
                    .iter()
                    .map(|p| (p - mean_pred).powi(2))
                    .sum::<f32>()
                    / epoch_predictions.len() as f32;
                variance.sqrt()
            };

            info!(
                "Epoch {}/{}: loss={:.4}, auroc={:.4}, mean_pred={:.6}, std_pred={:.6}, lr={:.6}",
                epoch + 1,
                config.epochs,
                mean_loss,
                epoch_auroc,
                mean_pred,
                std_pred,
                current_lr
            );
        }

        // Evaluate on validation set periodically
        if (epoch + 1) % config.val_frequency == 0 {
            let (train_metrics, _) =
                evaluate_model_with_calibration(model, train_loader, train_weights, calibrator);

            let (val_metrics, _) =
                evaluate_model_with_calibration(model, val_loader, val_weights, calibrator);

            info!(
                "Epoch {}/{}: Train loss={:.4}, acc={:.2}%, auroc={:.4} | Val loss={:.4}, acc={:.2}%, auroc={:.4}",
                epoch + 1, config.epochs,
                train_metrics.loss, train_metrics.accuracy * 100.0, train_metrics.auroc,
                val_metrics.loss, val_metrics.accuracy * 100.0, val_metrics.auroc
            );

            // Early stopping check
            if train_metrics.auroc < 0.51 && epoch > 10 {
                warn!("Model not learning (AUROC ~0.5), consider adjusting parameters");
            }
        }
    }
}

/// Evaluate the model with sample weights and calibration, returning a full set of metrics.
pub fn evaluate_model_with_calibration(
    model: &mut LateFusionNet,
    loader: &mut impl DataLoader<MultiModalBatch>,
    sample_weights: &[f32],
    calibrator: &BayesianCalibrator,
) -> (EvaluationMetrics, Vec<f32>) {
    loader.reset();

    // Return metrics AND calibration data
    let mut total_loss = 0.0;

    // Need to collect all predictions and targets to calculate metrics like AUROC
    let mut all_logits = Vec::new();
    let mut all_targets = Vec::new();
    let mut all_weights = Vec::new();
    let mut sample_idx = 0;

    while let Some(batch) = loader.next_batch(32) {
        let batch_size = batch.targets.len();

        for i in 0..batch_size {
            let sample_weight = if sample_idx < sample_weights.len() {
                sample_weights[sample_idx]
            } else {
                1.0
            };
            sample_idx += 1;

            // Prepare inputs
            let tabular_input = if !batch.tabular.is_empty() {
                batch.tabular[i].clone()
            } else {
                vec![] // Empty vector for disabled modality
            };

            let temporal_input = if !batch.temporal.is_empty() {
                batch.temporal[i].clone()
            } else {
                vec![] // Empty sequence for disabled modality
            };

            let text_input = if !batch.text.is_empty() {
                batch.text[i].clone()
            } else {
                vec![] // Empty token sequence for disabled modality
            };

            let image_input = if !batch.images.is_empty() {
                batch.images[i].clone()
            } else {
                ndarray::Array3::<f32>::zeros((0, 0, 0)) // Empty 3D array for disabled modality
            };

            let logit = model.forward((tabular_input, temporal_input, text_input, image_input))[0];

            all_logits.push(logit);
            all_targets.push(batch.targets[i]);
            all_weights.push(sample_weight);
        }
    }

    // Convert logits to probabilities
    let all_predictions: Vec<f32> = all_logits
        .iter()
        .map(|&z| 1.0 / (1.0 + (-z).exp()))
        .collect();

    // Calculate loss using logits
    let (loss, _) = weighted_binary_cross_entropy(&all_logits, &all_targets, &all_weights);

    // AUROC should be calculated on the raw, uncalibrated scores to measure
    // the model's intrinsic ability to rank predictions.
    let auroc = calculate_auroc(&all_predictions, &all_targets);

    // Apply calibration to probabilities
    let calibrated_predictions: Vec<f32> = all_predictions
        .iter()
        .map(|&p| calibrator.calibrate(p))
        .collect();

    // --- METRIC CALCULATIONS ---
    // Calculate metrics using calibrated probabilities
    // Use a threshold of 0.5 on the *calibrated* predictions for classification metrics
    let confusion = ConfusionMatrix::from_predictions(&calibrated_predictions, &all_targets, 0.5);

    let accuracy = (confusion.tp + confusion.tn) as f32
        / (confusion.tp + confusion.tn + confusion.fp + confusion.fn_) as f32;

    let ppv = confusion.ppv();
    let npv = confusion.npv();

    let sensitivity = if confusion.tp + confusion.fn_ > 0 {
        confusion.tp as f32 / (confusion.tp + confusion.fn_) as f32
    } else {
        0.0
    };

    let specificity = if confusion.tn + confusion.fp > 0 {
        confusion.tn as f32 / (confusion.tn + confusion.fp) as f32
    } else {
        0.0
    };

    let metrics = EvaluationMetrics::new(loss, accuracy, ppv, npv, sensitivity, specificity, auroc);

    // You can now print or plot this calibration_data. A perfect calibration
    // would have pairs like (0.1, 0.1), (0.2, 0.2), etc.

    (metrics, calibrated_predictions)
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
        let batch_size = batch.targets.len();

        for i in 0..batch_size {
            // Get the cause of mortality
            let cause = if sample_idx < causes.len() {
                causes[sample_idx]
            } else {
                4 // Default to \"other causes\"
            };

            sample_idx += 1;

            let tabular_input = if !batch.tabular.is_empty() {
                batch.tabular[i].clone()
            } else {
                vec![] // Empty vector for disabled modality
            };

            let temporal_input = if !batch.temporal.is_empty() {
                batch.temporal[i].clone()
            } else {
                vec![] // Empty sequence for disabled modality
            };

            let text_input = if !batch.text.is_empty() {
                batch.text[i].clone()
            } else {
                vec![] // Empty token sequence for disabled modality
            };

            let image_input = if !batch.images.is_empty() {
                batch.images[i].clone()
            } else {
                ndarray::Array3::<f32>::zeros((0, 0, 0)) // Empty 3D array for disabled modality
            };

            // Forward pass
            let prediction =
                model.forward((tabular_input, temporal_input, text_input, image_input))[0];

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
