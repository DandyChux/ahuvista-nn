use tracing::{debug, info, warn};

use crate::{
    config::TrainingParams,
    core::{Module, PrivacyEngine, fusion::LateFusionNet},
    data::{DataLoader, MultiModalBatch, loader::MultiModalDataLoader},
    training::{
        balancing::{BayesianCalibrator, weighted_binary_cross_entropy},
        loss::binary_cross_entropy,
        metrics::{ConfusionMatrix, EvaluationMetrics, calculate_auprc, calculate_auroc},
        optimizer::Adam,
    },
};

/// Train the model with sample weights and calibration
pub fn train_model_with_weights(
    model: &mut LateFusionNet,
    train_loader: &mut MultiModalDataLoader,
    val_loader: &mut impl DataLoader<MultiModalBatch>,
    config: &TrainingParams,
    train_weights: &mut [f32],
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

    // Initialize Adam optimizer — determine parameter count from the gradient vector
    model.zero_grad();

    let num_params = model.gradients().len();
    let mut optimizer =
        Adam::with_weight_decay(num_params, config.learning_rate, config.weight_decay);

    info!(
        "Adam optimizer initialized: {} parameters, lr={}, weight_decay={}",
        num_params, config.learning_rate, config.weight_decay
    );

    let mut total_batches: usize = 0;

    for epoch in 0..config.epochs {
        // Shuffle training data at the start of each epoch
        train_loader.shuffle();

        // Training phase
        train_loader.reset();

        let mut epoch_predictions: Vec<f32> = Vec::new();
        let mut epoch_targets: Vec<f32> = Vec::new();
        let mut epoch_logits: Vec<f32> = Vec::new();
        let mut epoch_weights: Vec<f32> = Vec::new();
        let mut epoch_losses: Vec<f32> = Vec::new();

        // Process batches
        while let Some(batch) = train_loader.next_batch(config.batch_size) {
            let batch_size = batch.targets.len();
            total_batches += 1;

            // Zero gradients before batch processing
            model.zero_grad();

            let mut batch_logits = Vec::with_capacity(batch_size);
            let mut batch_targets = Vec::with_capacity(batch_size);
            let mut batch_weights = Vec::with_capacity(batch_size);
            let mut batch_loss_sum = 0.0_f32;

            // Single-pass: forward, compute per-sample gradient, backward — no re-forward needed
            for i in 0..batch_size {
                let sample_weight = batch
                    .indices
                    .get(i)
                    .and_then(|&index| train_weights.get(index))
                    .copied()
                    .unwrap_or(1.0);

                let sequence_input = batch.sequences[i].clone();

                let text_input = batch
                    .text
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| vec![]);

                let image_input = batch
                    .images
                    .get(i)
                    .and_then(|images| images.first())
                    .cloned()
                    .unwrap_or_else(|| ndarray::Array3::<f32>::zeros((0, 0, 0)));

                let logit_output = model.forward((sequence_input, text_input, image_input));
                let logit = logit_output[0];
                let target = batch.targets[i];

                // Compute probability via sigmoid
                let prob = 1.0 / (1.0 + (-logit).exp());

                // BCE gradient for this sample: (prob - target) * weight
                let grad = (prob - target) * sample_weight;

                // Backward pass — accumulates gradients inside the model
                model.backward(vec![grad]);

                // Numerically stable per-sample BCE loss: max(z,0) - z*y + log(1 + exp(-|z|))
                let sample_loss = if logit >= 0.0 {
                    logit - logit * target + (1.0 + (-logit).exp()).ln()
                } else {
                    -logit * target + (1.0 + logit.exp()).ln()
                };
                batch_loss_sum += sample_loss * sample_weight;

                batch_logits.push(logit);
                batch_targets.push(target);
                batch_weights.push(sample_weight);

                // Debug logging for first 3 samples of first batch
                if epoch == 0 && total_batches == 1 && i < 3 {
                    debug!(
                        "Sample {}: logit={:.6}, prob={:.6}, target={}, loss_grad={:.6}",
                        i, logit, prob, target, grad
                    );
                }
            }

            // Compute mean batch loss (weighted)
            let batch_loss = batch_loss_sum / batch_size as f32;

            // Convert logits to probabilities for metrics
            let batch_predictions: Vec<f32> = batch_logits
                .iter()
                .map(|&z| 1.0 / (1.0 + (-z).exp()))
                .collect();

            // Store for epoch metrics
            epoch_predictions.extend(&batch_predictions);
            epoch_targets.extend(&batch_targets);
            epoch_logits.extend(&batch_logits);
            epoch_weights.extend(&batch_weights);
            epoch_losses.push(batch_loss);

            // Check prediction variance (important for debugging)
            let pred_variance = if batch_predictions.len() > 1 {
                let mean = batch_predictions.iter().sum::<f32>() / batch_predictions.len() as f32;
                batch_predictions
                    .iter()
                    .map(|p| (p - mean).powi(2))
                    .sum::<f32>()
                    / batch_predictions.len() as f32
            } else {
                0.0
            };

            // Log batch progress every 100 batches
            if total_batches % 100 == 0 {
                debug!(
                    "Epoch {} batch {}: loss={:.4}, pred_variance={:.8}",
                    epoch + 1,
                    total_batches,
                    batch_loss,
                    pred_variance
                );
            }

            // Get accumulated gradients and average by weight sum
            let mut model_grads = model.gradients();
            let weight_sum: f32 = batch_weights.iter().sum();
            if weight_sum > 0.0 {
                for grad in &mut model_grads {
                    *grad /= weight_sum;
                }
            }

            // Gradient clipping (applied to raw gradients BEFORE Adam)
            let grad_norm: f32 = model_grads.iter().map(|g| g * g).sum::<f32>().sqrt();

            if grad_norm < 1e-8 {
                warn!("WARNING: Gradient norm too small: {:.2e}", grad_norm);
            } else if grad_norm > config.gradient_clip_norm {
                let scale = config.gradient_clip_norm / grad_norm;
                for grad in &mut model_grads {
                    *grad *= scale;
                }
                debug!(
                    "Clipped gradients: norm was {:.4}, scaled to {:.4}",
                    grad_norm, config.gradient_clip_norm
                );
            }

            // Apply privacy noise if enabled
            if let Some(ref engine) = privacy_engine {
                engine.privatize_gradients(&mut model_grads, config.gradient_clip_norm);

                let noisy_grad_norm: f32 = model_grads.iter().map(|g| g * g).sum::<f32>().sqrt();
                if noisy_grad_norm / grad_norm > 10.0 {
                    warn!(
                        "Privacy noise may be too large: original_norm={:.4}, noisy_norm={:.4}",
                        grad_norm, noisy_grad_norm
                    );
                }
            }

            // Adam step with weight decay: needs current params for decoupled decay
            let current_params = model.get_all_parameters();
            let adam_updates = optimizer.step(&model_grads, &current_params);

            model.apply_gradients(&adam_updates, 1.0); // lr=1.0; Adam already includes the step size

            // Zero gradients after applying
            model.zero_grad();
        }

        // Calculate epoch metrics
        let train_loss = if epoch_losses.is_empty() {
            0.0
        } else {
            epoch_losses.iter().sum::<f32>() / epoch_losses.len() as f32
        };

        // Evaluate on validation set periodically
        if (epoch + 1) % config.val_frequency == 0 {
            let (val_metrics, _) =
                evaluate_model_with_calibration(model, val_loader, val_weights, calibrator);

            info!(
                "Epoch {}/{}: Train loss={:.4} | Val [loss={:.4}, auroc={:.4}, mean={:.6}, std={:.6}] | lr={:.6}",
                epoch + 1,
                config.epochs,
                train_loss,
                val_metrics.loss,
                val_metrics.auroc,
                val_metrics.mean_pred,
                val_metrics.std_pred,
                optimizer.lr()
            );

            // Early stopping check based on validation AUROC
            if val_metrics.auroc < 0.55 && epoch > 10 {
                warn!("Model not learning well (AUROC < 0.55), consider adjusting parameters");
            }
        }

        if config.use_adaptive_lr && (epoch + 1) % config.lr_decay_epochs == 0 {
            let new_lr = (optimizer.lr() * config.lr_decay_rate).max(config.min_learning_rate);
            optimizer.set_lr(new_lr);
            info!(
                "Learning rate decayed to: {:.6} (epoch {})",
                new_lr,
                epoch + 1
            );
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

            let sequence_input = batch.sequences[i].clone();

            let text_input = batch
                .text
                .get(i)
                .cloned()
                .unwrap_or_else(|| vec![]);

            let image_input = batch
                .images
                .get(i)
                .and_then(|images| images.first())
                .cloned()
                .unwrap_or_else(|| ndarray::Array3::<f32>::zeros((0, 0, 0)));

            let logit = model.forward((sequence_input, text_input, image_input))[0];

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

    let auprc = calculate_auprc(&all_predictions, &all_targets);

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

    let mean_prediction =
        calibrated_predictions.iter().sum::<f32>() / calibrated_predictions.len() as f32;

    let std_prediction = calibrated_predictions
        .iter()
        .map(|&x| (x - mean_prediction).powi(2))
        .sum::<f32>()
        / calibrated_predictions.len() as f32;

    let f1_score = if ppv + sensitivity > 0.0 {
        2.0 * ppv * sensitivity / (ppv + sensitivity)
    } else {
        0.0
    };

    let mcc = if confusion.tp + confusion.fp > 0
        && confusion.tp + confusion.fn_ > 0
        && confusion.tn + confusion.fp > 0
        && confusion.tn + confusion.fn_ > 0
    {
        let numerator =
            (confusion.tp * confusion.tn) as f32 - (confusion.fp * confusion.fn_) as f32;
        let denominator = ((confusion.tp + confusion.fp) as f32
            * (confusion.tp + confusion.fn_) as f32
            * (confusion.tn + confusion.fp) as f32
            * (confusion.tn + confusion.fn_) as f32)
            .sqrt();
        numerator / denominator
    } else {
        0.0
    };

    // Calculate average of specifity and sensitivity
    let balanced_accuracy = (sensitivity + specificity) / 2.0;

    let metrics = EvaluationMetrics::new(
        loss,
        accuracy,
        ppv,
        npv,
        sensitivity,
        specificity,
        auroc,
        auprc,
        mean_prediction,
        std_prediction,
        f1_score,
        mcc,
        balanced_accuracy,
    );

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

            let sequence_input = batch.sequences[i].clone();

            let text_input = batch
                .text
                .get(i)
                .cloned()
                .unwrap_or_else(|| vec![]);

            let image_input = batch
                .images
                .get(i)
                .and_then(|images| images.first())
                .cloned()
                .unwrap_or_else(|| ndarray::Array3::<f32>::zeros((0, 0, 0)));

            let prediction = model.forward((sequence_input, text_input, image_input))[0];

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
