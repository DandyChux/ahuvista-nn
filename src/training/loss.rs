//! Loss functions and gradient calculations.
//!
//! Provides implementations of various loss functions for training
//! the neural network, with a focus on binary cross-entropy for
//! classification tasks.

/// Binary cross-entropy loss.
///
/// # Arguments
/// * `prediction` - Model output (predicted probability)
/// * `target` - Target value (0.0 or 1.0)
///
/// # Returns
/// The BCE loss value
pub fn binary_cross_entropy(prediction: f32, target: f32) -> f32 {
    // Clip prediction to avoid log(0) or log(1)
    let pred_clipped = prediction.max(1e-7).min(1.0 - 1e-7);

    -target * pred_clipped.ln() - (1.0 - target) * (1.0 - pred_clipped).ln()
}

/// Gradient of binary cross-entropy loss with respect to the prediction.
///
/// # Arguments
/// * `prediction` - Model output (predicted probability)
/// * `target` - Target value (0.0 or 1.0)
///
/// # Returns
/// The gradient value
pub fn binary_cross_entropy_gradient(prediction: f32, target: f32) -> f32 {
    // Clip prediction to avoid division by zero
    let pred_clipped = prediction.max(1e-7).min(1.0 - 1e-7);

    // ∂BCE/∂pred = (pred - target) / (pred * (1 - pred))
    (pred_clipped - target) / (pred_clipped * (1.0 - pred_clipped))
}

/// Mean squared error loss.
///
/// # Arguments
/// * `predictions` - Model outputs
/// * `targets` - Target values
///
/// # Returns
/// The MSE loss value
pub fn mean_squared_error(predictions: &[f32], targets: &[f32]) -> f32 {
    assert_eq!(
        predictions.len(),
        targets.len(),
        "Predictions and targets must have same length"
    );

    let sum_squared_error: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&pred, &target)| (pred - target).powi(2))
        .sum();

    sum_squared_error / predictions.len() as f32
}

/// Gradient of mean squared error loss with respect to predictions.
///
/// # Arguments
/// * `predictions` - Model outputs
/// * `targets` - Target values
///
/// # Returns
/// Vector of gradient values
pub fn mean_squared_error_gradient(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    assert_eq!(
        predictions.len(),
        targets.len(),
        "Predictions and targets must have same length"
    );

    predictions
        .iter()
        .zip(targets.iter())
        .map(|(&pred, &target)| 2.0 * (pred - target) / predictions.len() as f32)
        .collect()
}

/// Calculate accuracy for binary classification.
///
/// # Arguments
/// * `predictions` - Model outputs (probabilities)
/// * `targets` - Target values (0.0 or 1.0)
/// * `threshold` - Classification threshold (default 0.5)
///
/// # Returns
/// Accuracy as a value between 0.0 and 1.0
pub fn binary_accuracy(predictions: &[f32], targets: &[f32], threshold: f32) -> f32 {
    assert_eq!(
        predictions.len(),
        targets.len(),
        "Predictions and targets must have same length"
    );

    if predictions.is_empty() {
        return 0.0;
    }

    let correct_count = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(&pred, &target)| {
            let pred_class = if pred >= threshold { 1.0 } else { 0.0 };
            (pred_class - target).abs() < 1e-6
        })
        .count();

    correct_count as f32 / predictions.len() as f32
}
