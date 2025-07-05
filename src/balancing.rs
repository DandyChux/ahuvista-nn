//! Balancing module for handling class imbalance in mortality prediction.
//!
//! Provides utilities for:
//! - Computing class weights
//! - Stratifying samples by demographic factors
//! - Applying weights during training
//! - Generating synthetic samples for minority classes

use ndarray::Array1;
use rand::Rng;
use std::collections::HashMap;

/// Compute class weights inversely proportional to class frequencies.
///
/// # Arguments
/// * `labels` - Vector of binary labels (0.0 for survival, 1.0 for mortality)
///
/// # Returns
/// HashMap mapping class label to weight value
pub fn compute_class_weights(labels: &[f32]) -> HashMap<f32, f32> {
    // Count occurrences of each class
    let mut class_counts: HashMap<f32, usize> = HashMap::new();

    for &label in labels {
        *class_counts.entry(label).or_insert(0) += 1;
    }

    let total_samples = labels.len();
    let num_classes = class_counts.len();

    // Compute weights inversely proportional to class frequencies
    let mut class_weights = HashMap::new();

    for (&class_label, &count) in &class_counts {
        let weight = total_samples as f32 / (num_classes as f32 * count as f32);
        class_weights.insert(class_label, weight);
    }

    class_weights
}

/// Compute sample weights based on class and optional demographic strata.
///
/// # Arguments
/// * `labels` - Vector of binary labels (0.0 for survival, 1.0 for mortality)
/// * `strata` - Optional vector of stratum identifiers for each sample
/// * `strata_multipliers` - Optional hashmap of stratum-specific multipliers
///
/// # Returns
/// Vector of sample weights matching the input length
pub fn compute_sample_weights(
    labels: &[f32],
    strata: Option<&[String]>,
    strata_multipliers: Option<&HashMap<String, f32>>,
) -> Vec<f32> {
    // Get base class weights
    let class_weights = compute_class_weights(labels);

    // Apply class weights to each sample
    let mut sample_weights = Vec::with_capacity(labels.len());

    for (i, &label) in labels.iter().enumerate() {
        let mut weight = *class_weights.get(&label).unwrap_or(&1.0);

        // Apply stratum-specific multiplier if available
        if let (Some(sample_strata), Some(multipliers)) = (strata, strata_multipliers) {
            if i < sample_strata.len() {
                let stratum = &sample_strata[i];
                if let Some(&multiplier) = multipliers.get(stratum) {
                    weight *= multiplier;
                }
            }
        }

        sample_weights.push(weight);
    }

    sample_weights
}

/// Apply sample weights to binary cross entropy loss.
///
/// # Arguments
/// * `loss` - Original BCE loss value
/// * `weight` - Sample weight
///
/// # Returns
/// Weighted loss value
pub fn apply_weight_to_loss(loss: f32, weight: f32) -> f32 {
    loss * weight
}

/// Apply sample weights to loss gradients.
///
/// # Arguments
/// * `grad` - Original gradient value
/// * `weight` - Sample weight
///
/// # Returns
/// Weighted gradient value
pub fn apply_weight_to_gradient(grad: f32, weight: f32) -> f32 {
    grad * weight
}

/// Structure for handling cause-specific mortality weighting.
pub struct CauseWeights {
    /// Mapping from cause ID to weight multiplier
    cause_weights: HashMap<usize, f32>,

    /// Default weight for causes not explicitly mapped
    default_weight: f32,
}

impl CauseWeights {
    /// Create a new CauseWeights with population-derived weightings.
    ///
    /// # Arguments
    /// * `cause_to_frequency` - Mapping from cause ID to frequency in population
    pub fn from_population_frequencies(cause_to_frequency: &HashMap<usize, f32>) -> Self {
        let mut cause_weights = HashMap::new();
        let total_frequency: f32 = cause_to_frequency.values().sum();

        // Inverse frequency weighting
        for (&cause, &freq) in cause_to_frequency {
            let relative_freq = freq / total_frequency;
            // Small causes get higher weights
            cause_weights.insert(cause, 1.0 / relative_freq.max(0.01)); // Avoid division by zero
        }

        CauseWeights {
            cause_weights,
            default_weight: 1.0,
        }
    }

    /// Apply cause-specific weight to a sample weight.
    ///
    /// # Arguments
    /// * `base_weight` - Original sample weight
    /// * `cause_id` - ID of the mortality cause
    ///
    /// # Returns
    /// Adjusted weight incorporating cause importance
    pub fn apply(&self, base_weight: f32, cause_id: usize) -> f32 {
        let cause_multiplier = self
            .cause_weights
            .get(&cause_id)
            .unwrap_or(&self.default_weight);
        base_weight * cause_multiplier
    }
}

/// Synthetic minority oversampling technique (SMOTE) for Rust.
///
/// Generates synthetic samples for minority classes to address class imbalance.
pub struct SmoteSynthesizer {
    /// Number of nearest neighbors to consider
    k_neighbors: usize,
}

impl SmoteSynthesizer {
    /// Create a new SMOTE synthesizer.
    ///
    /// # Arguments
    /// * `k_neighbors` - Number of nearest neighbors to use (typically 5)
    pub fn new(k_neighbors: usize) -> Self {
        SmoteSynthesizer {
            k_neighbors: k_neighbors.max(1), // Ensure at least 1 neighbor
        }
    }

    /// Generate synthetic samples for the minority class.
    ///
    /// # Arguments
    /// * `minority_samples` - Tabular features of minority class samples
    /// * `n_synthetic` - Number of synthetic samples to generate
    ///
    /// # Returns
    /// Vector of synthetic samples
    pub fn generate_synthetic_samples(
        &self,
        minority_samples: &[Vec<f32>],
        n_synthetic: usize,
    ) -> Vec<Vec<f32>> {
        if minority_samples.is_empty() || n_synthetic == 0 {
            return Vec::new();
        }

        let mut rng = rand::thread_rng();
        let mut synthetic_samples = Vec::with_capacity(n_synthetic);

        // Simple SMOTE implementation (for demonstration)
        for _ in 0..n_synthetic {
            // Random sample from minority class
            let sample_idx = rng.gen_range(0..minority_samples.len());
            let sample = &minority_samples[sample_idx];

            // Find a "neighbor" (randomly for simplicity)
            let neighbor_idx = loop {
                let idx = rng.gen_range(0..minority_samples.len());
                if idx != sample_idx {
                    break idx;
                }
            };
            let neighbor = &minority_samples[neighbor_idx];

            // Generate synthetic sample between sample and neighbor
            let mut synthetic = Vec::with_capacity(sample.len());

            for i in 0..sample.len() {
                let diff = neighbor[i] - sample[i];
                let gap = rng.gen::<f32>(); // Random number between 0 and 1
                synthetic.push(sample[i] + gap * diff);
            }

            synthetic_samples.push(synthetic);
        }

        synthetic_samples
    }
}

/// Bayesian prior adjustment for model calibration using population statistics.
pub struct BayesianCalibrator {
    /// Population prevalence of the condition
    population_prevalence: f32,
}

impl BayesianCalibrator {
    /// Create a new Bayesian calibrator.
    ///
    /// # Arguments
    /// * `population_prevalence` - Population-level prevalence (e.g., 0.001 for 0.1%)
    pub fn new(population_prevalence: f32) -> Self {
        BayesianCalibrator {
            population_prevalence: population_prevalence.max(1e-6).min(1.0 - 1e-6),
        }
    }

    /// Calibrate a model prediction using population prevalence.
    ///
    /// # Arguments
    /// * `raw_prediction` - Uncalibrated model output
    ///
    /// # Returns
    /// Calibrated prediction probability
    pub fn calibrate(&self, raw_prediction: f32) -> f32 {
        // Clip prediction to avoid numerical issues
        let p = raw_prediction.max(1e-6).min(1.0 - 1e-6);

        // Simple Bayesian adjustment (simplified for demonstration)
        // In a real implementation, you would use proper Bayesian methods
        let odds_ratio = p / (1.0 - p);
        let pop_odds = self.population_prevalence / (1.0 - self.population_prevalence);

        // Adjust odds by population prevalence
        let adjusted_odds = odds_ratio * pop_odds;

        // Convert back to probability
        adjusted_odds / (1.0 + adjusted_odds)
    }
}

/// Modify loss calculation with per-sample weights.
///
/// # Arguments
/// * `model_predictions` - Raw model outputs (probabilities)
/// * `targets` - Target values (0.0 or 1.0)
/// * `sample_weights` - Weight for each sample
///
/// # Returns
/// (weighted_loss, weighted_gradients)
pub fn weighted_binary_cross_entropy(
    model_predictions: &[f32],
    targets: &[f32],
    sample_weights: &[f32],
) -> (f32, Vec<f32>) {
    assert_eq!(
        model_predictions.len(),
        targets.len(),
        "Predictions and targets must have same length"
    );
    assert_eq!(
        model_predictions.len(),
        sample_weights.len(),
        "Predictions and weights must have same length"
    );

    let mut total_weighted_loss = 0.0;
    let mut weighted_gradients = Vec::with_capacity(model_predictions.len());

    for i in 0..model_predictions.len() {
        let pred = model_predictions[i].max(1e-7).min(1.0 - 1e-7); // Clip prediction
        let target = targets[i];
        let weight = sample_weights[i];

        // Calculate BCE loss for this sample
        let loss = -target * pred.ln() - (1.0 - target) * (1.0 - pred).ln();
        let weighted_loss = loss * weight;
        total_weighted_loss += weighted_loss;

        // Calculate gradient and apply weight
        let grad = (pred - target) / (pred * (1.0 - pred));
        let weighted_grad = grad * weight;
        weighted_gradients.push(weighted_grad);
    }

    let mean_weighted_loss = total_weighted_loss / model_predictions.len() as f32;

    (mean_weighted_loss, weighted_gradients)
}
