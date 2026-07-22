//! Balancing module for handling class imbalance in mortality prediction.
//!
//! Provides utilities for:
//! - Computing class weights
//! - Stratifying samples by demographic factors
//! - Applying weights during training
//! - Generating synthetic samples for minority classes

use ndarray::{Array1, Array3};
use rand::Rng;
use std::collections::HashMap;

/// Compute class weights inversely proportional to class frequencies.
///
/// # Arguments
/// * `labels` - Vector of binary labels (0.0 for survival, 1.0 for mortality)
///
/// # Returns
/// HashMap mapping class label to weight value
pub fn compute_class_weights(labels: &[f32]) -> HashMap<i32, f32> {
    // Count occurrences of each class
    let mut class_counts: HashMap<i32, usize> = HashMap::new();

    for &label in labels {
        // Convert f32 to i32 for use as HashMap key (0 or 1 for binary classification)
        let class_key = label.round() as i32;
        *class_counts.entry(class_key).or_insert(0) += 1;
    }

    let total_samples = labels.len();
    let num_classes = class_counts.len();

    // Compute weights inversely proportional to class frequencies
    let mut class_weights = HashMap::new();

    for (&class_label, &count) in &class_counts {
        // Use sqrt(total/count) instead of raw inverse frequency
        let weight = (total_samples as f32 / (num_classes as f32 * count as f32)).sqrt();
        // Cap weight ratio to prevent extreme gradients
        let capped_weight = weight.min(3.0);
        class_weights.insert(class_label, capped_weight);
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
        let class_key = label.round() as i32;
        let mut weight = *class_weights.get(&class_key).unwrap_or(&1.0);

        // Apply stratum-specific multiplier if available
        if let (Some(sample_strata), Some(multipliers)) = (strata, strata_multipliers) {
            if i < sample_strata.len() {
                let stratum = &sample_strata[i];
                if let Some(&multiplier) = multipliers.get(stratum) {
                    weight *= multiplier;
                }
            }
        }

        // Cap final weight to prevent extreme gradients
        let capped_weight = weight.min(10.0);
        sample_weights.push(capped_weight);
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
            cause_weights.insert(cause, (1.0 / relative_freq.max(0.01)).sqrt());
        }

        // Compute mean weight for unseen causes
        // This ensures unseen causes are treated as "average" importance
        let mean_weight = if cause_weights.is_empty() {
            1.0
        } else {
            cause_weights.values().sum::<f32>() / cause_weights.len() as f32
        };

        CauseWeights {
            cause_weights,
            default_weight: mean_weight,
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

    /// Apply cause-specific weights to base sample weights (only for positive/mortality samples)
    ///
    /// # Arguments
    /// * `base_weights` - Base sample weights from class/strata weighting
    /// * `causes` - Cause ID for each sample
    /// * `cause_weights` - CauseWeights instance with cause-specific multipliers
    /// * `targets` - Target values (0.0 for survival, 1.0 for mortality)
    /// * `max_weight` - Optional cap on final weight to prevent extreme gradients
    ///
    /// # Returns
    /// Vector of final sample weights
    pub fn apply_cause_weights(
        base_weights: &[f32],
        causes: &[usize],
        cause_weights: &CauseWeights,
        targets: &[f32],
        max_weight: Option<f32>,
    ) -> Vec<f32> {
        base_weights
            .iter()
            .zip(causes.iter())
            .zip(targets.iter())
            .map(|((&base_weight, &cause), &target)| {
                let weight = if target > 0.5 {
                    // Only apply cause weights to mortality cases
                    cause_weights.apply(base_weight, cause)
                } else {
                    // Survival cases keep base weight (class + strata only)
                    base_weight
                };
                // Apply optional cap
                match max_weight {
                    Some(cap) => weight.min(cap),
                    None => weight,
                }
            })
            .collect()
    }

    /// Generate synthetic samples for minority class using SMOTE
    pub fn generate_synthetic_samples(
        &self,
        tabular_data: &[Vec<f32>],
        temporal_data: &[Vec<Vec<f32>>],
        text_data: &[Vec<usize>],
        image_data: &[Array3<f32>],
        targets: &[f32],
        n_synthetic: usize,
    ) -> (
        Vec<Vec<f32>>,
        Vec<Vec<Vec<f32>>>,
        Vec<Vec<usize>>,
        Vec<Array3<f32>>,
        Vec<f32>,
    ) {
        // Identify minority class samples (assuming 1.0 is the minority class)
        let minority_indices: Vec<usize> = targets
            .iter()
            .enumerate()
            .filter(|&(_, &target)| target > 0.5)
            .map(|(i, _)| i)
            .collect();

        if minority_indices.is_empty() {
            // No minority samples to synthesize from
            return (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        }

        // Extract minority class tabular data
        let minority_tabular: Vec<Vec<f32>> = minority_indices
            .iter()
            .map(|&i| tabular_data[i].clone())
            .collect();

        // Generate synthetic tabular samples
        let synthetic_tabular = self.generate_synthetic_tabular(&minority_tabular, n_synthetic);

        // For simplicity, we'll just duplicate existing temporal, text and image data
        // In a real implementation, you would use modality-specific generation methods
        let mut synthetic_temporal = Vec::with_capacity(n_synthetic);
        let mut synthetic_text = Vec::with_capacity(n_synthetic);
        let mut synthetic_images = Vec::with_capacity(n_synthetic);

        for i in 0..n_synthetic {
            // Pick a random minority sample to duplicate
            let source_idx = minority_indices[i % minority_indices.len()];

            synthetic_temporal.push(temporal_data[source_idx].clone());
            synthetic_text.push(text_data[source_idx].clone());
            synthetic_images.push(image_data[source_idx].clone());
        }

        // All synthetic samples have target = 1.0
        let synthetic_targets = vec![1.0; n_synthetic];

        (
            synthetic_tabular,
            synthetic_temporal,
            synthetic_text,
            synthetic_images,
            synthetic_targets,
        )
    }

    /// Generate synthetic tabular samples using SMOTE algorithm
    fn generate_synthetic_tabular(
        &self,
        minority_samples: &[Vec<f32>],
        n_synthetic: usize,
    ) -> Vec<Vec<f32>> {
        let mut rng = rand::rng();
        let mut synthetic_samples = Vec::with_capacity(n_synthetic);

        if minority_samples.is_empty() {
            return synthetic_samples;
        }

        for _ in 0..n_synthetic {
            // Pick a random minority sample
            let sample_idx = rng.random_range(0..minority_samples.len());
            let base_sample = &minority_samples[sample_idx];

            // Pick another random minority sample for interpolation
            let neighbor_idx = rng.random_range(0..minority_samples.len());
            let neighbor_sample = &minority_samples[neighbor_idx];

            // Generate synthetic sample by interpolation
            let alpha = rng.random::<f32>();
            let synthetic_sample: Vec<f32> = base_sample
                .iter()
                .zip(neighbor_sample.iter())
                .map(|(&base, &neighbor)| base + alpha * (neighbor - base))
                .collect();

            synthetic_samples.push(synthetic_sample);
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

        // Bayes' theorem:
        // P(Y=1|X, π_p) = P(X|Y=1) * π_p / P(X)
        //
        // The likelihood ratio (LR) can be derived from the model's prediction:
        // LR = P(X|Y=1) / P(X|Y=0) = [p / (1-p)] * [(1-π_s) / π_s]
        //
        // Then the calibrated probability is:
        // P(Y=1|X, π_p) = LR * π_p / (LR * π_p + (1 - π_p))
        //
        // Convert model prediction to likelihood ratio
        // LR = p / (1 - p) when trained on balanced data
        let likelihood_ratio = p / (1.0 - p);

        // Apply Bayes' theorem with population prevalence as prior
        // P(Y=1|X) = LR * P(Y=1) / (LR * P(Y=1) + P(Y=0))
        //          = LR * π / (LR * π + (1 - π))
        let numerator = likelihood_ratio * self.population_prevalence;
        let denominator = numerator + (1.0 - self.population_prevalence);

        // Return calibrated posterior probability
        let calibrated = numerator / denominator;

        // Ensure output is in valid probability range
        calibrated.max(1e-6).min(1.0 - 1e-6)
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
    logits: &[f32],
    targets: &[f32],
    sample_weights: &[f32],
) -> (f32, Vec<f32>) {
    assert_eq!(logits.len(), targets.len());
    assert_eq!(logits.len(), sample_weights.len());

    let mut total_weighted_loss = 0.0;
    let mut weighted_gradients = Vec::with_capacity(logits.len());

    for i in 0..logits.len() {
        let z = logits[i];
        let target = targets[i];
        let weight = sample_weights[i];

        // Debug: check for invalid values
        // if i < 5 || z.is_nan() || z.is_infinite() || weight < 0.0 {
        //     tracing::debug!(
        //         "BCE sample {}: z={}, target={}, weight={}",
        //         i,
        //         z,
        //         target,
        //         weight
        //     );
        // }

        // Numerically stable BCE loss calculation
        // loss = max(z, 0) - z * y + log(1 + exp(-|z|))
        let loss = if z >= 0.0 {
            z - z * target + (1.0 + (-z).exp()).ln()
        } else {
            -z * target + (1.0 + z.exp()).ln()
        };

        // Debug: check for negative individual loss
        if loss < 0.0 {
            tracing::warn!("Negative loss! z={}, target={}, loss={}", z, target, loss);
        }

        let weighted_loss = loss * weight;
        total_weighted_loss += weighted_loss;

        // Calculate gradient: p - y where p = sigmoid(z)
        let p = 1.0 / (1.0 + (-z).exp());
        let grad = p - target;
        let weighted_grad = grad * weight;
        weighted_gradients.push(weighted_grad);
    }

    let mean_weighted_loss = total_weighted_loss / logits.len() as f32;

    (mean_weighted_loss, weighted_gradients)
}
