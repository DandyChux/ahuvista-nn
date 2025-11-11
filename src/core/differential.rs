use rand_distr::{Distribution, Normal};

pub struct PrivacyEngine {
    noise_multiplier: f32, // Proportional to 1/epsilon
}

impl PrivacyEngine {
    pub fn new(
        epsilon: f32,
        delta: f32,
        num_samples: usize,
        batch_size: usize,
        epochs: usize,
    ) -> Self {
        // This function calculates the noise multiplier (`sigma` in DP literature)
        // needed to achieve a given (epsilon, delta) privacy budget for DP-SGD.
        // The calculation uses an approximation based on standard DP composition theorems.

        // For a more precise calculation, a dedicated privacy accountant library
        // implementing methods like Rényi Differential Privacy (RDP) is recommended.
        // This implementation provides a reasonable estimate for many common scenarios.

        if epsilon <= 0.0 {
            panic!("Privacy budget epsilon must be positive.");
        }
        if delta <= 0.0 || delta >= 1.0 {
            panic!("Privacy budget delta must be in the range (0, 1).");
        }
        if num_samples == 0 || batch_size == 0 || epochs == 0 {
            panic!("num_samples, batch_size, and epochs must be positive.");
        }
        if batch_size > num_samples {
            panic!("batch_size cannot be greater than num_samples.");
        }

        // Sampling probability of a single data point in a batch.
        let sampling_rate = batch_size as f32 / num_samples as f32;

        // Total number of steps (gradient updates) across all epochs.
        let steps = (epochs as f32 * (num_samples as f32 / batch_size as f32)).ceil() as f32;

        // This approximation for the noise multiplier is derived from the privacy
        // amplification by subsampling and the advanced composition theorem for the
        // Gaussian mechanism.
        // The formula is approximately: noise_multiplier ~ q * sqrt(2 * T * log(1/delta)) / epsilon,
        // where q is the sampling_rate and T is the number of steps.
        // This provides an estimate of the required noise for the given privacy parameters.
        let noise_multiplier =
            (sampling_rate * (2.0 * steps * (1.0 / delta).ln()).sqrt()) / epsilon;

        Self { noise_multiplier }
    }

    /// Clips and adds noise to a set of gradients.
    pub fn privatize_gradients(&self, gradients: &mut Vec<f32>, l2_norm_clip: f32) {
        // Step 1: Gradient Clipping (limits the influence of a single sample)
        let l2_norm = gradients.iter().map(|g| g.powi(2)).sum::<f32>().sqrt();
        if l2_norm > l2_norm_clip {
            for grad in gradients.iter_mut() {
                *grad = (*grad * l2_norm_clip) / l2_norm;
            }
        }

        // Step 2: Add Gaussian Noise
        let noise_std_dev = self.noise_multiplier * l2_norm_clip;
        let noise_dist = Normal::new(0.0, noise_std_dev).unwrap();
        let mut rng = rand::rng();

        for grad in gradients.iter_mut() {
            *grad += noise_dist.sample(&mut rng);
        }
    }
}
