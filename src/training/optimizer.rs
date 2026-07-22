/// Adam optimizer (Adaptive Moment Estimation).
///
/// Maintains per-parameter first and second moment estimates to adapt
/// the learning rate for each weight individually. This allows the optimizer
/// to make larger updates for infrequent parameters and smaller updates for
/// frequent ones, which is critical for escaping base-rate collapse in
/// imbalanced classification.
///
/// Reference: Kingma & Ba, "Adam: A Method for Stochastic Optimization", 2015
#[derive(Debug)]
pub struct Adam {
    /// Base learning rate (step size)
    lr: f32,

    /// Exponential decay rate for first moment estimates
    beta1: f32,

    /// Exponential decay rate for second moment estimates
    beta2: f32,

    /// Small constant for numerical stability
    epsilon: f32,

    /// Weight decay coefficient. 0.0 = no regularization
    weight_decay: f32,

    /// First moment vector (mean of gradients)
    m: Vec<f32>,

    /// Second moment vector (mean of squared gradients)
    v: Vec<f32>,

    /// Timestep counter (incremented each call to `step`)
    t: usize,
}

impl Adam {
    /// Create a new Adam optimizer.
    ///
    /// # Arguments
    /// * `num_params` - Number of model parameters (length of gradient vector)
    /// * `lr` - Learning rate (typical: 0.001)
    /// * `beta1` - First moment decay (typical: 0.9)
    /// * `beta2` - Second moment decay (typical: 0.999)
    /// * `epsilon` - Numerical stability constant (typical: 1e-8)
    pub fn new(num_params: usize, lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon,
            weight_decay: 0.0,
            m: vec![0.0; num_params],
            v: vec![0.0; num_params],
            t: 0,
        }
    }

    /// Create an Adam optimizer with standard defaults (β1=0.9, β2=0.999, ε=1e-8).
    pub fn with_defaults(num_params: usize, lr: f32) -> Self {
        Self::new(num_params, lr, 0.9, 0.999, 1e-8)
    }

    /// Create an AdamW optimizer with weight decay.
    pub fn with_weight_decay(num_params: usize, lr: f32, weight_decay: f32) -> Self {
        let mut opt = Self::with_defaults(num_params, lr);
        opt.weight_decay = weight_decay;
        opt
    }

    /// Compute Adam-adjusted parameter updates from raw gradients.
    ///
    /// Returns a vector of updates that should be applied via
    /// `model.apply_gradients(&updates, 1.0)` — the learning rate is already
    /// baked into the returned values.
    ///
    /// # Arguments
    /// * `gradients` - Raw (averaged, optionally clipped) gradients from backprop
    pub fn step(&mut self, gradients: &[f32], params: &[f32]) -> Vec<f32> {
        assert_eq!(
            gradients.len(),
            self.m.len(),
            "Gradient length ({}) does not match optimizer state ({})",
            gradients.len(),
            self.m.len()
        );

        self.t += 1;

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        let mut updates = Vec::with_capacity(gradients.len());

        for i in 0..gradients.len() {
            let g = gradients[i];

            // Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;

            // Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            // Bias-corrected moment estimates
            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;

            // Adam update: lr * m_hat / (sqrt(v_hat) + epsilon)
            // AdamW: adam_update + weight_decay * current_param
            let adam_update = self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
            let decay_update = self.lr * self.weight_decay * params[i];

            updates.push(adam_update + decay_update);
        }

        updates
    }

    /// Update the learning rate (e.g., for LR scheduling on top of Adam).
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    /// Get the current learning rate.
    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// Get the current timestep.
    pub fn timestep(&self) -> usize {
        self.t
    }
}
