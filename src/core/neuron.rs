use crate::core::Module;
use rand::Rng;

#[derive(Debug)]
pub enum WeightType {
    F32(Vec<f32>),
    I8 {
        quantized_weights: Vec<i8>,
        scale: f32, // De-quantization scale
    },
}

impl WeightType {
    pub fn len(&self) -> usize {
        match self {
            WeightType::F32(weights) => weights.len(),
            WeightType::I8 {
                quantized_weights, ..
            } => quantized_weights.len(),
        }
    }
}

impl Iterator for WeightType {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            WeightType::F32(weights) => weights.pop(),
            WeightType::I8 {
                quantized_weights,
                scale,
            } => quantized_weights.pop().map(|w| w as f32 * *scale),
        }
    }
}

/// The basic computational unit of a neural network.
///
/// A `Neuron` has weights for each input, a bias term, and applies
/// a sigmoid activation function to produce its output.
/// Also stores intermediate values and gradients needed for backpropagation.
#[derive(Debug)]
pub struct Neuron {
    /// Weights for each input connection
    pub weights: WeightType,

    /// Bias term
    pub bias: f32,

    /// Cached input from last forward pass
    last_input: Vec<f32>,

    /// Cached pre-activation value from last forward pass
    last_z: f32,

    /// Accumulated gradients for weights
    grad_w: Vec<f32>,

    /// Accumulated gradient for bias
    grad_b: f32,
}

impl Neuron {
    /// Create a new neuron with the specified number of inputs.
    ///
    /// Weights and bias are initialized randomly in the range [-0.5, 0.5],
    /// which helps prevent saturation early in training.
    ///
    /// # Arguments
    /// * `num_inputs` - The number of input connections
    pub fn new(num_inputs: usize) -> Self {
        let mut rng = rand::rng();

        Neuron {
            weights: WeightType::F32(
                (0..num_inputs)
                    .map(|_| rng.random_range(-0.5..0.5))
                    .collect(),
            ),
            bias: rng.random_range(-0.5..0.5),
            last_input: vec![0.0; num_inputs],
            last_z: 0.0,
            grad_w: vec![0.0; num_inputs],
            grad_b: 0.0,
        }
    }

    /// Get all parameters (weights and bias) as a flat vector.
    pub fn get_parameters(&self) -> Vec<f32> {
        let mut params = match &self.weights {
            WeightType::F32(weights) => weights.clone(),
            WeightType::I8 {
                quantized_weights,
                scale,
            } => quantized_weights
                .iter()
                .map(|&w| w as f32 * scale)
                .collect(),
        };
        params.push(self.bias);
        params
    }

    /// Set all parameters (weights and bias) from a flat vector.
    /// Returns the number of parameters consumed.
    pub fn set_parameters(&mut self, params: &[f32]) -> Result<usize, Box<dyn std::error::Error>> {
        let num_weights = self.weights.len();
        let expected_len = num_weights + 1;

        if params.len() < expected_len {
            return Err(format!(
                "Not enough parameters provided. Expected {}, got {}",
                expected_len,
                params.len()
            )
            .into());
        }

        let new_weights = &params[..num_weights];
        self.bias = params[num_weights];

        match &mut self.weights {
            WeightType::F32(weights) => {
                weights.copy_from_slice(new_weights);
            }
            WeightType::I8 {
                quantized_weights,
                scale,
            } => {
                // Re-quantize the weights
                let min_val = new_weights.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val = new_weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let abs_max = min_val.abs().max(max_val.abs());
                *scale = abs_max / 127.0;
                let inv_scale = if *scale != 0.0 { 1.0 / *scale } else { 1.0 };

                for (i, &float_weight) in new_weights.iter().enumerate() {
                    let quantized = (float_weight * inv_scale).round();
                    quantized_weights[i] = quantized.max(-127.0).min(127.0) as i8;
                }
            }
        }

        Ok(expected_len)
    }

    /// Compute the weighted sum of inputs plus bias (pre-activation).
    ///
    /// # Arguments
    /// * `inputs` - Input values to the neuron
    fn activate(&self, inputs: &[f32]) -> f32 {
        match &self.weights {
            WeightType::F32(weights) => {
                inputs
                    .iter()
                    .zip(weights.iter())
                    .map(|(x, w)| x * w)
                    .sum::<f32>()
                    + self.bias
            }
            WeightType::I8 {
                quantized_weights,
                scale,
            } => {
                inputs
                    .iter()
                    .zip(quantized_weights.iter())
                    .map(|(x, w)| x * (*w as f32 * scale))
                    .sum::<f32>()
                    + self.bias
            }
        }
    }

    /// Apply the sigmoid activation function to a value.
    ///
    /// # Arguments
    /// * `x` - The value to apply the sigmoid function to
    fn sigmoid(x: f32) -> f32 {
        crate::utils::sigmoid(x)
    }

    /// Compute the derivative of the sigmoid function.
    ///
    /// # Arguments
    /// * `x` - The sigmoid output value to compute the derivative at
    fn sigmoid_derivative(x: f32) -> f32 {
        crate::utils::sigmoid_derivative(x)
    }
}

impl Module<Vec<f32>, f32> for Neuron {
    /// Perform a forward pass on the neuron.
    ///
    /// # Arguments
    /// * `input` - The input values
    ///
    /// # Returns
    /// The neuron's output after sigmoid activation
    fn forward(&mut self, input: Vec<f32>) -> f32 {
        // Store inputs for backpropagation
        self.last_input.clone_from_slice(&input);

        // Compute and store pre-activation
        let z = self.activate(&input);
        self.last_z = z;

        // Apply sigmoid activation
        Self::sigmoid(z)
    }

    /// Backpropagate gradients through this neuron.
    ///
    /// # Arguments
    /// * `grad_output` - Gradient of loss with respect to this neuron's output
    ///
    /// # Returns
    /// Vector of gradients with respect to each input
    fn backward(&mut self, grad_output: f32) -> Vec<f32> {
        let sigmoid_z = Self::sigmoid(self.last_z);
        let sigmoid_derivative = Self::sigmoid_derivative(sigmoid_z);

        // Gradient of loss with respect to pre-activation z
        let dz = grad_output * sigmoid_derivative;

        // Gradient with respect to each weight
        let mut grad_inputs = vec![0.0; self.weights.len()];

        match &self.weights {
            WeightType::F32(weights) => {
                for i in 0..weights.len() {
                    // Accumulate gradient for weight i
                    self.grad_w[i] += dz * self.last_input[i];

                    // Compute gradient for input i
                    grad_inputs[i] = dz * weights[i];
                }
            }
            WeightType::I8 {
                quantized_weights,
                scale,
            } => {
                for i in 0..quantized_weights.len() {
                    // Accumulate gradient for weight i
                    self.grad_w[i] += dz * self.last_input[i];

                    // Compute gradient for input i (de-quantize the weight)
                    grad_inputs[i] = dz * (quantized_weights[i] as f32 * scale);
                }
            }
        }

        // Accumulate gradient for bias
        self.grad_b += dz;

        grad_inputs
    }

    /// Get the gradients of the neuron's parameters.
    /// Returns a vector containing gradients for all weights followed by the bias gradient.
    fn gradients(&self) -> Vec<f32> {
        let mut grads = self.grad_w.clone();
        grads.push(self.grad_b);
        grads
    }

    /// Apply pre-computed gradients to update parameters.
    fn apply_gradients(&mut self, gradients: &[f32], learning_rate: f32) {
        // Update weights (all but last gradient)
        match &mut self.weights {
            WeightType::F32(weights) => {
                for (i, &grad) in gradients[..weights.len()].iter().enumerate() {
                    weights[i] -= learning_rate * grad;
                }
            }
            WeightType::I8 {
                quantized_weights,
                scale,
            } => {
                // For quantized weights, need to dequantize, update, and requantize
                // First, convert quantized weights to f32
                let mut float_weights: Vec<f32> = quantized_weights
                    .iter()
                    .map(|&w| w as f32 * *scale)
                    .collect();

                // Apply gradients to the float weights
                for (i, &grad) in gradients[..float_weights.len()].iter().enumerate() {
                    float_weights[i] -= learning_rate * grad;
                }

                // Re-quantize the weights
                // Find min and max for new scale calculation
                let min_val = float_weights.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val = float_weights
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                // Calculate new scale (symmetric quantization)
                let abs_max = min_val.abs().max(max_val.abs());
                *scale = abs_max / 127.0; // Scale for i8 range [-127, 127]

                // Avoid division by zero
                let inv_scale = if *scale != 0.0 { 1.0 / *scale } else { 1.0 };

                // Quantize the float weights back to i8
                for (i, &float_weight) in float_weights.iter().enumerate() {
                    let quantized = (float_weight * inv_scale).round();
                    quantized_weights[i] = quantized.max(-127.0).min(127.0) as i8;
                }
            }
        }

        // Update bias (last gradient)
        if let Some(&bias_grad) = gradients.last() {
            self.bias -= learning_rate * bias_grad;
        }
    }

    /// Reset accumulated gradients to zero.
    fn zero_grad(&mut self) {
        for grad in &mut self.grad_w {
            *grad = 0.0;
        }
        self.grad_b = 0.0;
    }
}
