use crate::core::Module;
use rand_distr::{Distribution, Normal};

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

/// Activation function type for neurons
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    /// No activation (linear/identity) - used for output layer logits
    None,
    /// Sigmoid activation - used for hidden layers
    Sigmoid,
    /// Tanh activation - alternative for hidden layers
    Tanh,
    /// ReLU activation - alternative for hidden layers
    ReLU,
    /// Leaky ReLU activation - alternative for hidden layers
    LeakyReLU,
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

    /// Activation function to apply
    pub activation: Activation,

    /// Cached input from last forward pass
    last_input: Vec<f32>,

    /// Cached pre-activation value from last forward pass
    last_z: f32,

    /// Cached post-activation value from last forward pass (for backward)
    last_output: f32,

    /// Accumulated gradients for weights
    grad_w: Vec<f32>,

    /// Accumulated gradient for bias
    grad_b: f32,
}

impl Neuron {
    /// Create a new neuron with the specified number of inputs and activation.
    ///
    /// Weights are initialized using Xavier/Glorot initialization.
    ///
    /// # Arguments
    /// * `num_inputs` - The number of input connections
    /// * `activation` - The activation function to use
    pub fn new(num_inputs: usize, activation: Activation) -> Self {
        let mut rng = rand::rng();

        // Xavier/Glorot initialization
        // For sigmoid/tanh: sqrt(2 / (fan_in + fan_out))
        // For ReLU: sqrt(2 / fan_in)
        // For linear: sqrt(1 / fan_in)
        let std = match activation {
            Activation::ReLU => (2.0 / num_inputs as f32).sqrt(),
            Activation::Sigmoid | Activation::Tanh => (2.0 / (num_inputs as f32 + 1.0)).sqrt(),
            Activation::None => (1.0 / num_inputs as f32).sqrt(),
            Activation::LeakyReLU => (2.0 / num_inputs as f32).sqrt(),
        };

        // Create a normal distribution
        let normal_dist = Normal::new(0.0, std).unwrap();

        Neuron {
            weights: WeightType::F32(
                (0..num_inputs)
                    .map(|_| normal_dist.sample(&mut rng))
                    .collect(),
            ),
            bias: 0.0,
            activation,
            last_input: vec![0.0; num_inputs],
            last_z: 0.0,
            last_output: 0.0,
            grad_w: vec![0.0; num_inputs],
            grad_b: 0.0,
        }
    }

    /// Create a new neuron with sigmoid activation (for backward compatibility).
    pub fn new_sigmoid(num_inputs: usize) -> Self {
        Self::new(num_inputs, Activation::Sigmoid)
    }

    /// Create a new neuron with linear activation (for output layers).
    pub fn new_linear(num_inputs: usize) -> Self {
        Self::new(num_inputs, Activation::None)
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
    fn linear_combination(&self, inputs: &[f32]) -> f32 {
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

    /// Apply the activation function.
    fn apply_activation(&self, x: f32) -> f32 {
        match self.activation {
            Activation::None => x,
            Activation::Sigmoid => super::sigmoid(x),
            Activation::Tanh => super::tanh(x),
            Activation::ReLU => super::relu(x),
            Activation::LeakyReLU => super::leaky_relu(x),
        }
    }

    /// Compute the derivative of the activation function.
    fn activation_derivative(&self, x: f32, output: f32) -> f32 {
        match self.activation {
            Activation::None => 1.0,
            Activation::Sigmoid => super::sigmoid_derivative(output),
            Activation::Tanh => super::tanh_derivative(output),
            Activation::ReLU => super::relu_derivative(x),
            Activation::LeakyReLU => super::leaky_relu_derivative(x),
        }
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
        // Validate input size
        if input.len() != self.weights.len() {
            panic!(
                "Input size mismatch: expected {} inputs, got {}",
                self.weights.len(),
                input.len()
            );
        }

        // Store inputs for backpropagation
        if self.last_input.len() != input.len() {
            self.last_input.resize(input.len(), 0.0);
        }
        self.last_input.copy_from_slice(&input);

        // Compute pre-activation
        let z = self.linear_combination(&input);
        self.last_z = z;

        // Apply activation and store output
        let output = self.apply_activation(z);
        self.last_output = output;

        output
    }

    /// Backpropagate gradients through this neuron.
    ///
    /// # Arguments
    /// * `grad_output` - Gradient of loss with respect to this neuron's output
    ///
    /// # Returns
    /// Vector of gradients with respect to each input
    fn backward(&mut self, grad_output: f32) -> Vec<f32> {
        // Validate that we have cached inputs
        if self.last_input.len() != self.weights.len() {
            panic!("Backward called without proper forward pass");
        }

        // Compute activation derivative
        let activation_deriv = self.activation_derivative(self.last_z, self.last_output);

        // Gradient of loss with respect to pre-activation z
        let dz = grad_output * activation_deriv;

        // Gradient with respect to each weight and input
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
