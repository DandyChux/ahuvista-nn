use crate::module::Module;
use rand::Rng;

/// The basic computational unit of a neural network.
///
/// A `Neuron` has weights for each input, a bias term, and applies
/// a sigmoid activation function to produce its output.
/// Also stores intermediate values and gradients needed for backpropagation.
#[derive(Debug)]
pub struct Neuron {
    /// Weights for each input connection
    pub weights: Vec<f32>,

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
        let mut rng = rand::thread_rng();

        Neuron {
            weights: (0..num_inputs).map(|_| rng.gen_range(-0.5..0.5)).collect(),
            bias: rng.gen_range(-0.5..0.5),
            last_input: vec![0.0; num_inputs],
            last_z: 0.0,
            grad_w: vec![0.0; num_inputs],
            grad_b: 0.0,
        }
    }

    /// Compute the weighted sum of inputs plus bias (pre-activation).
    ///
    /// # Arguments
    /// * `inputs` - Input values to the neuron
    pub fn activate(&self, inputs: &[f32]) -> f32 {
        inputs
            .iter()
            .zip(&self.weights)
            .map(|(x, w)| x * w)
            .sum::<f32>()
            + self.bias
    }

    /// Apply the sigmoid activation function to a value.
    ///
    /// # Arguments
    /// * `x` - The value to apply the sigmoid function to
    pub fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Compute the derivative of the sigmoid function.
    ///
    /// # Arguments
    /// * `x` - The sigmoid output value to compute the derivative at
    pub fn sigmoid_derivative(sigmoid_x: f32) -> f32 {
        sigmoid_x * (1.0 - sigmoid_x)
    }

    /// Perform a forward pass on the neuron.
    ///
    /// # Arguments
    /// * `inputs` - The input values
    ///
    /// # Returns
    /// The neuron's output after sigmoid activation
    pub fn forward(&mut self, inputs: &[f32]) -> f32 {
        // Store inputs for backpropagation
        self.last_input.clone_from_slice(inputs);

        // Compute and store pre-activation
        let z = self.activate(inputs);
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
    pub fn backward(&mut self, grad_output: f32) -> Vec<f32> {
        let sigmoid_z = Self::sigmoid(self.last_z);
        let sigmoid_derivative = Self::sigmoid_derivative(sigmoid_z);

        // Gradient of loss with respect to pre-activation z
        let dz = grad_output * sigmoid_derivative;

        // Gradient with respect to each weight
        let mut grad_inputs = vec![0.0; self.weights.len()];

        for i in 0..self.weights.len() {
            // Accumulate gradient for weight i
            self.grad_w[i] += dz * self.last_input[i];

            // Compute gradient for input i
            grad_inputs[i] = dz * self.weights[i];
        }

        // Accumulate gradient for bias
        self.grad_b += dz;

        grad_inputs
    }

    /// Update weights and bias using accumulated gradients.
    ///
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn update(&mut self, lr: f32) {
        // Update weights
        for i in 0..self.weights.len() {
            self.weights[i] -= lr * self.grad_w[i];
        }

        // Update bias
        self.bias -= lr * self.grad_b;
    }

    /// Reset accumulated gradients to zero.
    pub fn zero_grad(&mut self) {
        for grad in &mut self.grad_w {
            *grad = 0.0;
        }
        self.grad_b = 0.0;
    }
}
