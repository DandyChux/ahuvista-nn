use crate::core::{Activation, Module, Network};

/// Neural network for processing tabular (structured) data.
///
/// This is essentially a standard MLP optimized for tabular features.
#[derive(Debug)]
pub struct TabularNet {
    /// The underlying MLP
    network: Network,
}

impl TabularNet {
    /// Create a new tabular network.
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_sizes` - Sizes of hidden layers
    /// * `output_size` - Size of output feature vector
    pub fn new(
        input_size: usize,
        hidden_sizes: &[usize],
        output_size: usize,
        hidden_activation: Option<Activation>,
        output_activation: Option<Activation>,
    ) -> Self {
        let mut layer_sizes = vec![input_size];
        layer_sizes.extend_from_slice(hidden_sizes);
        layer_sizes.push(output_size);

        TabularNet {
            network: Network::from_layer_sizes(&layer_sizes, hidden_activation, output_activation),
        }
    }

    /// Create a tabular network for regression (ReLU hidden, linear output).
    pub fn new_for_regression(
        input_size: usize,
        hidden_sizes: &[usize],
        output_size: usize,
    ) -> Self {
        Self::new(input_size, hidden_sizes, output_size, None, None)
    }

    /// Create a tabular network for binary classification (ReLU hidden, sigmoid output).
    pub fn new_for_binary_classification(input_size: usize, hidden_sizes: &[usize]) -> Self {
        Self::new(input_size, hidden_sizes, 1, None, Some(Activation::Sigmoid))
    }

    /// Create a tabular network for multi-class classification (ReLU hidden, linear output for softmax).
    pub fn new_for_classification(
        input_size: usize,
        hidden_sizes: &[usize],
        num_classes: usize,
    ) -> Self {
        Self::new(input_size, hidden_sizes, num_classes, None, None)
    }

    /// Get all parameters from the network.
    pub fn get_parameters(&self) -> Vec<f32> {
        self.network.get_parameters()
    }

    /// Set all parameters in the network.
    /// Returns the number of parameters consumed.
    pub fn set_parameters(&mut self, params: &[f32]) -> Result<usize, Box<dyn std::error::Error>> {
        self.network.set_parameters(params)
    }

    /// Get the output size of this network
    pub fn output_size(&self) -> usize {
        self.network.output_size()
    }
}

impl Module<Vec<f32>, Vec<f32>> for TabularNet {
    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        self.network.forward(input)
    }

    fn backward(&mut self, grad_output: Vec<f32>) -> Vec<f32> {
        self.network.backward(grad_output)
    }

    fn gradients(&self) -> Vec<f32> {
        self.network.gradients()
    }

    fn apply_gradients(&mut self, gradients: &[f32], lr: f32) {
        self.network.apply_gradients(gradients, lr)
    }

    fn zero_grad(&mut self) {
        self.network.zero_grad();
    }
}
