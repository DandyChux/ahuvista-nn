use crate::module::Module;
use crate::network::Network;

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
    pub fn new(input_size: usize, hidden_sizes: &[usize], output_size: usize) -> Self {
        let mut layer_sizes = vec![input_size];
        layer_sizes.extend_from_slice(hidden_sizes);
        layer_sizes.push(output_size);

        TabularNet {
            network: Network::from_layer_sizes(&layer_sizes),
        }
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

    fn update(&mut self, lr: f32) {
        self.network.update(lr);
    }

    fn zero_grad(&mut self) {
        self.network.zero_grad();
    }
}
