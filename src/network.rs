use crate::layer::Layer;
use crate::module::Module;

/// A multi-layer perceptron (MLP) neural network.
///
/// Consists of a sequence of layers, where each layer's output
/// becomes the input to the next layer.
#[derive(Debug)]
pub struct Network {
    /// The sequence of layers in this network
    pub layers: Vec<Layer>,

    /// Cached intermediate outputs from last forward pass
    layer_outputs: Vec<Vec<f32>>,
}

impl Network {
    /// Create a new empty network with no layers.
    pub fn new() -> Self {
        Network {
            layers: Vec::new(),
            layer_outputs: Vec::new(),
        }
    }

    /// Add a layer to the end of the network.
    ///
    /// # Arguments
    /// * `layer` - The layer to add
    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
        self.layer_outputs.push(Vec::new());
    }

    /// Create a new network with the specified layer sizes.
    ///
    /// # Arguments
    /// * `layer_sizes` - Vector of integers where each element is the number of neurons in a layer.
    ///                   The first element is the input size, and the last is the output size.
    pub fn from_layer_sizes(layer_sizes: &[usize]) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "Need at least input and output sizes"
        );

        let mut network = Network::new();

        for i in 0..layer_sizes.len() - 1 {
            let layer = Layer::new(layer_sizes[i + 1], layer_sizes[i]);
            network.add_layer(layer);
        }

        network
    }

    /// Get the input size of the network
    pub fn input_size(&self) -> usize {
        if self.layers.is_empty() {
            0
        } else {
            self.layers[0].neurons[0].weights.len()
        }
    }

    /// Get the output size of the network
    pub fn output_size(&self) -> usize {
        if self.layers.is_empty() {
            0
        } else {
            self.layers.last().unwrap().size()
        }
    }
}

impl Module<Vec<f32>, Vec<f32>> for Network {
    fn forward(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        self.layer_outputs = Vec::with_capacity(self.layers.len() + 1);
        self.layer_outputs.push(inputs.clone());

        let mut current_inputs = inputs;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let outputs = layer.forward(current_inputs);
            self.layer_outputs.push(outputs.clone());
            current_inputs = outputs;
        }

        current_inputs
    }

    fn backward(&mut self, grad_output: Vec<f32>) -> Vec<f32> {
        assert!(
            !self.layers.is_empty(),
            "Cannot backpropagate through empty network"
        );
        assert_eq!(
            grad_output.len(),
            self.layers.last().unwrap().size(),
            "Gradient size must match output size"
        );

        let mut current_grad = grad_output;

        // Backpropagate through layers in reverse order
        for layer in self.layers.iter_mut().rev() {
            current_grad = layer.backward(current_grad);
        }

        current_grad
    }

    fn update(&mut self, lr: f32) {
        for layer in &mut self.layers {
            layer.update(lr);
        }
    }

    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }
}
