use crate::core::{Activation, Module, Neuron};

/// A layer of neurons in a neural network.
///
/// Each neuron in the layer receives the same inputs and
/// processes them independently, producing a vector of outputs.
#[derive(Debug)]
pub struct Layer {
    /// The neurons in this layer
    pub neurons: Vec<Neuron>,

    /// The activation function used by this layer
    pub activation: Activation,
}

impl Layer {
    /// Create a new layer with the specified number of neurons, inputs, and activation.
    ///
    /// # Arguments
    /// * `num_neurons` - Number of neurons in the layer
    /// * `num_inputs_per_neuron` - Number of inputs each neuron receives
    /// * `activation` - Activation function for this layer
    pub fn new(num_neurons: usize, num_inputs_per_neuron: usize, activation: Activation) -> Self {
        Layer {
            neurons: (0..num_neurons)
                .map(|_| Neuron::new(num_inputs_per_neuron, activation))
                .collect(),
            activation,
        }
    }

    /// Create a new layer with sigmoid activation (for backward compatibility).
    pub fn new_sigmoid(num_neurons: usize, num_inputs_per_neuron: usize) -> Self {
        Self::new(num_neurons, num_inputs_per_neuron, Activation::Sigmoid)
    }

    /// Create a new linear layer (no activation) for output layers.
    pub fn new_linear(num_neurons: usize, num_inputs_per_neuron: usize) -> Self {
        Self::new(num_neurons, num_inputs_per_neuron, Activation::None)
    }

    /// Get the number of neurons in this layer
    pub fn size(&self) -> usize {
        self.neurons.len()
    }

    /// Get all parameters from all neurons in the layer.
    pub fn get_parameters(&self) -> Vec<f32> {
        self.neurons
            .iter()
            .flat_map(|neuron| neuron.get_parameters())
            .collect()
    }

    /// Set all parameters for all neurons in the layer.
    /// Returns the number of parameters consumed.
    pub fn set_parameters(&mut self, params: &[f32]) -> Result<usize, Box<dyn std::error::Error>> {
        let mut offset = 0;
        for neuron in &mut self.neurons {
            let consumed = neuron.set_parameters(&params[offset..])?;
            offset += consumed;
        }
        Ok(offset)
    }
}

impl Module<Vec<f32>, Vec<f32>> for Layer {
    fn forward(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter_mut()
            .map(|neuron| neuron.forward(inputs.clone()))
            .collect()
    }

    fn backward(&mut self, grad_outputs: Vec<f32>) -> Vec<f32> {
        assert_eq!(
            grad_outputs.len(),
            self.neurons.len(),
            "Gradient size must match number of neurons"
        );

        let n_inputs = self.neurons[0].weights.len();
        let mut grad_inputs = vec![0.0; n_inputs];

        // For each neuron, accumulate its contribution to the input gradients
        for (neuron, &grad_output) in self.neurons.iter_mut().zip(&grad_outputs) {
            let neuron_grad_inputs = neuron.backward(grad_output);

            // Sum up gradients from all neurons for each input
            for i in 0..n_inputs {
                grad_inputs[i] += neuron_grad_inputs[i];
            }
        }

        grad_inputs
    }

    fn gradients(&self) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.gradients())
            .collect::<Vec<Vec<f32>>>()
            .concat()
    }

    fn apply_gradients(&mut self, gradients: &[f32], learning_rate: f32) {
        let mut offset = 0;
        for neuron in &mut self.neurons {
            let num_params = neuron.weights.len() + 1; // weights + bias
            neuron.apply_gradients(&gradients[offset..offset + num_params], learning_rate);
            offset += num_params;
        }
    }

    fn zero_grad(&mut self) {
        for neuron in &mut self.neurons {
            neuron.zero_grad();
        }
    }
}
