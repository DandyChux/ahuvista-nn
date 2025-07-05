use crate::module::Module;
use crate::neuron::Neuron;

/// A layer of neurons in a neural network.
///
/// Each neuron in the layer receives the same inputs and
/// processes them independently, producing a vector of outputs.
#[derive(Debug)]
pub struct Layer {
    /// The neurons in this layer
    pub neurons: Vec<Neuron>,
}

impl Layer {
    /// Create a new layer with the specified number of neurons and inputs per neuron.
    ///
    /// # Arguments
    /// * `num_neurons` - Number of neurons in the layer
    /// * `num_inputs_per_neuron` - Number of inputs each neuron receives
    pub fn new(num_neurons: usize, num_inputs_per_neuron: usize) -> Self {
        Layer {
            neurons: (0..num_neurons)
                .map(|_| Neuron::new(num_inputs_per_neuron))
                .collect(),
        }
    }

    /// Get the number of neurons in this layer
    pub fn size(&self) -> usize {
        self.neurons.len()
    }
}

impl Module<Vec<f32>, Vec<f32>> for Layer {
    fn forward(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter_mut()
            .map(|neuron| neuron.forward(&inputs))
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

    fn update(&mut self, lr: f32) {
        for neuron in &mut self.neurons {
            neuron.update(lr);
        }
    }

    fn zero_grad(&mut self) {
        for neuron in &mut self.neurons {
            neuron.zero_grad();
        }
    }
}
