use crate::neuron::Neuron;

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(num_neurons: usize, num_inputs_per_neuron: usize) -> Self {
        let neurons = (0..num_neurons).map(|_| Neuron::new(num_inputs_per_neuron)).collect();
        Layer { neurons }
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons.iter().map(|neuron| neuron.forward(inputs)).collect()
    }
}