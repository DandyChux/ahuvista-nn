use crate::layer::Layer;

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new() -> Self {
        Network { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.layers.iter().fold(inputs.to_vec(), |inputs, layer| layer.forward(&inputs))
    }
}