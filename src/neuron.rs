use rand::Rng;

pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl Neuron {
    pub fn new(num_inputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..num_inputs).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bias = rng.gen_range(-1.0..1.0);
        Neuron { weights, bias }
    }

    pub fn activate(&self, inputs: &[f64]) -> f64 {
        let sum: f64 = self.weights.iter().zip(inputs).map(|(w, i)| w * i).sum();
        sum + self.bias
    }

    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn forward(&self, inputs: &[f64]) -> f64 {
        Self::sigmoid(self.activate(inputs))
    }
}