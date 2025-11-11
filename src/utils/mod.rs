pub mod error;

use rand::Rng;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(sigmoid_x: f32) -> f32 {
    sigmoid_x * (1.0 - sigmoid_x)
}

/// Computes the hyperbolic tangent of a value.
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}

/// Computes the derivative of the tanh function from its output.
pub fn tanh_derivative(tanh_x: f32) -> f32 {
    1.0 - tanh_x.powi(2)
}

pub fn initialize_weights(num_weights: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    (0..num_weights)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect()
}
