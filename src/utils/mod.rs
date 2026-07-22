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

pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

pub fn leaky_relu(x: f32) -> f32 {
    x.max(0.1 * x)
}

pub fn relu_derivative(relu_x: f32) -> f32 {
    if relu_x > 0.0 { 1.0 } else { 0.0 }
}

pub fn leaky_relu_derivative(leaky_relu_x: f32) -> f32 {
    if leaky_relu_x > 0.0 { 1.0 } else { 0.1 }
}
