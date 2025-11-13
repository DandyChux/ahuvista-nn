use crate::{
    core::{LSTMLayer, Layer, Module},
    utils::{tanh, tanh_derivative},
};
use rand::Rng;

/// Simple RNN cell for processing sequence data.
///
/// This implements a basic RNN with tanh activation for the hidden state.
/// It's designed to be memory-efficient for deployment on low-compute devices.
#[derive(Debug)]
pub struct SimpleRnnCell {
    /// Input size
    input_size: usize,

    /// Hidden state size
    pub hidden_size: usize,

    /// Input-to-hidden weights
    pub w_ih: Vec<f32>,

    /// Hidden-to-hidden weights
    pub w_hh: Vec<f32>,

    /// Hidden bias
    pub b_h: Vec<f32>,

    /// Cached inputs and hidden states for backpropagation
    pub cached_inputs: Vec<Vec<f32>>,
    pub cached_hiddens: Vec<Vec<f32>>,

    /// Accumulated gradients
    pub grad_w_ih: Vec<f32>,
    pub grad_w_hh: Vec<f32>,
    pub grad_b_h: Vec<f32>,
}

impl SimpleRnnCell {
    /// Create a new RNN cell.
    ///
    /// # Arguments
    /// * `input_size` - Size of input vectors
    /// * `hidden_size` - Size of hidden state vectors
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization scaled for tanh activation
        let ih_scale = (6.0 / (input_size + hidden_size) as f32).sqrt();
        let hh_scale = (6.0 / (hidden_size + hidden_size) as f32).sqrt();

        SimpleRnnCell {
            input_size,
            hidden_size,
            w_ih: (0..input_size * hidden_size)
                .map(|_| rng.random_range(-ih_scale..ih_scale))
                .collect(),
            w_hh: (0..hidden_size * hidden_size)
                .map(|_| rng.random_range(-hh_scale..hh_scale))
                .collect(),
            b_h: vec![0.0; hidden_size],
            cached_inputs: Vec::new(),
            cached_hiddens: Vec::new(),
            grad_w_ih: vec![0.0; input_size * hidden_size],
            grad_w_hh: vec![0.0; hidden_size * hidden_size],
            grad_b_h: vec![0.0; hidden_size],
        }
    }

    /// Get all parameters from the cell.
    pub fn get_parameters(&self) -> Vec<f32> {
        let mut params = self.w_ih.clone();
        params.extend_from_slice(&self.w_hh);
        params.extend_from_slice(&self.b_h);
        params
    }

    /// Set all parameters in the cell.
    /// Returns the number of parameters consumed.
    pub fn set_parameters(&mut self, params: &[f32]) -> Result<usize, Box<dyn std::error::Error>> {
        let w_ih_size = self.w_ih.len();
        let w_hh_size = self.w_hh.len();
        let b_h_size = self.b_h.len();
        let expected_len = w_ih_size + w_hh_size + b_h_size;

        if params.len() < expected_len {
            return Err(format!(
                "Not enough parameters. Expected {}, got {}",
                expected_len,
                params.len()
            )
            .into());
        }

        self.w_ih.copy_from_slice(&params[0..w_ih_size]);
        self.w_hh
            .copy_from_slice(&params[w_ih_size..w_ih_size + w_hh_size]);
        self.b_h
            .copy_from_slice(&params[w_ih_size + w_hh_size..expected_len]);

        Ok(expected_len)
    }

    /// Apply tanh activation element-wise
    fn tanh(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&xi| tanh(xi)).collect()
    }

    /// Compute derivative of tanh
    pub fn tanh_derivative(tanh_x: &[f32]) -> Vec<f32> {
        tanh_x
            .iter()
            .map(|&tanhxi| tanh_derivative(tanhxi))
            .collect()
    }

    /// Process one time step.
    ///
    /// # Arguments
    /// * `x` - Input vector at this time step
    /// * `h_prev` - Hidden state from previous time step
    ///
    /// # Returns
    /// New hidden state vector
    pub fn forward_step(&self, x: &[f32], h_prev: &[f32]) -> Vec<f32> {
        let mut next_h = vec![0.0; self.hidden_size];

        // Compute input contribution
        for i in 0..self.hidden_size {
            for j in 0..self.input_size {
                next_h[i] += x[j] * self.w_ih[i * self.input_size + j];
            }
        }

        // Add hidden state contribution
        for i in 0..self.hidden_size {
            for j in 0..self.hidden_size {
                next_h[i] += h_prev[j] * self.w_hh[i * self.hidden_size + j];
            }
            next_h[i] += self.b_h[i];
        }

        // Apply tanh activation
        Self::tanh(&next_h)
    }

    /// Zero the hidden state for a new sequence
    pub fn reset_state(&mut self) {
        self.cached_inputs.clear();
        self.cached_hiddens.clear();
    }
}

/// Network for processing temporal (time series) data.
///
/// Uses a simple RNN to process sequential data and extract meaningful features.
#[derive(Debug)]
pub struct TemporalNet {
    lstm_layer: LSTMLayer,
    // Optional: A final linear layer to transform the LSTM output
    output_layer: Layer,
}

impl TemporalNet {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            lstm_layer: LSTMLayer::new(input_size, hidden_size),
            output_layer: Layer::new(output_size, hidden_size), // Takes final hidden state as input
        }
    }

    /// Get all parameters from the network.
    pub fn get_parameters(&self) -> Vec<f32> {
        let mut params = self.lstm_layer.get_parameters();
        params.extend(self.output_layer.get_parameters());
        params
    }

    /// Set all parameters in the network.
    /// Returns the number of parameters consumed.
    pub fn set_parameters(&mut self, params: &[f32]) -> Result<usize, Box<dyn std::error::Error>> {
        let mut offset = 0;
        offset += self.lstm_layer.set_parameters(&params[offset..])?;
        offset += self.output_layer.set_parameters(&params[offset..])?;
        Ok(offset)
    }
}

impl Module<Vec<Vec<f32>>, Vec<f32>> for TemporalNet {
    fn forward(&mut self, sequence: Vec<Vec<f32>>) -> Vec<f32> {
        // Process the sequence through the LSTM layer
        let final_hidden_state = self.lstm_layer.forward_sequence(&sequence);

        // Pass the final state through the output layer
        self.output_layer.forward(final_hidden_state.clone())
    }

    fn backward(&mut self, grad_output: Vec<f32>) -> Vec<Vec<f32>> {
        // Backpropagate through the output layer first
        let grad_hidden_state = self.output_layer.backward(grad_output);

        // Then, backpropagate through the entire sequence in the LSTM layer
        let grad_sequence = self.lstm_layer.backward_sequence(&grad_hidden_state);

        grad_sequence
    }

    fn zero_grad(&mut self) {
        self.lstm_layer.input_gate.zero_grad();
        self.lstm_layer.forget_gate.zero_grad();
        self.lstm_layer.output_gate.zero_grad();
        self.lstm_layer.cell_gate.zero_grad();
        self.output_layer.zero_grad();
    }

    fn gradients(&self) -> Vec<f32> {
        // Collect gradients from all layers
        let mut grads = Vec::new();
        grads.extend(self.lstm_layer.input_gate.gradients());
        grads.extend(self.lstm_layer.forget_gate.gradients());
        grads.extend(self.lstm_layer.output_gate.gradients());
        grads.extend(self.lstm_layer.cell_gate.gradients());
        grads.extend(self.output_layer.gradients());
        grads
    }

    fn apply_gradients(&mut self, gradients: &[f32], lr: f32) {
        // Calculate sizes for each layer's gradients
        let input_gate_size = self.lstm_layer.input_gate.gradients().len();
        let forget_gate_size = self.lstm_layer.forget_gate.gradients().len();
        let output_gate_size = self.lstm_layer.output_gate.gradients().len();
        let cell_gate_size = self.lstm_layer.cell_gate.gradients().len();

        let mut offset = 0;

        // Apply gradients to each layer
        self.lstm_layer
            .input_gate
            .apply_gradients(&gradients[offset..offset + input_gate_size], lr);
        offset += input_gate_size;

        self.lstm_layer
            .forget_gate
            .apply_gradients(&gradients[offset..offset + forget_gate_size], lr);
        offset += forget_gate_size;

        self.lstm_layer
            .output_gate
            .apply_gradients(&gradients[offset..offset + output_gate_size], lr);
        offset += output_gate_size;

        self.lstm_layer
            .cell_gate
            .apply_gradients(&gradients[offset..offset + cell_gate_size], lr);
        offset += cell_gate_size;

        self.output_layer.apply_gradients(&gradients[offset..], lr);
    }
}
