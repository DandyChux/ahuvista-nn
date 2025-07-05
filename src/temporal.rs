use crate::module::Module;
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
    hidden_size: usize,

    /// Input-to-hidden weights
    w_ih: Vec<f32>,

    /// Hidden-to-hidden weights
    w_hh: Vec<f32>,

    /// Hidden bias
    b_h: Vec<f32>,

    /// Cached inputs and hidden states for backpropagation
    cached_inputs: Vec<Vec<f32>>,
    cached_hiddens: Vec<Vec<f32>>,

    /// Accumulated gradients
    grad_w_ih: Vec<f32>,
    grad_w_hh: Vec<f32>,
    grad_b_h: Vec<f32>,
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
                .map(|_| rng.gen_range(-ih_scale..ih_scale))
                .collect(),
            w_hh: (0..hidden_size * hidden_size)
                .map(|_| rng.gen_range(-hh_scale..hh_scale))
                .collect(),
            b_h: vec![0.0; hidden_size],
            cached_inputs: Vec::new(),
            cached_hiddens: Vec::new(),
            grad_w_ih: vec![0.0; input_size * hidden_size],
            grad_w_hh: vec![0.0; hidden_size * hidden_size],
            grad_b_h: vec![0.0; hidden_size],
        }
    }

    /// Apply tanh activation element-wise
    fn tanh(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&xi| xi.tanh()).collect()
    }

    /// Compute derivative of tanh
    fn tanh_derivative(tanh_x: &[f32]) -> Vec<f32> {
        tanh_x.iter().map(|&tanhxi| 1.0 - tanhxi * tanhxi).collect()
    }

    /// Process one time step.
    ///
    /// # Arguments
    /// * `x` - Input vector at this time step
    /// * `h_prev` - Hidden state from previous time step
    ///
    /// # Returns
    /// New hidden state vector
    fn forward_step(&self, x: &[f32], h_prev: &[f32]) -> Vec<f32> {
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
    /// The RNN cell
    rnn: SimpleRnnCell,

    /// Last computed final hidden state
    last_hidden: Vec<f32>,
}

impl TemporalNet {
    /// Create a new temporal network.
    ///
    /// # Arguments
    /// * `input_size` - Size of input feature vectors at each time step
    /// * `hidden_size` - Size of RNN hidden state (and output feature vector)
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        TemporalNet {
            rnn: SimpleRnnCell::new(input_size, hidden_size),
            last_hidden: vec![0.0; hidden_size],
        }
    }

    /// Get the output size of this network
    pub fn output_size(&self) -> usize {
        self.rnn.hidden_size
    }
}

impl Module<Vec<Vec<f32>>, Vec<f32>> for TemporalNet {
    fn forward(&mut self, input_sequence: Vec<Vec<f32>>) -> Vec<f32> {
        if input_sequence.is_empty() {
            return vec![0.0; self.rnn.hidden_size];
        }

        self.rnn.reset_state();
        let mut h = vec![0.0; self.rnn.hidden_size];

        // Store input sequence
        self.rnn.cached_inputs = input_sequence.clone();

        // Initialize hidden states with initial zero state
        self.rnn.cached_hiddens.push(h.clone());

        // Process each time step
        for x in &input_sequence {
            h = self.rnn.forward_step(x, &h);
            self.rnn.cached_hiddens.push(h.clone());
        }

        // Store the final hidden state
        self.last_hidden = h.clone();
        h
    }

    fn backward(&mut self, grad_output: Vec<f32>) -> Vec<Vec<f32>> {
        let seq_len = self.rnn.cached_inputs.len();
        if seq_len == 0 {
            return Vec::new();
        }

        let mut dh_next = grad_output;
        let mut dx_sequence = vec![vec![0.0; self.rnn.input_size]; seq_len];

        // Backpropagate through time
        for t in (0..seq_len).rev() {
            let x_t = &self.rnn.cached_inputs[t];
            let h_prev = &self.rnn.cached_hiddens[t];
            let h = &self.rnn.cached_hiddens[t + 1];

            // Gradient through tanh
            let dtanh = SimpleRnnCell::tanh_derivative(h);
            let dh = dtanh
                .iter()
                .zip(&dh_next)
                .map(|(&dt, &dn)| dt * dn)
                .collect::<Vec<_>>();

            // Accumulate gradients for parameters
            for i in 0..self.rnn.hidden_size {
                // Bias gradient
                self.rnn.grad_b_h[i] += dh[i];

                // Input weights gradient
                for j in 0..self.rnn.input_size {
                    self.rnn.grad_w_ih[i * self.rnn.input_size + j] += dh[i] * x_t[j];
                }

                // Hidden weights gradient
                for j in 0..self.rnn.hidden_size {
                    self.rnn.grad_w_hh[i * self.rnn.hidden_size + j] += dh[i] * h_prev[j];
                }
            }

            // Compute gradient for inputs at this time step
            for j in 0..self.rnn.input_size {
                for i in 0..self.rnn.hidden_size {
                    dx_sequence[t][j] += dh[i] * self.rnn.w_ih[i * self.rnn.input_size + j];
                }
            }

            // Compute gradient for previous hidden state
            let mut dh_prev = vec![0.0; self.rnn.hidden_size];
            for j in 0..self.rnn.hidden_size {
                for i in 0..self.rnn.hidden_size {
                    dh_prev[j] += dh[i] * self.rnn.w_hh[i * self.rnn.hidden_size + j];
                }
            }

            // Prepare for next iteration
            dh_next = dh_prev;
        }

        dx_sequence
    }

    fn update(&mut self, lr: f32) {
        // Update input weights
        for i in 0..self.rnn.w_ih.len() {
            self.rnn.w_ih[i] -= lr * self.rnn.grad_w_ih[i];
        }

        // Update hidden weights
        for i in 0..self.rnn.w_hh.len() {
            self.rnn.w_hh[i] -= lr * self.rnn.grad_w_hh[i];
        }

        // Update biases
        for i in 0..self.rnn.b_h.len() {
            self.rnn.b_h[i] -= lr * self.rnn.grad_b_h[i];
        }
    }

    fn zero_grad(&mut self) {
        for grad in &mut self.rnn.grad_w_ih {
            *grad = 0.0;
        }

        for grad in &mut self.rnn.grad_w_hh {
            *grad = 0.0;
        }

        for grad in &mut self.rnn.grad_b_h {
            *grad = 0.0;
        }
    }
}
