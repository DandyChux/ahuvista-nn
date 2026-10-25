use crate::core::{Activation, Layer, Module, sigmoid, sigmoid_derivative, tanh, tanh_derivative};
use rand::Rng;

/// A single Long Short-Term Memory (LSTM) layer.
///
/// This layer is designed to be a building block for recurrent neural networks (RNNs)
/// and is optimized for sequence processing tasks where long-term dependencies are crucial.
#[derive(Debug)]
pub struct LSTMLayer {
    // Four internal layers for the gates
    pub input_gate: Layer,
    pub forget_gate: Layer,
    pub output_gate: Layer,
    pub cell_gate: Layer, // Computes the candidate cell state

    // State vectors
    pub hidden_state: Vec<f32>,
    pub cell_state: Vec<f32>,

    // Cached values for backpropagation
    cached_inputs: Vec<Vec<f32>>,
    cached_hidden_states: Vec<Vec<f32>>,
    cached_cell_states: Vec<Vec<f32>>,
    cached_input_gate_outputs: Vec<Vec<f32>>,
    cached_forget_gate_outputs: Vec<Vec<f32>>,
    cached_output_gate_outputs: Vec<Vec<f32>>,
    cached_cell_gate_outputs: Vec<Vec<f32>>,
}

impl LSTMLayer {
    /// Creates a new `LSTMLayer`.
    ///
    /// # Arguments
    ///
    /// * `input_size` - The number of features in the input.
    /// * `hidden_size` - The number of features in the hidden state.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let combined_input_size = input_size + hidden_size;

        Self {
            input_gate: Layer::new(hidden_size, combined_input_size, Activation::Sigmoid),
            forget_gate: Layer::new(hidden_size, combined_input_size, Activation::Sigmoid),
            output_gate: Layer::new(hidden_size, combined_input_size, Activation::Sigmoid),
            cell_gate: Layer::new(hidden_size, combined_input_size, Activation::Tanh),
            hidden_state: vec![0.0; hidden_size],
            cell_state: vec![0.0; hidden_size],
            cached_inputs: Vec::new(),
            cached_hidden_states: Vec::new(),
            cached_cell_states: Vec::new(),
            cached_input_gate_outputs: Vec::new(),
            cached_forget_gate_outputs: Vec::new(),
            cached_output_gate_outputs: Vec::new(),
            cached_cell_gate_outputs: Vec::new(),
        }
    }

    /// Get all parameters from all gates.
    pub fn get_parameters(&self) -> Vec<f32> {
        let mut params = Vec::new();
        params.extend(self.input_gate.get_parameters());
        params.extend(self.forget_gate.get_parameters());
        params.extend(self.output_gate.get_parameters());
        params.extend(self.cell_gate.get_parameters());
        params
    }

    /// Set all parameters for all gates.
    /// Returns the number of parameters consumed.
    pub fn set_parameters(&mut self, params: &[f32]) -> Result<usize, Box<dyn std::error::Error>> {
        let mut offset = 0;
        offset += self.input_gate.set_parameters(&params[offset..])?;
        offset += self.forget_gate.set_parameters(&params[offset..])?;
        offset += self.output_gate.set_parameters(&params[offset..])?;
        offset += self.cell_gate.set_parameters(&params[offset..])?;
        Ok(offset)
    }

    /// Processes a single time step of a sequence.
    ///
    /// # Arguments
    ///
    /// * `input` - The input vector for the current time step.
    ///
    /// # Returns
    ///
    /// The hidden state for the current time step.
    pub fn forward_step(&mut self, input: &Vec<f32>) -> Vec<f32> {
        let combined_input = [self.hidden_state.clone(), input.clone()].concat();

        // Input gate
        let input_gate_output = self
            .input_gate
            .forward(combined_input.clone())
            .iter()
            .map(|&x| sigmoid(x))
            .collect::<Vec<f32>>();

        // Forget gate
        let forget_gate_output = self
            .forget_gate
            .forward(combined_input.clone())
            .iter()
            .map(|&x| sigmoid(x))
            .collect::<Vec<f32>>();

        // Output gate
        let output_gate_output = self
            .output_gate
            .forward(combined_input.clone())
            .iter()
            .map(|&x| sigmoid(x))
            .collect::<Vec<f32>>();

        // Cell gate (candidate cell state)
        let cell_gate_output = self
            .cell_gate
            .forward(combined_input.clone())
            .iter()
            .map(|&x| tanh(x))
            .collect::<Vec<f32>>();

        // Update cell state
        self.cell_state = self
            .cell_state
            .iter()
            .zip(forget_gate_output.iter())
            .map(|(&cs, &fg)| cs * fg)
            .zip(input_gate_output.iter().zip(cell_gate_output.iter()))
            .map(|(cs_fg, (ig, cg))| cs_fg + ig * cg)
            .collect();

        // Update hidden state
        self.hidden_state = self
            .cell_state
            .iter()
            .map(|&cs| tanh(cs))
            .zip(output_gate_output.iter())
            .map(|(tanh_cs, &og)| tanh_cs * og)
            .collect();

        // Cache values for backpropagation
        self.cached_inputs.push(input.clone());
        self.cached_hidden_states.push(self.hidden_state.clone());
        self.cached_cell_states.push(self.cell_state.clone());
        self.cached_input_gate_outputs
            .push(input_gate_output.clone());
        self.cached_forget_gate_outputs
            .push(forget_gate_output.clone());
        self.cached_output_gate_outputs
            .push(output_gate_output.clone());
        self.cached_cell_gate_outputs.push(cell_gate_output.clone());

        self.hidden_state.clone()
    }

    /// Processes an entire sequence of inputs.
    ///
    /// # Arguments
    ///
    /// * `sequence` - A vector of input vectors, where each inner vector represents a time step.
    ///
    /// # Returns
    ///
    /// The final hidden state after processing the entire sequence.
    pub fn forward_sequence(&mut self, sequence: &Vec<Vec<f32>>) -> Vec<f32> {
        self.reset_state();
        for input in sequence {
            self.forward_step(input);
        }
        self.hidden_state.clone()
    }

    /// Performs backpropagation for a single time step.
    ///
    /// # Arguments
    ///
    /// * `grad_hidden_next` - The gradient of the loss with respect to the next hidden state.
    /// * `grad_cell_next` - The gradient of the loss with respect to the next cell state.
    ///
    /// # Returns
    ///
    /// A tuple containing the gradients with respect to the input, the previous hidden state, and the previous cell state.
    fn backward_step(
        &mut self,
        grad_hidden_next: &Vec<f32>,
        grad_cell_next: &Vec<f32>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Unpack cached values for this time step
        let input = self.cached_inputs.pop().unwrap();
        let _hidden_state = self.cached_hidden_states.pop().unwrap();
        let cell_state = self.cached_cell_states.pop().unwrap();
        let prev_cell_state = if self.cached_cell_states.is_empty() {
            vec![0.0; self.cell_state.len()]
        } else {
            self.cached_cell_states.last().unwrap().clone()
        };
        let input_gate_output = self.cached_input_gate_outputs.pop().unwrap();
        let forget_gate_output = self.cached_forget_gate_outputs.pop().unwrap();
        let output_gate_output = self.cached_output_gate_outputs.pop().unwrap();
        let cell_gate_output = self.cached_cell_gate_outputs.pop().unwrap();

        // Gradient of loss w.r.t. output gate
        let grad_output_gate = grad_hidden_next
            .iter()
            .zip(cell_state.iter())
            .map(|(&g, &cs)| g * tanh(cs))
            .collect::<Vec<f32>>();
        let grad_output_gate_pre_activation = grad_output_gate
            .iter()
            .zip(output_gate_output.iter())
            .map(|(&g, &o)| g * sigmoid_derivative(o))
            .collect::<Vec<f32>>();

        // Gradient of loss w.r.t. cell state
        let grad_cell_state = grad_hidden_next
            .iter()
            .zip(output_gate_output.iter())
            .zip(cell_state.iter())
            .map(|((&g, &o), &cs)| g * o * tanh_derivative(tanh(cs)))
            .zip(grad_cell_next.iter())
            .map(|(g, &g_next)| g + g_next)
            .collect::<Vec<f32>>();

        // Gradient of loss w.r.t. cell gate
        let grad_cell_gate = grad_cell_state
            .iter()
            .zip(input_gate_output.iter())
            .map(|(&g, &i)| g * i)
            .collect::<Vec<f32>>();
        let grad_cell_gate_pre_activation = grad_cell_gate
            .iter()
            .zip(cell_gate_output.iter())
            .map(|(&g, &c)| g * tanh_derivative(c))
            .collect::<Vec<f32>>();

        // Gradient of loss w.r.t. input gate
        let grad_input_gate = grad_cell_state
            .iter()
            .zip(cell_gate_output.iter())
            .map(|(&g, &c)| g * c)
            .collect::<Vec<f32>>();
        let grad_input_gate_pre_activation = grad_input_gate
            .iter()
            .zip(input_gate_output.iter())
            .map(|(&g, &i)| g * sigmoid_derivative(i))
            .collect::<Vec<f32>>();

        // Gradient of loss w.r.t. forget gate
        let grad_forget_gate = grad_cell_state
            .iter()
            .zip(prev_cell_state.iter())
            .map(|(&g, &pcs)| g * pcs)
            .collect::<Vec<f32>>();
        let grad_forget_gate_pre_activation = grad_forget_gate
            .iter()
            .zip(forget_gate_output.iter())
            .map(|(&g, &f)| g * sigmoid_derivative(f))
            .collect::<Vec<f32>>();

        // Backpropagate through the gates
        let grad_combined_input_input_gate =
            self.input_gate.backward(grad_input_gate_pre_activation);
        let grad_combined_input_forget_gate =
            self.forget_gate.backward(grad_forget_gate_pre_activation);
        let grad_combined_input_output_gate =
            self.output_gate.backward(grad_output_gate_pre_activation);
        let grad_combined_input_cell_gate = self.cell_gate.backward(grad_cell_gate_pre_activation);

        // Sum the gradients for the combined input
        let grad_combined_input = grad_combined_input_input_gate
            .iter()
            .zip(grad_combined_input_forget_gate.iter())
            .zip(grad_combined_input_output_gate.iter())
            .zip(grad_combined_input_cell_gate.iter())
            .map(|(((&a, &b), &c), &d)| a + b + c + d)
            .collect::<Vec<f32>>();

        // Split the gradients for the input and the previous hidden state
        let hidden_size = self.hidden_state.len();
        let grad_prev_hidden = grad_combined_input[..hidden_size].to_vec();
        let grad_input = grad_combined_input[hidden_size..].to_vec();

        // Gradient of loss w.r.t. previous cell state
        let grad_prev_cell = grad_cell_state
            .iter()
            .zip(forget_gate_output.iter())
            .map(|(&g, &f)| g * f)
            .collect::<Vec<f32>>();

        (grad_input, grad_prev_hidden, grad_prev_cell)
    }

    /// Performs backpropagation for an entire sequence.
    ///
    /// # Arguments
    ///
    /// * `grad_final_hidden` - The gradient of the loss with respect to the final hidden state.
    ///
    /// # Returns
    ///
    /// The gradients with respect to the input sequence.
    pub fn backward_sequence(&mut self, grad_final_hidden: &Vec<f32>) -> Vec<Vec<f32>> {
        let mut grad_hidden = grad_final_hidden.clone();
        let mut grad_cell = vec![0.0; self.cell_state.len()];
        let mut grad_sequence = Vec::new();

        for _ in 0..self.cached_inputs.len() {
            let (grad_input, grad_prev_hidden, grad_prev_cell) =
                self.backward_step(&grad_hidden, &grad_cell);
            grad_sequence.push(grad_input);
            grad_hidden = grad_prev_hidden;
            grad_cell = grad_prev_cell;
        }

        grad_sequence.reverse();
        grad_sequence
    }

    /// Resets the hidden and cell states to zero.
    pub fn reset_state(&mut self) {
        self.hidden_state.fill(0.0);
        self.cell_state.fill(0.0);
        self.cached_inputs.clear();
        self.cached_hidden_states.clear();
        self.cached_cell_states.clear();
        self.cached_input_gate_outputs.clear();
        self.cached_forget_gate_outputs.clear();
        self.cached_output_gate_outputs.clear();
        self.cached_cell_gate_outputs.clear();
    }
}
