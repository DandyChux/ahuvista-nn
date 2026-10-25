use crate::core::{Activation, Layer, Module, sigmoid, tanh};

/// A single Gated Recurrent Unit layer.
///
/// The GRU processes only dynamic features. Static features are handled by
/// `SequenceNet` and concatenated with the final hidden state before the
/// prediction layer.
#[derive(Debug)]
pub struct GruLayer {
    pub input_size: usize,
    pub hidden_size: usize,

    /// Update gate.
    ///
    /// Determines how much of the candidate state should replace the
    /// previous hidden state.
    pub update_gate: Layer,

    /// Reset gate.
    ///
    /// Determines how much of the previous hidden state contributes to the
    /// candidate state.
    pub reset_gate: Layer,

    /// Candidate hidden-state computation.
    pub candidate_gate: Layer,

    /// Hidden state after the most recent timestep.
    pub hidden_state: Vec<f32>,

    /// Cached values used during backpropagation.
    cached_inputs: Vec<Vec<f32>>,
    cached_previous_hidden_states: Vec<Vec<f32>>,
    cached_update_gates: Vec<Vec<f32>>,
    cached_reset_gates: Vec<Vec<f32>>,
    cached_candidate_states: Vec<Vec<f32>>,
}

impl GruLayer {
    /// Create a new GRU layer.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let combined_size = input_size + hidden_size;

        Self {
            input_size,
            hidden_size,

            update_gate: Layer::new(hidden_size, combined_size, Activation::Sigmoid),

            reset_gate: Layer::new(hidden_size, combined_size, Activation::Sigmoid),

            candidate_gate: Layer::new(hidden_size, combined_size, Activation::Tanh),

            hidden_state: vec![0.0; hidden_size],

            cached_inputs: Vec::new(),
            cached_previous_hidden_states: Vec::new(),
            cached_update_gates: Vec::new(),
            cached_reset_gates: Vec::new(),
            cached_candidate_states: Vec::new(),
        }
    }

    /// Reset the hidden state and all backpropagation caches.
    pub fn reset_state(&mut self) {
        self.hidden_state.fill(0.0);

        self.cached_inputs.clear();
        self.cached_previous_hidden_states.clear();
        self.cached_update_gates.clear();
        self.cached_reset_gates.clear();
        self.cached_candidate_states.clear();
    }

    /// Process one timestep.
    pub fn forward_step(&mut self, input: &[f32]) -> Vec<f32> {
        assert_eq!(
            input.len(),
            self.input_size,
            "GRU input size mismatch: expected {}, got {}",
            self.input_size,
            input.len()
        );

        let previous_hidden = self.hidden_state.clone();

        // The update and reset gates receive [input, previous_hidden].
        let mut combined_input = Vec::with_capacity(self.input_size + self.hidden_size);

        combined_input.extend_from_slice(input);
        combined_input.extend_from_slice(&previous_hidden);

        let update_gate = self.update_gate.forward(combined_input.clone());
        let reset_gate = self.reset_gate.forward(combined_input);

        // The Layer already applies sigmoid activation.
        let reset_hidden: Vec<f32> = reset_gate
            .iter()
            .zip(previous_hidden.iter())
            .map(|(&r, &h)| r * h)
            .collect();

        // The candidate receives [input, reset_gate * previous_hidden].
        let mut candidate_input = Vec::with_capacity(self.input_size + self.hidden_size);

        candidate_input.extend_from_slice(input);
        candidate_input.extend_from_slice(&reset_hidden);

        // The Layer already applies tanh activation.
        let candidate_state = self.candidate_gate.forward(candidate_input);

        // h_t = (1 - z_t) * h_(t-1) + z_t * n_t
        let next_hidden: Vec<f32> = update_gate
            .iter()
            .zip(candidate_state.iter())
            .zip(previous_hidden.iter())
            .map(|((&z, &n), &h)| (1.0 - z) * h + z * n)
            .collect();

        self.hidden_state = next_hidden.clone();

        self.cached_inputs.push(input.to_vec());
        self.cached_previous_hidden_states.push(previous_hidden);
        self.cached_update_gates.push(update_gate);
        self.cached_reset_gates.push(reset_gate);
        self.cached_candidate_states.push(candidate_state);

        next_hidden
    }

    /// Process an entire sequence and return its final hidden state.
    pub fn forward_sequence(&mut self, sequence: &[Vec<f32>]) -> Vec<f32> {
        self.reset_state();

        for input in sequence {
            self.forward_step(input);
        }

        self.hidden_state.clone()
    }

    /// Backpropagate through one timestep.
    ///
    /// The cached values are consumed in reverse chronological order.
    fn backward_step(&mut self, grad_hidden: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let input = self
            .cached_inputs
            .pop()
            .expect("GRU backward called without a matching forward pass");

        let previous_hidden = self
            .cached_previous_hidden_states
            .pop()
            .expect("Missing cached previous hidden state");

        let update_gate = self
            .cached_update_gates
            .pop()
            .expect("Missing cached update gate");

        let reset_gate = self
            .cached_reset_gates
            .pop()
            .expect("Missing cached reset gate");

        let candidate_state = self
            .cached_candidate_states
            .pop()
            .expect("Missing cached candidate state");

        // Gradient of h_t = (1 - z_t) * h_prev + z_t * n_t.
        let grad_update_gate: Vec<f32> = grad_hidden
            .iter()
            .zip(candidate_state.iter())
            .zip(previous_hidden.iter())
            .map(|((&dh, &n), &h)| dh * (n - h))
            .collect();

        let grad_candidate_state: Vec<f32> = grad_hidden
            .iter()
            .zip(update_gate.iter())
            .map(|(&dh, &z)| dh * z)
            .collect();

        let mut grad_previous_hidden: Vec<f32> = grad_hidden
            .iter()
            .zip(update_gate.iter())
            .map(|(&dh, &z)| dh * (1.0 - z))
            .collect();

        // Backpropagate through the candidate layer.
        //
        // candidate input = [input, reset_gate * previous_hidden]
        let grad_candidate_input = self.candidate_gate.backward(grad_candidate_state);

        let grad_input_from_candidate = &grad_candidate_input[..self.input_size];

        let grad_reset_hidden = &grad_candidate_input[self.input_size..];

        let mut grad_input = grad_input_from_candidate.to_vec();

        // reset_hidden = reset_gate * previous_hidden
        let grad_reset_gate: Vec<f32> = grad_reset_hidden
            .iter()
            .zip(previous_hidden.iter())
            .map(|(&g, &h)| g * h)
            .collect();

        for (gradient, &reset_value) in grad_previous_hidden
            .iter_mut()
            .zip(grad_reset_hidden.iter())
        {
            *gradient += reset_value;
        }

        // Backpropagate through the reset gate.
        //
        // reset_gate receives [input, previous_hidden].
        let grad_reset_gate_input = self.reset_gate.backward(grad_reset_gate);

        for (gradient, &gate_gradient) in grad_input
            .iter_mut()
            .zip(grad_reset_gate_input[..self.input_size].iter())
        {
            *gradient += gate_gradient;
        }

        for (gradient, &gate_gradient) in grad_previous_hidden
            .iter_mut()
            .zip(grad_reset_gate_input[self.input_size..].iter())
        {
            *gradient += gate_gradient;
        }

        // Backpropagate through the update gate.
        let grad_update_gate_input = self.update_gate.backward(grad_update_gate);

        for (gradient, &gate_gradient) in grad_input
            .iter_mut()
            .zip(grad_update_gate_input[..self.input_size].iter())
        {
            *gradient += gate_gradient;
        }

        for (gradient, &gate_gradient) in grad_previous_hidden
            .iter_mut()
            .zip(grad_update_gate_input[self.input_size..].iter())
        {
            *gradient += gate_gradient;
        }

        (grad_input, grad_previous_hidden)
    }

    /// Backpropagate through a complete sequence.
    ///
    /// Returns one input gradient per timestep.
    pub fn backward_sequence(&mut self, grad_final_hidden: &[f32]) -> Vec<Vec<f32>> {
        assert_eq!(
            grad_final_hidden.len(),
            self.hidden_size,
            "GRU hidden gradient size mismatch"
        );

        let sequence_length = self.cached_inputs.len();
        let mut gradients = vec![vec![0.0; self.input_size]; sequence_length];

        let mut grad_hidden = grad_final_hidden.to_vec();

        for index in (0..sequence_length).rev() {
            let (grad_input, grad_previous_hidden) = self.backward_step(&grad_hidden);

            gradients[index] = grad_input;
            grad_hidden = grad_previous_hidden;
        }

        // Reset the recurrent state after consuming the forward cache.
        self.hidden_state.fill(0.0);

        gradients
    }

    /// Get all GRU parameters in gate order.
    pub fn get_parameters(&self) -> Vec<f32> {
        let mut parameters = Vec::new();

        parameters.extend(self.update_gate.get_parameters());
        parameters.extend(self.reset_gate.get_parameters());
        parameters.extend(self.candidate_gate.get_parameters());

        parameters
    }

    /// Set all GRU parameters in gate order.
    pub fn set_parameters(
        &mut self,
        parameters: &[f32],
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let mut offset = 0;

        offset += self.update_gate.set_parameters(&parameters[offset..])?;
        offset += self.reset_gate.set_parameters(&parameters[offset..])?;
        offset += self.candidate_gate.set_parameters(&parameters[offset..])?;

        Ok(offset)
    }

    /// Return the number of trainable GRU parameters.
    pub fn parameter_count(&self) -> usize {
        self.get_parameters().len()
    }

    /// Return accumulated gradients in the same order as `get_parameters`.
    pub fn gradients(&self) -> Vec<f32> {
        let mut gradients = Vec::new();

        gradients.extend(self.update_gate.gradients());
        gradients.extend(self.reset_gate.gradients());
        gradients.extend(self.candidate_gate.gradients());

        gradients
    }

    /// Apply gradients in the same order as `get_parameters`.
    pub fn apply_gradients(&mut self, gradients: &[f32], learning_rate: f32) {
        let update_size = self.update_gate.gradients().len();
        let reset_size = self.reset_gate.gradients().len();
        let candidate_size = self.candidate_gate.gradients().len();

        let expected_size = update_size + reset_size + candidate_size;

        assert_eq!(
            gradients.len(),
            expected_size,
            "GRU gradient size mismatch: expected {}, got {}",
            expected_size,
            gradients.len()
        );

        let mut offset = 0;

        self.update_gate
            .apply_gradients(&gradients[offset..offset + update_size], learning_rate);
        offset += update_size;

        self.reset_gate
            .apply_gradients(&gradients[offset..offset + reset_size], learning_rate);
        offset += reset_size;

        self.candidate_gate
            .apply_gradients(&gradients[offset..offset + candidate_size], learning_rate);
    }

    /// Clear accumulated parameter gradients.
    pub fn zero_grad(&mut self) {
        self.update_gate.zero_grad();
        self.reset_gate.zero_grad();
        self.candidate_gate.zero_grad();
    }
}
