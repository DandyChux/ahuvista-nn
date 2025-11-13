/// Core trait that defines the interface for all neural network modules.
///
/// Modules must implement five primary operations:
/// - `forward`: Process inputs and produce outputs
/// - `backward`: Backpropagate gradients and update internal parameters
/// - `gradients`: Return the gradients of the module's parameters
/// - `apply_gradients`: Apply pre-computed gradients to update parameters.
/// - `zero_grad`: Reset accumulated gradients to zero, typically called at the start of a new batch.
///
/// Type parameters:
/// - `I`: Input type
/// - `O`: Output type
pub trait Module<I, O> {
    /// Perform a forward pass with the given input, returning the output.
    /// This method also stores any intermediate values needed for backpropagation.
    fn forward(&mut self, input: I) -> O;

    /// Backpropagate gradients with respect to the output.
    /// Returns gradients with respect to the input.
    fn backward(&mut self, grad_output: O) -> I;

    /// Get the gradients of the module's parameters.
    /// Returns a vector of gradients corresponding to each parameter.
    fn gradients(&self) -> Vec<f32>;

    /// Apply pre-computed gradients to update parameters.
    fn apply_gradients(&mut self, gradients: &[f32], learning_rate: f32);

    /// Reset accumulated gradients to zero, typically called at the start of a new batch.
    fn zero_grad(&mut self) {
        // Default implementation does nothing - override as needed
    }
}
