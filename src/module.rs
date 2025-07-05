/// Core trait that defines the interface for all neural network modules.
///
/// Modules must implement three primary operations:
/// - `forward`: Process inputs and produce outputs
/// - `backward`: Backpropagate gradients and update internal parameters
/// - `update`: Apply accumulated gradients to update parameters
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

    /// Update model parameters using accumulated gradients with learning rate `lr`.
    /// Usually called after one or more backward passes to apply gradient descent.
    fn update(&mut self, lr: f32);

    /// Reset accumulated gradients to zero, typically called at the start of a new batch.
    fn zero_grad(&mut self) {
        // Default implementation does nothing - override as needed
    }
}
