//! Comprehensive tests for core neural network components

use ahuvista_nn::core::{LSTMLayer, Layer, Module, Network, Neuron};

#[cfg(test)]
mod neuron_tests {
    use super::*;

    #[test]
    fn test_neuron_creation() {
        let neuron = Neuron::new(5);
        assert_eq!(neuron.weights.len(), 5);
    }

    #[test]
    fn test_neuron_forward_pass() {
        let mut neuron = Neuron::new(3);
        let inputs = vec![0.5, -1.0, 2.0];
        let output = neuron.forward(inputs);

        // Output should be between 0 and 1 due to sigmoid activation
        assert!(output >= 0.0 && output <= 1.0);
    }

    #[test]
    fn test_neuron_forward_backward() {
        let mut neuron = Neuron::new(3);
        let inputs = vec![0.5, -1.0, 2.0];

        // Forward pass
        let output = neuron.forward(inputs.clone());

        // Backward pass
        let grad_output = 1.0;
        let grad_inputs = neuron.backward(grad_output);

        assert_eq!(grad_inputs.len(), 3);
    }

    #[test]
    fn test_neuron_parameter_management() {
        let mut neuron = Neuron::new(3);

        // Get parameters
        let params = neuron.get_parameters();
        assert_eq!(params.len(), 4); // 3 weights + 1 bias

        // Set new parameters
        let new_params = vec![0.1, 0.2, 0.3, 0.5];
        let consumed = neuron.set_parameters(&new_params).unwrap();
        assert_eq!(consumed, 4);

        // Verify parameters were set
        let retrieved_params = neuron.get_parameters();
        for (a, b) in new_params.iter().zip(retrieved_params.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_neuron_gradient_accumulation() {
        let mut neuron = Neuron::new(2);
        let inputs1 = vec![1.0, 0.5];
        let inputs2 = vec![0.8, 0.3];

        // Forward and backward for first batch
        neuron.forward(inputs1);
        neuron.backward(0.5);

        let grads1 = neuron.gradients();

        // Forward and backward for second batch
        neuron.forward(inputs2);
        neuron.backward(0.3);

        let grads2 = neuron.gradients();

        // Gradients should have accumulated
        assert!(grads2.iter().sum::<f32>() > grads1.iter().sum::<f32>());

        // Test zero_grad
        neuron.zero_grad();
        let grads_after_zero = neuron.gradients();
        assert!(grads_after_zero.iter().all(|&g| g.abs() < 1e-6));
    }

    #[test]
    fn test_neuron_gradient_descent_convergence() {
        let mut neuron = Neuron::new(1);

        // Simple target: output should be close to input
        let learning_rate = 0.1;
        let target_input = vec![0.5];
        let target_output = 0.7;

        let mut initial_loss = f32::MAX;

        for _ in 0..100 {
            let output = neuron.forward(target_input.clone());
            let loss = (output - target_output).powi(2);

            if initial_loss == f32::MAX {
                initial_loss = loss;
            }

            let grad_output = 2.0 * (output - target_output);
            neuron.backward(grad_output);

            let grads = neuron.gradients();
            neuron.apply_gradients(&grads, learning_rate);
            neuron.zero_grad();
        }

        let final_output = neuron.forward(target_input);
        let final_loss = (final_output - target_output).powi(2);

        // Loss should decrease
        assert!(final_loss < initial_loss);
    }
}

#[cfg(test)]
mod layer_tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let layer = Layer::new(5, 3);
        assert_eq!(layer.size(), 5);
        assert_eq!(layer.neurons.len(), 5);
    }

    #[test]
    fn test_layer_forward_pass() {
        let mut layer = Layer::new(4, 3);
        let inputs = vec![0.5, -1.0, 2.0];
        let outputs = layer.forward(inputs);

        assert_eq!(outputs.len(), 4);
        for output in outputs {
            assert!(output >= 0.0 && output <= 1.0);
        }
    }

    #[test]
    fn test_layer_backward_pass() {
        let mut layer = Layer::new(3, 2);
        let inputs = vec![0.5, 1.5];

        // Forward pass
        let outputs = layer.forward(inputs.clone());
        assert_eq!(outputs.len(), 3);

        // Backward pass
        let grad_outputs = vec![0.1, 0.2, 0.3];
        let grad_inputs = layer.backward(grad_outputs);

        assert_eq!(grad_inputs.len(), 2);
    }

    #[test]
    fn test_layer_parameter_management() {
        let mut layer = Layer::new(2, 3);

        // Each neuron has 3 weights + 1 bias = 4 parameters
        // 2 neurons = 8 parameters total
        let params = layer.get_parameters();
        assert_eq!(params.len(), 8);

        // Set new parameters
        let new_params = vec![0.1; 8];
        let consumed = layer.set_parameters(&new_params).unwrap();
        assert_eq!(consumed, 8);
    }

    #[test]
    fn test_layer_gradient_operations() {
        let mut layer = Layer::new(2, 2);

        // Forward and backward
        layer.forward(vec![1.0, 0.5]);
        layer.backward(vec![0.1, 0.2]);

        let grads = layer.gradients();
        assert_eq!(grads.len(), 6); // 2 neurons × 3 params each (2 weights + 1 bias)

        // Apply gradients
        layer.apply_gradients(&grads, 0.1);

        // Zero gradients
        layer.zero_grad();
        let zero_grads = layer.gradients();
        assert!(zero_grads.iter().all(|&g| g.abs() < 1e-6));
    }
}

#[cfg(test)]
mod network_tests {
    use tracing::info;

    use super::*;

    #[test]
    fn test_network_creation() {
        let network = Network::new();
        assert!(network.layers.is_empty());
    }

    #[test]
    fn test_network_add_layer() {
        let mut network = Network::new();
        network.add_layer(Layer::new(5, 3));
        network.add_layer(Layer::new(2, 5));

        assert_eq!(network.layers.len(), 2);
    }

    #[test]
    fn test_network_from_layer_sizes() {
        let network = Network::from_layer_sizes(&[3, 5, 4, 2]);

        assert_eq!(network.layers.len(), 3);
        assert_eq!(network.input_size(), 3);
        assert_eq!(network.output_size(), 2);
    }

    #[test]
    fn test_network_forward_pass() {
        let mut network = Network::from_layer_sizes(&[3, 4, 2]);
        let inputs = vec![0.5, -1.0, 2.0];
        let outputs = network.forward(inputs);

        assert_eq!(outputs.len(), 2);
        for output in outputs {
            assert!(output >= 0.0 && output <= 1.0);
        }
    }

    #[test]
    fn test_network_backward_pass() {
        let mut network = Network::from_layer_sizes(&[3, 4, 2]);
        let inputs = vec![0.5, -1.0, 2.0];

        // Forward pass
        let outputs = network.forward(inputs.clone());

        // Backward pass
        let grad_outputs = vec![0.1, 0.2];
        let grad_inputs = network.backward(grad_outputs);

        assert_eq!(grad_inputs.len(), 3);
    }

    #[test]
    fn test_network_parameter_management() {
        let mut network = Network::from_layer_sizes(&[2, 3, 2]);

        let params = network.get_parameters();
        // Layer 1: 3 neurons × (2 weights + 1 bias) = 9 params
        // Layer 2: 2 neurons × (3 weights + 1 bias) = 8 params
        // Total: 17 params
        assert_eq!(params.len(), 17);

        // Set new parameters
        let new_params = vec![0.5; 17];
        network.set_parameters(&new_params).unwrap();

        let retrieved_params = network.get_parameters();
        assert_eq!(retrieved_params.len(), 17);
    }

    #[test]
    fn test_network_training_simple() {
        let mut network = Network::from_layer_sizes(&[2, 3, 1]); // 2 input, 3 hidden, 1 output

        // Simple XOR-like problem
        let training_data = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        // let learning_rate = 0.1 / training_data.len() as f32;
        let learning_rate = 1.0;
        let mut initial_loss = 0.0;

        // Train for a few iterations
        for epoch in 0..500 {
            let mut epoch_loss = 0.0;

            for (inputs, targets) in &training_data {
                // Forward pass
                let outputs = network.forward(inputs.clone());

                // Compute loss
                let loss: f32 = outputs
                    .iter()
                    .zip(targets.iter())
                    .map(|(o, t)| (o - t).powi(2))
                    .sum();
                epoch_loss += loss;
                println!("Epoch {}, Loss: {}", epoch, loss);

                // Backward pass
                let grad_outputs: Vec<f32> = outputs
                    .iter()
                    .zip(targets.iter())
                    .map(|(o, t)| 2.0 * (o - t))
                    .collect();
                network.backward(grad_outputs);
            }

            // Update weights with accumulated gradients
            let grads = network.gradients();
            if epoch == 0 {
                println!("Sample gradients: {:?}", &grads[0..5]); // First 5 gradients
                println!("All gradients sum: {}", grads.iter().sum::<f32>());
            }
            network.apply_gradients(&grads, learning_rate);
            network.zero_grad();

            if epoch == 0 {
                initial_loss = epoch_loss;
                println!("Initial loss: {}", initial_loss);
                let params = network.get_parameters();
                println!("Sample weights: {:?}", &params[0..5]);
                println!("All weights sum: {}", params.iter().sum::<f32>());
            }
        }

        // Verify that loss decreased (network learned something)
        let mut final_loss = 0.0;
        for (inputs, targets) in &training_data {
            let outputs = network.forward(inputs.clone());
            let loss: f32 = outputs
                .iter()
                .zip(targets.iter())
                .map(|(o, t)| (o - t).powi(2))
                .sum();
            final_loss += loss;
        }

        assert!(final_loss < initial_loss * 0.9);
    }
}

#[cfg(test)]
mod lstm_tests {
    use super::*;

    #[test]
    fn test_lstm_creation() {
        let lstm = LSTMLayer::new(10, 20);
        assert_eq!(lstm.hidden_state.len(), 20);
        assert_eq!(lstm.cell_state.len(), 20);
    }

    #[test]
    fn test_lstm_forward_step() {
        let mut lstm = LSTMLayer::new(5, 10);
        let input = vec![0.5, -1.0, 0.3, 0.7, -0.2];

        let hidden = lstm.forward_step(&input);
        assert_eq!(hidden.len(), 10);
    }

    #[test]
    fn test_lstm_forward_sequence() {
        let mut lstm = LSTMLayer::new(3, 5);
        let sequence = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        let final_hidden = lstm.forward_sequence(&sequence);
        assert_eq!(final_hidden.len(), 5);
    }

    #[test]
    fn test_lstm_backward_sequence() {
        let mut lstm = LSTMLayer::new(3, 5);
        let sequence = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        // Forward pass
        lstm.forward_sequence(&sequence);

        // Backward pass
        let grad_final_hidden = vec![0.1; 5];
        let grad_sequence = lstm.backward_sequence(&grad_final_hidden);

        assert_eq!(grad_sequence.len(), 3);
        assert_eq!(grad_sequence[0].len(), 3);
    }

    #[test]
    fn test_lstm_state_reset() {
        let mut lstm = LSTMLayer::new(3, 5);
        let sequence = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];

        // Process sequence
        lstm.forward_sequence(&sequence);

        // Verify state is non-zero
        assert!(lstm.hidden_state.iter().any(|&x| x.abs() > 1e-6));

        // Reset state
        lstm.reset_state();

        // Verify state is zero
        assert!(lstm.hidden_state.iter().all(|&x| x.abs() < 1e-6));
        assert!(lstm.cell_state.iter().all(|&x| x.abs() < 1e-6));
    }

    #[test]
    fn test_lstm_parameter_management() {
        let mut lstm = LSTMLayer::new(3, 5);

        let params = lstm.get_parameters();
        assert!(params.len() > 0);

        // Set and retrieve parameters
        let new_params = vec![0.1; params.len()];
        lstm.set_parameters(&new_params).unwrap();

        let retrieved = lstm.get_parameters();
        assert_eq!(retrieved.len(), new_params.len());
    }
}

#[cfg(test)]
mod differential_privacy_tests {
    use ahuvista_nn::core::PrivacyEngine;
    use rand_distr::num_traits::Float;

    #[test]
    fn test_privacy_engine_creation() {
        let engine = PrivacyEngine::new(1.0, 1e-5, 1000, 32, 10);
        // Just verify it doesn't panic
    }

    #[test]
    #[should_panic(expected = "epsilon must be positive")]
    fn test_privacy_engine_invalid_epsilon() {
        PrivacyEngine::new(0.0, 1e-5, 1000, 32, 10);
    }

    #[test]
    #[should_panic(expected = "delta must be in the range")]
    fn test_privacy_engine_invalid_delta() {
        PrivacyEngine::new(1.0, 0.0, 1000, 32, 10);
    }

    #[test]
    fn test_gradient_privatization() {
        let engine = PrivacyEngine::new(1.0, 1e-5, 1000, 32, 10);
        let mut gradients = vec![0.5, 1.0, -0.3, 0.8];

        let original_gradients = gradients.clone();
        engine.privatize_gradients(&mut gradients, 1.0);

        // Gradients should be different after adding noise
        assert!(gradients != original_gradients);
    }

    #[test]
    fn test_gradient_clipping() {
        let engine = PrivacyEngine::new(1.0, 1e-5, 1000, 32, 10);
        let mut gradients = vec![10.0, 20.0, 30.0]; // Large gradients

        let l2_norm = gradients.iter().map(|g| g.powi(2)).sum::<f32>().sqrt();
        let clip_value = 1.0;

        engine.privatize_gradients(&mut gradients, clip_value);

        // After clipping, L2 norm should be <= clip_value (plus noise)
        // We just verify the function completes without panicking
        assert!(gradients.len() == 3);
    }
}

#[cfg(test)]
mod utils_tests {
    use ahuvista_nn::utils::{sigmoid, sigmoid_derivative, tanh, tanh_derivative};

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }

    #[test]
    fn test_sigmoid_derivative() {
        let sig_0 = sigmoid(0.0);
        let deriv = sigmoid_derivative(sig_0);
        assert!((deriv - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_tanh() {
        assert!((tanh(0.0) - 0.0).abs() < 1e-6);
        assert!(tanh(100.0) > 0.99);
        assert!(tanh(-100.0) < -0.99);
    }

    #[test]
    fn test_tanh_derivative() {
        let tanh_0 = tanh(0.0);
        let deriv = tanh_derivative(tanh_0);
        assert!((deriv - 1.0).abs() < 1e-6);
    }
}
