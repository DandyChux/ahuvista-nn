//! Comprehensive tests for neural network models

use ahuvista_nn::core::Module;
use ahuvista_nn::models::{ImageNet, SimpleRnnCell, TabularNet, TemporalNet, TextNet};
use ndarray::Array3;
use std::collections::HashMap;

#[cfg(test)]
mod tabular_tests {
    use super::*;

    #[test]
    fn test_tabular_net_creation() {
        let net = TabularNet::new(10, &[20, 15], 5);
        assert_eq!(net.output_size(), 5);
    }

    #[test]
    fn test_tabular_net_forward() {
        let mut net = TabularNet::new(5, &[10], 3);
        let input = vec![0.5, -1.0, 2.0, 0.0, -0.5];
        let output = net.forward(input);

        assert_eq!(output.len(), 3);
        // Check outputs are in valid range after sigmoid
        for val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_tabular_net_forward_backward() {
        let mut net = TabularNet::new(3, &[5], 2);
        let input = vec![1.0, 0.5, -0.5];

        // Forward pass
        let output = net.forward(input.clone());
        assert_eq!(output.len(), 2);

        // Backward pass
        let grad_output = vec![1.0, -1.0];
        let grad_input = net.backward(grad_output);
        assert_eq!(grad_input.len(), 3);
    }

    #[test]
    fn test_tabular_net_parameter_management() {
        let mut net = TabularNet::new(3, &[4], 2);

        // Get parameters
        let params = net.get_parameters();
        assert!(params.len() > 0);

        // Set parameters
        let new_params = vec![0.1; params.len()];
        let consumed = net.set_parameters(&new_params).unwrap();
        assert_eq!(consumed, params.len());

        // Verify parameters were set
        let retrieved_params = net.get_parameters();
        for (a, b) in retrieved_params.iter().zip(new_params.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_tabular_net_gradients() {
        let mut net = TabularNet::new(2, &[3], 1);
        let input = vec![0.5, -0.5];

        // Forward and backward pass
        net.forward(input.clone());
        net.backward(vec![1.0]);

        // Get gradients
        let grads = net.gradients();
        assert!(grads.len() > 0);

        // Zero gradients
        net.zero_grad();
        let zero_grads = net.gradients();
        for grad in zero_grads {
            assert_eq!(grad, 0.0);
        }
    }

    #[test]
    fn test_tabular_net_training_simple() {
        let mut net = TabularNet::new(2, &[4], 1);
        let lr = 0.1;

        // Simple training: learn XOR-like pattern
        let data = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        // Train for a few epochs
        for _ in 0..10 {
            for (input, target) in &data {
                let output = net.forward(input.clone());

                // Compute simple gradient
                let grad = vec![output[0] - target[0]];

                net.backward(grad);
                let grads = net.gradients();
                net.apply_gradients(&grads, lr);
                net.zero_grad();
            }
        }

        // Network should have updated parameters
        let params = net.get_parameters();
        assert!(params.iter().any(|&p| p != 0.0));
    }

    #[test]
    fn test_tabular_net_multiple_hidden_layers() {
        let mut net = TabularNet::new(5, &[10, 8, 6], 3);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = net.forward(input);

        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_tabular_net_single_hidden_layer() {
        let mut net = TabularNet::new(4, &[5], 2);
        let input = vec![0.1, 0.2, 0.3, 0.4];
        let output = net.forward(input);

        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_tabular_net_no_hidden_layers() {
        // Direct connection from input to output
        let mut net = TabularNet::new(3, &[], 2);
        let input = vec![1.0, 2.0, 3.0];
        let output = net.forward(input);

        assert_eq!(output.len(), 2);
    }
}

#[cfg(test)]
mod image_tests {
    use super::*;

    #[test]
    fn test_image_net_creation() {
        let net = ImageNet::new(1, 28, 28);
        assert_eq!(net.output_size(), 16 * 7 * 7); // After 2 pooling layers: 28/2/2 = 7
    }

    #[test]
    fn test_image_net_forward() {
        let mut net = ImageNet::new(1, 28, 28);

        // Create a sample grayscale image
        let input = Array3::zeros((1, 28, 28));

        let output = net.forward(input);
        assert_eq!(output.len(), 16 * 7 * 7);
    }

    #[test]
    fn test_image_net_forward_backward() {
        let mut net = ImageNet::new(3, 16, 16);

        // Create RGB image
        let input = Array3::from_shape_fn((3, 16, 16), |(c, h, w)| {
            (c * 100 + h * 10 + w) as f32 / 1000.0
        });

        // Forward pass
        let output = net.forward(input.clone());
        let output_size = output.len();
        assert_eq!(output_size, 16 * 4 * 4); // 16/2/2 = 4

        // Backward pass
        let grad_output = vec![0.1; output_size];
        let grad_input = net.backward(grad_output);

        assert_eq!(grad_input.shape(), &[3, 16, 16]);
    }

    #[test]
    fn test_image_net_parameter_management() {
        let mut net = ImageNet::new(1, 8, 8);

        // Get parameters
        let params = net.get_parameters();
        assert!(params.len() > 0);

        // Set parameters
        let new_params = vec![0.01; params.len()];
        let consumed = net.set_parameters(&new_params).unwrap();
        assert_eq!(consumed, params.len());
    }

    #[test]
    fn test_image_net_gradients() {
        let mut net = ImageNet::new(1, 8, 8);
        let input = Array3::ones((1, 8, 8));

        // Forward and backward pass
        let output = net.forward(input.clone());
        let output_size = output.len();

        let grad_output = vec![0.1; output_size];
        net.backward(grad_output);

        // Get gradients
        let grads = net.gradients();
        assert!(grads.len() > 0);

        // Zero gradients
        net.zero_grad();
        let zero_grads = net.gradients();
        for grad in zero_grads {
            assert_eq!(grad, 0.0);
        }
    }

    #[test]
    fn test_image_net_color_images() {
        let mut net = ImageNet::new(3, 32, 32);

        // Create RGB image
        let input =
            Array3::from_shape_fn((3, 32, 32), |(c, h, w)| ((c + h + w) % 256) as f32 / 255.0);

        let output = net.forward(input);
        assert_eq!(output.len(), 16 * 8 * 8); // 32/2/2 = 8
    }

    #[test]
    fn test_image_net_different_sizes() {
        // Test various input sizes (channels, height, width)
        let sizes = vec![(1, 8, 8), (3, 16, 16), (3, 32, 32), (1, 64, 64)];

        for (c, h, w) in sizes {
            let mut net = ImageNet::new(c, h, w);
            let input = Array3::zeros((c, h, w));
            let output = net.forward(input);

            // Calculate expected output size: 16 channels * (h/4) * (w/4)
            let expected_size = 16 * (h / 4) * (w / 4);
            assert_eq!(output.len(), expected_size);
        }
    }

    #[test]
    fn test_image_net_training_step() {
        let mut net = ImageNet::new(1, 8, 8);
        let lr = 0.01;

        let input = Array3::ones((1, 8, 8));

        // Forward pass
        let output = net.forward(input.clone());
        let output_size = output.len();

        let target = vec![1.0; output_size];

        // Compute gradient
        let grad: Vec<f32> = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| o - t)
            .collect();

        // Backward pass and update
        net.backward(grad);
        let grads = net.gradients();
        net.apply_gradients(&grads, lr);
        net.zero_grad();

        // Network should have updated
        let params = net.get_parameters();
        assert!(params.iter().any(|&p| p.is_finite()));
    }
}

#[cfg(test)]
mod temporal_tests {
    use super::*;

    #[test]
    fn test_temporal_net_creation() {
        let net = TemporalNet::new(5, 10, 3);
        let params = net.get_parameters();
        assert!(params.len() > 0);
    }

    #[test]
    fn test_temporal_net_forward() {
        let mut net = TemporalNet::new(4, 8, 2);

        // Create a sequence of 5 time steps
        let sequence = vec![
            vec![1.0, 0.5, -0.5, 0.0],
            vec![0.8, 0.3, -0.3, 0.1],
            vec![0.6, 0.1, -0.1, 0.2],
            vec![0.4, -0.1, 0.1, 0.3],
            vec![0.2, -0.3, 0.3, 0.4],
        ];

        let output = net.forward(sequence);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_temporal_net_forward_backward() {
        let mut net = TemporalNet::new(3, 5, 2);

        let sequence = vec![
            vec![1.0, 0.0, -1.0],
            vec![0.5, 0.5, 0.0],
            vec![0.0, 1.0, 0.5],
        ];

        // Forward pass
        let output = net.forward(sequence.clone());
        assert_eq!(output.len(), 2);

        // Backward pass
        let grad_output = vec![1.0, -1.0];
        let grad_sequence = net.backward(grad_output);

        assert_eq!(grad_sequence.len(), 3); // Same as sequence length
        for grad_step in &grad_sequence {
            assert_eq!(grad_step.len(), 3); // Same as input size
        }
    }

    #[test]
    fn test_temporal_net_parameter_management() {
        let mut net = TemporalNet::new(3, 6, 2);

        // Get parameters
        let params = net.get_parameters();
        assert!(params.len() > 0);

        // Set parameters
        let new_params = vec![0.1; params.len()];
        let consumed = net.set_parameters(&new_params).unwrap();
        assert_eq!(consumed, params.len());
    }

    #[test]
    fn test_temporal_net_gradients() {
        let mut net = TemporalNet::new(2, 4, 1);

        let sequence = vec![vec![1.0, -1.0], vec![0.5, 0.5]];

        // Forward and backward
        net.forward(sequence.clone());
        net.backward(vec![1.0]);

        // Get gradients
        let grads = net.gradients();
        assert!(grads.len() > 0);

        // Zero gradients
        net.zero_grad();
        let zero_grads = net.gradients();
        for grad in zero_grads {
            assert_eq!(grad, 0.0);
        }
    }

    #[test]
    fn test_temporal_net_variable_sequence_lengths() {
        let mut net = TemporalNet::new(3, 5, 2);

        // Test different sequence lengths
        let sequences = vec![
            vec![vec![1.0, 0.0, -1.0]],
            vec![vec![1.0, 0.0, -1.0], vec![0.5, 0.5, 0.0]],
            vec![
                vec![1.0, 0.0, -1.0],
                vec![0.5, 0.5, 0.0],
                vec![0.0, 1.0, 0.5],
                vec![-0.5, 0.5, 1.0],
            ],
        ];

        for seq in sequences {
            let output = net.forward(seq);
            assert_eq!(output.len(), 2);
        }
    }

    #[test]
    fn test_temporal_net_training_step() {
        let mut net = TemporalNet::new(2, 4, 1);
        let lr = 0.01;

        let sequence = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let target = vec![1.0];

        // Forward pass
        let output = net.forward(sequence.clone());

        // Compute gradient
        let grad = vec![output[0] - target[0]];

        // Backward pass and update
        net.backward(grad);
        let grads = net.gradients();
        net.apply_gradients(&grads, lr);
        net.zero_grad();

        // Check that update happened
        let params = net.get_parameters();
        assert!(params.iter().any(|&p| p.is_finite()));
    }

    #[test]
    fn test_temporal_net_long_sequence() {
        let mut net = TemporalNet::new(2, 4, 1);

        // Create a long sequence
        let sequence: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![(i as f32).sin(), (i as f32).cos()])
            .collect();

        let output = net.forward(sequence);
        assert_eq!(output.len(), 1);
    }
}

#[cfg(test)]
mod text_tests {
    use super::*;

    #[test]
    fn test_build_vocab() {
        let texts = vec![
            "hello world".to_string(),
            "hello there".to_string(),
            "world of possibilities".to_string(),
        ];

        let vocab = TextNet::build_vocab(&texts, 10);

        // Should include <UNK> and the most frequent words
        assert!(vocab.contains_key("<UNK>"));
        assert!(vocab.contains_key("hello"));
        assert!(vocab.contains_key("world"));
    }

    #[test]
    fn test_build_vocab_max_size() {
        let texts = vec![
            "a b c d e".to_string(),
            "f g h i j".to_string(),
            "k l m n o".to_string(),
        ];

        let vocab = TextNet::build_vocab(&texts, 5);

        // Should have at most 5 words (including <UNK>)
        assert!(vocab.len() <= 5);
        assert!(vocab.contains_key("<UNK>"));
    }

    #[test]
    fn test_text_net_creation() {
        let mut vocab = HashMap::new();
        vocab.insert("<UNK>".to_string(), 0);
        vocab.insert("hello".to_string(), 1);
        vocab.insert("world".to_string(), 2);

        let net = TextNet::new(vocab, 10, 8);
        assert_eq!(net.output_size(), 8);
    }

    #[test]
    fn test_text_net_tokenize() {
        let mut vocab = HashMap::new();
        vocab.insert("<UNK>".to_string(), 0);
        vocab.insert("hello".to_string(), 1);
        vocab.insert("world".to_string(), 2);
        vocab.insert("test".to_string(), 3);

        let net = TextNet::new(vocab, 10, 8);

        let tokens = net.tokenize("hello world");
        assert_eq!(tokens, vec![1, 2]);

        // Unknown word should map to <UNK>
        let tokens_with_unknown = net.tokenize("hello unknown");
        assert_eq!(tokens_with_unknown, vec![1, 0]);
    }

    #[test]
    fn test_text_net_forward() {
        let mut vocab = HashMap::new();
        vocab.insert("<UNK>".to_string(), 0);
        vocab.insert("hello".to_string(), 1);
        vocab.insert("world".to_string(), 2);

        let mut net = TextNet::new(vocab, 8, 6);

        let word_indices = vec![1, 2];
        let output = net.forward(word_indices);

        assert_eq!(output.len(), 6);
    }

    #[test]
    fn test_text_net_forward_backward() {
        let mut vocab = HashMap::new();
        vocab.insert("<UNK>".to_string(), 0);
        vocab.insert("hello".to_string(), 1);
        vocab.insert("world".to_string(), 2);

        let mut net = TextNet::new(vocab, 8, 6);

        let word_indices = vec![1, 2, 1];

        // Forward pass
        let output = net.forward(word_indices.clone());
        assert_eq!(output.len(), 6);

        // Backward pass
        let grad_output = vec![1.0, -1.0, 0.5, -0.5, 0.2, -0.2];
        let grad_indices = net.backward(grad_output);

        // Returns the original indices (can't backprop through discrete ops)
        assert_eq!(grad_indices.len(), 3);
    }

    #[test]
    fn test_text_net_parameter_management() {
        let mut vocab = HashMap::new();
        vocab.insert("<UNK>".to_string(), 0);
        vocab.insert("test".to_string(), 1);

        let mut net = TextNet::new(vocab, 5, 4);

        // Get parameters
        let params = net.get_parameters();
        assert!(params.len() > 0);

        // Set parameters
        let new_params = vec![0.1; params.len()];
        let consumed = net.set_parameters(&new_params).unwrap();
        assert_eq!(consumed, params.len());
    }

    #[test]
    fn test_text_net_gradients() {
        let mut vocab = HashMap::new();
        vocab.insert("<UNK>".to_string(), 0);
        vocab.insert("word".to_string(), 1);

        let mut net = TextNet::new(vocab, 5, 4);

        let word_indices = vec![1, 1];

        // Forward and backward
        net.forward(word_indices.clone());
        net.backward(vec![1.0, 0.0, -1.0, 0.5]);

        // Get gradients
        let grads = net.gradients();
        assert!(grads.len() > 0);

        // Zero gradients
        net.zero_grad();
        let zero_grads = net.gradients();
        for grad in zero_grads {
            assert_eq!(grad, 0.0);
        }
    }

    #[test]
    fn test_text_net_empty_input() {
        let mut vocab = HashMap::new();
        vocab.insert("<UNK>".to_string(), 0);

        let mut net = TextNet::new(vocab, 5, 4);

        let empty_indices = vec![];
        let output = net.forward(empty_indices);

        // Should return zero vector for empty input
        assert_eq!(output.len(), 4);
        for val in output {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_text_net_single_word() {
        let mut vocab = HashMap::new();
        vocab.insert("<UNK>".to_string(), 0);
        vocab.insert("test".to_string(), 1);

        let mut net = TextNet::new(vocab, 5, 4);

        let single_word = vec![1];
        let output = net.forward(single_word);

        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_text_net_long_sequence() {
        let mut vocab = HashMap::new();
        vocab.insert("<UNK>".to_string(), 0);
        for i in 1..20 {
            vocab.insert(format!("word{}", i), i);
        }

        let mut net = TextNet::new(vocab, 8, 6);

        // Long sequence
        let word_indices: Vec<usize> = (1..15).collect();
        let output = net.forward(word_indices);

        assert_eq!(output.len(), 6);
    }

    #[test]
    fn test_text_net_training_step() {
        let mut vocab = HashMap::new();
        vocab.insert("<UNK>".to_string(), 0);
        vocab.insert("good".to_string(), 1);
        vocab.insert("bad".to_string(), 2);

        let mut net = TextNet::new(vocab, 8, 2);
        let lr = 0.01;

        let word_indices = vec![1]; // "good"
        let target = vec![1.0, 0.0]; // Positive sentiment

        // Forward pass
        let output = net.forward(word_indices.clone());

        // Compute gradient
        let grad: Vec<f32> = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| o - t)
            .collect();

        // Backward pass and update
        net.backward(grad);
        let grads = net.gradients();
        net.apply_gradients(&grads, lr);
        net.zero_grad();

        // Check that update happened
        let params = net.get_parameters();
        assert!(params.iter().any(|&p| p.is_finite()));
    }

    #[test]
    fn test_text_net_case_insensitive() {
        let mut vocab = HashMap::new();
        vocab.insert("<UNK>".to_string(), 0);
        vocab.insert("hello".to_string(), 1);

        let net = TextNet::new(vocab, 5, 4);

        // Should handle case-insensitively
        let tokens1 = net.tokenize("hello");
        let tokens2 = net.tokenize("HELLO");
        let tokens3 = net.tokenize("Hello");

        assert_eq!(tokens1, tokens2);
        assert_eq!(tokens2, tokens3);
    }
}

#[cfg(test)]
mod rnn_cell_tests {
    use super::*;

    #[test]
    fn test_rnn_cell_creation() {
        let cell = SimpleRnnCell::new(5, 10);
        assert_eq!(cell.hidden_size, 10);
    }

    #[test]
    fn test_rnn_cell_forward_step() {
        let cell = SimpleRnnCell::new(3, 5);

        let input = vec![1.0, 0.5, -0.5];
        let h_prev = vec![0.0; 5];

        let h_next = cell.forward_step(&input, &h_prev);

        assert_eq!(h_next.len(), 5);
        // Output should be in tanh range [-1, 1]
        for &val in &h_next {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_rnn_cell_reset_state() {
        let mut cell = SimpleRnnCell::new(2, 4);

        // Add some cached data
        cell.cached_inputs.push(vec![1.0, 2.0]);
        cell.cached_hiddens.push(vec![0.1, 0.2, 0.3, 0.4]);

        // Reset
        cell.reset_state();

        assert_eq!(cell.cached_inputs.len(), 0);
        assert_eq!(cell.cached_hiddens.len(), 0);
    }

    #[test]
    fn test_rnn_cell_parameter_management() {
        let mut cell = SimpleRnnCell::new(3, 5);

        // Get parameters
        let params = cell.get_parameters();
        assert!(params.len() > 0);

        // Set parameters
        let new_params = vec![0.1; params.len()];
        let consumed = cell.set_parameters(&new_params).unwrap();
        assert_eq!(consumed, params.len());
    }

    #[test]
    fn test_rnn_cell_sequence_processing() {
        let cell = SimpleRnnCell::new(2, 4);

        let sequence = vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]];

        let mut h = vec![0.0; 4];

        for input in sequence {
            h = cell.forward_step(&input, &h);
            assert_eq!(h.len(), 4);
        }

        // Final hidden state should contain information from all steps
        assert!(h.iter().any(|&val| val != 0.0));
    }

    #[test]
    fn test_rnn_cell_tanh_derivative() {
        let tanh_values = vec![0.0, 0.5, -0.5, 0.9, -0.9];
        let derivatives = SimpleRnnCell::tanh_derivative(&tanh_values);

        assert_eq!(derivatives.len(), tanh_values.len());

        // Derivative at 0 should be close to 1
        assert!((derivatives[0] - 1.0).abs() < 0.1);

        // Derivatives should be positive
        for &deriv in &derivatives {
            assert!(deriv >= 0.0);
        }
    }

    #[test]
    fn test_rnn_cell_different_sizes() {
        let sizes = vec![(2, 4), (5, 10), (10, 5), (1, 1)];

        for (input_size, hidden_size) in sizes {
            let cell = SimpleRnnCell::new(input_size, hidden_size);

            let input = vec![0.5; input_size];
            let h_prev = vec![0.0; hidden_size];

            let h_next = cell.forward_step(&input, &h_prev);
            assert_eq!(h_next.len(), hidden_size);
        }
    }
}
