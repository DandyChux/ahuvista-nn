//! Comprehensive tests for processing, training, and evaluation

use ahuvista_nn::core::{Module, Network};
use ahuvista_nn::training::*;

#[cfg(test)]
mod loss_tests {
    use super::*;

    #[test]
    fn test_binary_cross_entropy() {
        // Test with perfect prediction
        let predictions = vec![1.0];
        let targets = vec![1.0];
        let loss = binary_cross_entropy(&predictions, &targets);
        assert!(loss < 0.01); // Should be very small

        // Test with opposite prediction
        let predictions = vec![0.01];
        let targets = vec![1.0];
        let loss = binary_cross_entropy(&predictions, &targets);
        assert!(loss > 4.0); // Should be large

        // Test with neutral prediction
        let predictions = vec![0.5];
        let targets = vec![1.0];
        let loss = binary_cross_entropy(&predictions, &targets);
        assert!(loss > 0.5 && loss < 1.0);
    }

    #[test]
    fn test_mse_loss() {
        let predictions = vec![0.5, 0.8, 0.2];
        let targets = vec![0.6, 0.7, 0.3];

        let loss = mean_squared_error(&predictions, &targets);
        let expected = ((0.1_f32.powi(2) + 0.1_f32.powi(2) + 0.1_f32.powi(2)) / 3.0);

        assert!((loss - expected).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_binary_cross_entropy() {
        let predictions = vec![0.8, 0.6];
        let targets = vec![1.0, 0.0];
        let weights = vec![2.0, 1.0];

        let loss = weighted_binary_cross_entropy(&predictions, &targets, &weights);
        assert!(loss > 0.0);
    }
}

#[cfg(test)]
mod metrics_tests {
    use super::*;

    #[test]
    fn test_accuracy() {
        let predictions = vec![0.8, 0.3, 0.9, 0.1];
        let targets = vec![1.0, 0.0, 1.0, 0.0];

        let acc = calculate_accuracy(&predictions, &targets, 0.5);
        assert_eq!(acc, 1.0); // All correct

        let predictions = vec![0.8, 0.7, 0.9, 0.6];
        let targets = vec![1.0, 0.0, 1.0, 0.0];

        let acc = calculate_accuracy(&predictions, &targets, 0.5);
        assert_eq!(acc, 0.5); // Half correct
    }

    // #[test]
    // fn test_precision_recall() {
    //     // True positives: 2, False positives: 1, False negatives: 1
    //     let predictions = vec![0.8, 0.3, 0.9, 0.7];
    //     let targets = vec![1.0, 0.0, 1.0, 0.0];

    //     let (precision, recall) = calculate_precision_recall(&predictions, &targets, 0.5);

    //     // TP=2, FP=1, FN=0
    //     let expected_precision = 2.0 / 3.0;
    //     let expected_recall = 2.0 / 2.0;

    //     assert!((precision - expected_precision).abs() < 1e-6);
    //     assert!((recall - expected_recall).abs() < 1e-6);
    // }

    // #[test]
    // fn test_f1_score() {
    //     let predictions = vec![0.9, 0.8, 0.2, 0.1];
    //     let targets = vec![1.0, 1.0, 0.0, 0.0];

    //     let f1 = calculate_f1_score(&predictions, &targets, 0.5);
    //     assert_eq!(f1, 1.0); // Perfect predictions
    // }

    // #[test]
    // fn test_confusion_matrix() {
    //     let predictions = vec![0.9, 0.8, 0.2, 0.1, 0.6, 0.4];
    //     let targets = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0];

    //     let (tp, fp, tn, fn_count) = calculate_confusion_matrix(&predictions, &targets, 0.5);

    //     assert_eq!(tp, 3); // Correctly predicted positive
    //     assert_eq!(tn, 3); // Correctly predicted negative
    //     assert_eq!(fp, 0); // Incorrectly predicted positive
    //     assert_eq!(fn_count, 0); // Incorrectly predicted negative
    // }
}

// #[cfg(test)]
// mod optimizer_tests {
//     use super::*;

//     #[test]
//     fn test_sgd_optimizer() {
//         let mut params = vec![1.0, 2.0, 3.0];
//         let gradients = vec![0.1, 0.2, 0.3];
//         let learning_rate = 0.1;

//         sgd_update(&mut params, &gradients, learning_rate);

//         assert_eq!(params[0], 0.99);
//         assert_eq!(params[1], 1.98);
//         assert_eq!(params[2], 2.97);
//     }

//     #[test]
//     fn test_momentum_optimizer() {
//         let mut params = vec![1.0, 2.0];
//         let gradients = vec![0.1, 0.2];
//         let mut velocity = vec![0.0, 0.0];
//         let learning_rate = 0.1;
//         let momentum = 0.9;

//         momentum_update(
//             &mut params,
//             &gradients,
//             &mut velocity,
//             learning_rate,
//             momentum,
//         );

//         // Velocity should be updated
//         assert!(velocity[0].abs() > 0.0);
//         assert!(velocity[1].abs() > 0.0);

//         // Params should be updated
//         assert!(params[0] < 1.0);
//         assert!(params[1] < 2.0);
//     }

//     #[test]
//     fn test_adam_optimizer() {
//         let mut params = vec![1.0, 2.0];
//         let gradients = vec![0.1, 0.2];
//         let mut m = vec![0.0, 0.0]; // First moment
//         let mut v = vec![0.0, 0.0]; // Second moment
//         let mut t = 0;
//         let learning_rate = 0.001;
//         let beta1 = 0.9;
//         let beta2 = 0.999;
//         let epsilon = 1e-8;

//         adam_update(
//             &mut params,
//             &gradients,
//             &mut m,
//             &mut v,
//             &mut t,
//             learning_rate,
//             beta1,
//             beta2,
//             epsilon,
//         );

//         assert_eq!(t, 1);
//         assert!(m[0].abs() > 0.0);
//         assert!(v[0].abs() > 0.0);
//         assert!(params[0] < 1.0);
//     }
// }

#[cfg(test)]
mod training_tests {
    use super::*;
    use ahuvista_nn::core::Layer;

    #[test]
    fn test_simple_training_loop() {
        let mut network = Network::from_layer_sizes(&[2, 4, 1]);

        // Simple training data (XOR problem)
        let training_data = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        let learning_rate = 0.5;
        let epochs = 100;

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        for epoch in 0..epochs {
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

                // Backward pass
                let grad_outputs: Vec<f32> = outputs
                    .iter()
                    .zip(targets.iter())
                    .map(|(o, t)| 2.0 * (o - t))
                    .collect();
                network.backward(grad_outputs);

                // Update weights
                let grads = network.gradients();
                network.apply_gradients(&grads, learning_rate);
                network.zero_grad();
            }

            if epoch == 0 {
                initial_loss = epoch_loss;
            }
            if epoch == epochs - 1 {
                final_loss = epoch_loss;
            }
        }

        // Loss should decrease significantly
        assert!(final_loss < initial_loss * 0.5);
    }

    #[test]
    fn test_batch_training() {
        let mut network = Network::from_layer_sizes(&[2, 3, 1]);

        let batch_inputs = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];

        let batch_targets = vec![vec![0.3], vec![0.7], vec![0.9]];

        let learning_rate = 0.1;

        let mut accumulated_grads = vec![0.0; network.get_parameters().len()];

        // Process batch
        for (inputs, targets) in batch_inputs.iter().zip(batch_targets.iter()) {
            network.forward(inputs.clone());

            let outputs = network.forward(inputs.clone());
            let grad_outputs: Vec<f32> = outputs
                .iter()
                .zip(targets.iter())
                .map(|(o, t)| 2.0 * (o - t))
                .collect();

            network.backward(grad_outputs);

            let grads = network.gradients();
            for (acc, g) in accumulated_grads.iter_mut().zip(grads.iter()) {
                *acc += g;
            }
        }

        // Average gradients
        for g in accumulated_grads.iter_mut() {
            *g /= batch_inputs.len() as f32;
        }

        // Apply averaged gradients
        network.apply_gradients(&accumulated_grads, learning_rate);
        network.zero_grad();

        // Test completed without panicking
        assert!(true);
    }

    #[test]
    fn test_learning_rate_schedule() {
        let initial_lr = 0.1;
        let decay_rate = 0.95;

        let mut current_lr = initial_lr;

        for epoch in 0..10 {
            // Simulate training
            current_lr *= decay_rate;

            if epoch == 9 {
                assert!(current_lr < initial_lr);
                assert!(current_lr > initial_lr * 0.5);
            }
        }
    }
}

// #[cfg(test)]
// mod regularization_tests {
//     use super::*;

//     #[test]
//     fn test_l2_regularization() {
//         let params = vec![1.0, -0.5, 2.0, -1.5];
//         let lambda = 0.01;

//         let l2_penalty = l2_regularization(&params, lambda);

//         // L2 = lambda * sum(w^2)
//         let expected = lambda * (1.0 + 0.25 + 4.0 + 2.25);
//         assert!((l2_penalty - expected).abs() < 1e-6);
//     }

//     #[test]
//     fn test_l1_regularization() {
//         let params = vec![1.0, -0.5, 2.0, -1.5];
//         let lambda = 0.01;

//         let l1_penalty = l1_regularization(&params, lambda);

//         // L1 = lambda * sum(|w|)
//         let expected = lambda * (1.0 + 0.5 + 2.0 + 1.5);
//         assert!((l1_penalty - expected).abs() < 1e-6);
//     }

//     #[test]
//     fn test_dropout() {
//         let mut activations = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//         let dropout_rate = 0.5;

//         apply_dropout(&mut activations, dropout_rate);

//         // Some activations should be zeroed, others scaled
//         let non_zero = activations.iter().filter(|&&x| x > 0.0).count();
//         let zero = activations.iter().filter(|&&x| x == 0.0).count();

//         assert!(zero > 0 || non_zero > 0);
//     }

//     #[test]
//     fn test_gradient_clipping() {
//         let mut gradients = vec![10.0, -20.0, 5.0, -15.0];
//         let max_norm = 10.0;

//         clip_gradients(&mut gradients, max_norm);

//         // L2 norm should be <= max_norm
//         let norm: f32 = gradients.iter().map(|g| g.powi(2)).sum::<f32>().sqrt();
//         assert!(norm <= max_norm + 1e-6);
//     }
// }

#[cfg(test)]
mod evaluation_tests {
    use super::*;

    #[test]
    fn test_cross_validation_splits() {
        let data_size = 100;
        let k_folds = 5;
        let fold_size = data_size / k_folds;

        for fold in 0..k_folds {
            let val_start = fold * fold_size;
            let val_end = (fold + 1) * fold_size;

            assert!(val_start < data_size);
            assert!(val_end <= data_size);
            assert!(val_end > val_start);
        }
    }

    // #[test]
    // fn test_roc_auc_calculation() {
    //     // Perfect predictions
    //     let predictions = vec![0.9, 0.8, 0.2, 0.1];
    //     let targets = vec![1.0, 1.0, 0.0, 0.0];

    //     let auc = calculate_roc_auc(&predictions, &targets);
    //     assert_eq!(auc, 1.0);

    //     // Random predictions
    //     let predictions = vec![0.5, 0.5, 0.5, 0.5];
    //     let targets = vec![1.0, 0.0, 1.0, 0.0];

    //     let auc = calculate_roc_auc(&predictions, &targets);
    //     assert!((auc - 0.5).abs() < 0.1);
    // }
}

// Helper functions for tests
fn binary_cross_entropy(predictions: &[f32], targets: &[f32]) -> f32 {
    let epsilon = 1e-7;
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| -t * (p + epsilon).ln() - (1.0 - t) * (1.0 - p + epsilon).ln())
        .sum::<f32>()
        / predictions.len() as f32
}

fn mean_squared_error(predictions: &[f32], targets: &[f32]) -> f32 {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / predictions.len() as f32
}

fn weighted_binary_cross_entropy(predictions: &[f32], targets: &[f32], weights: &[f32]) -> f32 {
    let epsilon = 1e-7;
    predictions
        .iter()
        .zip(targets.iter())
        .zip(weights.iter())
        .map(|((p, t), w)| w * (-t * (p + epsilon).ln() - (1.0 - t) * (1.0 - p + epsilon).ln()))
        .sum::<f32>()
        / weights.iter().sum::<f32>()
}

fn calculate_accuracy(predictions: &[f32], targets: &[f32], threshold: f32) -> f32 {
    let correct = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(p, t)| {
            let pred_class = if **p > threshold { 1.0 } else { 0.0 };
            pred_class == **t
        })
        .count();

    correct as f32 / predictions.len() as f32
}

// fn calculate_precision_recall(predictions: &[f32], targets: &[f32], threshold: f32) -> (f32, f32) {
//     let (mut tp, mut fp, mut fn_count) = (0, 0, 0);

//     for (p, t) in predictions.iter().zip(targets.iter()) {
//         let pred_class = if *p > threshold { 1.0 } else { 0.0 };

//         if pred_class == 1.0 && *t == 1.0 {
//             tp += 1;
//         } else if pred_class == 1.0 && *t == 0.0 {
//             fp += 1;
//         } else if pred_class == 0.0 && *t == 1.0 {
//             fn_count += 1;
//         } else if pred_class == 0.0 && *t == 0.0 {
//             tn += 1;
//         }
//     }

//     let precision = if tp + fp == 0.0 {
//         0.0
//     } else {
//         tp as f32 / (tp + fp) as f32
//     };
//     let recall = if tp + fn_count == 0.0 {
//         0.0
//     } else {
//         tp as f32 / (tp + fn_count) as f32
//     };

//     (precision, recall)
// }
