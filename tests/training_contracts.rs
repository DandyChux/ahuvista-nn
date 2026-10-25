use ahuvista_nn::training::loss::{
    binary_accuracy, binary_cross_entropy, binary_cross_entropy_gradient, mean_squared_error,
    mean_squared_error_gradient,
};
use ahuvista_nn::training::metrics::{ConfusionMatrix, calculate_auprc, calculate_auroc};

#[test]
fn losses_have_expected_values_and_finite_boundary_behavior() {
    assert!(binary_cross_entropy(0.999, 1.0) < binary_cross_entropy(0.1, 1.0));
    assert!(binary_cross_entropy(0.0, 1.0).is_finite());
    assert!(binary_cross_entropy_gradient(0.5, 1.0) < 0.0);
    assert!((mean_squared_error(&[0.0, 1.0], &[0.0, 0.0]) - 0.5).abs() < 1e-6);
    assert_eq!(
        mean_squared_error_gradient(&[0.0, 1.0], &[0.0, 0.0]),
        vec![0.0, 1.0]
    );
}

#[test]
fn binary_metrics_and_confusion_matrix_match_labels() {
    let predictions = [0.9, 0.8, 0.1, 0.2];
    let targets = [1.0, 0.0, 1.0, 0.0];

    assert_eq!(binary_accuracy(&predictions, &targets, 0.5), 0.5);

    let matrix = ConfusionMatrix::from_predictions(&predictions, &targets, 0.5);

    assert_eq!((matrix.tp, matrix.fp, matrix.tn, matrix.fn_), (1, 1, 1, 1));
    assert!((matrix.ppv() - 0.5).abs() < 1e-6);
    assert!((matrix.npv() - 0.5).abs() < 1e-6);
}

#[test]
fn ranking_metrics_are_valid_for_perfect_and_undefined_inputs() {
    let perfect = calculate_auroc(&[0.9, 0.8, 0.2, 0.1], &[1.0, 1.0, 0.0, 0.0]);

    assert!((perfect - 1.0).abs() < 1e-6);
    assert_eq!(calculate_auroc(&[0.5, 0.6], &[1.0, 1.0]), 0.5);
    assert!(calculate_auprc(&[0.9, 0.8, 0.2, 0.1], &[1.0, 1.0, 0.0, 0.0]).is_finite());
}
