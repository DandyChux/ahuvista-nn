use ahuvista_nn::core::Module;
use ahuvista_nn::models::{GruLayer, PatientSequence, SequenceNet};

#[test]
fn patient_sequence_reports_dimensions_and_validates_widths() {
    let input = PatientSequence::new(vec![1.0, 2.0], vec![vec![3.0, 4.0], vec![5.0, 6.0]]);

    assert_eq!(input.static_size(), 2);
    assert_eq!(input.dynamic_size(), 2);
    assert_eq!(input.sequence_length(), 2);
    assert!(input.validate(2, 2).is_ok());
    assert!(input.validate(1, 2).is_err());
    assert!(input.validate(2, 1).is_err());
}

#[test]
fn unified_sequence_model_combines_gru_and_static_features() {
    let mut model = SequenceNet::new_for_binary_classification(2, 3, 4);

    assert_eq!(model.dynamic_size(), 2);
    assert_eq!(model.static_size(), 3);
    assert_eq!(model.output_size(), 1);

    let input = PatientSequence::new(vec![0.1, 0.2, 0.3], vec![vec![0.4, 0.5], vec![0.6, 0.7]]);
    let output = model.forward(input.clone());

    assert_eq!(output.len(), 1);
    assert!(output[0].is_finite());

    let gradient = model.backward(vec![1.0]);

    assert_eq!(gradient.static_features.len(), 3);
    assert_eq!(gradient.dynamic_features.len(), 2);
    assert!(gradient.dynamic_features.iter().all(|step| step.len() == 2));
}

#[test]
fn sequence_model_parameters_round_trip_and_gradients_can_be_cleared() {
    let mut model = SequenceNet::new(2, 2, 3, 1);
    let parameters = model.get_parameters();

    assert!(!parameters.is_empty());
    assert_eq!(model.set_parameters(&parameters).unwrap(), parameters.len());

    model.forward(PatientSequence::new(vec![1.0, 2.0], vec![vec![3.0, 4.0]]));
    model.backward(vec![1.0]);

    assert!(model.gradients().iter().any(|value| *value != 0.0));

    model.zero_grad();

    assert!(model.gradients().iter().all(|value| *value == 0.0));
}

#[test]
fn gru_handles_empty_and_variable_length_sequences() {
    let mut gru = GruLayer::new(2, 3);

    assert_eq!(gru.forward_sequence(&[]).len(), 3);
    assert_eq!(gru.forward_sequence(&[vec![1.0, 2.0]]).len(), 3);
    assert_eq!(
        gru.forward_sequence(&[vec![1.0, 2.0], vec![2.0, 3.0]])
            .len(),
        3
    );
}
