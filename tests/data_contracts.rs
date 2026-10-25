use ahuvista_nn::data::{
    DataLoader, MultiModalDataLoader, PatientData, PatientObservation, PreparedDataset,
};
use ahuvista_nn::models::PatientSequence;

fn sequence(static_features: Vec<f32>, dynamic_features: Vec<Vec<f32>>) -> PatientSequence {
    PatientSequence::new(static_features, dynamic_features)
}

#[test]
fn patient_data_maps_static_and_observations_to_one_sequence() {
    let mut patient = PatientData {
        patient_id: "p1".into(),
        age: Some(30.0),
        systolic_bp: Some(120.0),
        pregnancies: Some(2),
        bmi: Some(24.5),
        has_complications: true,
        ..Default::default()
    };

    patient.observations = vec![
        PatientObservation {
            timestamp: "2025-01-01".into(),
            feature_name: "heart_rate".into(),
            value: 80.0,
        },
        PatientObservation {
            timestamp: "2025-01-02".into(),
            feature_name: "temperature".into(),
            value: 37.0,
        },
    ];

    let static_names = [
        "age",
        "systolic_bp",
        "pregnancies",
        "has_complications",
        "bmi",
    ]
    .into_iter()
    .map(String::from)
    .collect::<Vec<_>>();
    let dynamic_names = ["heart_rate", "temperature"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let result = patient.to_patient_sequence(&static_names, &dynamic_names);

    assert_eq!(result.static_features, vec![30.0, 120.0, 2.0, 1.0, 24.5]);
    assert_eq!(
        result.dynamic_features,
        vec![vec![80.0, 0.0], vec![0.0, 37.0]]
    );
}

#[test]
fn prepared_dataset_rejects_alignment_and_width_errors() {
    let first = sequence(vec![1.0, 2.0], vec![vec![3.0, 4.0]]);
    let mismatched = sequence(vec![1.0], vec![vec![3.0, 4.0]]);

    assert!(PreparedDataset::new(vec![first.clone()], vec![], vec![], vec![]).is_err());
    assert!(
        PreparedDataset::new(
            vec![first.clone(), first.clone()],
            vec![],
            vec![],
            vec![0.0, 1.0]
        )
        .is_ok()
    );
    assert!(PreparedDataset::new(vec![first, mismatched], vec![], vec![], vec![0.0, 1.0]).is_err());
}

#[test]
fn prepared_dataset_split_preserves_aligned_fields() {
    let sequences = vec![
        sequence(vec![1.0], vec![vec![10.0]]),
        sequence(vec![2.0], vec![vec![20.0]]),
        sequence(vec![3.0], vec![vec![30.0]]),
    ];
    let text = vec![vec![1], vec![2], vec![3]];
    let images = vec![ndarray::Array3::zeros((1, 1, 1)); 3];
    let dataset = PreparedDataset::new(sequences, text, images, vec![0.0, 1.0, 0.0]).unwrap();
    let (train, validation) = dataset.split(2.0 / 3.0).unwrap();

    assert_eq!(train.len(), 2);
    assert_eq!(validation.len(), 1);
    assert_eq!(train.text, vec![vec![1], vec![2]]);
    assert_eq!(validation.targets, vec![0.0]);
}

#[test]
fn multimodal_loader_batches_and_resets() {
    let sequences = (0..3)
        .map(|value| sequence(vec![value as f32], vec![]))
        .collect();
    let dataset = PreparedDataset::new(sequences, vec![], vec![], vec![0.0, 1.0, 0.0]).unwrap();
    let mut loader = MultiModalDataLoader::new(dataset);

    assert_eq!(loader.next_batch(2).unwrap().len(), 2);
    assert_eq!(loader.next_batch(2).unwrap().len(), 1);
    assert!(loader.next_batch(1).is_none());
    loader.reset();
    assert_eq!(loader.next_batch(3).unwrap().targets, vec![0.0, 1.0, 0.0]);
    assert!(loader.next_batch(0).is_none());
}
