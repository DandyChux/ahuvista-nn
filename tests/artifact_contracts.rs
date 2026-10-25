use ahuvista_nn::config::ModalityConfig;
use ahuvista_nn::data::FeatureStats;
use ahuvista_nn::model_artifact::{ModelArtifact, ModelSpec};

fn spec() -> ModelSpec {
    ModelSpec {
        model_id: "test-model".into(),
        model_version: "1".into(),
        modalities: ModalityConfig {
            use_sequence: true,
            use_text: false,
            use_image: false,
        },
        sequence_dynamic_input_size: Some(2),
        sequence_static_input_size: Some(1),
        sequence_hidden_size: Some(3),
        sequence_output_size: Some(1),
        text_embedding_size: None,
        text_hidden_size: None,
        text_vocab: Default::default(),
        image_channels: None,
        image_height: None,
        image_width: None,
        classifier_hidden_sizes: vec![2],
        static_feature_names: vec!["age".into()],
        dynamic_feature_names: vec!["hr".into(), "bp".into()],
        static_normalization: vec![FeatureStats::default()],
        dynamic_normalization: vec![],
        fhir_mapping_version: None,
    }
}

#[test]
fn model_spec_validation_checks_dimensions_and_feature_names() {
    assert!(spec().validate().is_ok());
    let mut invalid = spec();
    invalid.dynamic_feature_names.pop();
    assert!(invalid.validate().is_err());
}

#[test]
fn model_artifact_serializes_and_loads_parameters() {
    let artifact = ModelArtifact::new(spec(), vec![0.1, 0.2]).unwrap();
    let path = std::env::temp_dir().join(format!("ahuvista-artifact-{}.json", std::process::id()));
    artifact.save(&path).unwrap();
    let loaded = ModelArtifact::load(&path).unwrap();
    assert_eq!(loaded.parameters, vec![0.1, 0.2]);
    assert_eq!(loaded.spec.sequence_params(), Some((2, 1, 3, 1)));
    std::fs::remove_file(path).unwrap();
}
