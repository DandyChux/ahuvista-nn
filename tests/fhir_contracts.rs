use ahuvista_nn::data::fhir::{
    Bundle, BundleEntry, CodeableConcept, Coding, DynamicFeatureMapping, FhirMappingConfig,
    FhirParser, Observation, Patient, Quantity, Reference, Resource, StaticFeatureMapping,
};

fn config() -> FhirMappingConfig {
    FhirMappingConfig {
        static_features: vec![StaticFeatureMapping {
            system: Some("loinc".into()),
            code: "AGE".into(),
            feature_index: 0,
            feature_name: "age".into(),
            default_value: 0.0,
            unit_conversion: None,
        }],
        dynamic_features: vec![DynamicFeatureMapping {
            system: Some("loinc".into()),
            code: "HR".into(),
            feature_index: 0,
            feature_name: "heart_rate".into(),
            default_value: 0.0,
            unit_conversion: None,
        }],
        condition_flags: vec![],
        text_document_types: vec![],
        max_sequence_length: 10,
    }
}

fn code(system: &str, value: &str) -> CodeableConcept {
    CodeableConcept {
        coding: vec![Coding {
            system: Some(system.into()),
            version: None,
            code: Some(value.into()),
            display: None,
        }],
        text: None,
    }
}

#[test]
fn fhir_parser_builds_ordered_unified_patient_sequence() {
    let bundle = Bundle {
        resource_type: "Bundle".into(),
        r#type: "collection".into(),
        entry: Some(vec![
            BundleEntry {
                resource: Some(Resource::Patient(Patient {
                    resource_type: "Patient".into(),
                    id: Some("p1".into()),
                    birth_date: None,
                    gender: None,
                })),
            },
            BundleEntry {
                resource: Some(Resource::Observation(Observation {
                    resource_type: "Observation".into(),
                    id: None,
                    status: "final".into(),
                    category: None,
                    code: code("loinc", "HR"),
                    subject: Some(Reference::new("Patient/p1")),
                    effective_date_time: Some("2025-01-02".into()),
                    effective_period: None,
                    value_quantity: Some(Quantity {
                        value: Some(80.0),
                        unit: None,
                        system: None,
                        code: None,
                    }),
                    value_codeable_concept: None,
                    value_string: None,
                    component: None,
                })),
            },
        ]),
    };

    let parsed = FhirParser::new(config()).parse_bundle(&bundle).unwrap();

    assert_eq!(parsed.patient_id, "p1");
    assert_eq!(parsed.sequence.static_features, vec![0.0]);
    assert_eq!(parsed.sequence.dynamic_features, vec![vec![80.0]]);
}

#[test]
fn fhir_parser_rejects_bundles_without_patient() {
    let bundle = Bundle {
        resource_type: "Bundle".into(),
        r#type: "collection".into(),
        entry: None,
    };

    assert!(FhirParser::new(config()).parse_bundle(&bundle).is_err());
}
