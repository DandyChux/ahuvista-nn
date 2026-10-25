use ahuvista_nn::data::db::DatabaseManager;
use ahuvista_nn::data::{ColumnMapping, PatientData, PatientObservation};
use ahuvista_nn::models::PatientSequence;
use std::{
    io::Write,
    sync::atomic::{AtomicUsize, Ordering},
};

fn patient(id: &str) -> PatientData {
    PatientData {
        patient_id: id.into(),
        age: Some(30.0),
        systolic_bp: Some(120.0),
        diastolic_bp: Some(80.0),
        glucose: Some(90.0),
        bmi: Some(24.0),
        pregnancies: Some(1),
        existing_outcome: Some(0.0),
        observations: vec![PatientObservation {
            timestamp: "2025-01-01".into(),
            feature_name: "heart_rate".into(),
            value: 80.0,
        }],
        clinical_notes: vec!["routine visit".into()],
        ..Default::default()
    }
}

#[test]
fn sqlite_round_trips_patient_sequence_and_outcome() {
    let mut db = DatabaseManager::in_memory().unwrap();
    db.upsert_patient(&patient("p1")).unwrap();
    db.upsert_patient(&patient("p2")).unwrap();

    assert_eq!(db.count_patients().unwrap(), 2);
    let loaded = db.get_patient("p1").unwrap().unwrap();
    assert_eq!(loaded.age, Some(30.0));
    assert_eq!(loaded.observations[0].feature_name, "heart_rate");
    assert_eq!(loaded.clinical_notes, vec!["routine visit"]);

    let sequences = db.build_patient_sequences().unwrap();
    assert_eq!(sequences.len(), 2);
    assert_eq!(sequences[0].1.dynamic_features, vec![vec![80.0]]);

    let (targets, causes) = db.build_outcomes().unwrap();
    assert_eq!(targets, vec![0.0, 0.0]);
    assert_eq!(causes, vec![4, 4]);
}

#[test]
fn clear_all_data_keeps_schema_and_removes_children_first() {
    let mut db = DatabaseManager::in_memory().unwrap();
    db.upsert_patient(&patient("p1")).unwrap();
    db.store_patient_sequences(&db.build_patient_sequences().unwrap())
        .unwrap();
    db.clear_all_data().unwrap();
    assert_eq!(db.count_patients().unwrap(), 0);
    assert_eq!(db.get_sample_counts().unwrap().sequences, 0);
}

#[test]
fn csv_import_maps_static_temporal_and_text_rows() {
    let mut db = DatabaseManager::in_memory().unwrap();
    let dir = tempfile_dir();
    let path = dir.join("patients.csv");
    let mut file = std::fs::File::create(&path).unwrap();
    writeln!(file, "PatientID,age,systolic_bp,pregnancies").unwrap();
    writeln!(file, "p1,28,115,1").unwrap();
    writeln!(file, "p2,34,130,2").unwrap();

    assert_eq!(
        db.import_csv(&path, "PatientID", Some("maternal")).unwrap(),
        2
    );
    assert_eq!(db.get_all_patient_ids().unwrap().len(), 2);
    assert_eq!(
        db.get_patient("p2").unwrap().unwrap().systolic_bp,
        Some(130.0)
    );
    std::fs::remove_dir_all(dir).unwrap();
}

fn tempfile_dir() -> std::path::PathBuf {
    static NEXT_TEMP_DIR: AtomicUsize = AtomicUsize::new(0);
    let suffix = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("ahuvista-test-{}-{}", std::process::id(), suffix));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn csv_import_maps_long_form_events_to_dynamic_observations() {
    let mut db = DatabaseManager::in_memory().unwrap();
    let dir = tempfile_dir();
    let path = dir.join("events.csv");
    let mut file = std::fs::File::create(&path).unwrap();
    writeln!(file, "patient_id,start,code,value,unit,visit_id,omop_table").unwrap();
    writeln!(file, "p1,2020-01-01,SNOMED/123,, ,v1,condition_occurrence").unwrap();
    writeln!(file, "p1,2020-01-02,LOINC/777-3,140,mg/dL,v1,measurement").unwrap();
    writeln!(file, "p2,2020-01-03,LOINC/777-3,130,mg/dL,v2,measurement").unwrap();

    let mapping = ColumnMapping {
        patient_id_column: Some("patient_id".into()),
        timestamp_column: Some("start".into()),
        event_feature_column: Some("code".into()),
        event_value_column: Some("value".into()),
        event_table_column: Some("omop_table".into()),
        event_visit_column: Some("visit_id".into()),
        ..Default::default()
    };

    assert_eq!(
        db.import_csv_with_mapping(&path, "patient_id", Some("ehrshot"), Some(&mapping))
            .unwrap(),
        3
    );

    let loaded = db.get_patient("p1").unwrap().unwrap();
    assert_eq!(loaded.observations.len(), 2);
    assert_eq!(
        loaded.observations[0].feature_name,
        "condition_occurrence:SNOMED/123"
    );
    assert_eq!(loaded.observations[0].value, 1.0);
    assert_eq!(loaded.observations[0].timestamp, "2020-01-01#visit:v1");
    assert_eq!(
        loaded.observations[1].feature_name,
        "measurement:LOINC/777-3"
    );
    assert_eq!(loaded.observations[1].value, 140.0);

    let sequences = db.build_patient_sequences().unwrap();
    assert_eq!(sequences.len(), 2);
    assert_eq!(sequences[0].1.dynamic_features.len(), 2);
    assert_eq!(sequences[1].1.dynamic_features.len(), 1);

    std::fs::remove_dir_all(dir).unwrap();
}

#[allow(dead_code)]
fn _sequence_type_is_public(sequence: PatientSequence) -> PatientSequence {
    sequence
}
