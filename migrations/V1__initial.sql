CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    age REAL,
    race TEXT,
    sex TEXT,
    insurance TEXT,
    systolic_bp REAL,
    diastolic_bp REAL,
    glucose REAL,
    bmi REAL,
    pregnancies INTEGER,
    risk_level TEXT,
    has_complications INTEGER NOT NULL DEFAULT 0,
    risk_score REAL NOT NULL DEFAULT 0,
    existing_outcome REAL,
    cause_code INTEGER,
    image_count INTEGER NOT NULL DEFAULT 0,
    has_ultrasound INTEGER NOT NULL DEFAULT 0,
    dataset_source TEXT
);

CREATE INDEX IF NOT EXISTS idx_patients_dataset_source
    ON patients(dataset_source);

CREATE INDEX IF NOT EXISTS idx_patients_pregnancies
    ON patients(pregnancies);

CREATE TABLE IF NOT EXISTS baseline_features (
    patient_id TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    value REAL,
    PRIMARY KEY (patient_id, feature_name),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_baseline_features_patient
    ON baseline_features(patient_id);

CREATE TABLE IF NOT EXISTS patient_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    value REAL,
    is_missing INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_patient_observations_patient_time
    ON patient_observations(patient_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_patient_observations_patient_time_feature
    ON patient_observations(patient_id, timestamp, feature_name);

CREATE TABLE IF NOT EXISTS outcomes (
    patient_id TEXT PRIMARY KEY,
    mortality REAL NOT NULL,
    cause_code INTEGER,
    outcome_time TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_outcomes_patient_id
	ON outcomes(patient_id);

CREATE TABLE IF NOT EXISTS patient_sequences (
    sample_id INTEGER PRIMARY KEY,
    patient_id TEXT NOT NULL UNIQUE,
    static_features BLOB NOT NULL,
    dynamic_features BLOB NOT NULL,
    static_feature_count INTEGER NOT NULL,
    dynamic_feature_count INTEGER NOT NULL,
    sequence_length INTEGER NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_patient_sequences_patient
    ON patient_sequences(patient_id);

CREATE TABLE IF NOT EXISTS clinical_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    note_type TEXT,
    content TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_clinical_notes_patient
    ON clinical_notes(patient_id);

CREATE TABLE IF NOT EXISTS text_data (
    sample_id INTEGER PRIMARY KEY,
    patient_id TEXT NOT NULL,
    document TEXT,
    token_sequence BLOB,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_text_data_patient_id
	ON text_data(patient_id);

CREATE TABLE IF NOT EXISTS text_vocabulary (
    word TEXT PRIMARY KEY,
    idx INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS text_metadata (
    id INTEGER PRIMARY KEY,
    vocab_size INTEGER NOT NULL,
    max_seq_length INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS image_data (
    sample_id INTEGER PRIMARY KEY,
    patient_id TEXT NOT NULL,
    image_blob BLOB NOT NULL,
    channels INTEGER NOT NULL,
    height INTEGER NOT NULL,
    width INTEGER NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_image_data_patient_id
    ON image_data(patient_id);

CREATE TABLE IF NOT EXISTS image_metadata (
    id INTEGER PRIMARY KEY,
    channels INTEGER NOT NULL,
    height INTEGER NOT NULL,
    width INTEGER NOT NULL,
    channel_means TEXT NOT NULL,
    channel_stds TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    risk_score REAL NOT NULL,
    calibrated_score REAL,
    risk_category TEXT NOT NULL,
    confidence REAL,
    predicted_at TEXT NOT NULL,
    fhir_resource TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_predictions_patient
    ON predictions(patient_id);

CREATE INDEX IF NOT EXISTS idx_predictions_predicted_at
    ON predictions(predicted_at);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    source_identifier TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    query_hash TEXT,
    record_count INTEGER NOT NULL DEFAULT 0,
    schema_version TEXT NOT NULL DEFAULT '1'
);

CREATE INDEX IF NOT EXISTS idx_ingestion_runs_source
    ON ingestion_runs(source_type, source_identifier);

CREATE TABLE IF NOT EXISTS prediction_origin (
    prediction_id TEXT PRIMARY KEY,
    patient_reference TEXT NOT NULL,
    model_reference TEXT NOT NULL,
    model_artifact_hash TEXT,
    input_source TEXT,
    profile_url TEXT,
    published_at TEXT,
    fhir_response TEXT,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS image_origin (
    sample_id INTEGER PRIMARY KEY,
    patient_id TEXT NOT NULL,
    path TEXT NOT NULL,
    association_method TEXT NOT NULL,
    FOREIGN KEY (sample_id) REFERENCES image_data(sample_id) ON DELETE CASCADE,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_image_origin_patient
    ON image_origin(patient_id);
