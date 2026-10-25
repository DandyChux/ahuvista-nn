//! SQLite database manager for handling large datasets (e.g., MIMIC-IV)
//! that don't fit in memory.
//!
//! ## Usage
//!
//! ```ignore
//! // Create/open database
//! let mut db = DatabaseManager::new("ahuvista.db")?;
//!
//! // Import a CSV file
//! db.import_csv("mimic_patients.csv", "patient_id")?;
//!
//! // Iterate over batches for training
//! for batch in db.batch_iter(32)? {
//!     let features = batch.iter().map(|p| p.to_raw_features()).collect();
//!     // train on batch...
//! }
//!
//! // Build normalizer from DB
//! let normalizer = db.build_normalizer()?;
//! ```

use crate::data::{
    FeatureStats, ImageData, NormalizationMethod, Normalizer, PatientData, PatientObservation,
    PreparedRecord, TextData, migrations,
};
use crate::data::source::ColumnMapping;
use crate::error::{AppError, AppResult};
use crate::models::PatientSequence;
use csv::ReaderBuilder;
use ndarray::Array3;
use rand::Rng;
use rusqlite::{Connection, params};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

#[derive(Debug, Clone, Copy)]
pub struct PreparedLoadOptions {
    pub use_sequence: bool,
    pub use_text: bool,
    pub use_image: bool,
}

/// Database manager
pub struct DatabaseManager {
    conn: Connection,
}

impl DatabaseManager {
    /// Open a connection to the SQLite database, creating it if it doesn't exist
    pub fn new<P: AsRef<Path>>(path: P) -> AppResult<Self> {
        let conn = Connection::open(path)?;

        // Enable foreign keys and WAL mode for better performance/safety
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA busy_timeout = 5000;",
        )?;

        let mut db = Self { conn };
        migrations::run(&mut db.conn)?;

        Ok(db)
    }

    /// Create an in-memory database (good for testing)
    pub fn in_memory() -> AppResult<Self> {
        let conn = Connection::open_in_memory()?;

        conn.execute_batch("PRAGMA foreign_keys = ON;")?;

        let mut db = Self { conn };
        migrations::run(&mut db.conn)?;

        Ok(db)
    }

    /// Clear all data from the database while preserving its schema.
    ///
    /// Tables are discovered from SQLite metadata and deleted in dependency order, so this
    /// continues to work when migrations add new tables with foreign keys.
    pub fn clear_all_data(&mut self) -> AppResult<()> {
        let tables = {
            let mut statement = self.conn.prepare(
                "SELECT name
                 FROM sqlite_master
                 WHERE type = 'table'
                   AND name NOT LIKE 'sqlite_%'
                   AND name != 'schema_migrations'",
            )?;

            statement
                .query_map([], |row| row.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?
        };

        let mut dependencies: HashMap<String, HashSet<String>> = tables
            .iter()
            .map(|table| (table.clone(), HashSet::new()))
            .collect();

        for table in &tables {
            let escaped_table = table.replace('"', "\"\"");
            let pragma = format!("PRAGMA foreign_key_list(\"{escaped_table}\")");
            let mut statement = self.conn.prepare(&pragma)?;
            let foreign_keys = statement.query_map([], |row| row.get::<_, String>(2))?;

            for foreign_key in foreign_keys {
                let parent = foreign_key?;
                if dependencies.contains_key(&parent) {
                    dependencies
                        .get_mut(table)
                        .expect("table was inserted into dependencies")
                        .insert(parent);
                }
            }
        }

        let mut delete_order = Vec::with_capacity(tables.len());

        while !dependencies.is_empty() {
            let ready: Vec<String> = dependencies
                .iter()
                .filter(|(_, parents)| parents.is_empty())
                .map(|(table, _)| table.clone())
                .collect();

            if ready.is_empty() {
                return Err(AppError::InvalidData(
                    "cannot clear database: tables contain a circular foreign-key dependency"
                        .to_string(),
                ));
            }

            for table in ready {
                dependencies.remove(&table);
                for parents in dependencies.values_mut() {
                    parents.remove(&table);
                }
                delete_order.push(table);
            }
        }

        let tx = self.conn.transaction()?;

        for table in delete_order.into_iter().rev() {
            let quoted_table = table.replace('"', "\"\"");
            let sql = format!("DELETE FROM \"{quoted_table}\"");
            tx.execute(&sql, [])?;
        }

        tx.commit()?;

        info!("Cleared all prepared data");

        Ok(())
    }

    pub fn load_prepared_records(
        &self,
        options: PreparedLoadOptions,
    ) -> AppResult<Vec<PreparedRecord>> {
        use std::collections::HashMap;

        let patient_ids =
            self.load_training_patient_ids()?;

        if patient_ids.is_empty() {
            return Ok(Vec::new());
        }

        let sequences =
            if options.use_sequence {
                self.load_sequences_by_patient()?
            } else {
                HashMap::new()
            };

        let text =
            if options.use_text {
                self.load_text_by_patient()?
            } else {
                HashMap::new()
            };

        let images =
            if options.use_image {
                self.load_images_by_patient()?
            } else {
                HashMap::new()
            };

        let outcomes =
            self.load_outcomes_by_patient()?;

        let mut records =
            Vec::with_capacity(patient_ids.len());

        for patient_id in patient_ids {
            let sequence =
                if options.use_sequence {
                    sequences
                        .get(&patient_id)
                        .cloned()
                        .ok_or_else(|| {
                            AppError::InvalidData(format!(
                                "Missing sequence for patient {}",
                                patient_id
                            ))
                        })?
                } else {
                    PatientSequence {
                        static_features: Vec::new(),
                        dynamic_features: Vec::new(),
                    }
                };

            let (target, cause_code) =
                outcomes
                    .get(&patient_id)
                    .copied()
                    .ok_or_else(|| {
                        AppError::InvalidData(format!(
                            "Missing outcome for patient {}",
                            patient_id
                        ))
                    })?;

            let images = if options.use_image {
                images
                    .get(&patient_id)
                    .cloned()
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            records.push(PreparedRecord {
                patient_id: patient_id.clone(),
                sequence,
                text: text
                    .get(&patient_id)
                    .cloned()
                    .unwrap_or_default(),
                images,
                target,
                cause_code: Some(cause_code as u32),
            });
        }

        Ok(records)
    }

    pub fn validate_prepared_data(
        &self,
        options: PreparedLoadOptions,
    ) -> AppResult<()> {
        let patient_count: i64 =
            self.conn.query_row(
                "SELECT COUNT(*) FROM patients",
                [],
                |row| row.get(0),
            )?;

        let outcome_count: i64 =
            self.conn.query_row(
                "SELECT COUNT(DISTINCT patient_id)
                 FROM outcomes",
                [],
                |row| row.get(0),
            )?;

        if patient_count == 0 {
            return Err(AppError::InvalidData(
                "No patients are stored".to_string(),
            ));
        }

        if outcome_count == 0 {
            return Err(AppError::InvalidData(
                "No outcomes are stored".to_string(),
            ));
        }

        if options.use_sequence {
            let missing_sequences: i64 =
                self.conn.query_row(
                    "SELECT COUNT(*)
                     FROM outcomes o
                     LEFT JOIN patient_sequences s
                       ON s.patient_id = o.patient_id
                     WHERE s.patient_id IS NULL",
                    [],
                    |row| row.get(0),
                )?;

            if missing_sequences > 0 {
                return Err(AppError::InvalidData(format!(
                    "{} outcome patients have no sequence",
                    missing_sequences
                )));
            }
        }

        if options.use_image {
            let missing_images: i64 =
                self.conn.query_row(
                    "SELECT COUNT(*)
                     FROM outcomes o
                     LEFT JOIN image_data i
                       ON i.patient_id = o.patient_id
                     WHERE i.patient_id IS NULL",
                    [],
                    |row| row.get(0),
                )?;

            if missing_images > 0 {
                return Err(AppError::InvalidData(format!(
                    "{} outcome patients have no image",
                    missing_images
                )));
            }
        }

        Ok(())
    }

    fn load_training_patient_ids(
        &self,
    ) -> AppResult<Vec<String>> {
        let mut statement = self.conn.prepare(
            "SELECT patient_id
             FROM outcomes
             ORDER BY patient_id",
        )?;

        let patient_ids =
            statement
                .query_map([], |row| {
                    row.get::<_, String>(0)
                })?
                .collect::<Result<Vec<_>, _>>()?;

        Ok(patient_ids)
    }

    fn load_sequences_by_patient(
        &self,
    ) -> AppResult<
        std::collections::HashMap<
            String,
            PatientSequence,
        >,
    > {
        use std::collections::HashMap;

        let mut statement = self.conn.prepare(
            "SELECT
                patient_id,
                static_features,
                dynamic_features
             FROM patient_sequences
             ORDER BY patient_id",
        )?;

        let rows =
            statement.query_map([], |row| {
                let patient_id: String =
                    row.get(0)?;

                let static_blob: Vec<u8> =
                    row.get(1)?;

                let dynamic_blob: Vec<u8> =
                    row.get(2)?;

                Ok((
                    patient_id,
                    static_blob,
                    dynamic_blob,
                ))
            })?;

        let mut sequences =
            HashMap::new();

        for row in rows {
            let (
                patient_id,
                static_blob,
                dynamic_blob,
            ) = row?;

            let sequence =
                PatientSequence {
                    static_features:
                        deserialize_f32_vec(&static_blob)?,
                    dynamic_features:
                        deserialize_dynamic_sequence(
                            &dynamic_blob,
                        )?,
                };

            if sequences
                .insert(patient_id.clone(), sequence)
                .is_some()
            {
                return Err(AppError::InvalidData(format!(
                    "Duplicate stored sequence for patient {}",
                    patient_id
                )));
            }
        }

        Ok(sequences)
    }

    fn load_text_by_patient(
        &self,
    ) -> AppResult<
        std::collections::HashMap<
            String,
            Vec<usize>,
        >,
    > {
        use std::collections::HashMap;

        let mut statement = self.conn.prepare(
            "SELECT patient_id, token_sequence
             FROM text_data
             ORDER BY patient_id",
        )?;

        let rows =
            statement.query_map([], |row| {
                let patient_id: String =
                    row.get(0)?;

                let tokens_blob: Vec<u8> =
                    row.get(1)?;

                Ok((patient_id, tokens_blob))
            })?;

        let mut text =
            HashMap::new();

        for row in rows {
            let (patient_id, tokens_blob) =
                row?;

            let tokens =
                deserialize_usize_vec(&tokens_blob)?;

            if text
                .insert(patient_id.clone(), tokens)
                .is_some()
            {
                return Err(AppError::InvalidData(format!(
                    "Duplicate text record for patient {}",
                    patient_id
                )));
            }
        }

        Ok(text)
    }

    fn load_images_by_patient(
        &self,
    ) -> AppResult<
        HashMap<String, Vec<Array3<f32>>>,
    > {
        let mut statement = self.conn.prepare(
            "SELECT
                patient_id,
                image_blob,
                channels,
                height,
                width
             FROM image_data
             ORDER BY patient_id, sample_id",
        )?;

        let rows =
            statement.query_map([], |row| {
                let patient_id: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                let channels: usize = row.get(2)?;
                let height: usize = row.get(3)?;
                let width: usize = row.get(4)?;

                let image =
                    deserialize_array3(
                        &blob,
                        channels,
                        height,
                        width,
                    )
                    .map_err(|error| {
                        rusqlite::Error::FromSqlConversionFailure(
                            1,
                            rusqlite::types::Type::Blob,
                            Box::new(error),
                        )
                    })?;

                Ok((patient_id, image))
            })?;

        let mut images: HashMap<String, Vec<Array3<f32>>> = HashMap::new();

        for row in rows {
            let (patient_id, image) =
                row?;

            images
                .entry(patient_id)
                .or_default()
                .push(image);
        }

        Ok(images)
    }

    fn load_outcomes_by_patient(
        &self,
    ) -> AppResult<
        std::collections::HashMap<
            String,
            (f32, usize),
        >,
    > {
        use std::collections::HashMap;

        let mut statement = self.conn.prepare(
            "SELECT patient_id, mortality, cause_code
             FROM outcomes
             ORDER BY patient_id",
        )?;

        let rows =
            statement.query_map([], |row| {
                let patient_id: String =
                    row.get(0)?;

                let target: f32 =
                    row.get(1)?;

                let cause_code: usize =
                    row.get::<_, i64>(2)? as usize;

                Ok((patient_id, target, cause_code))
            })?;

        let mut outcomes =
            HashMap::new();

        for row in rows {
            let (
                patient_id,
                target,
                cause_code,
            ) = row?;

            if outcomes
                .insert(
                    patient_id.clone(),
                    (target, cause_code),
                )
                .is_some()
            {
                return Err(AppError::InvalidData(format!(
                    "Duplicate outcome for patient {}",
                    patient_id
                )));
            }
        }

        Ok(outcomes)
    }

    fn load_image_metadata(
        &self,
    ) -> AppResult<Option<(usize, usize, usize)>> {
        let result =
            self.conn.query_row(
                "SELECT channels, height, width
                 FROM image_metadata
                 WHERE id = 1",
                [],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                    ))
                },
            );

        match result {
            Ok(metadata) => Ok(Some(metadata)),
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                Ok(None)
            }
            Err(error) => Err(error.into()),
        }
    }

    // =========================================================================
    // Multi-Modal Data Storage
    // =========================================================================

    /// Build unified static-plus-dynamic patient sequences from normalized tables.
    ///
    /// Static features are composed from:
    /// - known columns in `patients`;
    /// - additional values in `baseline_features`.
    ///
    /// Dynamic features are composed from:
    /// - rows in `patient_observations`;
    /// - one timestep per distinct observation timestamp.
    ///
    /// The returned vector is ordered by patient ID. Targets and all other
    /// modalities must use this same ordering.
    pub fn build_patient_sequences(&self) -> AppResult<Vec<(String, PatientSequence)>> {
        use std::collections::BTreeMap;

        let mut patient_ids: Vec<String> = self
            .conn
            .prepare("SELECT patient_id FROM patients ORDER BY patient_id")?
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;

        if patient_ids.is_empty() {
            return Ok(Vec::new());
        }

        let static_feature_names = vec![
            "age".to_string(),
            "systolic_bp".to_string(),
            "diastolic_bp".to_string(),
            "glucose".to_string(),
            "bmi".to_string(),
            "pregnancies".to_string(),
            "has_complications".to_string(),
            "risk_score".to_string(),
        ];

        let extra_static_names: Vec<String> = self
            .conn
            .prepare(
                "SELECT DISTINCT feature_name
                 FROM baseline_features
                 ORDER BY feature_name",
            )?
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;

        let mut all_static_names = static_feature_names;
        all_static_names.extend(extra_static_names);

        let dynamic_feature_names: Vec<String> = self
            .conn
            .prepare(
                "SELECT DISTINCT feature_name
                 FROM patient_observations
                 ORDER BY feature_name",
            )?
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;

        let dynamic_feature_indices: HashMap<String, usize> = dynamic_feature_names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), index))
            .collect();

        #[derive(Default)]
        struct PatientFeatures {
            age: f32,
            systolic_bp: f32,
            diastolic_bp: f32,
            glucose: f32,
            bmi: f32,
            pregnancies: f32,
            has_complications: f32,
            risk_score: f32,
            extra: HashMap<String, f32>,
            dynamic: BTreeMap<String, Vec<f32>>,
        }

        let mut features_by_patient: HashMap<String, PatientFeatures> = patient_ids
            .iter()
            .map(|id| (id.clone(), PatientFeatures::default()))
            .collect();

        // Load scalar patient features in one query.
        {
            let mut statement = self.conn.prepare(
                "SELECT
                    patient_id,
                    age,
                    systolic_bp,
                    diastolic_bp,
                    glucose,
                    bmi,
                    pregnancies,
                    has_complications,
                    risk_score
                 FROM patients",
            )?;

            let rows = statement.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<f32>>(1)?.unwrap_or_default(),
                    row.get::<_, Option<f32>>(2)?.unwrap_or_default(),
                    row.get::<_, Option<f32>>(3)?.unwrap_or_default(),
                    row.get::<_, Option<f32>>(4)?.unwrap_or_default(),
                    row.get::<_, Option<f32>>(5)?.unwrap_or_default(),
                    row.get::<_, Option<u32>>(6)?.unwrap_or_default() as f32,
                    if row.get::<_, bool>(7)? { 1.0 } else { 0.0 },
                    row.get::<_, f32>(8)?,
                ))
            })?;

            for row in rows {
                let (
                    patient_id,
                    age,
                    systolic_bp,
                    diastolic_bp,
                    glucose,
                    bmi,
                    pregnancies,
                    has_complications,
                    risk_score,
                ) = row?;

                if let Some(features) = features_by_patient.get_mut(&patient_id) {
                    features.age = age;
                    features.systolic_bp = systolic_bp;
                    features.diastolic_bp = diastolic_bp;
                    features.glucose = glucose;
                    features.bmi = bmi;
                    features.pregnancies = pregnancies;
                    features.has_complications = has_complications;
                    features.risk_score = risk_score;
                }
            }
        }

        // Load all baseline features in one query.
        {
            let mut statement = self.conn.prepare(
                "SELECT patient_id, feature_name, value
                 FROM baseline_features
                 WHERE value IS NOT NULL",
            )?;

            let rows = statement.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, f32>(2)?,
                ))
            })?;

            for row in rows {
                let (patient_id, feature_name, value) = row?;

                if let Some(features) = features_by_patient.get_mut(&patient_id) {
                    features.extra.insert(feature_name, value);
                }
            }
        }

        // Load all observations in one query, already ordered for efficient grouping.
        if !dynamic_feature_names.is_empty() {
            let mut statement = self.conn.prepare(
                "SELECT patient_id, timestamp, feature_name, value
                 FROM patient_observations
                 WHERE value IS NOT NULL
                 ORDER BY patient_id, timestamp, observation_id",
            )?;

            let rows = statement.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, f32>(3)?,
                ))
            })?;

            for row in rows {
                let (patient_id, timestamp, feature_name, value) = row?;

                let Some(&feature_index) = dynamic_feature_indices.get(&feature_name) else {
                    continue;
                };

                let Some(features) = features_by_patient.get_mut(&patient_id) else {
                    continue;
                };

                let timestep = features
                    .dynamic
                    .entry(timestamp)
                    .or_insert_with(|| vec![0.0; dynamic_feature_names.len()]);

                timestep[feature_index] = value;
            }
        }

        let mut sequences: Vec<(String, PatientSequence)> = Vec::with_capacity(patient_ids.len());

        for patient_id in patient_ids.drain(..) {
            let features = features_by_patient
            	.remove(&patient_id)
            	.ok_or_else(|| {
	                AppError::InvalidData(format!(
	                    "patient '{}' was not found while building sequences",
	                    patient_id
	                ))
             	})?;

            let static_features = all_static_names
                .iter()
                .map(|name| match name.as_str() {
                    "age" => features.age,
                    "systolic_bp" => features.systolic_bp,
                    "diastolic_bp" => features.diastolic_bp,
                    "glucose" => features.glucose,
                    "bmi" => features.bmi,
                    "pregnancies" => features.pregnancies,
                    "has_complications" => features.has_complications,
                    "risk_score" => features.risk_score,
                    _ => features.extra.get(name).copied().unwrap_or_default(),
                })
                .collect();

            let mut dynamic_features: Vec<Vec<f32>> = features.dynamic.into_values().collect();

            // Preserve the existing max-sequence behavior.
            const MAX_SEQUENCE_LENGTH: usize = 512;

            if dynamic_features.len() > MAX_SEQUENCE_LENGTH {
                let start = dynamic_features.len() - MAX_SEQUENCE_LENGTH;
                dynamic_features = dynamic_features.split_off(start);
            }

            sequences.push((
            	patient_id,
             	PatientSequence {
              		static_features,
		            dynamic_features
              	}
            ));
        }

        info!(
            "Built {} patient sequences using bulk database reads",
            sequences.len()
        );

        Ok(sequences)
    }

    pub fn store_patient_sequences(
        &mut self,
        sequences: &[(String, PatientSequence)],
    ) -> AppResult<()> {
        let tx = self.conn.transaction()?;

        tx.execute(
            "DELETE FROM patient_sequences",
            [],
        )?;

        let mut statement = tx.prepare(
            "INSERT INTO patient_sequences (
                sample_id,
                patient_id,
                static_features,
                dynamic_features,
                static_feature_count,
                dynamic_feature_count,
                sequence_length
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)",
        )?;

        for (sample_id, (patient_id, sequence))
            in sequences.iter().enumerate()
        {
            let static_blob =
                serialize_f32_vec(
                    &sequence.static_features,
                );

            let dynamic_blob =
                serialize_dynamic_sequence(
                    &sequence.dynamic_features,
                );

            let dynamic_feature_count =
                sequence
                    .dynamic_features
                    .first()
                    .map(Vec::len)
                    .unwrap_or(0);

            statement.execute(params![
                sample_id as i64,
                patient_id,
                static_blob,
                dynamic_blob,
                sequence.static_features.len() as i64,
                dynamic_feature_count as i64,
                sequence.dynamic_features.len() as i64,
            ])?;
        }

        drop(statement);
        tx.commit()?;

        Ok(())
    }

    /// Record where a normalized dataset came from for reproducibility and audit.
    pub fn record_ingestion_run(
        &self,
        source_type: &str,
        source_identifier: &str,
        record_count: usize,
    ) -> AppResult<()> {
        self.conn.execute(
            "INSERT INTO ingestion_runs (source_type, source_identifier, retrieved_at, record_count)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![source_type, source_identifier, chrono::Utc::now().to_rfc3339(), record_count as i64],
        )?;
        Ok(())
    }

    /// Store or replace one patient's unified sequence.
    pub fn store_fhir_patient_sequence(
        &mut self,
        patient_id: &str,
        sequence: &PatientSequence,
    ) -> AppResult<()> {
        let tx = self.conn.transaction()?;
        let static_blob = serialize_f32_vec(&sequence.static_features);
        let dynamic_blob = serialize_dynamic_sequence(&sequence.dynamic_features);
        let dynamic_feature_count = sequence
            .dynamic_features
            .first()
            .map(|timestep| timestep.len())
            .unwrap_or(0);

        tx.execute(
            "INSERT INTO patient_sequences (
                sample_id,
                patient_id,
                static_features,
                dynamic_features,
                static_feature_count,
                dynamic_feature_count,
                sequence_length
            ) VALUES (
                (SELECT COALESCE(MAX(sample_id), -1) + 1 FROM patient_sequences),
                ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(patient_id) DO UPDATE SET
                static_features = excluded.static_features,
                dynamic_features = excluded.dynamic_features,
                static_feature_count = excluded.static_feature_count,
                dynamic_feature_count = excluded.dynamic_feature_count,
                sequence_length = excluded.sequence_length",
            params![
                patient_id,
                static_blob,
                dynamic_blob,
                sequence.static_features.len() as i64,
                dynamic_feature_count as i64,
                sequence.dynamic_features.len() as i64,
            ],
        )?;

        tx.commit()?;
        Ok(())
    }

    /// Build TextData from the clinical_notes table
    /// Tokenizes notes and builds vocabulary
    pub fn build_text_data(&self, max_vocab_size: usize) -> AppResult<TextData> {
        // Get all clinical notes
        let notes: Vec<(String, String)> = self
            .conn
            .prepare(
                "SELECT clinical_notes.patient_id, clinical_notes.content
                 FROM clinical_notes
                 INNER JOIN patients ON patients.patient_id = clinical_notes.patient_id
                 WHERE clinical_notes.content IS NOT NULL
                   AND trim(clinical_notes.content) <> ''
                 ORDER BY clinical_notes.patient_id, clinical_notes.id",
            )?
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();

        if notes.is_empty() {
            info!("No clinical notes found in database");
            return Ok(TextData::new());
        }

        // Combine notes per patient into single documents
        let mut patient_docs: HashMap<String, String> = HashMap::new();
        for (patient_id, content) in notes {
            patient_docs
                .entry(patient_id)
                .and_modify(|doc| {
                    doc.push(' ');
                    doc.push_str(&content);
                })
                .or_insert(content);
        }

        // Sort by patient ID for consistent ordering
        let mut patient_ids: Vec<String> = patient_docs.keys().cloned().collect();
        patient_ids.sort();

        let documents: Vec<String> = patient_ids
            .iter()
            .map(|pid| patient_docs.get(pid).cloned().unwrap_or_default())
            .collect();

        // Tokenize and build vocabulary
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        let mut tokenized: Vec<Vec<String>> = Vec::new();

        for doc in &documents {
            let tokens: Vec<String> = doc
                .to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect();

            for token in &tokens {
                *word_counts.entry(token.clone()).or_insert(0) += 1;
            }
            tokenized.push(tokens);
        }

        // Build vocabulary from most frequent words
        let mut word_freq: Vec<(String, usize)> = word_counts.into_iter().collect();
        word_freq.sort_by(|a, b| b.1.cmp(&a.1));

        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut reverse_vocab: HashMap<usize, String> = HashMap::new();

        // Reserve index 0 for padding, 1 for unknown
        vocab.insert("<PAD>".to_string(), 0);
        vocab.insert("<UNK>".to_string(), 1);
        reverse_vocab.insert(0, "<PAD>".to_string());
        reverse_vocab.insert(1, "<UNK>".to_string());

        for (i, (word, _count)) in word_freq.iter().take(max_vocab_size - 2).enumerate() {
            let idx = i + 2;
            vocab.insert(word.clone(), idx);
            reverse_vocab.insert(idx, word.clone());
        }

        // Convert tokens to indices
        let mut token_sequences: Vec<Vec<usize>> = Vec::new();
        let mut max_seq_length: usize = 0;

        for tokens in &tokenized {
            let indices: Vec<usize> = tokens
                .iter()
                .map(|t| *vocab.get(t).unwrap_or(&1)) // 1 = <UNK>
                .collect();
            max_seq_length = max_seq_length.max(indices.len());
            token_sequences.push(indices);
        }

        info!(
            "Built TextData with {} documents, vocab size {}, max length {}",
            documents.len(),
            vocab.len(),
            max_seq_length
        );

        Ok(TextData {
            patient_ids: patient_ids.clone(),
            documents,
            token_sequences,
            vocab,
            reverse_vocab,
            max_seq_length,
            current_idx: 0,
        })
    }

    /// Store text data in the database
    pub fn store_text_data(&mut self, data: &TextData) -> AppResult<()> {
        let tx = self.conn.transaction()?;

        // Clear existing text data
        tx.execute("DELETE FROM text_data", [])?;
        tx.execute("DELETE FROM text_vocabulary", [])?;
        tx.execute("DELETE FROM text_metadata", [])?;

        // Store metadata
        tx.execute(
            "INSERT INTO text_metadata (id, vocab_size, max_seq_length) VALUES (1, ?, ?)",
            params![data.vocab_size(), data.max_seq_length],
        )?;

        // Store vocabulary
        {
            let mut stmt = tx.prepare("INSERT INTO text_vocabulary (word, idx) VALUES (?, ?)")?;
            for (word, idx) in &data.vocab {
                stmt.execute(params![word, *idx as i64])?;
            }
        }

        // Store documents and token sequences
        {
            let mut stmt = tx.prepare(
                "INSERT INTO text_data (
                    sample_id,
                    patient_id,
                    document,
                    token_sequence
                )
                SELECT ?1, ?2, ?3, ?4
                ON CONFLICT(patient_id) DO UPDATE SET
                    sample_id = excluded.sample_id,
                    document = excluded.document,
                    token_sequence = excluded.token_sequence",
            )?;

            if data.patient_ids.len() != data.documents.len()
                || data.documents.len() != data.token_sequences.len()
            {
                return Err(AppError::InvalidData(
                    "text patient IDs, documents, and token sequences are misaligned".to_string(),
                ));
            }

            for (idx, ((patient_id, doc), tokens)) in data
                .patient_ids
                .iter()
                .zip(data.documents.iter())
                .zip(data.token_sequences.iter())
                .enumerate()
            {
                let tokens_blob = serialize_usize_vec(tokens);

                stmt.execute(params![idx as i64, patient_id, doc, tokens_blob])?;
            }
        }

        tx.commit()?;
        info!("Stored {} text documents in database", data.documents.len());

        Ok(())
    }

    /// Build ImageData from image files referenced in the patients table
    /// Note: Images are loaded from files, not stored in the database
    pub fn build_image_data<P: AsRef<Path>>(
        &self,
        image_dir: P,
        target_size: (usize, usize),
        grayscale: bool,
    ) -> AppResult<ImageData> {
        let image_dir = image_dir.as_ref();
        let image_config = crate::data::source::ImageRoot {
            path: image_dir.display().to_string(),
            patient_id_pattern: None,
            patient_id_depth: Some(1),
        };
        let records = crate::data::source::discover_associated_images(image_dir, &image_config)?;
        self.build_image_data_for_records(&records, target_size, grayscale)
    }

    pub fn build_image_data_for_records(
        &self,
        records: &[(String, PathBuf)],
        target_size: (usize, usize),
        grayscale: bool,
    ) -> AppResult<ImageData> {
        use std::fs;

        if records.is_empty() {
            info!("No associated image files found");
            return Ok(ImageData::new());
        }

        let image_paths: Vec<PathBuf> = records.iter().map(|(_, path)| path.clone()).collect();
        let patient_ids: Vec<String> = records
            .iter()
            .map(|(patient_id, _)| patient_id.clone())
            .collect();

        if image_paths.is_empty() {
            info!("No associated image files found");
            return Ok(ImageData::new());
        }

        // Load and process images
        let channels = if grayscale { 1 } else { 3 };
        let mut images: Vec<Array3<f32>> = Vec::with_capacity(image_paths.len());
        let mut channel_sums = vec![0.0f64; channels];
        let mut channel_sq_sums = vec![0.0f64; channels];
        let mut pixel_count: usize = 0;

        for path in &image_paths {
            match image::open(path) {
                Ok(img) => {
                    let img = img.resize_exact(
                        target_size.0 as u32,
                        target_size.1 as u32,
                        image::imageops::FilterType::Triangle,
                    );

                    let img = if grayscale {
                        image::DynamicImage::ImageLuma8(img.to_luma8())
                    } else {
                        image::DynamicImage::ImageRgb8(img.to_rgb8())
                    };

                    // Convert to Array3<f32> (channels × height × width)
                    let (width, height) = (target_size.0, target_size.1);
                    let mut arr = Array3::<f32>::zeros((channels, height, width));

                    match &img {
                        image::DynamicImage::ImageLuma8(gray) => {
                            for (y, row) in gray.rows().enumerate() {
                                for (x, pixel) in row.enumerate() {
                                    let val = pixel.0[0] as f32 / 255.0;
                                    arr[[0, y, x]] = val;
                                    channel_sums[0] += val as f64;
                                    channel_sq_sums[0] += (val * val) as f64;
                                }
                            }
                        }
                        image::DynamicImage::ImageRgb8(rgb) => {
                            for (y, row) in rgb.rows().enumerate() {
                                for (x, pixel) in row.enumerate() {
                                    for c in 0..3 {
                                        let val = pixel.0[c] as f32 / 255.0;
                                        arr[[c, y, x]] = val;
                                        channel_sums[c] += val as f64;
                                        channel_sq_sums[c] += (val * val) as f64;
                                    }
                                }
                            }
                        }
                        _ => continue,
                    }

                    pixel_count += width * height;
                    images.push(arr);
                }
                Err(e) => {
                    warn!("Failed to load image {:?}: {}", path, e);
                }
            }
        }

        // Compute channel means and stds
        let channel_means: Vec<f32> = channel_sums
            .iter()
            .map(|&sum| (sum / pixel_count as f64) as f32)
            .collect();

        let channel_stds: Vec<f32> = channel_sq_sums
            .iter()
            .zip(&channel_sums)
            .map(|(&sq_sum, &sum)| {
                let mean = sum / pixel_count as f64;
                let variance = (sq_sum / pixel_count as f64) - (mean * mean);
                variance.sqrt().max(1e-8) as f32
            })
            .collect();

        info!(
            "Built ImageData with {} images, size {}x{}, {} channels",
            images.len(),
            target_size.0,
            target_size.1,
            channels
        );

        Ok(ImageData {
            images,
            patient_ids,
            image_size: target_size,
            channels,
            channel_means,
            channel_stds,
            current_idx: 0,
        })
    }

    /// Store image data in the database
    pub fn store_image_data(&mut self, data: &ImageData) -> AppResult<()> {
        let tx = self.conn.transaction()?;

        // Clear existing image data
        tx.execute("DELETE FROM image_data", [])?;
        tx.execute("DELETE FROM image_metadata", [])?;

        // Store metadata
        let channel_means_json = serde_json::to_string(&data.channel_means).unwrap_or_default();
        let channel_stds_json = serde_json::to_string(&data.channel_stds).unwrap_or_default();
        tx.execute(
                "INSERT INTO image_metadata (id, channels, height, width, channel_means, channel_stds) VALUES (1, ?, ?, ?, ?, ?)",
                params![data.channels, data.image_size.0, data.image_size.1, channel_means_json, channel_stds_json],
            )?;

        // Store each image
        {
            let mut stmt = tx.prepare(
                    "INSERT INTO image_data (sample_id, patient_id, image_blob, channels, height, width) VALUES (?, ?, ?, ?, ?, ?)",
                )?;

            for (idx, image) in data.images.iter().enumerate() {
                let image_blob = serialize_array3(image);
                let patient_id = data
                    .patient_ids
                    .get(idx)
                    .filter(|id| !id.is_empty())
                    .ok_or_else(|| {
                        AppError::InvalidData(format!(
                            "Missing patient association for image {idx}"
                        ))
                    })?;
                stmt.execute(params![
                    idx as i64,
                    patient_id,
                    image_blob,
                    data.channels,
                    data.image_size.0,
                    data.image_size.1
                ])?;
            }
        }

        tx.commit()?;
        info!("Stored {} images in database", data.images.len());
        Ok(())
    }

    /// Build outcomes from the patients table
    /// Returns (targets, causes) vectors aligned with patient order
    /// Uses existing_outcome as the target, and derives cause_code from risk_level
    pub fn build_outcomes(&self) -> AppResult<(Vec<f32>, Vec<usize>)> {
        // Get all patients in consistent order
        let rows: Vec<(String, Option<f32>, Option<i32>, Option<String>)> = self
	        .conn
	        .prepare(
	            "SELECT patient_id, existing_outcome, cause_code, risk_level FROM patients ORDER BY patient_id",
	        )?
	        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)))?
	        .filter_map(|r| r.ok())
	        .collect();

        if rows.is_empty() {
            info!("No patients found in database for outcomes");
            return Ok((Vec::new(), Vec::new()));
        }

        let mut targets: Vec<f32> = Vec::with_capacity(rows.len());
        let mut causes: Vec<usize> = Vec::with_capacity(rows.len());
        let mut rng = rand::rng();

        for (_patient_id, existing_outcome, cause_code, risk_level) in rows {
            // Use existing_outcome as target (default to 0.0 if not present)
            let target = match existing_outcome {
                Some(outcome) => outcome,
                None => {
                    // generated_count += 1;
                    // Generate synthetic outcome based on risk level
                    let mortality_prob = match risk_level.as_deref() {
                        Some(level) => {
                            let level_lower = level.to_lowercase();
                            if level_lower.contains("high") {
                                0.3 // 30% mortality for high risk
                            } else if level_lower.contains("mid") || level_lower.contains("medium")
                            {
                                0.1 // 10% mortality for medium risk
                            } else {
                                0.02 // 2% mortality for low risk
                            }
                        }
                        None => 0.1, // 10% default
                    };
                    if rng.random::<f32>() < mortality_prob {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
            targets.push(target);

            // Use cause_code if present, otherwise derive from risk_level or generate
            let cause = match cause_code {
                Some(code) => code as usize,
                None => {
                    if target > 0.5 {
                        // Generate random cause for mortality cases
                        match risk_level.as_deref() {
                            Some(level) => {
                                let level_lower = level.to_lowercase();
                                if level_lower.contains("high") {
                                    rng.random_range(0..3) // More severe causes
                                } else {
                                    rng.random_range(2..5) // Less severe causes
                                }
                            }
                            None => rng.random_range(0..5),
                        }
                    } else {
                        4 // No cause for survivors
                    }
                }
            };
            causes.push(cause);
        }

        info!(
            "Built outcomes for {} patients ({} with non-zero targets)",
            targets.len(),
            targets.iter().filter(|&&t| t != 0.0).count()
        );

        Ok((targets, causes))
    }

    /// Store outcomes in the database
    pub fn store_outcomes(&mut self, targets: &[f32], causes: &[usize]) -> AppResult<()> {
        if targets.len() != causes.len() {
            return Err(AppError::InvalidData(
                "Outcome and cause counts must match".to_string(),
            ));
        }

        let patient_ids: Vec<String> = self
            .conn
            .prepare("SELECT patient_id FROM patients ORDER BY patient_id")?
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        if patient_ids.len() < targets.len() {
            return Err(AppError::InvalidData(
                "Outcome count exceeds patient count".to_string(),
            ));
        }

        let tx = self.conn.transaction()?;

        // Clear existing outcomes
        tx.execute("DELETE FROM outcomes", [])?;

        // Store each outcome
        {
            let mut stmt = tx.prepare(
                "INSERT INTO outcomes (patient_id, mortality, cause_code) VALUES (?, ?, ?)",
            )?;

            for (idx, (target, cause)) in targets.iter().zip(causes.iter()).enumerate() {
                let patient_id = &patient_ids[idx];
                stmt.execute(params![patient_id, *target, *cause as i64])?;
            }
        }

        tx.commit()?;

        info!("Stored {} outcomes in database", targets.len());

        Ok(())
    }

    pub fn get_sample_counts(
        &self,
    ) -> AppResult<SampleCounts> {
        let sequences: i64 =
            self.conn.query_row(
                "SELECT COUNT(*) FROM patient_sequences",
                [],
                |row| row.get(0),
            )?;

        let text: i64 =
            self.conn.query_row(
                "SELECT COUNT(DISTINCT patient_id)
                 FROM text_data",
                [],
                |row| row.get(0),
            )?;

        let image: i64 =
            self.conn.query_row(
                "SELECT COUNT(DISTINCT patient_id)
                 FROM image_data",
                [],
                |row| row.get(0),
            )?;

        let outcomes: i64 =
            self.conn.query_row(
                "SELECT COUNT(DISTINCT patient_id)
                 FROM outcomes",
                [],
                |row| row.get(0),
            )?;

        Ok(SampleCounts {
            sequences: sequences as usize,
            text: text as usize,
            image: image as usize,
            outcomes: outcomes as usize,
        })
    }

    // =========================================================================
    // CSV Import Methods
    // =========================================================================

    /// Import a CSV file into the database
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file
    /// * `patient_id_column` - Name of the column containing patient IDs
    /// Import a CSV using the built-in column heuristics.
    ///
    /// This compatibility wrapper preserves the original public API. Use
    /// `import_csv_with_mapping` when a manifest declares source semantics.
    pub fn import_csv<P: AsRef<Path>>(
        &mut self,
        path: P,
        patient_id_column: &str,
        dataset_source: Option<&str>,
    ) -> AppResult<usize> {
        self.import_csv_with_mapping(path, patient_id_column, dataset_source, None)
    }

    /// Import a CSV file using an optional manifest-declared semantic mapping.
    ///
    /// * `path` - Path to the CSV file
    /// * `patient_id_column` - Fallback name of the patient ID column
    /// * `dataset_source` - Optional label for the data source (e.g., "MIMIC-IV")
    /// * `mapping` - Optional manifest-declared semantic column mapping
    pub fn import_csv_with_mapping<P: AsRef<Path>>(
        &mut self,
        path: P,
        patient_id_column: &str,
        dataset_source: Option<&str>,
        mapping: Option<&ColumnMapping>,
    ) -> AppResult<usize> {
        let path = path.as_ref();

        let file = File::open(path)?;
        let mut reader = ReaderBuilder::new()
            .flexible(true)
            .from_reader(file);

        let headers = reader
            .headers()?
            .iter()
            .map(str::trim)
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();

        if headers.is_empty() {
            return Err(AppError::InvalidData(format!(
                "CSV file has no headers: {}",
                path.display()
            )));
        }

        let id_column_index = find_patient_id_column(
            &headers,
            mapping
                .and_then(|value| value.patient_id_column.as_deref())
                .unwrap_or(patient_id_column),
        );

        let timestamp_column_index = mapping
            .and_then(|value| value.timestamp_column.as_deref())
            .and_then(|column| find_column(&headers, column))
            .or_else(|| {
                if mapping
                    .map(|value| value.disable_timestamp_detection)
                    .unwrap_or(false)
                {
                    None
                } else {
                    headers.iter().position(|header| {
                        matches!(normalize_column_name(header).as_str(),
                            "timestamp" | "datetime" | "date_time" |
                            "effective_date_time" | "observation_time" |
                            "charttime" | "storetime" | "admittime" |
                            "dischtime" | "intime" | "outtime")
                    })
                }
            });

        let note_column_index = mapping
            .and_then(|value| value.note_column.as_deref())
            .and_then(|column| find_column(&headers, column))
            .or_else(|| {
                headers.iter().position(|header| {
                    let normalized = normalize_column_name(header);
                    normalized.contains("note")
                        || normalized.contains("comment")
                        || normalized.contains("description")
                        || normalized == "text"
                        || normalized == "clinical_note"
                })
            });

        let note_type_column_index = mapping
            .and_then(|value| value.note_type_column.as_deref())
            .and_then(|column| find_column(&headers, column))
            .or_else(|| {
                headers.iter().position(|header| {
                    matches!(normalize_column_name(header).as_str(),
                        "note_type" | "note_category" | "note_kind")
                })
            });

        let has_timestamp =
            timestamp_column_index.is_some();

        let has_notes =
            note_column_index.is_some();

        let known_columns =
            known_column_mappings();

        let feature_blacklist =
            feature_blacklist();

        let outcome_column = mapping
            .and_then(|value| value.outcome_column.as_deref())
            .and_then(|column| find_column(&headers, column));

        let mut ignored_columns = mapping
            .map(|value| {
                value
                    .ignored_columns
                    .iter()
                    .filter_map(|column| find_column(&headers, column))
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default();

        let long_form_event = build_long_form_event_columns(
            &headers,
            mapping,
            timestamp_column_index,
        )?;

        if let Some(event) = &long_form_event {
            ignored_columns.insert(event.feature_column);
            if let Some(index) = event.value_column {
                ignored_columns.insert(index);
            }
            if let Some(index) = event.table_column {
                ignored_columns.insert(index);
            }
            if let Some(index) = event.visit_column {
                ignored_columns.insert(index);
            }
        }

        let column_actions = build_column_actions(
            &headers,
            id_column_index,
            outcome_column,
            note_column_index,
            timestamp_column_index,
            &known_columns,
            &feature_blacklist,
            mapping,
            &ignored_columns,
        );

        let measurement_columns =
            if has_timestamp && long_form_event.is_none() {
                headers
                    .iter()
                    .enumerate()
                    .filter(|(index, header)| {
                        Some(*index) != id_column_index
                            && Some(*index) != timestamp_column_index
                            && Some(*index) != note_column_index
                            && !ignored_columns.contains(index)
                            && !known_columns.contains_key(
                                normalize_column_name(header).as_str(),
                            )
                            && !feature_blacklist.contains(
                                normalize_column_name(header).as_str(),
                            )
                    })
                    .map(|(index, header)| {
                        (index, header.clone())
                    })
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };

        let mut generated_patient_id = 0usize;
        let mut batch = Vec::with_capacity(10_000);
        let mut imported = 0usize;

        for record_result in reader.records() {
            let record = record_result?;

            let patient_id =
                resolve_patient_id(
                    &record,
                    id_column_index,
                    &mut generated_patient_id,
                );

            let mut patient = PatientData {
                patient_id,
                ..PatientData::default()
            };

            if let Some(event) = &long_form_event {
                import_long_form_event(&record, event, &mut patient);
            } else {
                // A wide export may contain static features, observations, and notes
                // in the same row. Import each modality independently.
                import_static_fields(&record, &column_actions, &mut patient);

                if has_timestamp {
                    import_observations(
                        &record,
                        timestamp_column_index,
                        &measurement_columns,
                        &mut patient,
                    );
                }
            }

            if has_notes {
                import_note(
                    &record,
                    note_column_index,
                    note_type_column_index,
                    &mut patient,
                );
            }

            batch.push(patient);

            if batch.len() >= 10_000 {
                self.insert_patient_batch(
                    &batch,
                    dataset_source,
                )?;

                imported += batch.len();
                batch.clear();

                info!(
                    "Imported {} records from {}",
                    imported,
                    path.display()
                );
            }
        }

        if !batch.is_empty() {
            self.insert_patient_batch(
                &batch,
                dataset_source,
            )?;

            imported += batch.len();
        }

        info!(
            "Imported {} records from {}",
            imported,
            path.display()
        );

        Ok(imported)
    }

    /// Insert a batch of patients in a single transaction.
    ///
    /// Statements are prepared once per transaction rather than once per patient. This is
    /// important for large CSV imports, where preparing statements repeatedly can dominate the
    /// cost of parsing and inserting the data.
    fn insert_patient_batch(
        &mut self,
        patients: &[PatientData],
        dataset_source: Option<&str>,
    ) -> AppResult<()> {
        let tx = self.conn.transaction()?;
        let mut patient_stmt = tx.prepare_cached(
            "INSERT INTO patients (
                patient_id, sex, age, race, insurance, systolic_bp, diastolic_bp,
                glucose, bmi, pregnancies, risk_level, has_complications,
                risk_score, existing_outcome, cause_code, image_count, has_ultrasound, dataset_source
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)
            ON CONFLICT(patient_id) DO UPDATE SET
	            sex = COALESCE(excluded.sex, patients.sex),
	            age = COALESCE(excluded.age, patients.age),
	            race = COALESCE(excluded.race, patients.race),
	            insurance = COALESCE(excluded.insurance, patients.insurance),
	            systolic_bp = COALESCE(excluded.systolic_bp, patients.systolic_bp),
	            diastolic_bp = COALESCE(excluded.diastolic_bp, patients.diastolic_bp),
	            glucose = COALESCE(excluded.glucose, patients.glucose),
	            bmi = COALESCE(excluded.bmi, patients.bmi),
	            pregnancies = COALESCE(excluded.pregnancies, patients.pregnancies),
	            risk_level = COALESCE(excluded.risk_level, patients.risk_level),
	            has_complications =
	                patients.has_complications
	                OR excluded.has_complications,
	            risk_score = MAX(patients.risk_score, excluded.risk_score),
	            existing_outcome =
	                COALESCE(excluded.existing_outcome, patients.existing_outcome),
	            cause_code =
	                COALESCE(excluded.cause_code, patients.cause_code),
	            image_count =
	                patients.image_count + excluded.image_count,
	            has_ultrasound =
	                patients.has_ultrasound OR excluded.has_ultrasound,
	            dataset_source =
	                COALESCE(excluded.dataset_source, patients.dataset_source)",
        )?;
        let mut feature_stmt = tx.prepare_cached(
            "INSERT OR REPLACE INTO baseline_features (patient_id, feature_name, value)
             VALUES (?, ?, ?)",
        )?;
        let mut observation_stmt = tx.prepare_cached(
            "INSERT INTO patient_observations (patient_id, timestamp, feature_name, value)
             VALUES (?, ?, ?, ?)",
        )?;
        let mut note_stmt =
            tx.prepare_cached("INSERT INTO clinical_notes (patient_id, content) VALUES (?, ?)")?;

        for patient in patients {
            patient_stmt.execute(params![
                patient.patient_id,
                patient.sex,
                patient.age,
                patient.race,
                patient.insurance,
                patient.systolic_bp,
                patient.diastolic_bp,
                patient.glucose,
                patient.bmi,
                patient.pregnancies,
                patient.risk_level,
                patient.has_complications,
                patient.risk_score,
                patient.existing_outcome,
                patient.cause_code,
                patient.image_count as i64,
                patient.has_ultrasound,
                dataset_source,
            ])?;

            for (name, value) in &patient.extra_features {
                feature_stmt.execute(params![patient.patient_id, name, value])?;
            }

            for observation in &patient.observations {
                observation_stmt.execute(params![
                    patient.patient_id,
                    observation.timestamp,
                    observation.feature_name,
                    observation.value,
                ])?;
            }

            for note in &patient.clinical_notes {
                note_stmt.execute(params![patient.patient_id, note])?;
            }
        }

        drop(note_stmt);
        drop(observation_stmt);
        drop(feature_stmt);
        drop(patient_stmt);

        tx.commit()?;
        Ok(())
    }

    /// Import multiple CSV files from a directory, including subdirectories.
    /// Determines `dataset_source` based on folder structure if not explicitly provided.
    pub fn import_csv_directory<P: AsRef<Path>>(
        &mut self,
        dir: P,
        patient_id_column: &str,
        dataset_source: Option<&str>,
    ) -> AppResult<usize> {
        let mut total = 0;
        let base_dir = dir.as_ref();

        // We'll use a stack to process directories recursively
        let mut dirs_to_visit = vec![base_dir.to_path_buf()];

        while let Some(current_dir) = dirs_to_visit.pop() {
            let entries = std::fs::read_dir(&current_dir).map_err(|e| {
                rusqlite::Error::InvalidParameterName(format!(
                    "Failed to read directory {}: {}",
                    current_dir.display(),
                    e
                ))
            })?;

            for entry in entries {
                let entry = entry.map_err(|e| {
                    rusqlite::Error::InvalidParameterName(format!("Directory entry error: {}", e))
                })?;
                let path = entry.path();

                if path.is_dir() {
                    dirs_to_visit.push(path);
                } else if path.extension().and_then(|s| s.to_str()) == Some("csv") {
                    info!("Importing: {}", path.display());

                    let derived_source = if let Ok(rel_path) = path.strip_prefix(base_dir) {
                        // If it's in a subfolder, use the top-level subfolder name.
                        // Example: "MyDataset/metadata.csv" -> "MyDataset"
                        let mut components = rel_path.components();
                        let first_component = components.next();
                        let second_component = components.next();

                        if second_component.is_some() {
                            // It's nested in a folder
                            first_component
                                .and_then(|c| c.as_os_str().to_str())
                                .map(|s| s.to_string())
                        } else {
                            // It's directly in the base dir, use the file stem
                            path.file_stem()
                                .and_then(|s| s.to_str())
                                .map(|s| s.to_string())
                        }
                    } else {
                        // Fallback
                        path.file_stem()
                            .and_then(|s| s.to_str())
                            .map(|s| s.to_string())
                    };

                    let source_to_use = dataset_source.or(derived_source.as_deref());

                    total += self.import_csv(&path, patient_id_column, source_to_use)?;
                }
            }
        }

        Ok(total)
    }

    // =========================================================================
    // Patient CRUD Operations
    // =========================================================================

    /// Upsert a patient record within a transaction
    fn upsert_patient_tx(
        tx: &rusqlite::Transaction,
        patient: &PatientData,
        dataset_source: Option<&str>,
    ) -> AppResult<()> {
        tx.execute(
            "INSERT INTO patients (
                patient_id, age, race, insurance, systolic_bp, diastolic_bp,
                glucose, bmi, pregnancies, risk_level, has_complications,
                risk_score, existing_outcome, cause_code, image_count, has_ultrasound, dataset_source
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)
            ON CONFLICT(patient_id) DO UPDATE SET
                age = coalesce(excluded.age, age),
                race = coalesce(excluded.race, race),
                insurance = coalesce(excluded.insurance, insurance),
                systolic_bp = coalesce(excluded.systolic_bp, systolic_bp),
                diastolic_bp = coalesce(excluded.diastolic_bp, diastolic_bp),
                glucose = coalesce(excluded.glucose, glucose),
                bmi = coalesce(excluded.bmi, bmi),
                pregnancies = coalesce(excluded.pregnancies, pregnancies),
                risk_level = coalesce(excluded.risk_level, risk_level),
                has_complications = has_complications OR excluded.has_complications,
                risk_score = max(risk_score, excluded.risk_score),
                existing_outcome = coalesce(excluded.existing_outcome, existing_outcome),
                cause_code = coalesce(excluded.cause_code, cause_code),
                image_count = image_count + excluded.image_count,
                has_ultrasound = has_ultrasound OR excluded.has_ultrasound,
                dataset_source = coalesce(excluded.dataset_source, dataset_source)
            ",
            params![
                patient.patient_id,
                patient.age,
                patient.race,
                patient.insurance,
                patient.systolic_bp,
                patient.diastolic_bp,
                patient.glucose,
                patient.bmi,
                patient.pregnancies,
                patient.risk_level,
                patient.has_complications,
                patient.risk_score,
                patient.existing_outcome,
                patient.cause_code,
                patient.image_count as i64,
                patient.has_ultrasound,
                dataset_source,
            ],
        )?;

        // Insert extra features
        let mut stmt = tx.prepare_cached(
            "INSERT OR REPLACE INTO baseline_features (patient_id, feature_name, value)
             VALUES (?, ?, ?)",
        )?;
        for (name, value) in &patient.extra_features {
            stmt.execute(params![patient.patient_id, name, value])?;
        }

        Ok(())
    }

    /// Upsert a patient record (insert or update if exists)
    pub fn upsert_patient(&mut self, patient: &PatientData) -> AppResult<()> {
        let tx = self.conn.transaction()?;
        Self::upsert_patient_tx(&tx, patient, None)?;

        // Insert temporal measurements
        {
            let mut stmt = tx.prepare(
                "INSERT INTO patient_observations (patient_id, timestamp, feature_name, value)
                 VALUES (?, ?, ?, ?)",
            )?;
            for m in &patient.observations {
                stmt.execute(params![
                    patient.patient_id,
                    m.timestamp,
                    m.feature_name,
                    m.value
                ])?;
            }
        }

        // Insert clinical notes
        {
            let mut stmt =
                tx.prepare("INSERT INTO clinical_notes (patient_id, content) VALUES (?, ?)")?;
            for note in &patient.clinical_notes {
                stmt.execute(params![patient.patient_id, note])?;
            }
        }

        tx.commit()?;
        Ok(())
    }

    /// Retrieve a single patient by ID
    pub fn get_patient(&self, patient_id: &str) -> AppResult<Option<PatientData>> {
        let mut stmt = self.conn.prepare(
            "SELECT age, systolic_bp, diastolic_bp, glucose, bmi,
                    pregnancies, risk_level, has_complications, risk_score,
                    existing_outcome, image_count, has_ultrasound
             FROM patients WHERE patient_id = ?1",
        )?;

        let mut rows = stmt.query(params![patient_id])?;

        if let Some(row) = rows.next()? {
            let mut patient = PatientData {
                patient_id: patient_id.to_string(),
                age: row.get(0)?,
                systolic_bp: row.get(1)?,
                diastolic_bp: row.get(2)?,
                glucose: row.get(3)?,
                bmi: row.get(4)?,
                pregnancies: row.get(5)?,
                risk_level: row.get(6)?,
                has_complications: row.get(7)?,
                risk_score: row.get(8)?,
                existing_outcome: row.get(9)?,
                image_count: row.get::<_, i64>(10)? as usize,
                has_ultrasound: row.get(11)?,
                ..Default::default()
            };

            // Load related data
            patient.extra_features = self.get_patient_features(patient_id)?;
            patient.observations = self.get_observations(patient_id)?;
            patient.clinical_notes = self.get_clinical_notes(patient_id)?;

            Ok(Some(patient))
        } else {
            Ok(None)
        }
    }

    /// Retrieve all patient IDs
    pub fn get_all_patient_ids(&self) -> AppResult<Vec<String>> {
        let mut stmt = self.conn.prepare(
        	"SELECT patient_id
         	FROM patients
          	ORDER BY patient_id"
        )?;

        let ids = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ids)
    }

    /// Get total count of patients
    pub fn count_patients(&self) -> AppResult<usize> {
        let mut stmt = self.conn.prepare("SELECT COUNT(*) FROM patients")?;
        let count: i64 = stmt.query_row([], |row| row.get(0))?;
        Ok(count as usize)
    }

    // =========================================================================
    // Batch Operations (for Training)
    // =========================================================================

    /// Get a batch of patients by offset and limit
    pub fn get_patient_batch(&self, offset: usize, limit: usize) -> AppResult<Vec<PatientData>> {
        let mut stmt = self
            .conn
            .prepare("SELECT patient_id FROM patients LIMIT ?1 OFFSET ?2")?;

        let ids: Vec<String> = stmt
            .query_map(params![limit as i64, offset as i64], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        let mut patients = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(patient) = self.get_patient(&id)? {
                patients.push(patient);
            }
        }

        Ok(patients)
    }

    /// Create a batch iterator for training
    pub fn batch_iter(&self, batch_size: usize) -> AppResult<BatchIterator> {
        let total = self.count_patients()?;
        Ok(BatchIterator {
            db_path: self.conn.path().map(|p| PathBuf::from(p)),
            batch_size,
            total,
            current_offset: 0,
        })
    }

    /// Get patients with outcomes (for supervised training)
    pub fn get_patients_with_outcomes(&self) -> AppResult<Vec<(PatientData, f32)>> {
        let mut stmt = self.conn.prepare(
            "SELECT patient_id, existing_outcome FROM patients WHERE existing_outcome IS NOT NULL",
        )?;

        let rows: Vec<(String, f32)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();

        let mut results = Vec::with_capacity(rows.len());
        for (id, outcome) in rows {
            if let Some(patient) = self.get_patient(&id)? {
                results.push((patient, outcome));
            }
        }

        Ok(results)
    }

    // =========================================================================
    // Normalization
    // =========================================================================

    /// Build a Normalizer from all data in the database
    pub fn build_normalizer(&self) -> AppResult<Normalizer> {
        self.build_normalizer_with_method(NormalizationMethod::ZScore)
    }

    /// Build a Normalizer with a specific method
    pub fn build_normalizer_with_method(
        &self,
        method: NormalizationMethod,
    ) -> AppResult<Normalizer> {
        // Compute stats for core numeric columns
        let core_columns = [
            "age",
            "systolic_bp",
            "diastolic_bp",
            "glucose",
            "bmi",
            "pregnancies",
            "risk_score",
        ];

        let mut feature_stats = HashMap::new();
        let mut feature_order = Vec::new();

        for col in &core_columns {
            if let Some(stats) = self.compute_column_stats(col)? {
                feature_order.push(col.to_string());
                feature_stats.insert(col.to_string(), stats);
            }
        }

        // Compute stats for dynamic features
        let extra_features = self.get_all_feature_names()?;
        for name in extra_features {
            if let Some(stats) = self.compute_extra_feature_stats(&name)? {
                feature_order.push(name.clone());
                feature_stats.insert(name, stats);
            }
        }

        let normalizer = Normalizer {
            feature_stats,
            feature_order,
            categorical_columns: HashSet::new(),
            exclude_columns: HashSet::new(),
            method,
            missing_value: 0.0,
        };

        info!("Built normalizer from database:\n{}", normalizer.summary());
        Ok(normalizer)
    }

    /// Compute statistics for a core column
    fn compute_column_stats(&self, column: &str) -> AppResult<Option<FeatureStats>> {
        let sql = format!(
            "SELECT AVG({col}),
                    AVG({col} * {col}) - AVG({col}) * AVG({col}),
                    MIN({col}),
                    MAX({col}),
                    COUNT({col})
             FROM patients WHERE {col} IS NOT NULL",
            col = column
        );

        let mut stmt = self.conn.prepare(&sql)?;
        let result: Option<(f64, f64, f64, f64, i64)> = stmt
            .query_row([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            })
            .ok();

        if let Some((mean, variance, min, max, count)) = result {
            if count > 0 {
                let std = (variance.max(0.0) as f32).sqrt().max(1e-8);
                return Ok(Some(FeatureStats {
                    name: column.to_string(),
                    mean: mean as f32,
                    std,
                    min: min as f32,
                    max: max as f32,
                    count: count as usize,
                }));
            }
        }

        Ok(None)
    }

    /// Compute statistics for an extra feature
    fn compute_extra_feature_stats(&self, feature_name: &str) -> AppResult<Option<FeatureStats>> {
        let mut stmt = self.conn.prepare(
            "SELECT AVG(value),
                    AVG(value * value) - AVG(value) * AVG(value),
                    MIN(value),
                    MAX(value),
                    COUNT(value)
             FROM baseline_features WHERE feature_name = ?",
        )?;

        let result: Option<(f64, f64, f64, f64, i64)> = stmt
            .query_row(params![feature_name], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            })
            .ok();

        if let Some((mean, variance, min, max, count)) = result {
            if count > 0 {
                let std = (variance.max(0.0) as f32).sqrt().max(1e-8);
                return Ok(Some(FeatureStats {
                    name: feature_name.to_string(),
                    mean: mean as f32,
                    std,
                    min: min as f32,
                    max: max as f32,
                    count: count as usize,
                }));
            }
        }

        Ok(None)
    }

    /// Get all unique extra feature names
    fn get_all_feature_names(&self) -> AppResult<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT feature_name FROM baseline_features")?;
        let rows = stmt.query_map([], |row| row.get(0))?;

        let mut names = Vec::new();
        for name in rows {
            names.push(name?);
        }
        Ok(names)
    }

    /// Save a normalizer to the database
    pub fn save_normalizer(&mut self, normalizer: &Normalizer) -> AppResult<()> {
        let _ = normalizer;
        Err(AppError::DomainError(
            "Normalization statistics are stored in model artifacts, not SQLite".to_string(),
        ))
    }

    /// Load a normalizer from the database
    pub fn load_normalizer(&self) -> AppResult<Option<Normalizer>> {
        Ok(None)
    }

    pub fn cleanup_non_maternal_patients(
        &mut self,
    ) -> AppResult<usize> {
        let tx = self.conn.transaction()?;

        let removed = tx.execute(
            "DELETE FROM patients
             WHERE LOWER(TRIM(COALESCE(sex, ''))) IN (
                 'male',
                 'm',
                 'man',
                 'boy'
             )",
            [],
        )?;

        tx.commit()?;

        info!(
            "Removed {} explicitly non-maternal patients",
            removed
        );

        Ok(removed)
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    fn get_patient_features(&self, patient_id: &str) -> AppResult<HashMap<String, f32>> {
        let mut stmt = self
            .conn
            .prepare("SELECT feature_name, value FROM baseline_features WHERE patient_id = ?")?;
        let rows = stmt.query_map(params![patient_id], |row| Ok((row.get(0)?, row.get(1)?)))?;

        let mut features = HashMap::new();
        for row in rows {
            let (name, value) = row?;
            features.insert(name, value);
        }
        Ok(features)
    }

    fn get_observations(&self, patient_id: &str) -> AppResult<Vec<PatientObservation>> {
        let mut stmt = self.conn.prepare(
            "SELECT timestamp, feature_name, value FROM patient_observations WHERE patient_id = ?
             ORDER BY timestamp, observation_id",
        )?;
        let rows = stmt.query_map(params![patient_id], |row| {
            Ok(PatientObservation {
                timestamp: row.get(0)?,
                feature_name: row.get(1)?,
                value: row.get(2)?,
            })
        })?;

        let mut measurements = Vec::new();
        for row in rows {
            measurements.push(row?);
        }
        Ok(measurements)
    }

    fn get_clinical_notes(&self, patient_id: &str) -> AppResult<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT content FROM clinical_notes WHERE patient_id = ?")?;
        let rows = stmt.query_map(params![patient_id], |row| row.get(0))?;

        let mut notes = Vec::new();
        for row in rows {
            notes.push(row?);
        }
        Ok(notes)
    }
}

// =============================================================================
// Helpers
// =============================================================================

#[derive(Debug, Clone)]
enum ColumnAction {
    Skip,
    Known(String),
    Extra(String),
}

/// Iterator that yields batches of patients from the database
///
/// This allows streaming training without loading all data into memory.
pub struct BatchIterator {
    db_path: Option<std::path::PathBuf>,
    batch_size: usize,
    total: usize,
    current_offset: usize,
}

impl Iterator for BatchIterator {
    type Item = AppResult<Vec<PatientData>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_offset >= self.total {
            return None;
        }

        // Open a new connection for each batch (allows the iterator to be Send)
        let conn = match &self.db_path {
            Some(path) => Connection::open(path),
            None => Connection::open_in_memory(),
        };

        let conn = conn.ok()?;

        let db = DatabaseManager { conn };
        let batch = db.get_patient_batch(self.current_offset, self.batch_size);

        self.current_offset += self.batch_size;

        Some(batch)
    }
}

impl BatchIterator {
    /// Get the total number of batches
    pub fn num_batches(&self) -> usize {
        (self.total + self.batch_size - 1) / self.batch_size
    }

    /// Reset the iterator to the beginning
    pub fn reset(&mut self) {
        self.current_offset = 0;
    }
}

/// Counts of samples for each modality in the database
#[derive(Debug, Clone, Default)]
pub struct SampleCounts {
    pub sequences: usize,
    pub text: usize,
    pub image: usize,
    pub outcomes: usize,
}

fn normalize_column_name(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '-', '.'], "_")
}

fn find_column(headers: &[String], configured_name: &str) -> Option<usize> {
    let configured = normalize_column_name(configured_name);
    headers
        .iter()
        .position(|header| normalize_column_name(header) == configured)
}

#[derive(Debug, Clone)]
struct LongFormEventColumns {
    timestamp_column: usize,
    feature_column: usize,
    value_column: Option<usize>,
    table_column: Option<usize>,
    visit_column: Option<usize>,
    presence_value: f32,
}

fn build_long_form_event_columns(
    headers: &[String],
    mapping: Option<&ColumnMapping>,
    timestamp_column: Option<usize>,
) -> AppResult<Option<LongFormEventColumns>> {
    let Some(mapping) = mapping else {
        return Ok(None);
    };

    let Some(feature_name) = mapping.event_feature_column.as_deref() else {
        return Ok(None);
    };

    let timestamp_column = timestamp_column.ok_or_else(|| {
        AppError::InvalidData(
            "long-form event mapping requires a timestamp_column".to_string(),
        )
    })?;

    let feature_column = find_column(headers, feature_name).ok_or_else(|| {
        AppError::InvalidData(format!(
            "long-form event feature column does not exist: {}",
            feature_name
        ))
    })?;

    let resolve_optional = |name: Option<&String>, label: &str| -> AppResult<Option<usize>> {
        name.map(|value| {
            find_column(headers, value).ok_or_else(|| {
                AppError::InvalidData(format!(
                    "long-form event {} column does not exist: {}",
                    label, value
                ))
            })
        })
        .transpose()
    };

    Ok(Some(LongFormEventColumns {
        timestamp_column,
        feature_column,
        value_column: resolve_optional(mapping.event_value_column.as_ref(), "value")?,
        table_column: resolve_optional(mapping.event_table_column.as_ref(), "table")?,
        visit_column: resolve_optional(mapping.event_visit_column.as_ref(), "visit")?,
        presence_value: if mapping.event_presence_value.is_finite() {
            mapping.event_presence_value
        } else {
            return Err(AppError::InvalidData(
                "long-form event presence value must be finite".to_string(),
            ));
        },
    }))
}

fn find_patient_id_column(
    headers: &[String],
    configured_name: &str,
) -> Option<usize> {
    let configured =
        normalize_column_name(configured_name);

    headers.iter().position(|header| {
        let normalized =
            normalize_column_name(header);

        normalized == configured
            || normalized == "patient_id"
            || normalized == "patientid"
            || normalized == "subject_id"
            || normalized == "subjectid"
    })
}

fn resolve_patient_id(
    record: &csv::StringRecord,
    id_column_index: Option<usize>,
    generated_patient_id: &mut usize,
) -> String {
    if let Some(index) = id_column_index {
        let value = record
            .get(index)
            .unwrap_or("")
            .trim();

        if !value.is_empty() {
            return value.to_string();
        }
    }

    *generated_patient_id += 1;
    format!("P{:06}", generated_patient_id)
}

fn known_column_mappings() -> HashMap<String, &'static str> {
    [
        ("age", "age"),
        ("anchor_age", "age"),
        ("race", "race"),
        ("sex", "sex"),
        ("gender", "sex"),
        ("insurance", "insurance"),
        ("systolic_bp", "systolic_bp"),
        ("systolicbp", "systolic_bp"),
        ("diastolic_bp", "diastolic_bp"),
        ("diastolicbp", "diastolic_bp"),
        ("glucose", "glucose"),
        ("blood_sugar", "glucose"),
        ("bloodsugar", "glucose"),
        ("bs", "glucose"),
        ("bmi", "bmi"),
        ("body_mass_index", "bmi"),
        ("bodymassindex", "bmi"),
        ("pregnancies", "pregnancies"),
        ("gravida", "pregnancies"),
        ("risk_level", "risk_level"),
        ("risklevel", "risk_level"),
        ("outcome", "existing_outcome"),
        ("mortality", "existing_outcome"),
        ("hospital_expire_flag", "existing_outcome"),
        ("mortality_label", "existing_outcome"),
        ("cause_code", "cause_code"),
        ("causecode", "cause_code"),
        ("cause", "cause_code"),
    ]
    .into_iter()
    .map(|(name, field)| (name.to_string(), field))
    .collect()
}

fn feature_blacklist() -> HashSet<String> {
    [
        "hadm_id",
        "anchor_year",
        "anchor_year_group",
        "dod",
        "subject_id",
        "subjectid",
        "row_id",
        "charttime",
        "storetime",
        "admittime",
        "dischtime",
        "intime",
        "outtime",
        "deathtime",
        "edregtime",
        "edouttime",
        "hospital_expire_flag",
        "expire_flag",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn build_column_actions(
    headers: &[String],
    patient_id_index: Option<usize>,
    outcome_index: Option<usize>,
    note_index: Option<usize>,
    timestamp_index: Option<usize>,
    known_columns: &HashMap<String, &'static str>,
    feature_blacklist: &HashSet<String>,
    mapping: Option<&ColumnMapping>,
    ignored_columns: &HashSet<usize>,
) -> Vec<ColumnAction> {
    headers
        .iter()
        .enumerate()
        .map(|(index, header)| {
            if Some(index) == patient_id_index
                || Some(index) == note_index
                || Some(index) == timestamp_index
                || ignored_columns.contains(&index)
            {
                return ColumnAction::Skip;
            }

            let normalized = normalize_column_name(header);

            if Some(index) == outcome_index {
                return ColumnAction::Known("existing_outcome".to_string());
            }

            if let Some(alias) = mapping.and_then(|value| {
                value.feature_aliases.iter().find_map(|(source, target)| {
                    (normalize_column_name(source) == normalized).then(|| target.clone())
                })
            }) {
                let alias = normalize_column_name(&alias);
                return if known_columns.values().any(|field| *field == alias)
                    || alias == "existing_outcome"
                {
                    ColumnAction::Known(alias)
                } else {
                    ColumnAction::Extra(alias)
                };
            }

            if let Some(field) = known_columns.get(&normalized) {
                ColumnAction::Known((*field).to_string())
            } else if feature_blacklist.contains(&normalized) {
                ColumnAction::Skip
            } else {
                ColumnAction::Extra(header.clone())
            }
        })
        .collect()
}

fn import_long_form_event(
    record: &csv::StringRecord,
    columns: &LongFormEventColumns,
    patient: &mut PatientData,
) {
    let Some(feature) = record
        .get(columns.feature_column)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return;
    };

    let table = columns
        .table_column
        .and_then(|index| record.get(index))
        .map(str::trim)
        .filter(|value| !value.is_empty());

    let feature_name = table
        .map(|table| format!("{table}:{feature}"))
        .unwrap_or_else(|| feature.to_string());

    let value = columns
        .value_column
        .and_then(|index| record.get(index))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(columns.presence_value);

    let timestamp = record
        .get(columns.timestamp_column)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("unknown");

    let timestamp = columns
        .visit_column
        .and_then(|index| record.get(index))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|visit| format!("{timestamp}#visit:{visit}"))
        .unwrap_or_else(|| timestamp.to_string());

    patient.observations.push(PatientObservation {
        timestamp,
        feature_name,
        value,
    });
}

fn import_observations(
    record: &csv::StringRecord,
    timestamp_column_index: Option<usize>,
    measurement_columns: &[(usize, String)],
    patient: &mut PatientData,
) {
    let timestamp = timestamp_column_index
        .and_then(|index| record.get(index))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("unknown")
        .to_string();

    for (column_index, feature_name) in measurement_columns {
        let Some(raw_value) =
            record.get(*column_index)
        else {
            continue;
        };

        let Ok(value) =
            raw_value.trim().parse::<f32>()
        else {
            continue;
        };

        if !value.is_finite() {
            continue;
        }

        patient.observations.push(
            PatientObservation {
                timestamp: timestamp.clone(),
                feature_name: feature_name.clone(),
                value,
            },
        );
    }
}

fn import_note(
    record: &csv::StringRecord,
    note_column_index: Option<usize>,
    note_type_column_index: Option<usize>,
    patient: &mut PatientData,
) {
    let Some(note_index) = note_column_index else {
        return;
    };

    let Some(content) =
        record.get(note_index)
    else {
        return;
    };

    let content = content.trim();

    if content.is_empty() {
        return;
    }

    let note = match note_type_column_index
        .and_then(|index| record.get(index))
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(note_type) => {
            format!("[{}] {}", note_type, content)
        }
        None => content.to_string(),
    };

    patient.clinical_notes.push(note);
}

fn import_static_fields(
    record: &csv::StringRecord,
    actions: &[ColumnAction],
    patient: &mut PatientData,
) {
    for (index, value) in record.iter().enumerate() {
        let value = value.trim();

        if value.is_empty() {
            continue;
        }

        let Some(action) = actions.get(index) else {
            continue;
        };

        match action {
            ColumnAction::Skip => {}

            ColumnAction::Known(field_name) => {
                assign_known_field(patient, field_name, value);
            }

            ColumnAction::Extra(name) => {
                if let Ok(value) =
                    value.parse::<f32>()
                {
                    if value.is_finite() {
                        patient
                            .extra_features
                            .insert(name.clone(), value);
                    }
                }
            }
        }
    }
}

fn assign_known_field(
    patient: &mut PatientData,
    field_name: &str,
    value: &str,
) {
    match field_name {
    	"sex" | "gender" => {
     		patient.sex = Some(value.to_string());
     	}

        "age" => {
            patient.age = value.parse::<f32>().ok();
        }

        "race" => {
            patient.race = Some(value.to_string());
        }

        "insurance" => {
            patient.insurance = Some(value.to_string());
        }

        "systolic_bp" => {
            patient.systolic_bp =
                value.parse::<f32>().ok();
        }

        "diastolic_bp" => {
            patient.diastolic_bp =
                value.parse::<f32>().ok();
        }

        "glucose" => {
            patient.glucose =
                value.parse::<f32>().ok();
        }

        "bmi" => {
            patient.bmi =
                value.parse::<f32>().ok();
        }

        "pregnancies" => {
            patient.pregnancies =
                value.parse::<u32>().ok();
        }

        "risk_level" => {
            patient.risk_level =
                Some(value.to_string());
        }

        "existing_outcome" => {
            patient.existing_outcome =
                value.parse::<f32>().ok();
        }

        "cause_code" => {
            patient.cause_code =
                value.parse::<u32>().ok();
        }

        _ => {}
    }
}

/// Serialize a Vec<f32> to bytes
fn serialize_f32_vec(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &val in data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

fn deserialize_f32_vec(bytes: &[u8]) -> AppResult<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(AppError::InvalidData(
            "Invalid f32 vector blob length".to_string(),
        ));
    }

    let mut values = Vec::with_capacity(bytes.len() / 4);

    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes(chunk.try_into().map_err(|_| {
            AppError::InvalidData("Invalid f32 vector value".to_string())
        })?));
    }

    Ok(values)
}

/// Serialize a Vec<usize> to bytes
fn serialize_usize_vec(data: &[usize]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for &val in data {
        bytes.extend_from_slice(&(val as u64).to_le_bytes());
    }
    bytes
}

/// Deserialize bytes to Vec<usize>
fn deserialize_usize_vec(bytes: &[u8]) -> AppResult<Vec<usize>> {
    if bytes.len() % 8 != 0 {
        return Err(AppError::InvalidData(
            "Invalid usize vector blob length".to_string(),
        ));
    }

    let mut values = Vec::with_capacity(bytes.len() / 8);

    for chunk in bytes.chunks_exact(8) {
        values.push(u64::from_le_bytes(
            chunk
                .try_into()
                .map_err(|_| AppError::InvalidData("Invalid usize vector value".to_string()))?,
        ) as usize);
    }

    Ok(values)
}

/// Serialize an Array3<f32> to bytes
fn serialize_array3(array: &Array3<f32>) -> Vec<u8> {
    let data: Vec<f32> = array.iter().cloned().collect();
    serialize_f32_vec(&data)
}

/// Deserialize bytes to Array3<f32>
fn deserialize_array3(
    bytes: &[u8],
    channels: usize,
    height: usize,
    width: usize,
) -> AppResult<Array3<f32>> {
    let data = deserialize_f32_vec(bytes)?;
    Array3::from_shape_vec((channels, height, width), data)
        .map_err(|_| AppError::InvalidData("Invalid Array3<f32> shape".to_string()))
}

fn serialize_dynamic_sequence(sequence: &[Vec<f32>]) -> Vec<u8> {
    let mut bytes = Vec::new();

    let sequence_length = sequence.len() as u64;
    bytes.extend_from_slice(&sequence_length.to_le_bytes());

    let feature_count = sequence.first().map(|timestep| timestep.len()).unwrap_or(0) as u64;

    bytes.extend_from_slice(&feature_count.to_le_bytes());

    for timestep in sequence {
        for &value in timestep {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    bytes
}

fn deserialize_dynamic_sequence(bytes: &[u8]) -> AppResult<Vec<Vec<f32>>> {
    if bytes.len() < 16 {
        return Err(AppError::InvalidData(
            "Dynamic sequence blob is too short".to_string(),
        ));
    }

    let sequence_length = u64::from_le_bytes(
        bytes[0..8]
            .try_into()
            .map_err(|_| AppError::InvalidData("Invalid sequence length header".to_string()))?,
    ) as usize;

    let feature_count = u64::from_le_bytes(
        bytes[8..16]
            .try_into()
            .map_err(|_| AppError::InvalidData("Invalid feature count header".to_string()))?,
    ) as usize;

    let expected_values = sequence_length
        .checked_mul(feature_count)
        .ok_or_else(|| AppError::InvalidData("Dynamic sequence dimensions overflow".to_string()))?;

    let expected_bytes = 16usize
        .checked_add(expected_values.checked_mul(4).ok_or_else(|| {
            AppError::InvalidData("Dynamic sequence byte size overflow".to_string())
        })?)
        .ok_or_else(|| AppError::InvalidData("Dynamic sequence byte size overflow".to_string()))?;

    if bytes.len() != expected_bytes {
        return Err(AppError::InvalidData(format!(
            "Invalid dynamic sequence blob size: expected {}, got {}",
            expected_bytes,
            bytes.len()
        )));
    }

    let mut sequence = Vec::with_capacity(sequence_length);
    let mut offset = 16;

    for _ in 0..sequence_length {
        let mut timestep = Vec::with_capacity(feature_count);

        for _ in 0..feature_count {
            let value =
                f32::from_le_bytes(bytes[offset..offset + 4].try_into().map_err(|_| {
                    AppError::InvalidData("Invalid dynamic feature value".to_string())
                })?);

            timestep.push(value);
            offset += 4;
        }

        sequence.push(timestep);
    }

    Ok(sequence)
}
