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
    DataPreparer, FeatureStats, GenericRecord, ImageData, NormalizationMethod, Normalizer,
    PatientData, TabularData, TemporalData, TemporalMeasurement, TextData, normalize_features,
};
use crate::utils::error::{AppError, AppResult};
use csv::ReaderBuilder;
use ndarray::Array3;
use rand::Rng;
use rusqlite::{Connection, Error as SQLError, params};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Database manager
pub struct DatabaseManager {
    conn: Connection,
}

impl DatabaseManager {
    /// Current schema version - increment when making schema changes
    const SCHEMA_VERSION: i32 = 1;

    /// Open a connection to the SQLite database, creating it if it doesn't exist
    pub fn new<P: AsRef<Path>>(path: P) -> AppResult<Self> {
        let conn = Connection::open(path)?;

        // Enable foreign keys and WAL mode for better performance/safety
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;",
        )?;

        let mut db = Self { conn };
        db.init_schema()?;
        db.run_migrations()?;
        Ok(db)
    }

    /// Create an in-memory database (good for testing)
    pub fn in_memory() -> AppResult<Self> {
        let conn = Connection::open_in_memory()?;
        let mut db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    /// Initialize the database schema
    fn init_schema(&mut self) -> AppResult<()> {
        let tx = self.conn.transaction()?;

        // 1. Main Patient Table
        tx.execute(
            "CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    age REAL,
                    blood_pressure REAL,
                    systolic_bp REAL,
                    diastolic_bp REAL,
                    glucose REAL,
                    bmi REAL,
                    pregnancies INTEGER,
                    risk_level TEXT,
                    has_complications BOOLEAN DEFAULT 0,
                    risk_score REAL DEFAULT 0,
                    existing_outcome REAL,
                    cause_code INTEGER,
                    image_count INTEGER DEFAULT 0,
                    has_ultrasound BOOLEAN DEFAULT 0,
                    dataset_source TEXT
                )",
            [],
        )?;

        // 2. Dynamic Features (for any additional columns)
        tx.execute(
            "CREATE TABLE IF NOT EXISTS patient_features (
                    patient_id TEXT,
                    feature_name TEXT,
                    value REAL,
                    PRIMARY KEY (patient_id, feature_name),
                    FOREIGN KEY(patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
                )",
            [],
        )?;

        // 3. Temporal Measurements
        tx.execute(
            "CREATE TABLE IF NOT EXISTS temporal_measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    timestamp TEXT,
                    measurement_type TEXT,
                    value REAL,
                    FOREIGN KEY(patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
                )",
            [],
        )?;
        tx.execute(
            "CREATE INDEX IF NOT EXISTS idx_temporal_patient ON temporal_measurements(patient_id)",
            [],
        )?;

        // 4. Clinical Notes
        tx.execute(
            "CREATE TABLE IF NOT EXISTS clinical_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    note_type TEXT,
                    content TEXT,
                    FOREIGN KEY(patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
                )",
            [],
        )?;

        // 5. Normalization parameters (persist for inference)
        tx.execute(
            "CREATE TABLE IF NOT EXISTS normalization_stats (
                    feature_name TEXT PRIMARY KEY,
                    mean REAL,
                    std REAL,
                    min REAL,
                    max REAL,
                    count INTEGER
                )",
            [],
        )?;

        // 6. Tabular data storage (for prepared data)
        tx.execute(
            "CREATE TABLE IF NOT EXISTS tabular_data (
                    sample_id INTEGER PRIMARY KEY,
                    patient_id TEXT,
                    features BLOB
                )",
            [],
        )?;

        // 7. Tabular metadata
        tx.execute(
            "CREATE TABLE IF NOT EXISTS tabular_metadata (
                    id INTEGER PRIMARY KEY,
                    feature_names TEXT,
                    feature_count INTEGER
                )",
            [],
        )?;

        // 8. Temporal sequences storage
        tx.execute(
            "CREATE TABLE IF NOT EXISTS temporal_sequences (
                    sample_id INTEGER PRIMARY KEY,
                    patient_id TEXT,
                    sequence_data BLOB,
                    seq_length INTEGER,
                    feature_dim INTEGER
                )",
            [],
        )?;

        // 9. Temporal metadata
        tx.execute(
            "CREATE TABLE IF NOT EXISTS temporal_metadata (
                    id INTEGER PRIMARY KEY,
                    feature_names TEXT,
                    max_seq_length INTEGER,
                    feature_dim INTEGER
                )",
            [],
        )?;

        // 10. Text data storage
        tx.execute(
            "CREATE TABLE IF NOT EXISTS text_data (
                    sample_id INTEGER PRIMARY KEY,
                    patient_id TEXT,
                    document TEXT,
                    token_sequence BLOB
                )",
            [],
        )?;

        // 11. Text vocabulary
        tx.execute(
            "CREATE TABLE IF NOT EXISTS text_vocabulary (
                    word TEXT PRIMARY KEY,
                    idx INTEGER
                )",
            [],
        )?;

        // 12. Text metadata
        tx.execute(
            "CREATE TABLE IF NOT EXISTS text_metadata (
                    id INTEGER PRIMARY KEY,
                    vocab_size INTEGER,
                    max_seq_length INTEGER
                )",
            [],
        )?;

        // 13. Image data storage (stores serialized arrays)
        tx.execute(
            "CREATE TABLE IF NOT EXISTS image_data (
                    sample_id INTEGER PRIMARY KEY,
                    patient_id TEXT,
                    image_blob BLOB,
                    channels INTEGER,
                    height INTEGER,
                    width INTEGER
                )",
            [],
        )?;

        // 14. Image metadata
        tx.execute(
            "CREATE TABLE IF NOT EXISTS image_metadata (
                    id INTEGER PRIMARY KEY,
                    channels INTEGER,
                    height INTEGER,
                    width INTEGER,
                    channel_means TEXT,
                    channel_stds TEXT
                )",
            [],
        )?;

        // 15. Outcomes storage
        tx.execute(
            "CREATE TABLE IF NOT EXISTS outcomes (
                    sample_id INTEGER PRIMARY KEY,
                    patient_id TEXT,
                    mortality REAL,
                    cause_code INTEGER
                )",
            [],
        )?;

        tx.commit()?;
        Ok(())
    }

    /// Check and run migrations if needed
    fn run_migrations(&mut self) -> AppResult<()> {
        // Create version table if it doesn't exist
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        // Get current version
        let current_version: i32 = self
            .conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM schema_version",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        // Run migrations
        if current_version < Self::SCHEMA_VERSION {
            info!(
                "Running database migrations from version {} to {}",
                current_version,
                Self::SCHEMA_VERSION
            );

            for version in (current_version + 1)..=Self::SCHEMA_VERSION {
                self.apply_migration(version)?;
            }
        }

        Ok(())
    }

    /// Apply a specific migration version
    fn apply_migration(&mut self, version: i32) -> AppResult<()> {
        let tx = self.conn.transaction()?;

        match version {
            1 => {
                // Initial schema - no migration needed
                debug!("Migration 1: Initial schema");
            }
            2 => {
                // Add new multi-modal data tables (already in init_schema for new DBs)
                // tx.execute("ALTER TABLE patients ADD COLUMN new_field TEXT", [])?;
                debug!("Migration 2: Added multi-modal data tables");
            }
            _ => {
                warn!("Unknown migration version: {}", version);
            }
        }

        tx.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
            params![version],
        )?;
        tx.commit()?;

        Ok(())
    }

    /// Clear all data from the database (but keep schema)
    pub fn clear_all_data(&mut self) -> AppResult<()> {
        let tx = self.conn.transaction()?;

        tx.execute("DELETE FROM outcomes", [])?;
        tx.execute("DELETE FROM image_data", [])?;
        tx.execute("DELETE FROM image_metadata", [])?;
        tx.execute("DELETE FROM text_data", [])?;
        tx.execute("DELETE FROM text_vocabulary", [])?;
        tx.execute("DELETE FROM text_metadata", [])?;
        tx.execute("DELETE FROM temporal_sequences", [])?;
        tx.execute("DELETE FROM temporal_metadata", [])?;
        tx.execute("DELETE FROM tabular_data", [])?;
        tx.execute("DELETE FROM tabular_metadata", [])?;
        tx.execute("DELETE FROM normalization_stats", [])?;
        tx.execute("DELETE FROM clinical_notes", [])?;
        tx.execute("DELETE FROM temporal_measurements", [])?;
        tx.execute("DELETE FROM patient_features", [])?;
        tx.execute("DELETE FROM patients", [])?;

        tx.commit()?;
        info!("Cleared all data from database");
        Ok(())
    }

    // =========================================================================
    // Multi-Modal Data Storage
    // =========================================================================

    /// Build TabularData from the patients and patient_features tables
    pub fn build_tabular_data(&self) -> AppResult<TabularData> {
        // Get all patient IDs in consistent order
        let patient_ids: Vec<String> = self
            .conn
            .prepare("SELECT patient_id FROM patients ORDER BY patient_id")?
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        if patient_ids.is_empty() {
            return Ok(TabularData::new());
        }

        // Get all unique extra feature names
        let extra_feature_names: Vec<String> = self
            .conn
            .prepare("SELECT DISTINCT feature_name FROM patient_features ORDER BY feature_name")?
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        // Build feature names list
        let base_feature_names = vec![
            "age".to_string(),
            "systolic_bp".to_string(),
            "diastolic_bp".to_string(),
            "glucose".to_string(),
            "bmi".to_string(),
            "pregnancies".to_string(),
            "has_complications".to_string(),
            "risk_score".to_string(),
        ];
        let mut feature_names = base_feature_names;
        feature_names.extend(extra_feature_names.clone());

        // Build feature vectors for each patient
        let mut data: Vec<Vec<f32>> = Vec::with_capacity(patient_ids.len());

        for patient_id in &patient_ids {
            // Get base patient data
            let row: (Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<i32>, bool, f32) = self
                .conn
                .query_row(
                    "SELECT age, COALESCE(systolic_bp, blood_pressure), diastolic_bp, glucose, bmi, pregnancies, has_complications, risk_score
                     FROM patients WHERE patient_id = ?",
                    params![patient_id],
                    |row| Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                        row.get(7)?,
                    )),
                )?;

            let mut features = vec![
                row.0.unwrap_or(0.0),
                row.1.unwrap_or(0.0),
                row.2.unwrap_or(0.0),
                row.3.unwrap_or(0.0),
                row.4.unwrap_or(0.0),
                row.5.map(|p| p as f32).unwrap_or(0.0),
                if row.6 { 1.0 } else { 0.0 },
                row.7,
            ];

            // Get extra features
            let extra_features: HashMap<String, f32> = self
                .conn
                .prepare("SELECT feature_name, value FROM patient_features WHERE patient_id = ?")?
                .query_map(params![patient_id], |row| Ok((row.get(0)?, row.get(1)?)))?
                .filter_map(|r| r.ok())
                .collect();

            // Add extra features in consistent order
            for name in &extra_feature_names {
                features.push(*extra_features.get(name).unwrap_or(&0.0));
            }

            data.push(features);
        }

        // Normalize features
        normalize_features(&mut data);

        info!(
            "Built TabularData with {} samples, {} features",
            data.len(),
            feature_names.len()
        );

        Ok(TabularData {
            data,
            patient_ids,
            feature_names,
            numeric_stats: HashMap::new(),
            categorical_mappings: HashMap::new(),
            categorical_features: Vec::new(),
            current_idx: 0,
        })
    }

    /// Store tabular data in the database
    pub fn store_tabular_data(&mut self, data: &TabularData) -> AppResult<()> {
        // First, get all existing patient IDs from database
        let existing_patient_ids: Vec<String> = self
            .conn
            .prepare("SELECT patient_id FROM patients ORDER BY rowid")?
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        let tx = self.conn.transaction()?;

        // Clear existing tabular data
        tx.execute("DELETE FROM tabular_data", [])?;
        tx.execute("DELETE FROM tabular_metadata", [])?;

        // Store metadata
        let feature_names_json = serde_json::to_string(&data.feature_names).unwrap_or_default();
        tx.execute(
            "INSERT INTO tabular_metadata (id, feature_names, feature_count) VALUES (1, ?, ?)",
            params![feature_names_json, data.feature_dim()],
        )?;

        // Store each sample
        {
            let mut stmt = tx.prepare(
                "INSERT INTO tabular_data (sample_id, patient_id, features) VALUES (?, ?, ?)",
            )?;

            for (idx, features) in data.data.iter().enumerate() {
                let features_blob = serialize_f32_vec(features);
                // Use patient ID from TabularData, then from database, or generate one if not available
                let patient_id = data
                    .patient_ids
                    .get(idx)
                    .cloned()
                    .or_else(|| existing_patient_ids.get(idx).cloned())
                    .unwrap_or_else(|| format!("P{:02}", idx));
                stmt.execute(params![idx as i64, patient_id, features_blob])?;
            }
        }

        tx.commit()?;
        info!("Stored {} tabular samples in database", data.data.len());
        Ok(())
    }

    /// Load tabular data from the database
    pub fn load_tabular_data(&self) -> AppResult<TabularData> {
        // Load metadata
        let (feature_names, feature_count): (String, usize) = self.conn.query_row(
            "SELECT feature_names, feature_count FROM tabular_metadata WHERE id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;

        let feature_names: Vec<String> = serde_json::from_str(&feature_names).unwrap_or_default();

        // Load samples
        let mut stmt = self
            .conn
            .prepare("SELECT features, patient_id FROM tabular_data ORDER BY sample_id")?;

        let data: Vec<Vec<f32>> = stmt
            .query_map([], |row| {
                let blob: Vec<u8> = row.get(0)?;
                Ok(deserialize_f32_vec(&blob))
            })?
            .filter_map(|r| r.ok())
            .collect();

        let patient_ids: Vec<String> = stmt
            .query_map([], |row| {
                let patient_id: String = row.get(1)?;
                Ok(patient_id)
            })?
            .filter_map(|r| r.ok())
            .collect();

        info!("Loaded {} tabular samples from database", data.len());

        Ok(TabularData {
            data,
            patient_ids,
            feature_names,
            numeric_stats: HashMap::new(),
            categorical_mappings: HashMap::new(),
            categorical_features: Vec::new(),
            current_idx: 0,
        })
    }

    /// Build TemporalData from the temporal_measurements table
    /// Groups measurements by patient, sorted by timestamp
    pub fn build_temporal_data(&self) -> AppResult<TemporalData> {
        // Get all patient IDs that have temporal measurements
        let patient_ids: Vec<String> = self
            .conn
            .prepare("SELECT DISTINCT patient_id FROM temporal_measurements ORDER BY patient_id")?
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        if patient_ids.is_empty() {
            info!("No temporal measurements found in database");
            return Ok(TemporalData::new());
        }

        // Get all unique measurement types (these become our features)
        let measurement_types: Vec<String> = self
            .conn
            .prepare(
                "SELECT DISTINCT measurement_type FROM temporal_measurements ORDER BY measurement_type",
            )?
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        let num_features = measurement_types.len();
        let type_to_idx: HashMap<String, usize> = measurement_types
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();

        let mut sequences: Vec<Vec<Vec<f32>>> = Vec::with_capacity(patient_ids.len());
        let mut max_seq_length: usize = 0;

        for patient_id in &patient_ids {
            // Get all measurements for this patient, grouped by timestamp
            let measurements: Vec<(String, String, f32)> = self
                .conn
                .prepare(
                    "SELECT timestamp, measurement_type, value
                     FROM temporal_measurements
                     WHERE patient_id = ?
                     ORDER BY timestamp",
                )?
                .query_map(params![patient_id], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })?
                .filter_map(|r| r.ok())
                .collect();

            // Group by timestamp to create time steps
            let mut timestamp_groups: Vec<(String, Vec<(String, f32)>)> = Vec::new();
            for (ts, mtype, value) in measurements {
                if let Some(last) = timestamp_groups.last_mut() {
                    if last.0 == ts {
                        last.1.push((mtype, value));
                        continue;
                    }
                }
                timestamp_groups.push((ts, vec![(mtype, value)]));
            }

            // Convert to feature vectors
            let mut patient_sequence: Vec<Vec<f32>> = Vec::new();
            for (_ts, group) in timestamp_groups {
                let mut feature_vec = vec![0.0; num_features];
                for (mtype, value) in group {
                    if let Some(&idx) = type_to_idx.get(&mtype) {
                        feature_vec[idx] = value;
                    }
                }
                patient_sequence.push(feature_vec);
            }

            max_seq_length = max_seq_length.max(patient_sequence.len());
            sequences.push(patient_sequence);
        }

        info!(
            "Built TemporalData with {} sequences, {} features, max length {}",
            sequences.len(),
            num_features,
            max_seq_length
        );

        Ok(TemporalData {
            sequences,
            feature_names: measurement_types,
            feature_stats: HashMap::new(),
            max_seq_length,
            current_idx: 0,
        })
    }

    /// Store temporal data in the database
    pub fn store_temporal_data(&mut self, data: &TemporalData) -> AppResult<()> {
        let tx = self.conn.transaction()?;

        // Clear existing temporal data
        tx.execute("DELETE FROM temporal_sequences", [])?;
        tx.execute("DELETE FROM temporal_metadata", [])?;

        // Store metadata
        let feature_names_json = serde_json::to_string(&data.feature_names).unwrap_or_default();
        tx.execute(
                "INSERT INTO temporal_metadata (id, feature_names, max_seq_length, feature_dim) VALUES (1, ?, ?, ?)",
                params![feature_names_json, data.max_seq_length, data.feature_dim()],
            )?;

        // Store each sequence
        {
            let mut stmt = tx.prepare(
                    "INSERT INTO temporal_sequences (sample_id, patient_id, sequence_data, seq_length, feature_dim) VALUES (?, ?, ?, ?, ?)",
                )?;

            for (idx, sequence) in data.sequences.iter().enumerate() {
                let sequence_blob = serialize_temporal_sequence(sequence);
                let patient_id = format!("P{:02}", idx);
                let seq_length = sequence.len();
                let feature_dim = if sequence.is_empty() {
                    0
                } else {
                    sequence[0].len()
                };
                stmt.execute(params![
                    idx as i64,
                    patient_id,
                    sequence_blob,
                    seq_length,
                    feature_dim
                ])?;
            }
        }

        tx.commit()?;
        info!(
            "Stored {} temporal sequences in database",
            data.sequences.len()
        );
        Ok(())
    }

    /// Load temporal data from the database
    pub fn load_temporal_data(&self) -> AppResult<TemporalData> {
        // Load metadata
        let (feature_names_json, max_seq_length, feature_dim): (String, usize, usize) = self
            .conn
            .query_row(
            "SELECT feature_names, max_seq_length, feature_dim FROM temporal_metadata WHERE id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;

        let feature_names: Vec<String> =
            serde_json::from_str(&feature_names_json).unwrap_or_default();

        // Load sequences
        let mut stmt = self.conn.prepare(
                "SELECT sequence_data, seq_length, feature_dim FROM temporal_sequences ORDER BY sample_id",
            )?;

        let sequences: Vec<Vec<Vec<f32>>> = stmt
            .query_map([], |row| {
                let blob: Vec<u8> = row.get(0)?;
                let seq_length: usize = row.get(1)?;
                let feature_dim: usize = row.get(2)?;
                Ok(deserialize_temporal_sequence(
                    &blob,
                    seq_length,
                    feature_dim,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();

        info!(
            "Loaded {} temporal sequences from database",
            sequences.len()
        );

        Ok(TemporalData {
            sequences,
            feature_names,
            feature_stats: HashMap::new(),
            max_seq_length,
            current_idx: 0,
        })
    }

    /// Build TextData from the clinical_notes table
    /// Tokenizes notes and builds vocabulary
    pub fn build_text_data(&self, max_vocab_size: usize) -> AppResult<TextData> {
        // Get all clinical notes
        let notes: Vec<(String, String)> = self
            .conn
            .prepare("SELECT patient_id, content FROM clinical_notes ORDER BY patient_id, id")?
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
                    "INSERT INTO text_data (sample_id, patient_id, document, token_sequence) VALUES (?, ?, ?, ?)",
                )?;

            for (idx, (doc, tokens)) in data
                .documents
                .iter()
                .zip(data.token_sequences.iter())
                .enumerate()
            {
                let tokens_blob = serialize_usize_vec(tokens);
                let patient_id = format!("P{:02}", idx);
                stmt.execute(params![idx as i64, patient_id, doc, tokens_blob])?;
            }
        }

        tx.commit()?;
        info!("Stored {} text documents in database", data.documents.len());
        Ok(())
    }

    /// Load text data from the database
    pub fn load_text_data(&self) -> AppResult<TextData> {
        // Load metadata
        let (vocab_size, max_seq_length): (usize, usize) = self.conn.query_row(
            "SELECT vocab_size, max_seq_length FROM text_metadata WHERE id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;

        // Load vocabulary
        let mut stmt = self.conn.prepare("SELECT word, idx FROM text_vocabulary")?;
        let vocab: HashMap<String, usize> = stmt
            .query_map([], |row| {
                let word: String = row.get(0)?;
                let idx: i64 = row.get(1)?;
                Ok((word, idx as usize))
            })?
            .filter_map(|r| r.ok())
            .collect();

        let reverse_vocab: HashMap<usize, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        // Load documents and token sequences
        let mut stmt = self
            .conn
            .prepare("SELECT document, token_sequence FROM text_data ORDER BY sample_id")?;

        let mut documents = Vec::new();
        let mut token_sequences = Vec::new();

        let rows: Vec<(String, Vec<u8>)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();

        for (doc, tokens_blob) in rows {
            documents.push(doc);
            token_sequences.push(deserialize_usize_vec(&tokens_blob));
        }

        info!("Loaded {} text documents from database", documents.len());

        Ok(TextData {
            documents,
            token_sequences,
            vocab,
            reverse_vocab,
            max_seq_length,
            current_idx: 0,
        })
    }

    /// Build ImageData from image files referenced in the patients table
    /// Note: Images are loaded from files, not stored in the database
    pub fn build_image_data<P: AsRef<Path>>(
        &self,
        image_dir: P,
        target_size: (usize, usize),
        grayscale: bool,
    ) -> AppResult<ImageData> {
        use std::fs;

        let image_dir = image_dir.as_ref();
        if !image_dir.exists() {
            info!("Image directory not found: {:?}", image_dir);
            return Ok(ImageData::new());
        }

        // Find all image files recursively
        let image_extensions = ["jpg", "jpeg", "png", "bmp", "gif"];
        let mut image_paths: Vec<PathBuf> = Vec::new();

        fn find_images(dir: &Path, extensions: &[&str], paths: &mut Vec<PathBuf>) {
            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let path = entry.path();
                    if path.is_dir() {
                        find_images(&path, extensions, paths);
                    } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                        if extensions.iter().any(|&e| e.eq_ignore_ascii_case(ext)) {
                            paths.push(path);
                        }
                    }
                }
            }
        }

        find_images(image_dir, &image_extensions, &mut image_paths);
        image_paths.sort();

        if image_paths.is_empty() {
            info!("No image files found in {:?}", image_dir);
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
                let patient_id = format!("P{:02}", idx);
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

    /// Load image data from the database
    pub fn load_image_data(&self) -> AppResult<ImageData> {
        // Load metadata
        let (channels, height, width, channel_means_json, channel_stds_json): (usize, usize, usize, String, String) =
                self.conn.query_row(
                    "SELECT channels, height, width, channel_means, channel_stds FROM image_metadata WHERE id = 1",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
                )?;

        let channel_means: Vec<f32> = serde_json::from_str(&channel_means_json).unwrap_or_default();
        let channel_stds: Vec<f32> = serde_json::from_str(&channel_stds_json).unwrap_or_default();

        // Load images
        let mut stmt = self.conn.prepare(
            "SELECT image_blob, channels, height, width FROM image_data ORDER BY sample_id",
        )?;

        let images: Vec<Array3<f32>> = stmt
            .query_map([], |row| {
                let blob: Vec<u8> = row.get(0)?;
                let c: usize = row.get(1)?;
                let h: usize = row.get(2)?;
                let w: usize = row.get(3)?;
                Ok(deserialize_array3(&blob, c, h, w))
            })?
            .filter_map(|r| r.ok())
            .collect();

        info!("Loaded {} images from database", images.len());

        Ok(ImageData {
            images,
            image_size: (height, width),
            channels,
            channel_means,
            channel_stds,
            current_idx: 0,
        })
    }

    /// Build outcomes from the patients table
    /// Returns (targets, causes) vectors aligned with patient order
    /// Uses existing_outcome as the target, and derives cause_code from risk_level
    pub fn build_outcomes(&self) -> AppResult<(Vec<f32>, Vec<usize>)> {
        // Get all patients in consistent order (same order as build_tabular_data)
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
        let patient_ids: Vec<String> = self
            .conn
            .prepare("SELECT patient_id FROM patients ORDER BY patient_id")?
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        let tx = self.conn.transaction()?;

        // Clear existing outcomes
        tx.execute("DELETE FROM outcomes", [])?;

        // Store each outcome
        {
            let mut stmt = tx.prepare(
                    "INSERT INTO outcomes (sample_id, patient_id, mortality, cause_code) VALUES (?, ?, ?, ?)",
                )?;

            for (idx, (target, cause)) in targets.iter().zip(causes.iter()).enumerate() {
                let patient_id = patient_ids
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("P{:02}", idx));
                stmt.execute(params![idx as i64, patient_id, *target, *cause as i64])?;
            }
        }

        tx.commit()?;
        info!("Stored {} outcomes in database", targets.len());
        Ok(())
    }

    /// Load outcomes from the database
    pub fn load_outcomes(&self) -> AppResult<(Vec<f32>, Vec<usize>)> {
        let mut stmt = self
            .conn
            .prepare("SELECT mortality, cause_code FROM outcomes ORDER BY sample_id")?;

        let mut targets = Vec::new();
        let mut causes = Vec::new();

        let rows: Vec<(f32, i64)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();

        for (mortality, cause) in rows {
            targets.push(mortality);
            causes.push(cause as usize);
        }

        info!("Loaded {} outcomes from database", targets.len());
        Ok((targets, causes))
    }

    /// Get the count of samples for each modality
    pub fn get_sample_counts(&self) -> AppResult<ModalityCounts> {
        let tabular: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM tabular_data", [], |row| row.get(0))
            .unwrap_or(0);

        let temporal: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM temporal_sequences", [], |row| {
                row.get(0)
            })
            .unwrap_or(0);

        let text: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM text_data", [], |row| row.get(0))
            .unwrap_or(0);

        let image: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM image_data", [], |row| row.get(0))
            .unwrap_or(0);

        let outcomes: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM outcomes", [], |row| row.get(0))
            .unwrap_or(0);

        Ok(ModalityCounts {
            tabular: tabular as usize,
            temporal: temporal as usize,
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
    /// * `dataset_source` - Optional label for the data source (e.g., "MIMIC-IV")
    pub fn import_csv<P: AsRef<Path>>(
        &mut self,
        path: P,
        patient_id_column: &str,
        dataset_source: Option<&str>,
    ) -> AppResult<usize> {
        let file = File::open(&path).map_err(|e| AppError::Io(e))?;
        let mut preparer = DataPreparer::new();

        let mut reader = ReaderBuilder::new().flexible(true).from_reader(file);

        let headers: Vec<String> = reader.headers()?.iter().map(|h| h.to_string()).collect();

        // Find patient ID and timestamp column index
        let id_col_idx = headers.iter().position(|h| {
            let h_lower = h.to_lowercase();
            h_lower.contains("patient")
                || h_lower.contains("subject")
                // || h_lower == "id"
                || h.eq_ignore_ascii_case(patient_id_column)
        });

        let timestamp_col_idx = headers.iter().position(|h| {
            let h_lower = h.to_lowercase();
            h_lower.contains("timestamp")
                || h_lower.contains("time")
                || h_lower.contains("date")
                || h_lower.contains("datetime")
        });

        let note_col_idx = headers.iter().position(|h| {
            let h_lower = h.to_lowercase();
            h_lower.contains("note")
                || h_lower.contains("comment")
                || h_lower.contains("description")
                || h_lower.contains("text")
        });

        let note_type_col_idx = headers.iter().position(|h| {
            let h_lower = h.to_lowercase();
            h_lower.contains("type") || h_lower.contains("category") || h_lower.contains("kind")
        });

        let is_temporal = timestamp_col_idx.is_some();
        let is_text = note_col_idx.is_some();

        // Known column mappings (case-insensitive)
        let known_columns: HashMap<&str, &str> = [
            ("age", "age"),
            ("bloodpressure", "blood_pressure"),
            ("bp", "blood_pressure"),
            ("systolicbp", "systolic_bp"),
            ("systolic_bp", "systolic_bp"),
            ("diastolicbp", "diastolic_bp"),
            ("diastolic_bp", "diastolic_bp"),
            ("glucose", "glucose"),
            ("bs", "glucose"),
            ("blood_sugar", "glucose"),
            ("bmi", "bmi"),
            ("bodymassindex", "bmi"),
            ("pregnancies", "pregnancies"),
            ("gravida", "pregnancies"),
            ("risklevel", "risk_level"),
            ("risk_level", "risk_level"),
            ("outcome", "existing_outcome"),
            ("mortality", "existing_outcome"),
            ("causecode", "cause_code"),
            ("cause_code", "cause_code"),
            ("cause", "cause_code"),
        ]
        .iter()
        .cloned()
        .collect();

        // For temporal data, identify measurement columns
        let measurement_cols: Vec<(usize, String)> = if is_temporal {
            headers
                .iter()
                .enumerate()
                .filter(|(i, h)| {
                    Some(*i) != id_col_idx
                        && Some(*i) != timestamp_col_idx
                        && !known_columns.contains_key(&h.to_lowercase().replace(" ", "_").as_str())
                })
                .map(|(i, name)| (i, name.clone()))
                .collect()
        } else {
            Vec::new()
        };

        const BATCH_SIZE: usize = 10_000;
        let mut batch: Vec<PatientData> = Vec::with_capacity(BATCH_SIZE);
        let mut total_count: usize = 0;

        for result in reader.records() {
            let record = result?;

            let patient_id = match id_col_idx {
                Some(idx) => {
                    let id = record.get(idx).unwrap_or("").trim().to_string();
                    if id.is_empty() {
                        preparer.generate_patient_id()
                    } else {
                        id
                    }
                }
                None => preparer.generate_patient_id(),
            };

            // Build patient data from CSV row
            let mut patient = PatientData {
                patient_id: patient_id.clone(),
                ..Default::default()
            };

            if is_temporal {
                let timestamp = timestamp_col_idx
                    .and_then(|idx| record.get(idx))
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default();

                for (col_idx, measurement_type) in &measurement_cols {
                    if let Some(value_str) = record.get(*col_idx) {
                        if let Ok(value) = value_str.trim().parse::<f32>() {
                            patient.temporal_measurements.push(TemporalMeasurement {
                                timestamp: timestamp.clone(),
                                measurement_type: measurement_type.clone(),
                                value,
                            });
                        }
                    }
                }
            } else if is_text {
                if let Some(text_idx) = note_col_idx {
                    if let Some(content) = record.get(text_idx) {
                        let content = content.trim();
                        if !content.is_empty() {
                            // Optionally prepend note type
                            let note = if let Some(type_idx) = note_type_col_idx {
                                let note_type = record.get(type_idx).unwrap_or("").trim();
                                if !note_type.is_empty() {
                                    format!("[{}] {}", note_type, content)
                                } else {
                                    content.to_string()
                                }
                            } else {
                                content.to_string()
                            };
                            patient.clinical_notes.push(note);
                        }
                    }
                }
            } else {
                for (i, value) in record.iter().enumerate() {
                    if Some(i) == id_col_idx || value.trim().is_empty() {
                        continue;
                    }

                    let col_name = &headers[i];
                    let col_lower = col_name.to_lowercase().replace(" ", "_");

                    if let Some(&field_name) = known_columns.get(col_lower.as_str()) {
                        // Parse and assign to known field
                        if let Ok(v) = value.trim().parse::<f32>() {
                            match field_name {
                                "age" => patient.age = Some(v),
                                "blood_pressure" => patient.blood_pressure = Some(v),
                                "systolic_bp" => patient.systolic_bp = Some(v),
                                "diastolic_bp" => patient.diastolic_bp = Some(v),
                                "glucose" => patient.glucose = Some(v),
                                "bmi" => patient.bmi = Some(v),
                                "pregnancies" => patient.pregnancies = Some(v as u32),
                                "existing_outcome" => patient.existing_outcome = Some(v),
                                "cause_code" => patient.cause_code = Some(v as u32),
                                _ => {}
                            }
                        } else if field_name == "risk_level" {
                            patient.risk_level = Some(value.trim().to_string());
                        }
                    } else {
                        // Try to parse as numeric and store as extra feature
                        if let Ok(v) = value.trim().parse::<f32>() {
                            patient.extra_features.insert(col_name.clone(), v);
                        }
                    }
                }
            }
            // Map known columns and store unknown numeric columns as extra_features

            batch.push(patient);

            // Flush batch when full
            if batch.len() >= BATCH_SIZE {
                self.insert_patient_batch(&batch, dataset_source)?;
                total_count += batch.len();
                batch.clear();

                if total_count % 100_000 == 0 {
                    info!("  Imported {} records so far...", total_count);
                }
            }
        }

        // Insert remaining records
        if !batch.is_empty() {
            self.insert_patient_batch(&batch, dataset_source)?;
            total_count += batch.len();
        }

        info!("Imported {} records from CSV", total_count);
        Ok(total_count)
    }

    /// Insert a batch of patients in a single transaction
    fn insert_patient_batch(
        &mut self,
        patients: &[PatientData],
        dataset_source: Option<&str>,
    ) -> AppResult<()> {
        for patient in patients {
            self.upsert_patient(patient)?;
        }
        Ok(())
    }

    /// Import multiple CSV files from a directory
    pub fn import_csv_directory<P: AsRef<Path>>(
        &mut self,
        dir: P,
        patient_id_column: &str,
        dataset_source: Option<&str>,
    ) -> AppResult<usize> {
        let mut total = 0;

        let entries = std::fs::read_dir(&dir).map_err(|e| {
            rusqlite::Error::InvalidParameterName(format!("Failed to read directory: {}", e))
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                rusqlite::Error::InvalidParameterName(format!("Directory entry error: {}", e))
            })?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("csv") {
                info!("Importing: {}", path.display());
                total += self.import_csv(&path, patient_id_column, dataset_source)?;
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
                patient_id, age, blood_pressure, systolic_bp, diastolic_bp,
                glucose, bmi, pregnancies, risk_level, has_complications,
                risk_score, existing_outcome, cause_code, image_count, has_ultrasound, dataset_source
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)
            ON CONFLICT(patient_id) DO UPDATE SET
                age = coalesce(excluded.age, age),
                blood_pressure = coalesce(excluded.blood_pressure, blood_pressure),
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
                patient.blood_pressure,
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
            "INSERT OR REPLACE INTO patient_features (patient_id, feature_name, value)
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
                "INSERT INTO temporal_measurements (patient_id, timestamp, measurement_type, value)
                 VALUES (?, ?, ?, ?)",
            )?;
            for m in &patient.temporal_measurements {
                stmt.execute(params![
                    patient.patient_id,
                    m.timestamp,
                    m.measurement_type,
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
            "SELECT age, blood_pressure, systolic_bp, diastolic_bp, glucose, bmi,
                    pregnancies, risk_level, has_complications, risk_score,
                    existing_outcome, image_count, has_ultrasound
             FROM patients WHERE patient_id = ?1",
        )?;

        let mut rows = stmt.query(params![patient_id])?;

        if let Some(row) = rows.next()? {
            let mut patient = PatientData {
                patient_id: patient_id.to_string(),
                age: row.get(0)?,
                blood_pressure: row.get(1)?,
                systolic_bp: row.get(2)?,
                diastolic_bp: row.get(3)?,
                glucose: row.get(4)?,
                bmi: row.get(5)?,
                pregnancies: row.get(6)?,
                risk_level: row.get(7)?,
                has_complications: row.get(8)?,
                risk_score: row.get(9)?,
                existing_outcome: row.get(10)?,
                image_count: row.get::<_, i64>(11)? as usize,
                has_ultrasound: row.get(12)?,
                ..Default::default()
            };

            // Load related data
            patient.extra_features = self.get_patient_features(patient_id)?;
            patient.temporal_measurements = self.get_temporal_measurements(patient_id)?;
            patient.clinical_notes = self.get_clinical_notes(patient_id)?;

            Ok(Some(patient))
        } else {
            Ok(None)
        }
    }

    /// Retrieve all patient IDs
    pub fn get_all_patient_ids(&self) -> AppResult<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT patient_id FROM patients")?;
        let rows = stmt.query_map([], |row| row.get(0))?;

        let mut ids = Vec::new();
        for id in rows {
            ids.push(id?);
        }
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
            "blood_pressure",
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
             FROM patient_features WHERE feature_name = ?",
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
            .prepare("SELECT DISTINCT feature_name FROM patient_features")?;
        let rows = stmt.query_map([], |row| row.get(0))?;

        let mut names = Vec::new();
        for name in rows {
            names.push(name?);
        }
        Ok(names)
    }

    /// Save a normalizer to the database
    pub fn save_normalizer(&mut self, normalizer: &Normalizer) -> AppResult<()> {
        let tx = self.conn.transaction()?;

        // Clear existing
        tx.execute("DELETE FROM normalization_stats", [])?;

        {
            // Insert new
            let mut stmt = tx.prepare(
                "INSERT INTO normalization_stats (feature_name, mean, std, min, max, count)
	             VALUES (?, ?, ?, ?, ?, ?)",
            )?;

            for (name, stats) in &normalizer.feature_stats {
                stmt.execute(params![
                    name,
                    stats.mean,
                    stats.std,
                    stats.min,
                    stats.max,
                    stats.count
                ])?;
            }
        }

        tx.commit()?;
        info!(
            "Saved normalizer with {} features to database",
            normalizer.feature_stats.len()
        );
        Ok(())
    }

    /// Load a normalizer from the database
    pub fn load_normalizer(&self) -> AppResult<Option<Normalizer>> {
        let mut stmt = self
            .conn
            .prepare("SELECT feature_name, mean, std, min, max, count FROM normalization_stats")?;

        let rows: Vec<FeatureStats> = stmt
            .query_map([], |row| {
                Ok(FeatureStats {
                    name: row.get(0)?,
                    mean: row.get(1)?,
                    std: row.get(2)?,
                    min: row.get(3)?,
                    max: row.get(4)?,
                    count: row.get(5)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        if rows.is_empty() {
            return Ok(None);
        }

        let mut feature_stats = HashMap::new();
        let mut feature_order = Vec::new();

        for stats in rows {
            feature_order.push(stats.name.clone());
            feature_stats.insert(stats.name.clone(), stats);
        }

        Ok(Some(Normalizer {
            feature_stats,
            feature_order,
            categorical_columns: HashSet::new(),
            exclude_columns: HashSet::new(),
            method: NormalizationMethod::ZScore,
            missing_value: 0.0,
        }))
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    fn get_patient_features(&self, patient_id: &str) -> AppResult<HashMap<String, f32>> {
        let mut stmt = self
            .conn
            .prepare("SELECT feature_name, value FROM patient_features WHERE patient_id = ?")?;
        let rows = stmt.query_map(params![patient_id], |row| Ok((row.get(0)?, row.get(1)?)))?;

        let mut features = HashMap::new();
        for row in rows {
            let (name, value) = row?;
            features.insert(name, value);
        }
        Ok(features)
    }

    fn get_temporal_measurements(&self, patient_id: &str) -> AppResult<Vec<TemporalMeasurement>> {
        let mut stmt = self.conn.prepare(
            "SELECT timestamp, measurement_type, value FROM temporal_measurements WHERE patient_id = ?",
        )?;
        let rows = stmt.query_map(params![patient_id], |row| {
            Ok(TemporalMeasurement {
                timestamp: row.get(0)?,
                measurement_type: row.get(1)?,
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
// Helper Types
// =============================================================================

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
pub struct ModalityCounts {
    pub tabular: usize,
    pub temporal: usize,
    pub text: usize,
    pub image: usize,
    pub outcomes: usize,
}

// =============================================================================
// Serialization Helpers
// =============================================================================

/// Serialize a Vec<f32> to bytes
fn serialize_f32_vec(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &val in data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Deserialize bytes to Vec<f32>
fn deserialize_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
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
fn deserialize_usize_vec(bytes: &[u8]) -> Vec<usize> {
    bytes
        .chunks_exact(8)
        .map(|chunk| {
            u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as usize
        })
        .collect()
}

/// Serialize a temporal sequence (Vec<Vec<f32>>) to bytes
fn serialize_temporal_sequence(sequence: &[Vec<f32>]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for timestep in sequence {
        bytes.extend(serialize_f32_vec(timestep));
    }
    bytes
}

/// Deserialize bytes to a temporal sequence
fn deserialize_temporal_sequence(
    bytes: &[u8],
    seq_length: usize,
    feature_dim: usize,
) -> Vec<Vec<f32>> {
    if feature_dim == 0 || seq_length == 0 {
        return Vec::new();
    }

    let step_size = feature_dim * 4; // 4 bytes per f32
    bytes
        .chunks(step_size)
        .take(seq_length)
        .map(|chunk| deserialize_f32_vec(chunk))
        .collect()
}

/// Serialize an Array3<f32> to bytes
fn serialize_array3(array: &Array3<f32>) -> Vec<u8> {
    let data: Vec<f32> = array.iter().cloned().collect();
    serialize_f32_vec(&data)
}

/// Deserialize bytes to Array3<f32>
fn deserialize_array3(bytes: &[u8], channels: usize, height: usize, width: usize) -> Array3<f32> {
    let data = deserialize_f32_vec(bytes);
    Array3::from_shape_vec((channels, height, width), data)
        .unwrap_or_else(|_| Array3::zeros((channels, height, width)))
}
