//! Data loading utilities for all modalities

use csv::{ReaderBuilder, Writer};
use ndarray::Array3;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

use super::{ImageData, TabularData, TemporalData, TextData};
use crate::data::db::DatabaseManager;
use crate::utils::error::{AppError, AppResult};

// ============================================================================
// Normalization Types
// ============================================================================

/// Statistics for normalizing a single numeric feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    pub name: String,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub count: usize,
}

impl Default for FeatureStats {
    fn default() -> Self {
        Self {
            name: String::new(),
            mean: 0.0,
            std: 1.0,
            min: 0.0,
            max: 1.0,
            count: 0,
        }
    }
}

/// Normalization method to use
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq)]
pub enum NormalizationMethod {
    #[default]
    ZScore,
    MinMax,
    None,
}

/// Dynamic normalizer that works with arbitrary features
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Normalizer {
    pub feature_stats: HashMap<String, FeatureStats>,
    pub feature_order: Vec<String>,
    pub categorical_columns: HashSet<String>,
    pub exclude_columns: HashSet<String>,
    pub method: NormalizationMethod,
    pub missing_value: f32,
}

impl Normalizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn summary(&self) -> String {
        format!(
            "Normalizer: {} features, method={:?}",
            self.feature_order.len(),
            self.method
        )
    }
}

// ============================================================================
// Patient Data Types
// ============================================================================

/// A generic record from any CSV file
#[derive(Debug, Clone, Default)]
pub struct GenericRecord {
    pub values: HashMap<String, String>,
    pub headers: Vec<String>,
}

/// A temporal measurement record
#[derive(Debug, Clone)]
pub struct TemporalMeasurement {
    pub timestamp: String,
    pub measurement_type: String,
    pub value: f32,
}

/// Patient data aggregated from all sources
#[derive(Debug, Default, Clone)]
pub struct PatientData {
    pub patient_id: String,
    pub age: Option<f32>,
    pub blood_pressure: Option<f32>,
    pub systolic_bp: Option<f32>,
    pub diastolic_bp: Option<f32>,
    pub glucose: Option<f32>,
    pub bmi: Option<f32>,
    pub pregnancies: Option<u32>,
    pub risk_level: Option<String>,
    pub existing_outcome: Option<f32>,
    pub cause_code: Option<u32>,
    pub extra_features: HashMap<String, f32>,
    pub temporal_measurements: Vec<TemporalMeasurement>,
    pub clinical_notes: Vec<String>,
    pub image_count: usize,
    pub has_ultrasound: bool,
    pub has_complications: bool,
    pub risk_score: f32,
}

impl PatientData {
    /// Convert patient data to a feature vector.
    /// If extra_keys is provided, uses those keys in order (with 0.0 for missing features).
    /// Otherwise, uses this patient's extra features sorted alphabetically.
    pub fn to_raw_features(&self, extra_keys: Option<&[String]>) -> Vec<f32> {
        let owned_keys: Vec<String>;
        let keys = match extra_keys {
            Some(keys) => keys,
            None => {
                owned_keys = {
                    let mut keys: Vec<_> = self.extra_features.keys().cloned().collect();
                    keys.sort();
                    keys
                };
                &owned_keys
            }
        };

        let mut features = vec![
            self.age.unwrap_or(0.0),
            self.systolic_bp.or(self.blood_pressure).unwrap_or(0.0),
            self.diastolic_bp.unwrap_or(0.0),
            self.glucose.unwrap_or(0.0),
            self.bmi.unwrap_or(0.0),
            self.pregnancies.map(|p| p as f32).unwrap_or(0.0),
            if self.has_complications { 1.0 } else { 0.0 },
            self.risk_score,
        ];

        // Add extra features in the order specified by keys
        // Use 0.0 for missing features
        for key in keys {
            features.push(*self.extra_features.get(key).unwrap_or(&0.0));
        }

        features
    }

    /// Merge another PatientData into this one (for combining data from multiple CSVs)
    pub fn merge(&mut self, other: &PatientData) {
        // Only merge if same patient
        if self.patient_id != other.patient_id {
            return;
        }

        // Merge optional fields (other takes precedence if Some)
        if other.age.is_some() {
            self.age = other.age;
        }
        if other.blood_pressure.is_some() {
            self.blood_pressure = other.blood_pressure;
        }
        if other.systolic_bp.is_some() {
            self.systolic_bp = other.systolic_bp;
        }
        if other.diastolic_bp.is_some() {
            self.diastolic_bp = other.diastolic_bp;
        }
        if other.glucose.is_some() {
            self.glucose = other.glucose;
        }
        if other.bmi.is_some() {
            self.bmi = other.bmi;
        }
        if other.pregnancies.is_some() {
            self.pregnancies = other.pregnancies;
        }
        if other.risk_level.is_some() {
            self.risk_level = other.risk_level.clone();
        }
        if other.existing_outcome.is_some() {
            self.existing_outcome = other.existing_outcome;
        }

        // Merge extra features
        for (k, v) in &other.extra_features {
            self.extra_features.insert(k.clone(), *v);
        }

        // Append temporal and text data
        self.temporal_measurements
            .extend(other.temporal_measurements.clone());
        self.clinical_notes.extend(other.clinical_notes.clone());

        // Update image counts
        self.image_count += other.image_count;
        self.has_ultrasound = self.has_ultrasound || other.has_ultrasound;
        self.has_complications = self.has_complications || other.has_complications;
        if other.risk_score > 0.0 {
            self.risk_score = other.risk_score;
        }
    }
}

// ============================================================================
// Data Preparer (for prepare binary - merges files into database)
// ============================================================================

/// Data preparation utility that merges patient data from multiple files
/// and stores them in the database.
///
/// Used by the `prepare` binary to:
/// 1. Load CSV files from datasets directory
/// 2. Merge records by patient_id across files
/// 3. Store merged data in SQLite database
pub struct DataPreparer {
    /// In-memory patient database for merging records across files
    patient_database: HashMap<String, PatientData>,
    /// Cached normalizer (built after all data is loaded)
    normalizer: Option<Normalizer>,
    /// Counter for generating unique patient IDs
    id_counter: usize,
}

impl DataPreparer {
    pub fn new() -> Self {
        Self {
            patient_database: HashMap::new(),
            normalizer: None,
            id_counter: 0,
        }
    }

    /// Generate a unique patient ID when one is not provided in the data
    pub fn generate_patient_id(&mut self) -> String {
        self.id_counter += 1;
        format!("P{:02}", self.id_counter)
    }

    /// Add or merge a patient record into the in-memory database
    pub fn upsert_patient(&mut self, patient: PatientData) {
        let patient_id = patient.patient_id.clone();

        if let Some(existing) = self.patient_database.get_mut(&patient_id) {
            existing.merge(&patient);
        } else {
            self.patient_database.insert(patient_id, patient);
        }
    }

    /// Get a patient by ID
    pub fn get_patient(&self, patient_id: &str) -> Option<&PatientData> {
        self.patient_database.get(patient_id)
    }

    pub fn get_patient_database(&self) -> &HashMap<String, PatientData> {
        &self.patient_database
    }

    /// Get all patient IDs
    pub fn get_all_patient_ids(&self) -> Vec<String> {
        self.patient_database.keys().cloned().collect()
    }

    /// Get count of patients in memory
    pub fn patient_count(&self) -> usize {
        self.patient_database.len()
    }

    /// Export all patients to the database
    pub fn export_to_database(&self, db: &mut DatabaseManager) -> AppResult<usize> {
        let mut count = 0;
        for patient in self.patient_database.values() {
            db.upsert_patient(patient)
                .map_err(|e| AppError::DataError(format!("Failed to upsert patient: {}", e)))?;
            count += 1;
        }
        info!("Exported {} patients to database", count);
        Ok(count)
    }

    /// Find all CSV files in a directory
    pub fn find_csv_files(&self, dir: &Path) -> AppResult<Vec<PathBuf>> {
        find_csv_files(dir)
    }

    /// Prepare all data in a directory (generates outcomes if needed)
    pub fn prepare_all_data(&mut self, data_dir: &Path) -> AppResult<()> {
        let outcomes_dir = data_dir.join("outcomes");

        if !outcomes_dir.exists() {
            fs::create_dir_all(&outcomes_dir)?;
        }

        let outcomes_file = outcomes_dir.join("outcomes.csv");
        if !outcomes_file.exists() {
            info!("Generating synthetic outcomes file...");
            generate_outcomes(data_dir, &outcomes_file)?;
        }

        Ok(())
    }
}

impl Default for DataPreparer {
    fn default() -> Self {
        Self::new()
    }
}

/// Common trait for data loaders that can provide batches of data
pub trait DataLoader<T> {
    /// Get the next batch of data
    fn next_batch(&mut self, batch_size: usize) -> Option<T>;
    /// Reset the data loader to start from the beginning
    fn reset(&mut self);
    /// Get the total number of samples
    fn len(&self) -> usize;
    /// Check if there are no samples
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Structure holding a batch of multi-modal data
pub struct MultiModalBatch {
    /// Tabular features
    pub tabular: Vec<Vec<f32>>,
    /// Temporal sequences
    pub temporal: Vec<Vec<Vec<f32>>>,
    /// Text token sequences
    pub text: Vec<Vec<usize>>,
    /// Image data
    pub images: Vec<ndarray::Array3<f32>>,
    /// Target values (0.0 or 1.0 for binary classification)
    pub targets: Vec<f32>,
}

/// Combined data loader for multi-modal data
pub struct MultiModalDataLoader {
    /// Tabular data
    pub tabular_data: TabularData,
    /// Temporal data
    pub temporal_data: TemporalData,
    /// Text data
    pub text_data: TextData,
    /// Image data
    pub image_data: ImageData,
    /// Target values
    pub targets: Vec<f32>,
    /// Current position in the dataset
    pub current_idx: usize,
}

impl MultiModalDataLoader {
    /// Create a new multi-modal data loader
    pub fn new(
        tabular_data: TabularData,
        temporal_data: TemporalData,
        text_data: TextData,
        image_data: ImageData,
        targets: Vec<f32>,
    ) -> Self {
        debug!("Target length: {}", targets.len());
        if !tabular_data.data.is_empty() {
            debug!("Tabular data has {} rows", tabular_data.data.len());
            assert_eq!(
                tabular_data.data.len(),
                targets.len(),
                "Tabular data and targets must have same length: {} vs {}",
                tabular_data.data.len(),
                targets.len()
            );
        }

        if !temporal_data.sequences.is_empty() {
            debug!(
                "Temporal data has {} sequences",
                temporal_data.sequences.len()
            );
            assert_eq!(
                temporal_data.sequences.len(),
                targets.len(),
                "Temporal data and targets must have same length: {} vs {}",
                temporal_data.sequences.len(),
                targets.len()
            );
        }

        if !text_data.documents.is_empty() {
            assert_eq!(
                text_data.documents.len(),
                targets.len(),
                "Text data and targets must have same length: {} vs {}",
                text_data.documents.len(),
                targets.len()
            );
        }

        if !image_data.images.is_empty() {
            assert_eq!(
                image_data.images.len(),
                targets.len(),
                "Image data and targets must have same length: {} vs {}",
                image_data.images.len(),
                targets.len()
            );
        }

        // At least one modality should have data
        let has_data = !tabular_data.data.is_empty()
            || !temporal_data.sequences.is_empty()
            || !text_data.documents.is_empty()
            || !image_data.images.is_empty();

        assert!(has_data, "At least one modality must have data");

        MultiModalDataLoader {
            tabular_data,
            temporal_data,
            text_data,
            image_data,
            targets,
            current_idx: 0,
        }
    }
}

impl DataLoader<MultiModalBatch> for MultiModalDataLoader {
    fn next_batch(&mut self, batch_size: usize) -> Option<MultiModalBatch> {
        if self.current_idx >= self.targets.len() {
            return None;
        }

        let end_idx = (self.current_idx + batch_size).min(self.targets.len());
        let batch_size = end_idx - self.current_idx;

        let mut batch = MultiModalBatch {
            tabular: Vec::with_capacity(batch_size),
            temporal: Vec::with_capacity(batch_size),
            text: Vec::with_capacity(batch_size),
            images: Vec::with_capacity(batch_size),
            targets: Vec::with_capacity(batch_size),
        };

        for i in self.current_idx..end_idx {
            if !self.tabular_data.data.is_empty() {
                batch.tabular.push(self.tabular_data.data[i].clone());
            }

            if !self.temporal_data.sequences.is_empty() {
                batch.temporal.push(self.temporal_data.sequences[i].clone());
            }

            if !self.text_data.token_sequences.is_empty() {
                batch.text.push(self.text_data.token_sequences[i].clone());
            }

            if !self.image_data.images.is_empty() {
                batch.images.push(self.image_data.images[i].clone());
            }
            batch.targets.push(self.targets[i]);
        }

        self.current_idx = end_idx;
        Some(batch)
    }

    fn reset(&mut self) {
        self.current_idx = 0;
    }

    fn len(&self) -> usize {
        self.targets.len()
    }
}

/// Configuration for loading datasets
pub struct DataConfig {
    /// Root directory containing all datasets
    pub data_dir: PathBuf,
    /// Whether to merge multiple files of the same modality
    pub merge_files: bool,
    /// Patient ID column name for joining data
    pub patient_id_column: String,
    /// Whether CSV files have headers
    pub has_headers: bool,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("datasets"),
            merge_files: true,
            patient_id_column: "patient_id".to_string(),
            has_headers: true,
        }
    }
}

/// Find all CSV files in a directory
pub fn find_csv_files(dir: &Path) -> AppResult<Vec<std::path::PathBuf>> {
    use std::fs;

    let mut csv_files = Vec::new();

    if !dir.exists() {
        return Ok(csv_files);
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext.eq_ignore_ascii_case("csv") {
                    csv_files.push(path);
                }
            }
        }
    }

    csv_files.sort();
    Ok(csv_files)
}

/// Prepare all data in a directory (generates outcomes if needed)
pub fn prepare_all_data(data_dir: &Path) -> AppResult<()> {
    use std::fs;

    let outcomes_dir = data_dir.join("outcomes");

    if !outcomes_dir.exists() {
        fs::create_dir_all(&outcomes_dir)?;
    }

    let outcomes_file = outcomes_dir.join("outcomes.csv");
    if !outcomes_file.exists() {
        tracing::info!("Generating synthetic outcomes file...");
        generate_outcomes(data_dir, &outcomes_file)?;
    }

    Ok(())
}

pub fn generate_outcomes(data_dir: &Path, output_path: &Path) -> AppResult<()> {
    use csv::Writer;
    use rand::Rng;

    let tabular_dir = data_dir.join("tabular");
    let csv_files = find_csv_files(&tabular_dir)?;

    if csv_files.is_empty() {
        return Err(AppError::from(io::Error::new(
            io::ErrorKind::NotFound,
            "No tabular data found to generate outcomes from",
        )));
    }

    let file = File::open(&csv_files[0])?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);
    let patient_count = reader.records().count();

    let file = File::create(output_path)?;
    let mut writer = Writer::from_writer(file);
    writer.write_record(["PatientID", "Mortality", "CauseCode"])?;

    let mut rng = rand::rng();
    for i in 0..patient_count {
        let patient_id = format!("P{:02}", i);
        let mortality = if rng.random::<f32>() < 0.1 { 1 } else { 0 };
        let cause_code = if mortality == 1 {
            rng.random_range(0..5)
        } else {
            4
        };
        writer.write_record(&[patient_id, mortality.to_string(), cause_code.to_string()])?;
    }

    writer.flush()?;
    tracing::info!("Generated outcomes for {} patients", patient_count);
    Ok(())
}

pub fn read_csv(path: &Path) -> AppResult<Vec<GenericRecord>> {
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new().flexible(true).from_reader(file);

    let headers: Vec<String> = reader.headers()?.iter().map(|h| h.to_string()).collect();

    let mut records = Vec::new();
    for result in reader.records() {
        let record = result?;
        let mut values = HashMap::new();

        for (i, field) in record.iter().enumerate() {
            if i < headers.len() {
                values.insert(headers[i].clone(), field.to_string());
            }
        }

        records.push(GenericRecord {
            headers: headers.clone(),
            values,
        });
    }

    Ok(records)
}

/// Load all tabular data from the tabular subfolder
pub fn load_all_tabular_data(config: &DataConfig) -> AppResult<TabularData> {
    let tabular_dir = config.data_dir.join("tabular");

    if !tabular_dir.exists() {
        return Err(AppError::from(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Tabular data directory not found: {:?}", tabular_dir),
        )));
    }

    let csv_files = find_csv_files(&tabular_dir)
        .map_err(|e| AppError::DataError(format!("Failed to find CSV files: {}", e)))?;

    if csv_files.is_empty() {
        return Err(AppError::from(io::Error::new(
            io::ErrorKind::InvalidData,
            "No CSV files found in tabular directory",
        )));
    }

    info!("  Found {} tabular data files:", csv_files.len());
    for file in &csv_files {
        info!(
            "    - {}",
            file.file_name().unwrap_or_default().to_string_lossy()
        );
    }

    if config.merge_files && csv_files.len() > 1 {
        merge_tabular_files(&csv_files, config)
    } else if csv_files.len() == 1 {
        info!("  Loading single tabular file: {}", csv_files[0].display());
        load_tabular_data(&csv_files[0])
    } else {
        Err(AppError::DataError(
            "Multiple CSV files found but merge_files is false".to_string(),
        ))
    }
}

/// Load all temporal data from the temporal subfolder
pub fn load_all_temporal_data(config: &DataConfig) -> AppResult<TemporalData> {
    let temporal_dir = config.data_dir.join("temporal");

    if !temporal_dir.exists() {
        return Err(AppError::Io(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Temporal data directory not found: {:?}", temporal_dir),
        )));
    }

    let csv_files = find_csv_files(&temporal_dir)?;

    if csv_files.is_empty() {
        return Err(AppError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "No CSV files found in temporal directory",
        )));
    }

    info!("  Found {} temporal data files:", csv_files.len());
    for file in &csv_files {
        info!(
            "    - {}",
            file.file_name().unwrap_or_default().to_string_lossy()
        );
    }

    if config.merge_files && csv_files.len() > 1 {
        merge_temporal_files(&csv_files, config)
    } else {
        // Load the first (or only) file
        load_temporal_data(&csv_files[0])
    }
}

/// Load all text data from the text subfolder
pub fn load_all_text_data(config: &DataConfig) -> AppResult<TextData> {
    let text_dir = config.data_dir.join("text");

    if !text_dir.exists() {
        return Err(AppError::from(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Text data directory not found: {:?}", text_dir),
        )));
    }

    let csv_files = find_csv_files(&text_dir)?;

    if csv_files.is_empty() {
        return Err(AppError::from(io::Error::new(
            io::ErrorKind::InvalidData,
            "No CSV files found in text directory",
        )));
    }

    info!("  Found {} text data files:", csv_files.len());
    for file in &csv_files {
        info!(
            "    - {}",
            file.file_name().unwrap_or_default().to_string_lossy()
        );
    }

    if config.merge_files && csv_files.len() > 1 {
        merge_text_files(&csv_files, config)
    } else {
        // Load the first (or only) file
        load_text_data(&csv_files[0])
    }
}

/// Load all image data from the image subfolder and its subdirectories
pub fn load_all_image_data(config: &DataConfig) -> AppResult<ImageData> {
    let image_dir = config.data_dir.join("image");

    if !image_dir.exists() {
        return Err(AppError::from(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Image data directory not found: {:?}", image_dir),
        )));
    }

    // Recursively find all image files
    let image_files = find_image_files_recursive(&image_dir)?;

    if image_files.is_empty() {
        return Err(AppError::from(io::Error::new(
            io::ErrorKind::InvalidData,
            "No image files found in image directory",
        )));
    }

    info!(
        "  Found {} image files across subdirectories",
        image_files.len()
    );

    // Group by subdirectory for reporting
    let mut dir_counts: HashMap<String, usize> = HashMap::new();
    for file in &image_files {
        let parent = file
            .parent()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "root".to_string());
        *dir_counts.entry(parent).or_insert(0) += 1;
    }

    for (dir, count) in dir_counts {
        info!("    - {}: {} images", dir, count);
    }

    load_image_data_from_files(&image_files)
}

/// Load outcomes/targets from the outcomes subfolder
pub fn load_all_outcomes(config: &DataConfig) -> AppResult<(Vec<f32>, Vec<usize>)> {
    prepare_all_data(&config.data_dir)?;

    let outcomes_dir = config.data_dir.join("outcomes");

    if !outcomes_dir.exists() {
        return Err(AppError::from(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Outcomes directory not found: {:?}", outcomes_dir),
        )));
    }

    let csv_files = find_csv_files(&outcomes_dir)?;

    if csv_files.is_empty() {
        return Err(AppError::from(io::Error::new(
            io::ErrorKind::InvalidData,
            "No CSV files found in outcomes directory",
        )));
    }

    // For outcomes, typically use just one file
    info!(
        "  Loading outcomes from: {}",
        csv_files[0]
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    );

    load_targets_and_causes(&csv_files[0])
}

/// Recursively find all image files in a directory and subdirectories
fn find_image_files_recursive(dir: &Path) -> AppResult<Vec<PathBuf>> {
    let mut image_files = Vec::new();
    find_image_files_recursive_impl(dir, &mut image_files)?;

    // Sort files for consistent ordering
    image_files.sort();
    Ok(image_files)
}

fn find_image_files_recursive_impl(dir: &Path, files: &mut Vec<PathBuf>) -> AppResult<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // Recurse into subdirectories
            find_image_files_recursive_impl(&path, files)?;
        } else if path.is_file() {
            if let Some(ext) = path.extension() {
                let ext = ext.to_str().unwrap_or("").to_lowercase();
                if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "bmp" | "tiff") {
                    files.push(path);
                }
            }
        }
    }
    Ok(())
}

/// Merge multiple tabular CSV files into a single TabularData
/// Uses DataPreparer to merge records by patient_id
fn merge_tabular_files(files: &[PathBuf], config: &DataConfig) -> AppResult<TabularData> {
    let mut preparer = DataPreparer::new();
    let mut all_feature_names: Vec<String> = Vec::new();
    let mut feature_name_set: HashSet<String> = HashSet::new();

    // Known column mappings (case-insensitive)
    let known_columns: HashMap<&str, &str> = [
        ("age", "age"),
        ("bloodpressure", "blood_pressure"),
        ("systolicbp", "systolic_bp"),
        ("bmi", "bmi"),
        ("diastolicbp", "diastolic_bp"),
        ("height", "height"),
        ("weight", "weight"),
        ("glucose", "glucose"),
        ("outcome", "outcome"),
        ("pregnancies", "pregnancies"),
        ("risklevel", "risk_level"),
    ]
    .iter()
    .cloned()
    .collect();

    // Process each file
    for file in files {
        info!("Processing file: {}", file.display());

        let f = File::open(file)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(config.has_headers)
            .flexible(true)
            .from_reader(f);

        let headers: Vec<String> = reader.headers()?.iter().map(|s| s.to_string()).collect();

        // Find patient ID column
        let id_col_idx = headers.iter().position(|h| {
            let h_lower = h.to_lowercase();
            h_lower.contains("patient")
                || h_lower.contains("subject")
                || h_lower == "id"
                || h.eq_ignore_ascii_case(&config.patient_id_column)
        });

        // Track feature names
        for (i, name) in headers.iter().enumerate() {
            if Some(i) != id_col_idx && !feature_name_set.contains(name) {
                feature_name_set.insert(name.clone());
                all_feature_names.push(name.clone());
            }
        }

        // Process records
        for result in reader.records() {
            let record = result?;
            let patient_id = if let Some(idx) = id_col_idx {
                let id = record.get(idx).unwrap_or("").trim().to_string();
                if id.is_empty() {
                    preparer.generate_patient_id()
                } else {
                    id
                }
            } else {
                preparer.generate_patient_id()
            };

            let mut patient = PatientData {
                patient_id: patient_id.clone(),
                ..Default::default()
            };

            // Map columns to PatientData fields
            for (i, value) in record.iter().enumerate() {
                if Some(i) == id_col_idx || value.trim().is_empty() {
                    continue;
                }

                let col_name = &headers[i];
                let col_lower = col_name.to_lowercase().replace(" ", "_");

                if let Some(&field_name) = known_columns.get(col_lower.as_str()) {
                    if let Ok(v) = value.trim().parse::<f32>() {
                        match field_name {
                            "age" => patient.age = Some(v),
                            "blood_pressure" => patient.blood_pressure = Some(v),
                            // ... etc
                            _ => {}
                        }
                    }
                } else {
                    // Store as extra feature
                    if let Ok(v) = value.trim().parse::<f32>() {
                        patient.extra_features.insert(col_name.clone(), v);
                    }
                }
            }

            // Merge into preparer's patient_database
            preparer.upsert_patient(patient);
        }
    }

    info!(
        "Merged {} patients from {} files",
        preparer.patient_count(),
        files.len()
    );

    // Convert to TabularData
    let patient_ids = preparer.get_all_patient_ids();
    let mut all_extra_keys: HashSet<String> = HashSet::new();

    for patient_id in &patient_ids {
        if let Some(patient) = preparer.get_patient(patient_id) {
            for key in patient.extra_features.keys() {
                all_extra_keys.insert(key.clone());
            }
        }
    }

    // Sort the keys for consistent ordering
    let mut sorted_extra_keys: Vec<String> = all_extra_keys.into_iter().collect();
    sorted_extra_keys.sort();

    info!("Total extra feature columns: {}", sorted_extra_keys.len());

    // Now convert all patients using the same set of keys
    let mut data: Vec<Vec<f32>> = Vec::new();
    for patient_id in &patient_ids {
        if let Some(patient) = preparer.get_patient(patient_id) {
            data.push(patient.to_raw_features(Some(&sorted_extra_keys)));
        }
    }

    // Build complete feature names list
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

    let mut complete_feature_names = base_feature_names;
    complete_feature_names.extend(sorted_extra_keys);

    // Normalize features
    normalize_features(&mut data);

    Ok(TabularData {
        data,
        patient_ids,
        feature_names: complete_feature_names,
        numeric_stats: HashMap::new(),
        categorical_mappings: HashMap::new(),
        categorical_features: Vec::new(),
        current_idx: 0,
    })
}

/// Merge multiple temporal CSV files using DataPreparer
fn merge_temporal_files(files: &[PathBuf], config: &DataConfig) -> AppResult<TemporalData> {
    let mut preparer = DataPreparer::new();
    let mut all_measurement_types: HashSet<String> = HashSet::new();

    // Process each CSV file
    for file in files {
        info!("    Processing temporal file: {}", file.display());

        let f = File::open(file)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(config.has_headers)
            .flexible(true)
            .from_reader(f);

        let headers: Vec<String> = reader.headers()?.iter().map(|s| s.to_string()).collect();

        // Find patient ID and timestamp columns
        let id_col_idx = headers.iter().position(|h| {
            let h_lower = h.to_lowercase();
            h_lower.contains("patient")
                || h_lower.contains("subject")
                || h_lower == "id"
                || h.eq_ignore_ascii_case(&config.patient_id_column)
        });

        let timestamp_col_idx = headers
            .iter()
            .position(|h| h.eq_ignore_ascii_case("timestamp") || h.eq_ignore_ascii_case("time"));

        // Collect measurement type columns (all columns except ID and timestamp)
        for (i, name) in headers.iter().enumerate() {
            if Some(i) != id_col_idx && timestamp_col_idx.map_or(true, |t| i != t) {
                all_measurement_types.insert(name.clone());
            }
        }

        // Process records
        for result in reader.records() {
            let record = result?;
            let patient_id = if let Some(idx) = id_col_idx {
                let id = record.get(idx).unwrap_or("").trim().to_string();
                if id.is_empty() {
                    preparer.generate_patient_id()
                } else {
                    id
                }
            } else {
                preparer.generate_patient_id()
            };

            let timestamp = timestamp_col_idx
                .and_then(|idx| record.get(idx))
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "0".to_string());

            // Create temporal measurements for this row
            let mut measurements = Vec::new();
            for (i, value) in record.iter().enumerate() {
                if Some(i) == id_col_idx || timestamp_col_idx.map_or(false, |t| i == t) {
                    continue;
                }

                if let Ok(v) = value.trim().parse::<f32>() {
                    measurements.push(TemporalMeasurement {
                        timestamp: timestamp.clone(),
                        measurement_type: headers[i].clone(),
                        value: v,
                    });
                }
            }

            // Get or create patient and add measurements
            let mut patient = preparer
                .get_patient(&patient_id)
                .cloned()
                .unwrap_or_else(|| PatientData {
                    patient_id: patient_id.clone(),
                    ..Default::default()
                });

            patient.temporal_measurements.extend(measurements);
            preparer.upsert_patient(patient);
        }
    }

    info!(
        "    Merged {} patients with temporal data from {} files",
        preparer.patient_count(),
        files.len()
    );

    // Convert DataPreparer's patient database to TemporalData format
    let patient_database = preparer.get_patient_database();
    let mut sequences: Vec<Vec<Vec<f32>>> = Vec::new();
    let measurement_types: Vec<String> = all_measurement_types.into_iter().collect();

    // Sort patient IDs for consistent ordering
    let mut patient_ids: Vec<String> = patient_database.keys().cloned().collect();
    patient_ids.sort();

    for patient_id in &patient_ids {
        let patient = &patient_database[patient_id];

        if patient.temporal_measurements.is_empty() {
            continue;
        }

        // Group measurements by timestamp
        let mut by_timestamp: HashMap<String, HashMap<String, f32>> = HashMap::new();
        for m in &patient.temporal_measurements {
            by_timestamp
                .entry(m.timestamp.clone())
                .or_default()
                .insert(m.measurement_type.clone(), m.value);
        }

        // Convert to sequence (sorted by timestamp)
        let mut timestamps: Vec<String> = by_timestamp.keys().cloned().collect();
        timestamps.sort();

        let sequence: Vec<Vec<f32>> = timestamps
            .iter()
            .map(|ts| {
                let measurements = &by_timestamp[ts];
                measurement_types
                    .iter()
                    .map(|mt| *measurements.get(mt).unwrap_or(&0.0))
                    .collect()
            })
            .collect();

        sequences.push(sequence);
    }

    let max_seq_length = sequences.iter().map(|s| s.len()).max().unwrap_or(0);

    Ok(TemporalData {
        sequences,
        feature_names: measurement_types,
        feature_stats: HashMap::new(),
        max_seq_length,
        current_idx: 0,
    })
}

/// Merge multiple text CSV files using DataPreparer
fn merge_text_files(files: &[PathBuf], config: &DataConfig) -> AppResult<TextData> {
    let mut preparer = DataPreparer::new();

    info!("  Merging {} text files...", files.len());

    // Process each CSV file
    for file in files {
        info!("    Processing text file: {}", file.display());

        let f = File::open(file)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(config.has_headers)
            .flexible(true)
            .from_reader(f);

        let headers: Vec<String> = reader.headers()?.iter().map(|s| s.to_string()).collect();

        // Find patient ID column
        let id_col_idx = headers
            .iter()
            .position(|h| h.eq_ignore_ascii_case(&config.patient_id_column))
            .ok_or_else(|| {
                AppError::DataError(format!(
                    "Patient ID column '{}' not found in {}",
                    config.patient_id_column,
                    file.display()
                ))
            })?;

        // Find text/notes column (look for common names)
        let text_col_idx = headers.iter().position(|h| {
            let lower = h.to_lowercase();
            lower.contains("note")
                || lower.contains("text")
                || lower.contains("clinical")
                || lower.contains("description")
                || lower.contains("comment")
        });

        // Process records
        for result in reader.records() {
            let record = result?;
            let patient_id = record.get(id_col_idx).unwrap_or("").trim().to_string();

            if patient_id.is_empty() {
                continue;
            }

            // Get text content
            let text_content = if let Some(text_idx) = text_col_idx {
                record.get(text_idx).unwrap_or("").trim().to_string()
            } else {
                // If no specific text column, concatenate all non-ID columns
                record
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != id_col_idx)
                    .map(|(_, v)| v.trim())
                    .filter(|v| !v.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ")
            };

            if text_content.is_empty() {
                continue;
            }

            // Get or create patient and add clinical note
            let mut patient = preparer
                .get_patient(&patient_id)
                .cloned()
                .unwrap_or_else(|| PatientData {
                    patient_id: patient_id.clone(),
                    ..Default::default()
                });

            patient.clinical_notes.push(text_content);
            preparer.upsert_patient(patient);
        }
    }

    info!(
        "    Merged {} patients with text data from {} files",
        preparer.patient_count(),
        files.len()
    );

    // Convert to TextData format
    let patient_database = preparer.get_patient_database();
    let mut documents: Vec<String> = Vec::new();

    // Sort patient IDs for consistent ordering
    let mut patient_ids: Vec<String> = patient_database.keys().cloned().collect();
    patient_ids.sort();

    for patient_id in &patient_ids {
        let patient = &patient_database[patient_id];

        if patient.clinical_notes.is_empty() {
            continue;
        }

        // Combine all notes for this patient
        let combined = patient.clinical_notes.join(" ");
        documents.push(combined);
    }

    // Create TextData with vocabulary
    TextData::from_documents(documents, 10000) // max 10k vocabulary
        .map_err(|e| AppError::DataError(format!("Failed to create text data: {}", e)))
}

/// Load image data from a list of file paths (images don't need patient-based merging)
fn load_image_data_from_files(paths: &[PathBuf]) -> AppResult<ImageData> {
    ImageData::from_files(paths, (224, 224), false)
        .map_err(|e| AppError::DataError(format!("Failed to load images: {}", e)))
}

/// Load tabular data from a CSV file with appropriate categorical features
pub fn load_tabular_data<P: AsRef<Path>>(path: P) -> AppResult<TabularData> {
    // Define which columns should be treated as categorical
    let categorical_features = vec![
        "ethnicity".to_string(),
        "education_level".to_string(),
        "insurance_type".to_string(),
        "delivery_type".to_string(),
        "prenatal_care_adequacy".to_string(),
        "location_type".to_string(), // urban/rural
    ];

    TabularData::from_csv(path, true, &categorical_features)
}

/// Load temporal data from a CSV file
pub fn load_temporal_data<P: AsRef<Path>>(path: P) -> AppResult<TemporalData> {
    // Load from a single CSV file where each patient has multiple rows
    // Expects columns: PatientID, Timestamp, and various vital signs/lab values
    TemporalData::from_single_csv(
        path,
        true, // has header
        "PatientID".to_string(),
        Some("Timestamp".to_string()),
    )
}

/// Load text data from a CSV file
pub fn load_text_data<P: AsRef<Path>>(path: P) -> AppResult<TextData> {
    // Use DataPreparer for consistent CSV reading
    let records = read_csv(path.as_ref())?;

    let mut documents = Vec::new();

    for record in records {
        // Assuming the CSV has columns like "PatientID" and "NoteText" or "ClinicalNotes"
        let note_text = record
            .values
            .get("NoteText")
            .or_else(|| record.values.get("ClinicalNotes"))
            .or_else(|| record.values.get("Notes"))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "No text column found (tried NoteText, ClinicalNotes, Notes)",
                )
            })?;

        documents.push(note_text.clone());
    }

    // Create text data with a reasonable vocabulary size
    TextData::from_documents(documents, 10000)
}

/// Load image data from a directory
pub fn load_image_data<P: AsRef<Path>>(dir_path: P) -> AppResult<ImageData> {
    // Collect all image file paths
    let paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir_path)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();

            // Check if it's an image file
            if path.is_file() {
                let ext = path.extension()?.to_str()?.to_lowercase();
                if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "bmp") {
                    Some(path)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    if paths.is_empty() {
        return Err(AppError::from(io::Error::new(
            io::ErrorKind::InvalidData,
            "No image files found in directory",
        )));
    }

    // Load images with standard size for medical imaging
    // Using 224x224 for compatibility with standard models
    ImageData::from_files(&paths, (224, 224), false) // false = keep RGB
}

/// Load target values and mortality causes from a CSV file
pub fn load_targets_and_causes<P: AsRef<Path>>(path: P) -> AppResult<(Vec<f32>, Vec<usize>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut targets = Vec::new();
    let mut causes = Vec::new();

    // Skip header
    let mut lines = reader.lines();
    if let Some(header) = lines.next() {
        let _ = header?; // Consume header
    }

    // Read each line
    for (line_num, line) in lines.enumerate() {
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();

        if parts.len() < 3 {
            return Err(AppError::from(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Line {} has insufficient columns", line_num + 2),
            )));
        }

        // Expected format: PatientID,Mortality(0/1),CauseCode
        let mortality = parts[1].trim().parse::<f32>().map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid mortality value on line {}: {}", line_num + 2, e),
            )
        })?;

        // Validate binary target
        if mortality != 0.0 && mortality != 1.0 {
            return Err(AppError::from(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Mortality must be 0 or 1, found {} on line {}",
                    mortality,
                    line_num + 2
                ),
            )));
        }

        targets.push(mortality);

        // Parse cause code (0-4, where 4 is "other/none")
        let cause = if mortality > 0.5 {
            parts[2].trim().parse::<usize>().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid cause code on line {}: {}", line_num + 2, e),
                )
            })?
        } else {
            4 // No cause for survivors
        };

        causes.push(cause);
    }

    if targets.is_empty() {
        return Err(AppError::from(io::Error::new(
            io::ErrorKind::InvalidData,
            "No data found in outcomes file",
        )));
    }

    Ok((targets, causes))
}

/// Create a multi-modal data loader from separate modality data
pub fn create_data_loader(
    tabular_data: Vec<Vec<f32>>,
    temporal_data: Vec<Vec<Vec<f32>>>,
    text_data: Vec<Vec<usize>>,
    image_data: Vec<Array3<f32>>,
    targets: Vec<f32>,
) -> MultiModalDataLoader {
    // Create data containers
    let tabular_container = TabularData {
        data: tabular_data,
        patient_ids: Vec::new(),
        feature_names: Vec::new(),
        numeric_stats: HashMap::new(),
        categorical_mappings: HashMap::new(),
        categorical_features: Vec::new(),
        current_idx: 0,
    };

    let temporal_container = TemporalData {
        sequences: temporal_data,
        feature_names: Vec::new(),
        feature_stats: HashMap::new(),
        max_seq_length: 0,
        current_idx: 0,
    };

    let text_container = TextData {
        documents: Vec::new(),
        token_sequences: text_data,
        vocab: HashMap::new(),
        reverse_vocab: HashMap::new(),
        max_seq_length: 0,
        current_idx: 0,
    };

    let image_container = ImageData {
        images: image_data,
        image_size: (224, 224), // Standard image size
        channels: 3,            // RGB images
        channel_means: Vec::new(),
        channel_stds: Vec::new(),
        current_idx: 0,
    };

    MultiModalDataLoader::new(
        tabular_container,
        temporal_container,
        text_container,
        image_container,
        targets,
    )
}

pub fn normalize_features(data: &mut Vec<Vec<f32>>) {
    if data.is_empty() {
        return;
    }

    let num_features = data[0].len();
    let num_samples = data.len();

    for feature_idx in 0..num_features {
        // Calculate mean
        let mean: f32 =
            data.iter().map(|sample| sample[feature_idx]).sum::<f32>() / num_samples as f32;

        // Calculate std dev
        let variance: f32 = data
            .iter()
            .map(|sample| (sample[feature_idx] - mean).powi(2))
            .sum::<f32>()
            / num_samples as f32;
        let std = variance.sqrt().max(1e-8); // Avoid division by zero

        // Normalize
        for sample in data.iter_mut() {
            sample[feature_idx] = (sample[feature_idx] - mean) / std;
        }
    }
}
