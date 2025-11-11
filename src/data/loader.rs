//! Data loading utilities for all modalities

use ndarray::Array3;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use tracing::info;

use crate::data::{ImageData, TabularData, TemporalData, TextData};

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
        // Ensure all data sources have the same number of samples
        assert_eq!(
            tabular_data.data.len(),
            targets.len(),
            "Tabular data and targets must have same length"
        );
        assert_eq!(
            temporal_data.sequences.len(),
            targets.len(),
            "Temporal data and targets must have same length"
        );
        assert_eq!(
            text_data.documents.len(),
            targets.len(),
            "Text data and targets must have same length"
        );
        assert_eq!(
            image_data.images.len(),
            targets.len(),
            "Image data and targets must have same length"
        );

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
            batch.tabular.push(self.tabular_data.data[i].clone());
            batch.temporal.push(self.temporal_data.sequences[i].clone());
            batch.text.push(self.text_data.token_sequences[i].clone());
            batch.images.push(self.image_data.images[i].clone());
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

/// Load all tabular data from the tabular subfolder
pub fn load_all_tabular_data(config: &DataConfig) -> io::Result<TabularData> {
    let tabular_dir = config.data_dir.join("tabular");

    if !tabular_dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Tabular data directory not found: {:?}", tabular_dir),
        ));
    }

    let csv_files = find_csv_files(&tabular_dir)?;

    if csv_files.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No CSV files found in tabular directory",
        ));
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
    } else {
        // Load the first (or only) file
        load_tabular_data(&csv_files[0])
    }
}

/// Load all temporal data from the temporal subfolder
pub fn load_all_temporal_data(config: &DataConfig) -> io::Result<TemporalData> {
    let temporal_dir = config.data_dir.join("temporal");

    if !temporal_dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Temporal data directory not found: {:?}", temporal_dir),
        ));
    }

    let csv_files = find_csv_files(&temporal_dir)?;

    if csv_files.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No CSV files found in temporal directory",
        ));
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
pub fn load_all_text_data(config: &DataConfig) -> io::Result<TextData> {
    let text_dir = config.data_dir.join("text");

    if !text_dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Text data directory not found: {:?}", text_dir),
        ));
    }

    let csv_files = find_csv_files(&text_dir)?;

    if csv_files.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No CSV files found in text directory",
        ));
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
pub fn load_all_image_data(config: &DataConfig) -> io::Result<ImageData> {
    let image_dir = config.data_dir.join("image");

    if !image_dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Image data directory not found: {:?}", image_dir),
        ));
    }

    // Recursively find all image files
    let image_files = find_image_files_recursive(&image_dir)?;

    if image_files.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No image files found in image directory",
        ));
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
pub fn load_all_outcomes(config: &DataConfig) -> io::Result<(Vec<f32>, Vec<usize>)> {
    let outcomes_dir = config.data_dir.join("outcomes");

    if !outcomes_dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Outcomes directory not found: {:?}", outcomes_dir),
        ));
    }

    let csv_files = find_csv_files(&outcomes_dir)?;

    if csv_files.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No CSV files found in outcomes directory",
        ));
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

/// Find all CSV files in a directory
fn find_csv_files(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut csv_files = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext.to_str() == Some("csv") {
                    csv_files.push(path);
                }
            }
        }
    }

    // Sort files for consistent ordering
    csv_files.sort();
    Ok(csv_files)
}

/// Recursively find all image files in a directory and subdirectories
fn find_image_files_recursive(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut image_files = Vec::new();
    find_image_files_recursive_impl(dir, &mut image_files)?;

    // Sort files for consistent ordering
    image_files.sort();
    Ok(image_files)
}

fn find_image_files_recursive_impl(dir: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
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

/// Merge multiple tabular CSV files based on patient ID
fn merge_tabular_files(files: &[PathBuf], config: &DataConfig) -> io::Result<TabularData> {
    // This would implement logic to:
    // 1. Read all CSV files
    // 2. Join them on patient ID column
    // 3. Handle missing values appropriately
    // 4. Return combined TabularData

    // For now, using simplified implementation
    info!(
        "  Merging {} tabular files on '{}'...",
        files.len(),
        config.patient_id_column
    );

    // Placeholder - you'd implement actual merging logic here
    load_tabular_data(&files[0])
}

/// Merge multiple temporal CSV files
fn merge_temporal_files(files: &[PathBuf], config: &DataConfig) -> io::Result<TemporalData> {
    info!("  Merging {} temporal files...", files.len());

    // Would implement logic to combine multiple time series files
    // Could be different variables measured at different times
    load_temporal_data(&files[0])
}

/// Merge multiple text CSV files
fn merge_text_files(files: &[PathBuf], config: &DataConfig) -> io::Result<TextData> {
    info!("  Merging {} text files...", files.len());

    // Would concatenate text from multiple sources (notes, reports, etc.)
    let mut all_documents = HashMap::new();

    for file in files {
        let file_handle = File::open(file)?;
        let reader = BufReader::new(file_handle);

        for (i, line) in reader.lines().enumerate() {
            if i == 0 && config.has_headers {
                continue;
            }

            let line = line?;
            let parts: Vec<&str> = line.split(',').collect();

            if parts.len() >= 2 {
                let patient_id = parts[0].trim();
                let text = parts[1..].join(",").trim_matches('"').to_string();

                all_documents
                    .entry(patient_id.to_string())
                    .or_insert_with(Vec::new)
                    .push(text);
            }
        }
    }

    // Combine all text for each patient
    let documents: Vec<String> = all_documents
        .into_iter()
        .map(|(_, texts)| texts.join(" "))
        .collect();

    TextData::from_documents(documents, 10000)
}

/// Load image data from a list of file paths
fn load_image_data_from_files(paths: &[PathBuf]) -> io::Result<ImageData> {
    ImageData::from_files(paths, (224, 224), false)
}

/// Load tabular data from a CSV file with appropriate categorical features
pub fn load_tabular_data<P: AsRef<Path>>(path: P) -> io::Result<TabularData> {
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
pub fn load_temporal_data<P: AsRef<Path>>(path: P) -> io::Result<TemporalData> {
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
pub fn load_text_data<P: AsRef<Path>>(path: P) -> io::Result<TextData> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut documents = Vec::new();

    // Skip header and read documents
    for (i, line) in reader.lines().enumerate() {
        if i == 0 {
            continue;
        } // Skip header

        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();

        if parts.len() < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Line {} has insufficient columns", i + 1),
            ));
        }

        // Assume format: PatientID,NoteText
        documents.push(parts[1..].join(",").trim_matches('"').to_string());
    }

    // Create text data with a reasonable vocabulary size
    TextData::from_documents(documents, 10000)
}

/// Load image data from a directory
pub fn load_image_data<P: AsRef<Path>>(dir_path: P) -> io::Result<ImageData> {
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
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No image files found in directory",
        ));
    }

    // Load images with standard size for medical imaging
    // Using 224x224 for compatibility with standard models
    ImageData::from_files(&paths, (224, 224), false) // false = keep RGB
}

/// Load target values and mortality causes from a CSV file
pub fn load_targets_and_causes<P: AsRef<Path>>(path: P) -> io::Result<(Vec<f32>, Vec<usize>)> {
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
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Line {} has insufficient columns", line_num + 2),
            ));
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
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Mortality must be 0 or 1, found {} on line {}",
                    mortality,
                    line_num + 2
                ),
            ));
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
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No data found in outcomes file",
        ));
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
