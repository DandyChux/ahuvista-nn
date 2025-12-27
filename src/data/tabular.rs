//! Tabular data processing module.
//!
//! Provides utilities for loading, preprocessing, and normalizing
//! structured tabular data (e.g., patient demographics, lab values).

use crate::data::DataLoader;
use crate::utils::error::{AppError, AppResult};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// Structure for handling tabular (structured) data
pub struct TabularData {
    /// The processed feature vectors
    pub data: Vec<Vec<f32>>,
    /// Patient IDs corresponding to each data row
    pub patient_ids: Vec<String>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Statistics for each numeric feature (mean, std_dev)
    pub numeric_stats: HashMap<String, (f32, f32)>,
    /// Mapping for each categorical feature (value -> index)
    pub categorical_mappings: HashMap<String, HashMap<String, usize>>,
    /// Which features are categorical (vs numeric)
    pub categorical_features: Vec<String>,
    /// Current index for batch iteration
    pub current_idx: usize,
}

impl TabularData {
    /// Create a new empty TabularData instance
    pub fn new() -> Self {
        TabularData {
            data: Vec::new(),
            patient_ids: Vec::new(),
            feature_names: Vec::new(),
            numeric_stats: HashMap::new(),
            categorical_mappings: HashMap::new(),
            categorical_features: Vec::new(),
            current_idx: 0,
        }
    }

    /// Load tabular data from a CSV file
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file
    /// * `has_header` - Whether the file has a header row
    /// * `categorical_features` - Names of features that should be treated as categorical
    ///
    /// # Returns
    /// Result containing TabularData or an error
    pub fn from_csv<P: AsRef<Path>>(
        path: P,
        has_header: bool,
        categorical_features: &[String],
    ) -> AppResult<Self> {
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Read header if present
        let header_line = if has_header {
            lines
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Empty CSV file"))??
        } else {
            String::new()
        };

        let feature_names = if has_header {
            header_line
                .split(',')
                .map(|s| s.trim().to_string())
                .collect()
        } else {
            // Generate default feature names (Feature0, Feature1, etc.)
            let first_line = lines
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Empty CSV file"))??;
            let num_features = first_line.split(',').count();

            (0..num_features)
                .map(|i| format!("Feature{}", i))
                .collect::<Vec<String>>()
        };

        // If we read the first line to count features, put it back
        if !has_header {
            // Reset the reader to start from the beginning
            let file = File::open(&path)?;
            let reader = BufReader::new(file);
            lines = reader.lines();
        }

        // Collect all data as strings first
        let mut raw_data: Vec<Vec<String>> = Vec::new();

        for line_result in lines {
            let line = line_result?;
            let values: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();

            if values.len() != feature_names.len() {
                return Err(AppError::Io(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Line has {} values, expected {}",
                        values.len(),
                        feature_names.len()
                    ),
                )));
            }

            raw_data.push(values);
        }

        // Create a set of categorical feature names for faster lookup
        let categorical_set: std::collections::HashSet<String> =
            categorical_features.iter().cloned().collect();

        // Detect which columns are categorical (if not explicitly specified)
        let mut all_categorical_features = Vec::new();

        for (i, name) in feature_names.iter().enumerate() {
            if categorical_set.contains(name) {
                all_categorical_features.push(name.clone());
            } else {
                // Check if the column contains any non-numeric values
                let has_non_numeric = raw_data.iter().any(|row| {
                    let val = &row[i];
                    val.parse::<f32>().is_err() && !val.is_empty()
                });

                if has_non_numeric {
                    all_categorical_features.push(name.clone());
                }
            }
        }

        // Process data
        let mut processed_data = TabularData {
            data: Vec::with_capacity(raw_data.len()),
            patient_ids: Vec::with_capacity(raw_data.len()),
            feature_names: feature_names.clone(),
            numeric_stats: HashMap::new(),
            categorical_mappings: HashMap::new(),
            categorical_features: all_categorical_features.clone(),
            current_idx: 0,
        };

        // Build categorical mappings and compute numeric stats
        processed_data.build_preprocessing_params(
            &raw_data,
            &feature_names,
            &all_categorical_features,
        );

        // Transform raw data to processed feature vectors
        for row in &raw_data {
            let processed_row =
                processed_data.process_row(row, &feature_names, &all_categorical_features);
            processed_data.data.push(processed_row);
        }

        Ok(processed_data)
    }

    /// Build preprocessing parameters (categorical mappings, numeric stats)
    fn build_preprocessing_params(
        &mut self,
        raw_data: &[Vec<String>],
        feature_names: &[String],
        categorical_features: &[String],
    ) {
        let categorical_set: std::collections::HashSet<String> =
            categorical_features.iter().cloned().collect();

        // For each feature
        for (i, name) in feature_names.iter().enumerate() {
            if categorical_set.contains(name) {
                // Build mapping for categorical feature
                let mut value_set = std::collections::HashSet::new();
                for row in raw_data {
                    value_set.insert(row[i].clone());
                }

                let mut mapping = HashMap::new();
                for (j, value) in value_set.iter().enumerate() {
                    mapping.insert(value.clone(), j);
                }

                self.categorical_mappings.insert(name.clone(), mapping);
            } else {
                // Compute statistics for numeric feature
                let mut sum = 0.0;
                let mut count = 0;

                for row in raw_data {
                    if let Ok(val) = row[i].parse::<f32>() {
                        sum += val;
                        count += 1;
                    }
                }

                let mean = if count > 0 { sum / count as f32 } else { 0.0 };

                // Compute standard deviation
                let mut sum_sq_diff = 0.0;
                for row in raw_data {
                    if let Ok(val) = row[i].parse::<f32>() {
                        let diff = val - mean;
                        sum_sq_diff += diff * diff;
                    }
                }

                let std_dev = if count > 1 {
                    (sum_sq_diff / (count - 1) as f32).sqrt()
                } else {
                    1.0 // Default to 1.0 if not enough data
                };

                self.numeric_stats.insert(name.clone(), (mean, std_dev));
            }
        }
    }

    /// Process a single row of raw data into a feature vector
    fn process_row(
        &self,
        row: &[String],
        feature_names: &[String],
        categorical_features: &[String],
    ) -> Vec<f32> {
        let categorical_set: std::collections::HashSet<String> =
            categorical_features.iter().cloned().collect();

        let mut processed = Vec::with_capacity(row.len());

        for (i, (name, value)) in feature_names.iter().zip(row.iter()).enumerate() {
            if categorical_set.contains(name) {
                // One-hot encode categorical feature
                let mapping = &self.categorical_mappings[name];
                let index = mapping.get(value).copied().unwrap_or(0);
                let num_categories = mapping.len();

                // For efficiency in low-compute environments, use index encoding
                // instead of full one-hot encoding
                processed.push(index as f32);
            } else {
                // Normalize numeric feature
                let parsed_value = value.parse::<f32>().unwrap_or(0.0);
                let (mean, std_dev) = self.numeric_stats.get(name).unwrap_or(&(0.0, 1.0));

                // Avoid division by zero
                let normalized = if *std_dev > 1e-6 {
                    (parsed_value - mean) / std_dev
                } else {
                    parsed_value - mean
                };

                processed.push(normalized);
            }
        }

        processed
    }

    /// Get the number of features after preprocessing
    pub fn feature_dim(&self) -> usize {
        if self.data.is_empty() {
            0
        } else {
            self.data[0].len()
        }
    }

    /// Preprocess a new row of data (for inference)
    pub fn preprocess_row(&self, row: &[String]) -> Vec<f32> {
        self.process_row(row, &self.feature_names, &self.categorical_features)
    }
}

impl DataLoader<Vec<Vec<f32>>> for TabularData {
    fn next_batch(&mut self, batch_size: usize) -> Option<Vec<Vec<f32>>> {
        if self.current_idx >= self.data.len() {
            return None;
        }

        let end_idx = (self.current_idx + batch_size).min(self.data.len());
        let batch = self.data[self.current_idx..end_idx].to_vec();

        self.current_idx = end_idx;
        Some(batch)
    }

    fn reset(&mut self) {
        self.current_idx = 0;
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}
