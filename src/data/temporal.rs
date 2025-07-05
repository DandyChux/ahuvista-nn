//! Temporal data processing module.
//!
//! Provides utilities for loading and preprocessing time series data
//! such as patient vitals, lab measurements over time, or sensor readings.

use crate::data::DataLoader;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// Structure for handling temporal (time series) data
pub struct TemporalData {
    /// Sequences of feature vectors, one sequence per sample
    pub sequences: Vec<Vec<Vec<f32>>>,

    /// Feature names for each time step
    pub feature_names: Vec<String>,

    /// Statistics for normalization (mean, std_dev) for each feature
    pub feature_stats: HashMap<String, (f32, f32)>,

    /// Maximum sequence length across all samples
    pub max_seq_length: usize,
}

impl TemporalData {
    /// Create a new empty TemporalData instance
    pub fn new() -> Self {
        TemporalData {
            sequences: Vec::new(),
            feature_names: Vec::new(),
            feature_stats: HashMap::new(),
            max_seq_length: 0,
        }
    }

    /// Load temporal data from CSV files
    ///
    /// # Arguments
    /// * `paths` - Paths to CSV files, one per sample
    /// * `has_header` - Whether files have header rows
    /// * `time_column` - Name or index of the column containing timestamps
    ///
    /// # Returns
    /// Result containing TemporalData or an error
    pub fn from_csv_files<P: AsRef<Path>>(
        paths: &[P],
        has_header: bool,
        time_column: Option<String>,
    ) -> io::Result<Self> {
        let mut all_sequences = Vec::with_capacity(paths.len());
        let mut feature_names = Vec::new();
        let mut all_values: HashMap<String, Vec<f32>> = HashMap::new();
        let mut max_length = 0;

        // Process each file (each represents one patient/sample)
        for (sample_idx, path) in paths.iter().enumerate() {
            let file = File::open(path)?;
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

            // Extract feature names
            let file_feature_names = if has_header {
                header_line
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect::<Vec<String>>()
            } else {
                // Generate default feature names
                let first_line = lines.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "Empty CSV file")
                })??;
                let num_features = first_line.split(',').count();

                (0..num_features)
                    .map(|i| format!("Feature{}", i))
                    .collect::<Vec<String>>()
            };

            // If this is the first file, set the feature names
            if sample_idx == 0 {
                feature_names = file_feature_names.clone();

                // Initialize value collectors for each feature
                for name in &feature_names {
                    all_values.insert(name.clone(), Vec::new());
                }
            } else if file_feature_names != feature_names {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "File {} has different features than previous files",
                        path.as_ref().display()
                    ),
                ));
            }

            // If we read the first line to count features, reset the reader
            if !has_header {
                let file = File::open(path)?;
                let reader = BufReader::new(file);
                lines = reader.lines();
            }

            // Find time column index if specified
            let time_col_idx = if let Some(time_col) = &time_column {
                if let Ok(idx) = time_col.parse::<usize>() {
                    Some(idx)
                } else {
                    feature_names.iter().position(|name| name == time_col)
                }
            } else {
                None
            };

            // Read and parse each line
            let mut sequence = Vec::new();

            for line_result in lines {
                let line = line_result?;
                let values: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();

                if values.len() != feature_names.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "Line has {} values, expected {}",
                            values.len(),
                            feature_names.len()
                        ),
                    ));
                }

                // Parse values to floats
                let mut feature_vector =
                    Vec::with_capacity(values.len() - time_col_idx.is_some() as usize);

                for (i, (name, value)) in feature_names.iter().zip(values.iter()).enumerate() {
                    // Skip time column if specified
                    if Some(i) == time_col_idx {
                        continue;
                    }

                    let parsed_value = value.parse::<f32>().unwrap_or(0.0);
                    feature_vector.push(parsed_value);

                    // Collect values for statistics calculation
                    all_values.get_mut(name).unwrap().push(parsed_value);
                }

                sequence.push(feature_vector);
            }

            // Update max sequence length
            max_length = max_length.max(sequence.len());

            all_sequences.push(sequence);
        }

        // Calculate statistics for each feature
        let mut feature_stats = HashMap::new();

        for (name, values) in &all_values {
            if values.is_empty() {
                feature_stats.insert(name.clone(), (0.0, 1.0));
                continue;
            }

            // Calculate mean
            let sum: f32 = values.iter().sum();
            let mean = sum / values.len() as f32;

            // Calculate standard deviation
            let variance: f32 =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
            let std_dev = variance.sqrt().max(1e-6); // Avoid division by zero

            feature_stats.insert(name.clone(), (mean, std_dev));
        }

        // Create temporal data object
        let mut temporal_data = TemporalData {
            sequences: all_sequences,
            feature_names,
            feature_stats,
            max_seq_length: max_length,
        };

        // Normalize sequences
        temporal_data.normalize_sequences();

        Ok(temporal_data)
    }

    /// Load multiple sequences from a single CSV file where each patient has multiple rows
    ///
    /// # Arguments
    /// * `path` - Path to CSV file
    /// * `has_header` - Whether the file has a header row
    /// * `patient_id_column` - Name or index of column containing patient IDs
    /// * `time_column` - Name or index of column containing timestamps
    ///
    /// # Returns
    /// Result containing TemporalData or an error
    pub fn from_single_csv<P: AsRef<Path>>(
        path: P,
        has_header: bool,
        patient_id_column: String,
        time_column: Option<String>,
    ) -> io::Result<Self> {
        let file = File::open(path)?;
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

        // Extract feature names
        let all_feature_names = if has_header {
            header_line
                .split(',')
                .map(|s| s.trim().to_string())
                .collect::<Vec<String>>()
        } else {
            // Generate default feature names
            let first_line = lines
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Empty CSV file"))??;
            let num_features = first_line.split(',').count();

            (0..num_features)
                .map(|i| format!("Feature{}", i))
                .collect::<Vec<String>>()
        };

        // Find column indices
        let patient_id_idx = if let Ok(idx) = patient_id_column.parse::<usize>() {
            idx
        } else {
            all_feature_names
                .iter()
                .position(|name| name == &patient_id_column)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Patient ID column '{}' not found", patient_id_column),
                    )
                })?
        };

        let time_col_idx = if let Some(time_col) = &time_column {
            if let Ok(idx) = time_col.parse::<usize>() {
                Some(idx)
            } else {
                all_feature_names.iter().position(|name| name == time_col)
            }
        } else {
            None
        };

        // Get feature names excluding patient ID and time
        let mut feature_names = Vec::new();
        for (i, name) in all_feature_names.iter().enumerate() {
            if i != patient_id_idx && Some(i) != time_col_idx {
                feature_names.push(name.clone());
            }
        }

        // If we read the first line to count features, reset the reader
        let mut lines = if !has_header {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            reader.lines()
        } else {
            // Skip header
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            let mut lines = reader.lines();
            lines.next();
            lines
        };

        // Group rows by patient ID
        let mut patient_data: HashMap<String, Vec<Vec<String>>> = HashMap::new();

        for line_result in lines {
            let line = line_result?;
            let values: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();

            if values.len() != all_feature_names.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Line has {} values, expected {}",
                        values.len(),
                        all_feature_names.len()
                    ),
                ));
            }

            let patient_id = values[patient_id_idx].clone();
            patient_data.entry(patient_id).or_default().push(values);
        }

        // Sort each patient's data by time if time column specified
        if let Some(time_idx) = time_col_idx {
            for sequences in patient_data.values_mut() {
                sequences.sort_by(|a, b| {
                    let time_a = &a[time_idx];
                    let time_b = &b[time_idx];

                    // Try parsing as numbers first
                    match (time_a.parse::<f32>(), time_b.parse::<f32>()) {
                        (Ok(ta), Ok(tb)) => {
                            ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
                        }
                        _ => time_a.cmp(time_b), // Fall back to string comparison
                    }
                });
            }
        }

        // Extract feature values for each patient
        let mut all_sequences = Vec::with_capacity(patient_data.len());
        let mut all_values: HashMap<String, Vec<f32>> = HashMap::new();
        let mut max_length = 0;

        // Initialize value collectors for each feature
        for name in &feature_names {
            all_values.insert(name.clone(), Vec::new());
        }

        for sequences in patient_data.values() {
            let mut patient_sequence = Vec::new();

            for row in sequences {
                let mut feature_vector = Vec::with_capacity(feature_names.len());

                for (i, name) in all_feature_names.iter().enumerate() {
                    if i != patient_id_idx && Some(i) != time_col_idx {
                        let parsed_value = row[i].parse::<f32>().unwrap_or(0.0);
                        feature_vector.push(parsed_value);

                        // Collect values for statistics calculation
                        if let Some(values) = all_values.get_mut(name) {
                            values.push(parsed_value);
                        }
                    }
                }

                patient_sequence.push(feature_vector);
            }

            // Update max sequence length
            max_length = max_length.max(patient_sequence.len());

            all_sequences.push(patient_sequence);
        }

        // Calculate statistics for each feature
        let mut feature_stats = HashMap::new();

        for (name, values) in &all_values {
            if values.is_empty() {
                feature_stats.insert(name.clone(), (0.0, 1.0));
                continue;
            }

            // Calculate mean
            let sum: f32 = values.iter().sum();
            let mean = sum / values.len() as f32;

            // Calculate standard deviation
            let variance: f32 =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
            let std_dev = variance.sqrt().max(1e-6); // Avoid division by zero

            feature_stats.insert(name.clone(), (mean, std_dev));
        }

        // Create temporal data object
        let mut temporal_data = TemporalData {
            sequences: all_sequences,
            feature_names,
            feature_stats,
            max_seq_length: max_length,
        };

        // Normalize sequences
        temporal_data.normalize_sequences();

        Ok(temporal_data)
    }

    /// Normalize all sequences using precomputed statistics
    fn normalize_sequences(&mut self) {
        for sequence in &mut self.sequences {
            for time_step in sequence {
                for (i, value) in time_step.iter_mut().enumerate() {
                    let name = &self.feature_names[i];
                    if let Some(&(mean, std_dev)) = self.feature_stats.get(name) {
                        *value = (*value - mean) / std_dev;
                    }
                }
            }
        }
    }

    /// Pad sequences to the maximum length
    pub fn pad_sequences(&mut self) {
        for sequence in &mut self.sequences {
            let seq_len = sequence.len();
            if seq_len < self.max_seq_length {
                let feature_dim = if sequence.is_empty() {
                    self.feature_names.len()
                } else {
                    sequence[0].len()
                };

                let padding = vec![0.0; feature_dim];
                for _ in 0..(self.max_seq_length - seq_len) {
                    sequence.push(padding.clone());
                }
            }
        }
    }

    /// Get the feature dimension
    pub fn feature_dim(&self) -> usize {
        self.feature_names.len()
    }

    /// Preprocess a new sequence (for inference)
    pub fn preprocess_sequence(&self, sequence: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut normalized = Vec::with_capacity(sequence.len());

        for time_step in &sequence {
            let mut norm_step = Vec::with_capacity(time_step.len());

            for (i, &value) in time_step.iter().enumerate() {
                let name = &self.feature_names[i];
                if let Some(&(mean, std_dev)) = self.feature_stats.get(name) {
                    norm_step.push((value - mean) / std_dev);
                } else {
                    norm_step.push(value);
                }
            }

            normalized.push(norm_step);
        }

        normalized
    }
}

impl DataLoader<Vec<Vec<Vec<f32>>>> for TemporalData {
    fn next_batch(&mut self, batch_size: usize) -> Option<Vec<Vec<Vec<f32>>>> {
        static mut CURRENT_IDX: usize = 0;

        unsafe {
            if CURRENT_IDX >= self.sequences.len() {
                return None;
            }

            let end_idx = (CURRENT_IDX + batch_size).min(self.sequences.len());
            let batch = self.sequences[CURRENT_IDX..end_idx].to_vec();

            CURRENT_IDX = end_idx;
            Some(batch)
        }
    }

    fn reset(&mut self) {
        unsafe {
            CURRENT_IDX = 0;
        }
    }

    fn len(&self) -> usize {
        self.sequences.len()
    }
}
