//! Static patient feature loading and preprocessing.
//!
//! This module handles non-recurring patient features such as demographics,
//! baseline genetics, pregnancy history, and admission-time information.
//!
//! It does not represent the complete model input. Dynamic observations are
//! combined with these features into `PatientSequence` by the database layer.

use crate::error::{AppError, AppResult};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct StaticFeatureData {
    /// Static feature vectors, one row per patient.
    pub data: Vec<Vec<f32>>,

    /// Patient IDs corresponding to each feature row.
    pub patient_ids: Vec<String>,

    /// Feature names in vector order.
    pub feature_names: Vec<String>,

    /// Numeric normalization statistics by feature name.
    pub numeric_stats: HashMap<String, (f32, f32)>,

    /// Categorical mappings by feature name.
    pub categorical_mappings: HashMap<String, HashMap<String, usize>>,

    /// Feature names treated as categorical.
    pub categorical_features: Vec<String>,

    pub current_idx: usize,
}

impl StaticFeatureData {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            patient_ids: Vec::new(),
            feature_names: Vec::new(),
            numeric_stats: HashMap::new(),
            categorical_mappings: HashMap::new(),
            categorical_features: Vec::new(),
            current_idx: 0,
        }
    }

    pub fn from_csv<P: AsRef<Path>>(
        path: P,
        has_header: bool,
        patient_id_column: &str,
        categorical_features: &[String],
    ) -> AppResult<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut first_line = String::new();

        reader.read_line(&mut first_line)?;

        if first_line.trim().is_empty() {
            return Err(AppError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "Empty static-feature CSV",
            )));
        }

        let headers: Vec<String> = if has_header {
            first_line
                .trim_end()
                .split(',')
                .map(|value| value.trim().to_string())
                .collect()
        } else {
            first_line
                .trim_end()
                .split(',')
                .enumerate()
                .map(|(index, _)| format!("Feature{index}"))
                .collect()
        };

        let patient_id_index = headers.iter().position(|name| name == patient_id_column);

        let feature_names: Vec<String> = headers
            .iter()
            .enumerate()
            .filter_map(|(index, name)| {
                if Some(index) == patient_id_index {
                    None
                } else {
                    Some(name.clone())
                }
            })
            .collect();

        let mut rows = Vec::new();
        let mut patient_ids = Vec::new();

        if !has_header {
            let values = first_line
                .trim_end()
                .split(',')
                .map(|value| value.trim().to_string())
                .collect::<Vec<_>>();

            Self::validate_row(&values, headers.len())?;

            let patient_id = patient_id_index
                .and_then(|index| values.get(index))
                .cloned()
                .unwrap_or_else(|| format!("sample-{}", rows.len()));

            patient_ids.push(patient_id);
            rows.push(values);
        }

        for line in reader.lines() {
            let line = line?;

            if line.trim().is_empty() {
                continue;
            }

            let values = line
                .split(',')
                .map(|value| value.trim().to_string())
                .collect::<Vec<_>>();

            Self::validate_row(&values, headers.len())?;

            let patient_id = patient_id_index
                .and_then(|index| values.get(index))
                .cloned()
                .unwrap_or_else(|| format!("sample-{}", rows.len()));

            patient_ids.push(patient_id);
            rows.push(values);
        }

        let categorical_set: std::collections::HashSet<String> =
            categorical_features.iter().cloned().collect();

        let mut result = Self {
            data: Vec::with_capacity(rows.len()),
            patient_ids,
            feature_names: feature_names.clone(),
            numeric_stats: HashMap::new(),
            categorical_mappings: HashMap::new(),
            categorical_features: Vec::new(),
            current_idx: 0,
        };

        let feature_indices: Vec<usize> = headers
            .iter()
            .enumerate()
            .filter_map(|(index, _)| {
                if Some(index) == patient_id_index {
                    None
                } else {
                    Some(index)
                }
            })
            .collect();

        for (feature_position, feature_name) in feature_names.iter().enumerate() {
            let source_index = feature_indices[feature_position];

            let explicitly_categorical = categorical_set.contains(feature_name);

            let contains_non_numeric = rows.iter().any(|row| {
                let value = row[source_index].trim();
                !value.is_empty() && value.parse::<f32>().is_err()
            });

            if explicitly_categorical || contains_non_numeric {
                result.categorical_features.push(feature_name.clone());

                let mut mapping = HashMap::new();

                for row in &rows {
                    let value = row[source_index].clone();

                    if !mapping.contains_key(&value) {
                        let next_index = mapping.len();
                        mapping.insert(value, next_index);
                    }
                }

                result
                    .categorical_mappings
                    .insert(feature_name.clone(), mapping);
            } else {
                let values: Vec<f32> = rows
                    .iter()
                    .filter_map(|row| row[source_index].parse::<f32>().ok())
                    .collect();

                let mean = if values.is_empty() {
                    0.0
                } else {
                    values.iter().sum::<f32>() / values.len() as f32
                };

                let variance = if values.len() > 1 {
                    values
                        .iter()
                        .map(|value| (value - mean).powi(2))
                        .sum::<f32>()
                        / (values.len() - 1) as f32
                } else {
                    0.0
                };

                result
                    .numeric_stats
                    .insert(feature_name.clone(), (mean, variance.sqrt().max(1e-6)));
            }
        }

        for row in &rows {
            let mut processed = Vec::with_capacity(feature_indices.len());

            for (feature_position, feature_name) in feature_names.iter().enumerate() {
                let source_index = feature_indices[feature_position];
                let raw_value = row[source_index].trim();

                if result
                    .categorical_features
                    .iter()
                    .any(|name| name == feature_name)
                {
                    let mapping =
                        result
                            .categorical_mappings
                            .get(feature_name)
                            .ok_or_else(|| {
                                AppError::InvalidData(format!(
                                    "Missing categorical mapping for '{}'",
                                    feature_name
                                ))
                            })?;

                    let encoded = mapping.get(raw_value).copied().unwrap_or(0);

                    processed.push(encoded as f32);
                } else {
                    let value = raw_value.parse::<f32>().unwrap_or(0.0);
                    let (mean, std) = result
                        .numeric_stats
                        .get(feature_name)
                        .copied()
                        .unwrap_or((0.0, 1.0));

                    processed.push((value - mean) / std.max(1e-6));
                }
            }

            result.data.push(processed);
        }

        Ok(result)
    }

    fn validate_row(row: &[String], expected_columns: usize) -> AppResult<()> {
        if row.len() != expected_columns {
            return Err(AppError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Static-feature row has {} values; expected {}",
                    row.len(),
                    expected_columns
                ),
            )));
        }

        Ok(())
    }

    pub fn feature_dim(&self) -> usize {
        self.feature_names.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn preprocess_row(&self, row: &[String]) -> AppResult<Vec<f32>> {
        if row.len() != self.feature_names.len() {
            return Err(AppError::InvalidData(format!(
                "Expected {} static features, got {}",
                self.feature_names.len(),
                row.len()
            )));
        }

        let mut result = Vec::with_capacity(row.len());

        for (index, value) in row.iter().enumerate() {
            let name = &self.feature_names[index];

            if self
                .categorical_features
                .iter()
                .any(|feature| feature == name)
            {
                let encoded = self
                    .categorical_mappings
                    .get(name)
                    .and_then(|mapping| mapping.get(value))
                    .copied()
                    .unwrap_or(0);

                result.push(encoded as f32);
            } else {
                let parsed = value.parse::<f32>().unwrap_or(0.0);
                let (mean, std) = self.numeric_stats.get(name).copied().unwrap_or((0.0, 1.0));

                result.push((parsed - mean) / std.max(1e-6));
            }
        }

        Ok(result)
    }
}

impl Default for StaticFeatureData {
    fn default() -> Self {
        Self::new()
    }
}
