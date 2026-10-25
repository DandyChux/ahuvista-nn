use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Statistics for one numeric feature.
///
/// These values must be calculated from training data only and persisted
/// with the trained model artifact for inference.
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
            max: 0.0,
            count: 0,
        }
    }
}

impl FeatureStats {
    pub fn new(name: impl Into<String>, values: &[f32]) -> Self {
        if values.is_empty() {
            return Self {
                name: name.into(),
                ..Self::default()
            };
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;

        let variance = if values.len() > 1 {
            values
                .iter()
                .map(|value| (*value - mean).powi(2))
                .sum::<f32>()
                / (values.len() - 1) as f32
        } else {
            0.0
        };

        Self {
            name: name.into(),
            mean,
            std: variance.sqrt().max(1e-6),
            min: values.iter().copied().fold(f32::INFINITY, f32::min),
            max: values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            count: values.len(),
        }
    }

    pub fn normalize(&self, value: f32) -> f32 {
        if self.std.is_finite() && self.std > 1e-6 {
            (value - self.mean) / self.std
        } else {
            value - self.mean
        }
    }

    pub fn denormalize(&self, value: f32) -> f32 {
        value * self.std.max(1e-6) + self.mean
    }
}

/// Supported normalization strategies.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum NormalizationMethod {
    #[default]
    ZScore,
    MinMax,
    None,
}

/// Persisted normalization metadata.
///
/// This is useful for raw-data preprocessing, but the model artifact should
/// also persist ordered static and dynamic statistics directly.
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
        Self {
            missing_value: 0.0,
            ..Self::default()
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "Normalizer: {} features, method={:?}",
            self.feature_order.len(),
            self.method
        )
    }

    pub fn normalize(&self, feature_name: &str, value: f32) -> f32 {
        let Some(stats) = self.feature_stats.get(feature_name) else {
            return value;
        };

        match self.method {
            NormalizationMethod::ZScore => stats.normalize(value),

            NormalizationMethod::MinMax => {
                let range = stats.max - stats.min;

                if range.abs() <= 1e-6 {
                    0.0
                } else {
                    (value - stats.min) / range
                }
            }

            NormalizationMethod::None => value,
        }
    }
}
