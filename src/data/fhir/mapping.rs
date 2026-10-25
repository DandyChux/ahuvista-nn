use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

use crate::error::FhirError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirMappingConfig {
    #[serde(default)]
    pub metadata: MappingMetadata,
    #[serde(default)]
    pub static_features: Vec<StaticFeatureMapping>,
    #[serde(default)]
    pub dynamic_features: Vec<DynamicFeatureMapping>,
    #[serde(default)]
    pub condition_flags: Vec<ConditionMapping>,
    #[serde(default)]
    pub text_document_types: Vec<String>,
    #[serde(default = "default_sequence_length")]
    pub max_sequence_length: usize,
}

fn default_sequence_length() -> usize { 128 }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MappingMetadata {
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub based_on_training_data: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticFeatureMapping {
    pub system: Option<String>,
    pub code: String,
    pub feature_index: usize,
    pub feature_name: String,
    #[serde(default)]
    pub default_value: f32,
    #[serde(default)]
    pub unit_conversion: Option<UnitConversion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicFeatureMapping {
    pub system: Option<String>,
    pub code: String,
    pub feature_index: usize,
    pub feature_name: String,
    #[serde(default)]
    pub default_value: f32,
    #[serde(default)]
    pub unit_conversion: Option<UnitConversion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitConversion {
    pub from_unit: String,
    pub to_unit: String,
    pub multiplier: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionMapping {
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub snomed_code: Option<String>,
    #[serde(default)]
    pub icd10_code: Option<String>,
    pub feature_index: usize,
    pub feature_name: String,
}

impl ConditionMapping {
    pub fn matches(&self, system: Option<&str>, code: &str) -> bool {
        if self.system.as_deref().is_some_and(|expected| Some(expected) != system) {
            return false;
        }

        [self.code.as_deref(), self.snomed_code.as_deref(), self.icd10_code.as_deref()]
            .into_iter()
            .flatten()
            .any(|candidate| candidate == code)
    }
}

impl FhirMappingConfig {
    pub fn load(path: &Path) -> Result<Self, FhirError> {
        let content = fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), FhirError> {
        validate_indices("static", self.static_features.iter().map(|mapping| mapping.feature_index))?;
        validate_indices("dynamic", self.dynamic_features.iter().map(|mapping| mapping.feature_index))?;

        for mapping in &self.condition_flags {
            if mapping.code.is_none() && mapping.snomed_code.is_none() && mapping.icd10_code.is_none() {
                return Err(FhirError::InvalidMapping(format!(
                    "condition mapping '{}' has no code",
                    mapping.feature_name
                )));
            }
        }

        Ok(())
    }

    pub fn static_feature_count(&self) -> usize {
        self.static_features.iter().map(|mapping| mapping.feature_index + 1).max().unwrap_or(0)
    }

    pub fn dynamic_feature_count(&self) -> usize {
        self.dynamic_features.iter().map(|mapping| mapping.feature_index + 1).max().unwrap_or(0)
    }
}

fn validate_indices<I>(kind: &str, indices: I) -> Result<(), FhirError>
where
    I: IntoIterator<Item = usize>,
{
    let mut seen = std::collections::HashSet::new();
    for index in indices {
        if !seen.insert(index) {
            return Err(FhirError::InvalidMapping(format!("duplicate {kind} feature index {index}")));
        }
    }
    Ok(())
}
