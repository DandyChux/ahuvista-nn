use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize, Serialize)]
pub struct FhirMappingConfig {
    /// Maps LOINC codes to tabular feature positions
    pub tabular_features: Vec<TabularFeatureMapping>,

    /// Maps LOINC codes to temporal feature positions
    pub temporal_features: TemporalFeatureMapping,

    /// Condition codes that map to binary flags
    pub condition_flags: Vec<ConditionMapping>,

    /// Document types to extract for text modality
    pub text_document_types: Vec<String>,

    /// Expected feature vector size (for validation)
    pub expected_tabular_size: usize,
    pub expected_temporal_feature_count: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TabularFeatureMapping {
    pub loinc_code: String,
    pub snomed_code: Option<String>, // Alternative code
    pub feature_index: usize,
    pub feature_name: String,
    pub default_value: f32, // If missing
    pub unit_conversion: Option<UnitConversion>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct UnitConversion {
    pub from_unit: String,
    pub to_unit: String,
    pub multiplier: f32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TemporalFeatureMapping {
    /// LOINC codes for temporal features, in order
    pub feature_codes: Vec<String>,
    /// Maximum sequence length
    pub max_sequence_length: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ConditionMapping {
    pub snomed_code: String,
    pub icd10_code: Option<String>,
    pub feature_index: usize,
    pub feature_name: String,
}

impl FhirMappingConfig {
    pub fn load(path: &std::path::Path) -> Result<Self, FhirError> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content).map_err(FhirError::from)
    }
}
