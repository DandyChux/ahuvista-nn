use serde::{Deserialize, Serialize};

use crate::models::PatientSequence;

/// Input data structure for predictions
#[derive(Debug, Clone, Deserialize)]
pub struct PredictionInput {
    #[serde(default)]
    pub sequence: Option<PatientSequence>,

    /// Text data as raw string (optional if modality not enabled)
    #[serde(default)]
    pub text: Option<String>,

    /// Path to image file (optional if modality not enabled)
    #[serde(default)]
    pub image_path: Option<String>,

    /// Optional patient identifier (for logging/tracking)
    #[serde(default)]
    pub patient_id: Option<String>,
}

/// Internal prediction result
///
/// This is not the FHIR wire format. It is converted to a FHIR
/// RiskAssessment bu the FHIR output layer
#[derive(Debug, Clone, Serialize)]
pub struct PredictionOutput {
    /// Raw model output score (0-1)
    pub risk_score: f32,

    /// Calibrated risk score if calibration was applied
    pub calibrated_score: Option<f32>,

    /// Risk category (MINIMAL, LOW, MODERATE, HIGH, CRITICAL)
    pub risk_category: String,

    /// Confidence in the prediction
    pub confidence: f32,

    /// Which modalities were used for this prediction
    pub modalities_used: Vec<String>,

    /// Optional patient identifier
    pub patient_id: Option<String>,

    /// Timestamp of prediction
    pub timestamp: String,
}
