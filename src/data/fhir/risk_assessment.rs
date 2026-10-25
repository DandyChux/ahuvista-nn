use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::prediction::PredictionOutput;

use super::resources::{CodeableConcept, Coding, Extension, Period, Reference};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    #[serde(rename = "resourceType")]
    pub resource_type: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    pub status: String,
    pub subject: Reference,

    #[serde(rename = "occurrenceDateTime")]
    pub occurrence_date_time: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<CodeableConcept>,

    pub prediction: Vec<FhirRiskPrediction>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub basis: Option<Vec<Reference>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub extension: Option<Vec<Extension>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirRiskPrediction {
    pub outcome: CodeableConcept,

    #[serde(rename = "probabilityDecimal", skip_serializing_if = "Option::is_none")]
    pub probability_decimal: Option<f32>,

    #[serde(rename = "qualitativeRisk", skip_serializing_if = "Option::is_none")]
    pub qualitative_risk: Option<CodeableConcept>,

    #[serde(rename = "whenPeriod", skip_serializing_if = "Option::is_none")]
    pub when_period: Option<Period>,
}

impl RiskAssessment {
    pub fn new(patient_id: &str, prediction: FhirRiskPrediction) -> Self {
        Self {
            resource_type: "RiskAssessment".to_string(),
            id: None,
            status: "final".to_string(),
            subject: Reference::new(format!("Patient/{patient_id}")),
            occurrence_date_time: Utc::now().to_rfc3339(),
            method: None,
            prediction: vec![prediction],
            basis: None,
            extension: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_method(
        mut self,
        system: impl Into<String>,
        code: impl Into<String>,
        display: impl Into<String>,
    ) -> Self {
        self.method = Some(CodeableConcept {
            coding: vec![Coding {
                version: None,
                system: Some(system.into()),
                code: Some(code.into()),
                display: Some(display.into()),
            }],
            text: None,
        });

        self
    }

    pub fn with_basis(mut self, references: impl IntoIterator<Item = String>) -> Self {
        let basis: Vec<Reference> = references.into_iter().map(Reference::new).collect();

        if !basis.is_empty() {
            self.basis = Some(basis);
        }

        self
    }
}

impl PredictionOutput {
    pub fn to_fhir_risk_assessment(
        &self,
        outcome_system: &str,
        outcome_code: &str,
        outcome_display: &str,
        model_system: &str,
        model_code: &str,
        model_display: &str,
    ) -> RiskAssessment {
        let patient_id = self.patient_id.as_deref().unwrap_or("unknown");

        let probability = self
            .calibrated_score
            .unwrap_or(self.risk_score)
            .clamp(0.0, 1.0);

        let prediction = FhirRiskPrediction {
            outcome: CodeableConcept {
                coding: vec![Coding {
                    version: None,
                    system: Some(outcome_system.to_string()),
                    code: Some(outcome_code.to_string()),
                    display: Some(outcome_display.to_string()),
                }],
                text: Some(outcome_display.to_string()),
            },
            probability_decimal: Some(probability),
            qualitative_risk: Some(qualitative_risk_concept(&self.risk_category)),
            when_period: None,
        };

        let mut assessment = RiskAssessment::new(patient_id, prediction).with_method(
            model_system,
            model_code,
            model_display,
        );

        let mut extensions = Vec::new();

        extensions.push(Extension {
            url: "https://ahuvista.example.org/fhir/StructureDefinition/model-version".to_string(),
            value_string: Some(model_code.to_string()),
            value_decimal: None,
            value_integer: None,
            value_boolean: None,
        });

        extensions.push(Extension {
            url: "https://ahuvista.example.org/fhir/StructureDefinition/confidence".to_string(),
            value_string: None,
            value_decimal: Some(self.confidence),
            value_integer: None,
            value_boolean: None,
        });

        extensions.push(Extension {
            url: "https://ahuvista.example.org/fhir/StructureDefinition/modalities-used"
                .to_string(),
            value_string: Some(self.modalities_used.join(",")),
            value_decimal: None,
            value_integer: None,
            value_boolean: None,
        });

        assessment.extension = Some(extensions);
        assessment.occurrence_date_time = self.timestamp.clone();

        assessment
    }
}

fn qualitative_risk_concept(category: &str) -> CodeableConcept {
    let normalized = category.to_lowercase();

    let code = match normalized.as_str() {
        "minimal" => "low",
        "low" => "low",
        "moderate" => "moderate",
        "high" => "high",
        "critical" => "high",
        _ => "unknown",
    };

    CodeableConcept {
        coding: vec![Coding {
            version: None,
            system: Some("http://terminology.hl7.org/CodeSystem/risk-probability".to_string()),
            code: Some(code.to_string()),
            display: Some(category.to_string()),
        }],
        text: Some(category.to_string()),
    }
}
