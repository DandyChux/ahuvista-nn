use chrono::Utc;
use serde::Serialize;

use crate::{
    core::PredictionOutput,
    data::resources::{Annotation, CodeableConcept, Coding, Reference},
};

/// Generates FHIR RiskAssessment from prediction output
pub struct RiskAssessmentBuilder {
    /// Your system's identifier
    pub system_url: String,
    /// Method code for your algorithm
    pub method_code: String,
    pub method_display: String,
}

impl RiskAssessmentBuilder {
    pub fn build(
        &self,
        prediction: &PredictionOutput,
        patient_reference: &str,
        encounter_reference: Option<&str>,
        basis_references: &[String], // Observations used as input
    ) -> RiskAssessmentOutput {
        RiskAssessmentOutput {
            resource_type: "RiskAssessment".to_string(),
            id: generate_uuid(),
            status: "final".to_string(),
            method: CodeableConcept {
                coding: Some(vec![Coding {
                    system: Some(self.system_url.clone()),
                    code: Some(self.method_code.clone()),
                    display: Some(self.method_display.clone()),
                }]),
                text: Some("Ahuvista-NN Maternal Risk Prediction".to_string()),
            },
            code: CodeableConcept {
                coding: Some(vec![Coding {
                    system: Some("http://snomed.info/sct".to_string()),
                    code: Some("313029009".to_string()), // "Risk assessment"
                    display: Some("Maternal mortality risk assessment".to_string()),
                }]),
                text: None,
            },
            subject: Reference {
                reference: Some(patient_reference.to_string()),
                display: prediction.patient_id.clone(),
            },
            encounter: encounter_reference.map(|e| Reference {
                reference: Some(e.to_string()),
                display: None,
            }),
            occurrence_date_time: Some(Utc::now().to_rfc3339()),
            performer: Reference {
                reference: Some("Device/ahuvista-nn".to_string()),
                display: Some("Ahuvista-NN Risk Prediction System".to_string()),
            },
            basis: basis_references
                .iter()
                .map(|r| Reference {
                    reference: Some(r.clone()),
                    display: None,
                })
                .collect(),
            prediction: vec![RiskPrediction {
                outcome: CodeableConcept {
                    coding: Some(vec![Coding {
                        system: Some("http://snomed.info/sct".to_string()),
                        code: Some("87309006".to_string()), // "Maternal death"
                        display: Some("Maternal mortality".to_string()),
                    }]),
                    text: None,
                },
                probability_decimal: Some(prediction.risk_score),
                qualitative_risk: CodeableConcept {
                    coding: Some(vec![Coding {
                        system: Some(
                            "http://terminology.hl7.org/CodeSystem/risk-probability".to_string(),
                        ),
                        code: Some(risk_category_to_fhir_code(&prediction.risk_category)),
                        display: Some(prediction.risk_category.clone()),
                    }]),
                    text: Some(prediction.risk_category.clone()),
                },
                rationale: Some(format!(
                    "Prediction based on {} modalities. Confidence: {:.1}%",
                    prediction.modalities_used.join(", "),
                    prediction.confidence * 100.0
                )),
            }],
            note: prediction.calibrated_score.map(|cal| {
                vec![Annotation {
                    text: format!("Calibrated score: {:.4}", cal),
                    time: Some(Utc::now().to_string()),
                    author_reference: None,
                    author_string: None,
                }]
            }),
        }
    }
}

fn risk_category_to_fhir_code(category: &str) -> String {
    match category.to_uppercase().as_str() {
        "MINIMAL" => "negligible",
        "LOW" => "low",
        "MODERATE" => "moderate",
        "HIGH" => "high",
        "CRITICAL" => "certain",
        _ => "moderate",
    }
    .to_string()
}
