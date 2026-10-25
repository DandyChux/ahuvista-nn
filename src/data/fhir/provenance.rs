use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::data::{CodeableConcept, Coding};

use super::resources::Reference;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    #[serde(rename = "resourceType")]
    pub resource_type: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    pub recorded: String,
    pub target: Vec<Reference>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<Vec<ProvenanceAgent>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity: Option<Vec<ProvenanceEntity>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<Vec<ProvenanceReason>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceAgent {
    pub who: Reference,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Vec<CodeableConcept>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceEntity {
    pub role: String,
    pub what: Reference,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceReason {
    pub coding: Vec<Coding>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl Provenance {
    pub fn for_risk_assessment(
        risk_assessment_reference: impl Into<String>,
        model_reference: impl Into<String>,
        model_display: impl Into<String>,
    ) -> Self {
        Self {
            resource_type: "Provenance".to_string(),
            id: None,
            recorded: Utc::now().to_rfc3339(),
            target: vec![Reference::new(risk_assessment_reference)],
            agent: Some(vec![ProvenanceAgent {
                who: Reference::with_display(model_reference, model_display),
                role: None,
            }]),
            entity: None,
            reason: None,
        }
    }

    pub fn with_source(mut self, source_reference: impl Into<String>) -> Self {
        self.entity = Some(vec![ProvenanceEntity {
            role: "source".to_string(),
            what: Reference::new(source_reference),
        }]);

        self
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
}
