use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    pub reference: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
}

impl Reference {
    pub fn new(reference: impl Into<String>) -> Self {
        Self {
            reference: Some(reference.into()),
            display: None,
        }
    }

    pub fn with_display(reference: impl Into<String>, display: impl Into<String>) -> Self {
        Self {
            reference: Some(reference.into()),
            display: Some(display.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coding {
    pub system: Option<String>,
    pub version: Option<String>,
    pub code: Option<String>,
    pub display: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeableConcept {
    #[serde(default)]
    pub coding: Vec<Coding>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quantity {
    pub value: Option<f64>,
    pub unit: Option<String>,
    pub system: Option<String>,
    pub code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Period {
    pub start: Option<String>,
    pub end: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Extension {
    pub url: String,

    #[serde(rename = "valueString", skip_serializing_if = "Option::is_none")]
    pub value_string: Option<String>,

    #[serde(rename = "valueDecimal", skip_serializing_if = "Option::is_none")]
    pub value_decimal: Option<f32>,

    #[serde(rename = "valueInteger", skip_serializing_if = "Option::is_none")]
    pub value_integer: Option<i32>,

    #[serde(rename = "valueBoolean", skip_serializing_if = "Option::is_none")]
    pub value_boolean: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Patient {
    pub resource_type: String,
    pub id: Option<String>,
    pub birth_date: Option<String>,
    pub gender: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Observation {
    pub resource_type: String,
    pub id: Option<String>,
    pub status: String,
    pub category: Option<Vec<CodeableConcept>>,
    pub code: CodeableConcept,
    pub subject: Option<Reference>,
    pub effective_date_time: Option<String>,
    pub effective_period: Option<Period>,
    pub value_quantity: Option<Quantity>,
    pub value_codeable_concept: Option<CodeableConcept>,
    pub value_string: Option<String>,
    pub component: Option<Vec<ObservationComponent>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ObservationComponent {
    pub code: CodeableConcept,
    pub value_quantity: Option<Quantity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Condition {
    pub resource_type: String,
    pub id: Option<String>,
    pub clinical_status: Option<CodeableConcept>,
    pub verification_status: Option<CodeableConcept>,
    pub code: Option<CodeableConcept>,
    pub subject: Option<Reference>,
    pub onset_date_time: Option<String>,
    pub recorded_date: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentReference {
    pub resource_type: String,
    pub id: Option<String>,
    pub status: String,
    pub r#type: Option<CodeableConcept>,
    pub subject: Option<Reference>,
    pub date: Option<String>,
    pub content: Vec<DocumentContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentContent {
    pub attachment: Attachment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Attachment {
    pub content_type: Option<String>,
    pub data: Option<String>,
    pub url: Option<String>,
    pub title: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticReport {
    pub resource_type: String,
    pub id: Option<String>,
    pub status: String,
    pub code: CodeableConcept,
    pub subject: Option<Reference>,
    pub effective_date_time: Option<String>,
    pub issued: Option<String>,
    pub conclusion: Option<String>,
    pub presented_form: Option<Vec<Attachment>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImagingStudy {
    pub resource_type: String,
    pub id: Option<String>,
    pub status: String,
    pub subject: Option<Reference>,
    pub started: Option<String>,
    pub endpoint: Option<Vec<Reference>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Encounter {
    pub resource_type: String,
    pub id: Option<String>,
    pub status: String,
    pub class: Option<Coding>,
    pub subject: Option<Reference>,
    pub period: Option<Period>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "resourceType")]
pub enum Resource {
    Patient(Patient),
    Observation(Observation),
    Condition(Condition),
    DocumentReference(DocumentReference),
    DiagnosticReport(DiagnosticReport),
    ImagingStudy(ImagingStudy),
    Encounter(Encounter),

    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Bundle {
    pub resource_type: String,
    pub r#type: String,
    pub id: Option<String>,
    pub entry: Option<Vec<BundleEntry>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleEntry {
    pub resource: Option<Resource>,
}
