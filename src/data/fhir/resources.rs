use serde::Deserialize;

// Core FHIR types
#[derive(Debug, Deserialize, Clone)]
pub struct CodeableConcept {
    pub coding: Option<Vec<Coding>>,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Coding {
    pub system: Option<String>,
    pub code: Option<String>,
    pub display: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Reference {
    pub reference: Option<String>,
    pub display: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Quantity {
    pub value: Option<f64>,
    pub unit: Option<String>,
    pub system: Option<String>,
    pub code: Option<String>,
}

// Patient Resource
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PatientResource {
    pub id: Option<String>,
    pub birth_date: Option<String>, // Calculate age from this
    pub gender: Option<String>,
    // Extensions often contain race, ethnicity in US Core
}

// Observation Resource (most complex - handles vitals AND labs)
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ObservationResource {
    pub id: Option<String>,
    pub status: String,
    pub category: Option<Vec<CodeableConcept>>,
    pub code: CodeableConcept, // LOINC code here
    pub subject: Option<Reference>,
    pub effective_date_time: Option<String>,
    pub effective_period: Option<Period>,
    pub value_quantity: Option<Quantity>,
    pub value_codeable_concept: Option<CodeableConcept>,
    pub value_string: Option<String>,
    pub component: Option<Vec<ObservationComponent>>, // For BP (systolic/diastolic)
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ObservationComponent {
    pub code: CodeableConcept,
    pub value_quantity: Option<Quantity>,
}

// DocumentReference (clinical notes)
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct DocumentReferenceResource {
    pub id: Option<String>,
    pub status: String,
    pub r#type: Option<CodeableConcept>,
    pub subject: Option<Reference>,
    pub date: Option<String>,
    pub content: Vec<DocumentContent>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct DocumentContent {
    pub attachment: Attachment,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Attachment {
    pub content_type: Option<String>,
    pub data: Option<String>, // Base64 encoded
    pub url: Option<String>,  // Or URL reference
}

// Bundle (container)
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Bundle {
    pub resource_type: String,
    pub r#type: String,
    pub entry: Option<Vec<BundleEntry>>,
}

#[derive(Debug, Deserialize)]
pub struct BundleEntry {
    pub resource: Option<FhirResource>,
}

// Period type - used for time ranges in FHIR
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Period {
    pub start: Option<String>, // ISO 8601 datetime
    pub end: Option<String>,
}

// FhirResource enum - discriminated union of all supported resources
// The #[serde(tag = "resourceType")] attribute uses the JSON's resourceType field
// to determine which variant to deserialize into
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "resourceType")]
pub enum FhirResource {
    Patient(PatientResource),
    Observation(ObservationResource),
    Condition(ConditionResource),
    DocumentReference(DocumentReferenceResource),
    DiagnosticReport(DiagnosticReportResource),
    ImagingStudy(ImagingStudyResource),
    Encounter(EncounterResource),
    // Catch-all for resources we don't need to parse
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ConditionResource {
    pub id: Option<String>,
    pub clinical_status: Option<CodeableConcept>,
    pub verification_status: Option<CodeableConcept>,
    pub code: Option<CodeableConcept>, // SNOMED/ICD-10 code
    pub subject: Option<Reference>,
    pub onset_date_time: Option<String>,
    pub recorded_date: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticReportResource {
    pub id: Option<String>,
    pub status: String,
    pub code: CodeableConcept,
    pub subject: Option<Reference>,
    pub effective_date_time: Option<String>,
    pub issued: Option<String>,
    pub conclusion: Option<String>, // Text summary - useful for text modality
    pub presented_form: Option<Vec<Attachment>>, // Full report attachment
    pub media: Option<Vec<DiagnosticMedia>>, // Image references
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticMedia {
    pub comment: Option<String>,
    pub link: Reference, // Reference to Media or ImagingStudy
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ImagingStudyResource {
    pub id: Option<String>,
    pub status: String,
    pub subject: Reference,
    pub started: Option<String>,
    pub endpoint: Option<Vec<Reference>>, // DICOM endpoints
    pub series: Option<Vec<ImagingSeries>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ImagingSeries {
    pub uid: String,
    pub modality: Coding,
    pub instance: Option<Vec<ImagingInstance>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ImagingInstance {
    pub uid: String,
    pub sop_class: Coding,
    pub title: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct EncounterResource {
    pub id: Option<String>,
    pub status: String,
    pub class: Option<Coding>,
    pub subject: Option<Reference>,
    pub period: Option<Period>,
}

/// FHIR Annotation type - for notes/comments
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Annotation {
    /// Who authored the annotation
    pub author_reference: Option<Reference>,
    pub author_string: Option<String>,

    /// When the annotation was made
    pub time: Option<String>,

    /// The annotation text (required)
    pub text: String,
}

impl Annotation {
    /// Create a simple text annotation
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            author_reference: None,
            author_string: None,
            time: None,
            text: text.into(),
        }
    }
}
