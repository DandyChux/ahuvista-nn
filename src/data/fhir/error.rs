use std::fmt;

#[derive(Debug)]
pub enum FhirError {
    /// Missing required resource in bundle
    MissingResource { resource_type: String },

    /// Missing required field in resource
    MissingField { resource: String, field: String },

    /// Invalid or unrecognized code
    InvalidCode { system: String, code: String },

    /// Feature mapping not found for code
    UnmappedCode { code: String, system: String },

    /// Date/time parsing error
    DateParseError { value: String, message: String },

    /// JSON parsing error
    JsonError(serde_json::Error),

    /// IO error (file reading, etc.)
    IoError(std::io::Error),

    /// HTTP error (when fetching from FHIR server)
    HttpError(String),

    /// Configuration error
    ConfigError(String),

    /// Validation error
    ValidationError { message: String },
}

impl fmt::Display for FhirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FhirError::MissingResource { resource_type } => {
                write!(f, "Missing required FHIR resource: {}", resource_type)
            }
            FhirError::MissingField { resource, field } => {
                write!(f, "Missing required field '{}' in {}", field, resource)
            }
            FhirError::InvalidCode { system, code } => {
                write!(f, "Invalid code '{}' in system '{}'", code, system)
            }
            FhirError::UnmappedCode { code, system } => {
                write!(
                    f,
                    "No feature mapping found for code '{}' ({})",
                    code, system
                )
            }
            FhirError::DateParseError { value, message } => {
                write!(f, "Failed to parse date '{}': {}", value, message)
            }
            FhirError::JsonError(e) => write!(f, "JSON error: {}", e),
            FhirError::IoError(e) => write!(f, "IO error: {}", e),
            FhirError::HttpError(msg) => write!(f, "HTTP error: {}", msg),
            FhirError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            FhirError::ValidationError { message } => {
                write!(f, "Validation error: {}", message)
            }
        }
    }
}

impl std::error::Error for FhirError {}

impl From<serde_json::Error> for FhirError {
    fn from(err: serde_json::Error) -> Self {
        FhirError::JsonError(err)
    }
}

impl From<std::io::Error> for FhirError {
    fn from(err: std::io::Error) -> Self {
        FhirError::IoError(err)
    }
}

// Convenience type alias
pub type FhirResult<T> = Result<T, FhirError>;
