use std::fmt::{self, Display, Formatter};
use std::io;

#[derive(Debug)]
pub enum FhirError {
    Io(io::Error),
    Json(serde_json::Error),
    MissingPatient,
    InvalidResource(String),
    InvalidMapping(String),
    UnsupportedResource(String),
}

impl fmt::Display for FhirError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "FHIR I/O error: {error}"),
            Self::Json(error) => write!(formatter, "FHIR JSON error: {error}"),
            Self::MissingPatient => {
                write!(formatter, "FHIR Bundle contains no Patient")
            }
            Self::InvalidResource(message) => {
                write!(formatter, "Invalid FHIR resource: {message}")
            }
            Self::InvalidMapping(message) => {
                write!(formatter, "Invalid FHIR mapping: {message}")
            }
            Self::UnsupportedResource(resource) => {
                write!(formatter, "Unsupported FHIR resource: {resource}")
            }
        }
    }
}

impl std::error::Error for FhirError {}

impl From<io::Error> for FhirError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for FhirError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Debug)]
pub enum AppError {
    Config(String),
    Io(std::io::Error),
    DataError(String),
    ModelError(String),
    CsvError(csv::Error),
    JsonError(serde_json::Error),
    /// Domain Error is related to business logic or data validation
    DomainError(String),
    ImageError(image::ImageError),
    DatabaseError(rusqlite::Error),
    MigrationError(refinery::Error),
    FhirError(crate::data::FhirError),
    InvalidData(String),
}

impl Display for AppError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::Config(err) => write!(f, "Configuration error: {}", err),
            AppError::Io(err) => write!(f, "I/O error: {}", err),
            AppError::DataError(msg) => write!(f, "Data loading or preprocessing error: {}", msg),
            AppError::ModelError(msg) => write!(f, "Model training or prediction error: {}", msg),
            AppError::CsvError(msg) => write!(f, "CSV processing error: {}", msg),
            AppError::JsonError(msg) => write!(f, "JSON processing error: {}", msg),
            AppError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            AppError::DomainError(msg) => write!(f, "Domain error: {}", msg),
            AppError::ImageError(msg) => write!(f, "Image processing error: {}", msg),
            AppError::DatabaseError(msg) => write!(f, "SQLite error: {}", msg),
            AppError::MigrationError(msg) => write!(f, "Migration error: {}", msg),
            AppError::FhirError(err) => write!(f, "FHIR error: {}", err),
        }
    }
}

macro_rules! impl_from_error {
    // Multi-type syntax: impl_from_error! { Type1 => Variant1, Type2 => Variant2, ... }
    ($($error_type:ty => $variant:ident),+ $(,)?) => {
        $(
            impl From<$error_type> for AppError {
                fn from(err: $error_type) -> Self {
                    AppError::$variant(err)
                }
            }
        )+
    };

    // Single-type syntax: impl_from_error!(Type, Variant)
    ($error_type:ty, $variant:ident) => {
        impl From<$error_type> for AppError {
            fn from(err: $error_type) -> Self {
                AppError::$variant(err)
            }
        }
    };

    // String conversion syntax: impl_from_error!(Type => Variant as String)
    // Converts any error type to String using Display trait
    ($error_type:ty => $variant:ident as String) => {
        impl From<$error_type> for AppError {
            fn from(err: $error_type) -> Self {
                AppError::$variant(err.to_string())
            }
        }
    };
}

impl std::error::Error for AppError {}

impl_from_error!(std::io::Error, Io);
impl_from_error!(csv::Error, CsvError);
impl_from_error!(serde_json::Error, JsonError);
impl_from_error!(String, DomainError);
impl_from_error!(image::ImageError, ImageError);
impl_from_error!(std::string::FromUtf8Error => DomainError as String);
impl_from_error!(rusqlite::Error, DatabaseError);
impl_from_error!(refinery::Error, MigrationError);
impl_from_error!(&str => DomainError as String);

pub type AppResult<T, E = AppError> = Result<T, E>;
