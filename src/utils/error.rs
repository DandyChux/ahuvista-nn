use std::fmt::{Display, Formatter};

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
    SqliteError(rusqlite::Error),
    #[cfg(feature = "fhir")]
    FhirError(crate::data::FhirError),
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
            AppError::DomainError(msg) => write!(f, "Domain error: {}", msg),
            AppError::ImageError(msg) => write!(f, "Image processing error: {}", msg),
            AppError::SqliteError(msg) => write!(f, "SQLite error: {}", msg),
            #[cfg(feature = "fhir")]
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
impl_from_error!(rusqlite::Error, SqliteError);
#[cfg(feature = "fhir")]
impl_from_error!(crate::data::FhirError, FhirError);
impl_from_error!(&str => DomainError as String);

pub type AppResult<T, E = AppError> = Result<T, E>;
