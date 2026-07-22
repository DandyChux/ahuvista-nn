use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::utils::error::AppResult;

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionAuditRecord {
    pub event_timestamp: DateTime<Utc>,
    pub model_version: String,
    pub patient_id_hash: String, // Hashed to protect privacy
    pub input_data_hash: String,
    pub raw_prediction: f32,
    pub calibrated_prediction: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingAuditRecord {
    pub event_timestamp: DateTime<Utc>,
    pub model_version: String,
    pub training_data_hash: String,
    pub training_loss: f32,
    pub training_accuracy: f32,
    pub validation_loss: f32,
    pub validation_accuracy: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingLogEntry {
    pub timestamp: DateTime<Utc>,
    pub level: String,
    pub message: String,
    pub epoch: Option<usize>,
    pub batch: Option<usize>,
    pub loss: Option<f32>,
    pub metrics: Option<serde_json::Value>,
}

pub struct AuditLogger {
    log_path: PathBuf,
}

impl AuditLogger {
    pub fn new(log_path: impl AsRef<Path>) -> Self {
        Self {
            log_path: log_path.as_ref().to_path_buf(),
        }
    }

    pub fn log<T: Serialize>(&self, record: &T) -> AppResult<()> {
        let json_record = serde_json::to_string(record)?;
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(&self.log_path)?;

        writeln!(file, "{}", json_record)?;

        Ok(())
    }
}
