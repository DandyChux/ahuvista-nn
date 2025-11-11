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

pub struct AuditLogger {
    log_path: PathBuf,
}

impl AuditLogger {
    pub fn new(log_path: impl AsRef<Path>) -> Self {
        Self {
            log_path: log_path.as_ref().to_path_buf(),
        }
    }

    pub fn log(&self, record: &PredictionAuditRecord) -> AppResult<()> {
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
