use serde::{Deserialize, Serialize};
use std::{collections::HashSet, fs, path::PathBuf, sync::LazyLock};

use crate::utils::error::AppResult;

/// Stores all configuration settings for the application.
#[derive(Debug, Clone)]
pub struct Settings {
    // Data settings
    pub data_dir: PathBuf,
    pub patient_id_column: String,

    // Training settings
    pub training: TrainingParams,

    // Model settings
    pub tabular_hidden_sizes: Vec<usize>,
    pub temporal_hidden_size: usize,
    pub text_embedding_dim: usize,

    // Modality settings
    pub modality_config: ModalityConfig,
}

impl Settings {
    /// Load settings from a JSON configuration file
    pub fn from_file(path: &str) -> AppResult<Self> {
        let contents = fs::read_to_string(path)?;
        let config: AppConfig = serde_json::from_str(&contents)?;
        Ok(config.into())
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("datasets"),
            patient_id_column: String::from("PatientID"),
            training: TrainingParams::default(),
            tabular_hidden_sizes: vec![128, 64],
            temporal_hidden_size: 128,
            text_embedding_dim: 128,
            modality_config: ModalityConfig::all(),
        }
    }
}

/// Configuration for which modalities to enable in the fusion network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityConfig {
    pub use_tabular: bool,
    pub use_temporal: bool,
    pub use_text: bool,
    pub use_image: bool,
}

impl ModalityConfig {
    /// Create a config with all modalities enabled
    pub fn all() -> Self {
        Self {
            use_tabular: true,
            use_temporal: true,
            use_text: true,
            use_image: true,
        }
    }

    /// Create a config with only specified modalities
    pub fn from_names(names: &[&str]) -> Self {
        let enabled: HashSet<&str> = names.iter().copied().collect();
        Self {
            use_tabular: enabled.contains("tabular"),
            use_temporal: enabled.contains("temporal"),
            use_text: enabled.contains("text"),
            use_image: enabled.contains("image"),
        }
    }

    /// Get the number of enabled modalities
    pub fn enabled_count(&self) -> usize {
        [
            self.use_tabular,
            self.use_temporal,
            self.use_text,
            self.use_image,
        ]
        .iter()
        .filter(|&&x| x)
        .count()
    }

    /// Validate that at least one modality is enabled
    pub fn validate(&self) -> Result<(), String> {
        if self.enabled_count() == 0 {
            Err("At least one modality must be enabled".to_string())
        } else {
            Ok(())
        }
    }

    /// Get list of enabled modality names
    pub fn enabled_modalities(&self) -> Vec<&str> {
        let mut modalities = Vec::new();
        if self.use_tabular {
            modalities.push("tabular");
        }
        if self.use_temporal {
            modalities.push("temporal");
        }
        if self.use_text {
            modalities.push("text");
        }
        if self.use_image {
            modalities.push("image");
        }
        modalities
    }
}

/// Full application configuration loaded from JSON file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub modalities: ModalityConfig,
    pub data_paths: DataPaths,
    pub training: TrainingParams,
    pub model: ModelParams,
}

/// Data file paths for each modality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPaths {
    pub data_dir: String,
    pub patient_id_column: String,

    // Optional paths - only used if corresponding modality is enabled
    pub tabular_files: Option<Vec<String>>,
    pub temporal_files: Option<Vec<String>>,
    pub text_files: Option<Vec<String>>,
    pub image_files: Option<Vec<String>>,
}

impl Default for DataPaths {
    fn default() -> Self {
        Self {
            data_dir: "datasets".to_string(),
            patient_id_column: "PatientID".to_string(),
            tabular_files: Some(vec!["tabular_data.csv".to_string()]),
            temporal_files: Some(vec!["temporal_data.csv".to_string()]),
            text_files: Some(vec!["text_data.txt".to_string()]),
            image_files: Some(vec!["images/".to_string()]),
        }
    }
}

/// Training hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    // Training settings
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub val_frequency: usize,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f32,

    // Privacy settings
    #[serde(default = "default_use_privacy")]
    pub use_privacy: bool,
    #[serde(default = "default_privacy_epsilon")]
    pub privacy_epsilon: f32,
    #[serde(default = "default_privacy_delta")]
    pub privacy_delta: f32,
    #[serde(default = "default_gradient_clip_norm")]
    pub gradient_clip_norm: f32,

    // Adaptive LR settings
    #[serde(default = "default_use_adaptive_lr")]
    pub use_adaptive_lr: bool,
    #[serde(default = "default_lr_decay_rate")]
    pub lr_decay_rate: f32,
    #[serde(default = "default_lr_decay_epochs", alias = "lr_decay_steps")]
    pub lr_decay_epochs: usize,
    #[serde(default = "default_min_learning_rate")]
    pub min_learning_rate: f32,
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            epochs: 20,
            batch_size: 256,
            learning_rate: 0.005,
            val_frequency: 1,
            weight_decay: 0.001,
            use_privacy: false,
            privacy_epsilon: 10.0,
            privacy_delta: 1e-5,
            gradient_clip_norm: 5.0,
            use_adaptive_lr: false,
            lr_decay_rate: 0.95,
            lr_decay_epochs: 5,
            min_learning_rate: 0.0001,
        }
    }
}

/// Model architecture parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParams {
    // Tabular network
    pub tabular_hidden_sizes: Vec<usize>,
    pub tabular_output_size: usize,

    // Temporal network
    pub temporal_hidden_size: usize,
    pub temporal_output_size: usize,

    // Text network
    pub text_embedding_dim: usize,
    pub text_hidden_size: usize,
    pub text_max_vocab: usize,

    // Image network
    pub image_channels: usize,
    pub image_height: usize,
    pub image_width: usize,

    // Classifier
    pub classifier_hidden_sizes: Vec<usize>,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            tabular_hidden_sizes: vec![64, 32],
            tabular_output_size: 16,
            temporal_hidden_size: 32,
            temporal_output_size: 16,
            text_embedding_dim: 50,
            text_hidden_size: 32,
            text_max_vocab: 5000,
            image_channels: 3,
            image_height: 224,
            image_width: 224,
            classifier_hidden_sizes: vec![64, 32],
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            modalities: ModalityConfig::all(),
            data_paths: DataPaths::default(),
            training: TrainingParams::default(),
            model: ModelParams::default(),
        }
    }
}

impl From<AppConfig> for Settings {
    fn from(config: AppConfig) -> Self {
        Settings {
            data_dir: PathBuf::from(config.data_paths.data_dir),
            patient_id_column: config.data_paths.patient_id_column,
            training: config.training, // Just move it directly!
            tabular_hidden_sizes: config.model.tabular_hidden_sizes,
            temporal_hidden_size: config.model.temporal_hidden_size,
            text_embedding_dim: config.model.text_embedding_dim,
            modality_config: config.modalities,
        }
    }
}

// Default functions
fn default_use_privacy() -> bool {
    false
}
fn default_privacy_epsilon() -> f32 {
    10.0
}
fn default_privacy_delta() -> f32 {
    1e-5
}
fn default_gradient_clip_norm() -> f32 {
    5.0
}
fn default_use_adaptive_lr() -> bool {
    true
}
fn default_lr_decay_rate() -> f32 {
    0.5
}
fn default_lr_decay_epochs() -> usize {
    5
}
fn default_min_learning_rate() -> f32 {
    0.0001
}
fn default_weight_decay() -> f32 {
    0.01
}

// pub static SETTINGS: LazyLock<Settings> = LazyLock::new(|| Settings::default());
