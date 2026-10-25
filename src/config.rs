use serde::{Deserialize, Serialize};
use std::{collections::HashSet, fs, path::PathBuf};

use crate::error::AppResult;

/// Stores all configuration settings for the application.
#[derive(Debug, Clone)]
pub struct Settings {
    // Data settings
    pub data_dir: PathBuf,
    pub patient_id_column: String,

    // Training settings
    pub training: TrainingParams,

    // Model settings
    pub model: ModelParams,

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
            patient_id_column: String::from("subject_id"),
            training: TrainingParams::default(),
            model: ModelParams::default(),
            modality_config: ModalityConfig::builder().use_sequence(true).use_text(true).build(),
        }
    }
}

/// Configuration for which modalities to enable in the fusion network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityConfig {
    pub use_sequence: bool,
    pub use_text: bool,
    pub use_image: bool,
}

impl ModalityConfig {
    pub fn builder() -> ModalityConfigBuilder {
        ModalityConfigBuilder::default()
    }

    pub fn default() -> Self {
        Self::builder().build()
    }

    pub fn from_names(names: &[&str]) -> Self {
        let enabled: HashSet<&str> = names.iter().copied().collect();

        Self {
            use_sequence: enabled.contains("sequence"),
            use_text: enabled.contains("text"),
            use_image: enabled.contains("image"),
        }
    }

    pub fn enabled_count(&self) -> usize {
        [self.use_sequence, self.use_text, self.use_image]
            .iter()
            .filter(|&&enabled| enabled)
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

        if self.use_sequence {
            modalities.push("sequence");
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

/// Builder for configuring enabled modalities.
#[derive(Debug, Clone, Default)]
pub struct ModalityConfigBuilder {
    use_sequence: bool,
    use_text: bool,
    use_image: bool,
}

impl ModalityConfigBuilder {
    pub fn use_sequence(mut self, enabled: bool) -> Self {
        self.use_sequence = enabled;
        self
    }

    pub fn use_text(mut self, enabled: bool) -> Self {
        self.use_text = enabled;
        self
    }

    pub fn use_image(mut self, enabled: bool) -> Self {
        self.use_image = enabled;
        self
    }

    pub fn build(self) -> ModalityConfig {
        ModalityConfig {
            use_sequence: self.use_sequence,
            use_text: self.use_text,
            use_image: self.use_image,
        }
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

    pub sequence_files: Option<Vec<String>>,
    pub text_files: Option<Vec<String>>,
    pub image_files: Option<Vec<String>>,
}

impl Default for DataPaths {
    fn default() -> Self {
        Self {
            data_dir: "datasets".to_string(),
            patient_id_column: "PatientID".to_string(),
            sequence_files: None,
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
    // Sequence network
    pub sequence_hidden_size: usize,
    pub sequence_output_size: usize,

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
            sequence_hidden_size: 32,
            sequence_output_size: 16,

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
            modalities: ModalityConfig::default(),
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
            training: config.training,
            model: config.model,
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
