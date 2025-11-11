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
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,

    // Model settings
    pub tabular_hidden_sizes: Vec<usize>,
    pub temporal_hidden_size: usize,
    pub text_embedding_dim: usize,

    // Modality settings
    pub modality_config: ModalityConfig,
}

impl Settings {
    /// Creates a new builder for constructing a `Settings` instance.
    pub fn new() -> SettingsBuilder {
        SettingsBuilder::default()
    }

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
            epochs: 20,
            batch_size: 32,
            learning_rate: 0.001,
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
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            epochs: 20,
            batch_size: 32,
            learning_rate: 0.001,
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
            epochs: config.training.epochs,
            batch_size: config.training.batch_size,
            learning_rate: config.training.learning_rate,
            tabular_hidden_sizes: config.model.tabular_hidden_sizes,
            temporal_hidden_size: config.model.temporal_hidden_size,
            text_embedding_dim: config.model.text_embedding_dim,
            modality_config: config.modalities,
        }
    }
}

/// A builder for creating `Settings` instances.
///
/// This builder allows for selectively overriding default configuration values.
#[derive(Default)]
pub struct SettingsBuilder {
    data_dir: Option<PathBuf>,
    patient_id_column: Option<String>,
    epochs: Option<usize>,
    batch_size: Option<usize>,
    learning_rate: Option<f32>,
    tabular_hidden_sizes: Option<Vec<usize>>,
    temporal_hidden_size: Option<usize>,
    text_embedding_dim: Option<usize>,
    modality_config: Option<ModalityConfig>,
}

impl SettingsBuilder {
    /// Sets the data directory. Default: "datasets".
    pub fn data_dir(mut self, data_dir: impl Into<PathBuf>) -> Self {
        self.data_dir = Some(data_dir.into());
        self
    }

    /// Sets the patient ID column name. Default: "PatientID".
    pub fn patient_id_column(mut self, patient_id_column: impl Into<String>) -> Self {
        self.patient_id_column = Some(patient_id_column.into());
        self
    }

    /// Sets the number of training epochs. Default: 20.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = Some(epochs);
        self
    }

    /// Sets the batch size for training. Default: 32.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Sets the learning rate. Default: 0.001.
    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = Some(learning_rate);
        self
    }

    /// Sets the hidden layer sizes for the tabular data network. Default: [64, 32].
    pub fn tabular_hidden_sizes(mut self, tabular_hidden_sizes: Vec<usize>) -> Self {
        self.tabular_hidden_sizes = Some(tabular_hidden_sizes);
        self
    }

    /// Sets the hidden layer size for the temporal data network. Default: 64.
    pub fn temporal_hidden_size(mut self, temporal_hidden_size: usize) -> Self {
        self.temporal_hidden_size = Some(temporal_hidden_size);
        self
    }

    /// Sets the embedding dimension for the text data. Default: 50.
    pub fn text_embedding_dim(mut self, text_embedding_dim: usize) -> Self {
        self.text_embedding_dim = Some(text_embedding_dim);
        self
    }

    /// Sets the modality configuration
    pub fn modality_config(mut self, modality_config: ModalityConfig) -> Self {
        self.modality_config = Some(modality_config);
        self
    }

    /// Builds the `Settings` struct with the specified values,
    /// falling back to defaults for any unset values.
    pub fn build(self) -> Settings {
        Settings {
            data_dir: self.data_dir.unwrap_or_else(|| PathBuf::from("datasets")),
            patient_id_column: self
                .patient_id_column
                .unwrap_or_else(|| "PatientID".to_string()),
            epochs: self.epochs.unwrap_or(20),
            batch_size: self.batch_size.unwrap_or(32),
            learning_rate: self.learning_rate.unwrap_or(0.001),
            tabular_hidden_sizes: self.tabular_hidden_sizes.unwrap_or_else(|| vec![64, 32]),
            temporal_hidden_size: self.temporal_hidden_size.unwrap_or(64),
            text_embedding_dim: self.text_embedding_dim.unwrap_or(50),
            modality_config: self.modality_config.unwrap_or_else(ModalityConfig::all),
        }
    }
}

pub static SETTINGS: LazyLock<Settings> = LazyLock::new(|| SettingsBuilder::default().build());
