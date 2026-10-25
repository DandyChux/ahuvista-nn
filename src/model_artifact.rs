use std::{collections::HashMap, fs, path::Path};

use serde::{Deserialize, Serialize};

use crate::{
    config::ModalityConfig,
    core::fusion::LateFusionNet,
    data::FeatureStats,
    error::{AppError, AppResult},
};

const MODEL_ARTIFACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub model_id: String,
    pub model_version: String,

    pub modalities: ModalityConfig,

    pub sequence_dynamic_input_size: Option<usize>,
    pub sequence_static_input_size: Option<usize>,
    pub sequence_hidden_size: Option<usize>,
    pub sequence_output_size: Option<usize>,

    pub text_embedding_size: Option<usize>,
    pub text_hidden_size: Option<usize>,
    pub text_vocab: HashMap<String, usize>,

    pub image_channels: Option<usize>,
    pub image_height: Option<usize>,
    pub image_width: Option<usize>,

    pub classifier_hidden_sizes: Vec<usize>,

    /// Feature names must be persisted in vector order.
    pub static_feature_names: Vec<String>,
    pub dynamic_feature_names: Vec<String>,

    /// Preprocessing statistics required during inference.
    pub static_normalization: Vec<FeatureStats>,
    pub dynamic_normalization: Vec<FeatureStats>,

    /// Optional mapping/configuration identifier for FHIR preprocessing.
    pub fhir_mapping_version: Option<String>,
}

impl ModelSpec {
    pub fn validate(&self) -> AppResult<()> {
        self.modalities.validate().map_err(AppError::Config)?;

        if self.modalities.use_sequence {
            let values = [
                self.sequence_dynamic_input_size,
                self.sequence_static_input_size,
                self.sequence_hidden_size,
                self.sequence_output_size,
            ];

            if values.iter().any(Option::is_none) {
                return Err(AppError::Config(
                    "Sequence model dimensions are incomplete".to_string(),
                ));
            }
        }

        if self.modalities.use_text {
            if self.text_embedding_size.is_none() || self.text_hidden_size.is_none() {
                return Err(AppError::Config(
                    "Text model dimensions are incomplete".to_string(),
                ));
            }

            if self.text_vocab.is_empty() {
                return Err(AppError::Config(
                    "Text modality is enabled but vocabulary is empty".to_string(),
                ));
            }
        }

        if self.modalities.use_image {
            let values = [self.image_channels, self.image_height, self.image_width];

            if values.iter().any(Option::is_none) {
                return Err(AppError::Config(
                    "Image model dimensions are incomplete".to_string(),
                ));
            }
        }

        if self.modalities.use_sequence {
            if self.static_feature_names.len() != self.sequence_static_input_size.unwrap() {
                return Err(AppError::Config(
                    "Static feature-name count does not match model input size".to_string(),
                ));
            }

            if self.dynamic_feature_names.len() != self.sequence_dynamic_input_size.unwrap() {
                return Err(AppError::Config(
                    "Dynamic feature-name count does not match model input size".to_string(),
                ));
            }
        }

        Ok(())
    }

    pub fn sequence_params(&self) -> Option<(usize, usize, usize, usize)> {
        if !self.modalities.use_sequence {
            return None;
        }

        Some((
            self.sequence_dynamic_input_size?,
            self.sequence_static_input_size?,
            self.sequence_hidden_size?,
            self.sequence_output_size?,
        ))
    }

    pub fn text_params(&self) -> Option<(HashMap<String, usize>, usize, usize)> {
        if !self.modalities.use_text {
            return None;
        }

        Some((
            self.text_vocab.clone(),
            self.text_embedding_size?,
            self.text_hidden_size?,
        ))
    }

    pub fn image_params(&self) -> Option<(usize, usize, usize)> {
        if !self.modalities.use_image {
            return None;
        }

        Some((self.image_channels?, self.image_height?, self.image_width?))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub artifact_version: u32,
    pub spec: ModelSpec,
    pub parameters: Vec<f32>,
}

impl ModelArtifact {
    pub fn new(spec: ModelSpec, parameters: Vec<f32>) -> AppResult<Self> {
        spec.validate()?;

        if parameters.is_empty() {
            return Err(AppError::ModelError(
                "Cannot create artifact with no parameters".to_string(),
            ));
        }

        Ok(Self {
            artifact_version: MODEL_ARTIFACT_VERSION,
            spec,
            parameters,
        })
    }

    pub fn from_model(spec: ModelSpec, model: &LateFusionNet) -> AppResult<Self> {
        Self::new(spec, model.get_all_parameters())
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> AppResult<()> {
        if self.artifact_version != MODEL_ARTIFACT_VERSION {
            return Err(AppError::ModelError(
                "Unsupported model artifact version".to_string(),
            ));
        }

        self.spec.validate()?;

        let json = serde_json::to_vec_pretty(self)?;
        fs::write(path, json)?;

        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> AppResult<Self> {
        let bytes = fs::read(path)?;
        let artifact: Self = serde_json::from_slice(&bytes)?;

        if artifact.artifact_version != MODEL_ARTIFACT_VERSION {
            return Err(AppError::ModelError(format!(
                "Unsupported model artifact version: {}",
                artifact.artifact_version
            )));
        }

        artifact.spec.validate()?;

        Ok(artifact)
    }

    pub fn instantiate(&self) -> AppResult<LateFusionNet> {
        let mut model = LateFusionNet::new(
            self.spec.modalities.clone(),
            self.spec.sequence_params(),
            self.spec.text_params(),
            self.spec.image_params(),
            &self.spec.classifier_hidden_sizes,
        )
        .map_err(AppError::Config)?;

        model
            .set_all_parameters(&self.parameters)
            .map_err(|error| AppError::ModelError(error.to_string()))?;

        Ok(model)
    }
}
