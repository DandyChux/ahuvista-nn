use crate::config::ModalityConfig;
use crate::core::{Activation, Layer, Module, Network};
use crate::models::{ImageNet, TabularNet, TemporalNet, TextNet};
use crate::utils::error::{AppError, AppResult};
use ndarray::{Array3, ArrayBase};
use std::collections::HashMap;
use tracing::{error, info};

/// Input for modular late fusion network with optional modalities
#[derive(Debug, Clone)]
pub struct MultiModalInput {
    pub tabular: Option<Vec<f32>>,
    pub temporal: Option<Vec<Vec<f32>>>,
    pub text: Option<Vec<usize>>,
    pub image: Option<Array3<f32>>,
}

impl MultiModalInput {
    /// Create a new empty input
    pub fn new() -> Self {
        Self {
            tabular: None,
            temporal: None,
            text: None,
            image: None,
        }
    }

    /// Add tabular data
    pub fn with_tabular(mut self, data: Vec<f32>) -> Self {
        self.tabular = Some(data);
        self
    }

    /// Add temporal data
    pub fn with_temporal(mut self, data: Vec<Vec<f32>>) -> Self {
        self.temporal = Some(data);
        self
    }

    /// Add text data
    pub fn with_text(mut self, data: Vec<usize>) -> Self {
        self.text = Some(data);
        self
    }

    /// Add image data
    pub fn with_image(mut self, data: Array3<f32>) -> Self {
        self.image = Some(data);
        self
    }
}

impl Default for MultiModalInput {
    fn default() -> Self {
        Self::new()
    }
}

/// Modular multimodal network that supports selective modality processing.
///
/// This network allows you to enable/disable specific modalities and will only
/// process the enabled ones. This provides:
/// - Memory efficiency: Only load networks for enabled modalities
/// - Flexibility: Run different modality combinations based on available data
/// - Easier testing: Test individual modalities in isolation
///
/// Implements late fusion by:
/// 1. Processing each modality separately through specialized networks
/// 2. Concatenating the feature vectors from each modality
/// 3. Passing the combined vector through a final classifier network
///
/// This design allows each subnetwork to learn modality-specific features
/// before combining them for the final prediction, making it more efficient
/// for low-compute environments.
#[derive(Debug)]
pub struct LateFusionNet {
    /// Config for enabled modalities
    config: ModalityConfig,

    /// Network for tabular data
    tabular_net: Option<TabularNet>,

    /// Network for temporal data
    temporal_net: Option<TemporalNet>,

    /// Network for text data
    text_net: Option<TextNet>,

    /// Network for image data
    image_net: Option<ImageNet>,

    /// Final classifier network
    classifier: Network,

    /// Cache for last outputs from each modality
    last_tabular_output: Option<Vec<f32>>,
    last_temporal_output: Option<Vec<f32>>,
    last_text_output: Option<Vec<f32>>,
    last_image_output: Option<Vec<f32>>,

    /// Cache for fused vector
    last_fused: Option<Vec<f32>>,

    /// Cached feature sizes for each modality
    tabular_output_size: usize,
    temporal_output_size: usize,
    text_output_size: usize,
    image_output_size: usize,
}

impl LateFusionNet {
    /// Create a new late fusion network with selective modalities
    ///
    /// Only the networks for enabled modalities will be created and used.
    ///
    /// # Arguments
    /// * `config` - Configuration specifying which modalities to enable
    /// * `tabular_params` - (input_size, hidden_sizes, output_size) if tabular is enabled
    /// * `temporal_params` - (input_size, hidden_size, output_size) if temporal is enabled
    /// * `text_params` - (vocab, embedding_dim, hidden_size) if text is enabled
    /// * `image_params` - (channels, height, width) if image is enabled
    /// * `classifier_hidden_sizes` - Hidden layer sizes for the final classifier
    pub fn new(
        config: ModalityConfig,
        tabular_params: Option<(usize, Vec<usize>, usize)>,
        temporal_params: Option<(usize, usize, usize)>,
        text_params: Option<(HashMap<String, usize>, usize, usize)>,
        image_params: Option<(usize, usize, usize)>,
        classifier_hidden_sizes: &[usize],
    ) -> AppResult<Self, String> {
        // Validate configuration
        config.validate()?;

        // Initialize enabled subnetworks and track their output sizes
        let mut tabular_output_size = 0;
        let mut temporal_output_size = 0;
        let mut text_output_size = 0;
        let mut image_output_size = 0;

        let tabular_net = if config.use_tabular {
            let (input_size, hidden_sizes, output_size) = tabular_params
                .ok_or("Tabular parameters required when tabular modality is enabled")?;
            tabular_output_size = output_size;
            Some(TabularNet::new(
                input_size,
                &hidden_sizes,
                output_size,
                Some(Activation::ReLU),
                None,
            ))
        } else {
            None
        };

        let temporal_net = if config.use_temporal {
            let (input_size, hidden_size, output_size) = temporal_params
                .ok_or("Temporal parameters required when temporal modality is enabled")?;
            temporal_output_size = output_size;
            Some(TemporalNet::new(input_size, hidden_size, output_size))
        } else {
            None
        };

        let text_net = if config.use_text {
            let (vocab, embedding_dim, hidden_size) =
                text_params.ok_or("Text parameters required when text modality is enabled")?;
            text_output_size = hidden_size;
            Some(TextNet::new(vocab, embedding_dim, hidden_size))
        } else {
            None
        };

        let image_net = if config.use_image {
            let (channels, height, width) =
                image_params.ok_or("Image parameters required when image modality is enabled")?;
            // Use a fixed output size for image features (e.g. 128)
            let image_feature_size = 128;
            let net = ImageNet::new(channels, height, width, image_feature_size, None);
            image_output_size = net.output_size();
            Some(net)
        } else {
            None
        };

        // Calculate total feature vector size based on enabled modalities
        let fused_size =
            tabular_output_size + temporal_output_size + text_output_size + image_output_size;

        if fused_size == 0 {
            return Err("No modalities enabled, cannot create classifier".to_string());
        }

        // Create classifier network
        let mut classifier = Network::new();
        let mut prev_size = fused_size;

        for &size in classifier_hidden_sizes {
            classifier.add_layer(Layer::new(size, prev_size, Activation::ReLU));
            prev_size = size;
        }

        // Final output layer (single neuron for binary classification)
        classifier.add_layer(Layer::new_linear(1, prev_size));

        Ok(LateFusionNet {
            config,
            tabular_net,
            temporal_net,
            text_net,
            image_net,
            classifier,
            last_tabular_output: None,
            last_temporal_output: None,
            last_text_output: None,
            last_image_output: None,
            last_fused: None,
            tabular_output_size,
            temporal_output_size,
            text_output_size,
            image_output_size,
        })
    }

    /// Fuse feature vectors from all modalities.
    ///
    /// # Arguments
    /// * `tabular_features` - Features from tabular network
    /// * `temporal_features` - Features from temporal network
    /// * `text_features` - Features from text network
    /// * `image_features` - Features from image network
    ///
    /// # Returns
    /// Combined feature vector
    fn fuse_features(
        &self,
        tabular_features: &[f32],
        temporal_features: &[f32],
        text_features: &[f32],
        image_features: &[f32],
    ) -> Vec<f32> {
        let mut fused = Vec::with_capacity(
            tabular_features.len()
                + temporal_features.len()
                + text_features.len()
                + image_features.len(),
        );

        fused.extend_from_slice(tabular_features);
        fused.extend_from_slice(temporal_features);
        fused.extend_from_slice(text_features);
        fused.extend_from_slice(image_features);

        fused
    }

    /// Process modalities and make a prediction with optional inputs.
    ///
    /// # Arguments
    /// * `input` - Multi-modal input with optional data for each modality
    ///
    /// # Returns
    /// Probability of positive outcome (0-1)
    ///
    /// # Errors
    /// Returns an error if:
    /// - An enabled modality is missing required data
    /// - Data is provided for a disabled modality (warning only)
    pub fn predict(&mut self, input: MultiModalInput) -> AppResult<f32, String> {
        // Validate that enabled modalities have provided data
        self.validate_input(&input)?;

        let output = self.forward_modular(input)?;
        Ok(output[0])
    }

    /// Validate that the input matches the configuration
    fn validate_input(&self, input: &MultiModalInput) -> Result<(), String> {
        if self.config.use_tabular && input.tabular.is_none() {
            return Err("Tabular data required but not provided".to_string());
        }
        if self.config.use_temporal && input.temporal.is_none() {
            return Err("Temporal data required but not provided".to_string());
        }
        if self.config.use_text && input.text.is_none() {
            return Err("Text data required but not provided".to_string());
        }
        if self.config.use_image && input.image.is_none() {
            return Err("Image data required but not provided".to_string());
        }
        Ok(())
    }

    /// Forward pass with modular input
    fn forward_modular(&mut self, input: MultiModalInput) -> Result<Vec<f32>, String> {
        let mut feature_vectors = Vec::new();

        // Process enabled modalities only
        if self.config.use_tabular {
            if let (Some(net), Some(data)) = (&mut self.tabular_net, input.tabular) {
                let features = net.forward(data);
                self.last_tabular_output = Some(features.clone());
                feature_vectors.push(features);
            }
        }

        if self.config.use_temporal {
            if let (Some(net), Some(data)) = (&mut self.temporal_net, input.temporal) {
                let features = net.forward(data);
                self.last_temporal_output = Some(features.clone());
                feature_vectors.push(features);
            }
        }

        if self.config.use_text {
            if let (Some(net), Some(data)) = (&mut self.text_net, input.text) {
                let features = net.forward(data);
                self.last_text_output = Some(features.clone());
                feature_vectors.push(features);
            }
        }

        if self.config.use_image {
            if let (Some(net), Some(data)) = (&mut self.image_net, input.image) {
                let features = net.forward(data);
                self.last_image_output = Some(features.clone());
                feature_vectors.push(features);
            }
        }

        // Fuse all feature vectors
        let fused = feature_vectors.concat();
        self.last_fused = Some(fused.clone());

        // Final classification
        Ok(self.classifier.forward(fused))
    }

    /// Get the modality configuration
    pub fn config(&self) -> &ModalityConfig {
        &self.config
    }

    /// Check if a specific modality is enabled
    pub fn is_enabled(&self, modality: &str) -> bool {
        match modality {
            "tabular" => self.config.use_tabular,
            "temporal" => self.config.use_temporal,
            "text" => self.config.use_text,
            "image" => self.config.use_image,
            _ => false,
        }
    }

    /// Get the list of enabled modalities
    pub fn enabled_modalities(&self) -> Vec<&str> {
        self.config.enabled_modalities()
    }

    /// Save the model to a file in a custom binary format.
    ///
    /// # Arguments
    /// * `path` - Path to save the model file
    ///
    /// # Returns
    /// Result indicating success or error
    ///
    /// # Format
    /// The file format is:
    /// - Magic bytes: "LFNN" (4 bytes)
    /// - Version: u32 (4 bytes)
    /// - Number of parameters: u64 (8 bytes)
    /// - Parameters: Vec<f32> (4 bytes each)
    /// - Text vocabulary size: u64 (8 bytes)
    /// - Text vocabulary entries: (key_len: u64, key: String, value: usize)*
    pub fn save(&self, path: &str) -> AppResult<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        // Write header
        file.write_all(b"LFNN")?; // Magic bytes
        file.write_all(&1u32.to_le_bytes())?; // Version

        // Write modality configuration
        file.write_all(&[
            self.config.use_tabular as u8,
            self.config.use_temporal as u8,
            self.config.use_text as u8,
            self.config.use_image as u8,
        ])?;

        // Get all parameters from the model
        let params = self.get_all_parameters();

        // Write number of parameters
        file.write_all(&(params.len() as u64).to_le_bytes())?;

        // Write all parameters
        for param in &params {
            file.write_all(&param.to_le_bytes())?;
        }

        // Save text vocabulary if text network is enabled
        if let Some(ref net) = self.text_net {
            file.write_all(&(net.vocab.len() as u64).to_le_bytes())?;
            for (word, idx) in &net.vocab {
                let word_bytes = word.as_bytes();
                file.write_all(&(word_bytes.len() as u64).to_le_bytes())?;
                file.write_all(word_bytes)?;
                file.write_all(&(*idx as u64).to_le_bytes())?;
            }
        } else {
            file.write_all(&0u64.to_le_bytes())?; // No vocabulary
        }

        info!("Model saved to {} ({} parameters)", path, params.len());
        Ok(())
    }

    /// Load the model from a file.
    ///
    /// # Arguments
    /// * `path` - Path to the model file
    ///
    /// # Returns
    /// Result indicating success or error
    pub fn load(&mut self, path: &str) -> AppResult<()> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;

        // Read and verify header
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"MLFN" {
            return Err("Invalid model file: wrong magic bytes".into());
        }

        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);

        if version != 1 {
            return Err(format!("Unsupported model version: {}", version).into());
        }

        // Read modality configuration
        let mut config_bytes = [0u8; 4];
        file.read_exact(&mut config_bytes)?;
        // TODO: Validate that loaded config matches current config

        // Read parameters
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)?;
        let param_count = u64::from_le_bytes(len_bytes) as usize;

        let mut params = Vec::with_capacity(param_count);
        for _ in 0..param_count {
            let mut param_bytes = [0u8; 4];
            file.read_exact(&mut param_bytes)?;
            params.push(f32::from_le_bytes(param_bytes));
        }

        self.set_all_parameters(&params).unwrap_or_else(|err| {
            error!("Failed to set parameters: {}", err);
        });

        // Load text vocabulary if present
        let mut vocab_len_bytes = [0u8; 8];
        file.read_exact(&mut vocab_len_bytes)?;
        let vocab_len = u64::from_le_bytes(vocab_len_bytes) as usize;

        if vocab_len > 0 {
            if let Some(ref mut net) = self.text_net {
                let mut vocab = HashMap::new();
                for _ in 0..vocab_len {
                    let mut word_len_bytes = [0u8; 8];
                    file.read_exact(&mut word_len_bytes)?;
                    let word_len = u64::from_le_bytes(word_len_bytes) as usize;

                    let mut word_bytes = vec![0u8; word_len];
                    file.read_exact(&mut word_bytes)?;
                    let word = String::from_utf8(word_bytes)?;

                    let mut idx_bytes = [0u8; 8];
                    file.read_exact(&mut idx_bytes)?;
                    let idx = u64::from_le_bytes(idx_bytes) as usize;

                    vocab.insert(word, idx);
                }
                net.vocab = vocab;
            }
        }

        info!("Model loaded from {} ({} parameters)", path, param_count);
        Ok(())
    }

    /// Get all parameters from enabled subnetworks
    fn get_all_parameters(&self) -> Vec<f32> {
        let mut params = Vec::new();

        if let Some(ref net) = self.tabular_net {
            params.extend(net.get_parameters());
        }
        if let Some(ref net) = self.temporal_net {
            params.extend(net.get_parameters());
        }
        if let Some(ref net) = self.text_net {
            params.extend(net.get_parameters());
        }
        if let Some(ref net) = self.image_net {
            params.extend(net.get_parameters());
        }
        params.extend(self.classifier.get_parameters());

        params
    }

    /// Set all parameters in enabled subnetworks
    fn set_all_parameters(&mut self, params: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        let mut offset = 0;

        if let Some(ref mut net) = self.tabular_net {
            offset += net.set_parameters(&params[offset..])?;
        }
        if let Some(ref mut net) = self.temporal_net {
            offset += net.set_parameters(&params[offset..])?;
        }
        if let Some(ref mut net) = self.text_net {
            offset += net.set_parameters(&params[offset..])?;
        }
        if let Some(ref mut net) = self.image_net {
            offset += net.set_parameters(&params[offset..])?;
        }
        offset += self.classifier.set_parameters(&params[offset..])?;

        if offset != params.len() {
            return Err(format!(
                "Parameter count mismatch: expected to use {}, but used {}",
                params.len(),
                offset
            )
            .into());
        }

        Ok(())
    }
}

impl Module<(Vec<f32>, Vec<Vec<f32>>, Vec<usize>, Array3<f32>), Vec<f32>> for LateFusionNet {
    fn forward(&mut self, input: (Vec<f32>, Vec<Vec<f32>>, Vec<usize>, Array3<f32>)) -> Vec<f32> {
        let (tabular_input, temporal_input, text_input, image_input) = input;

        // Process each modality independently, providing an empty vec if the modality is disabled.
        let tabular_features = if let Some(ref mut net) = self.tabular_net {
            net.forward(tabular_input)
        } else {
            vec![]
        };

        let temporal_features = if let Some(ref mut net) = self.temporal_net {
            net.forward(temporal_input)
        } else {
            vec![]
        };

        let text_features = if let Some(ref mut net) = self.text_net {
            net.forward(text_input)
        } else {
            vec![]
        };

        let image_features = if let Some(ref mut net) = self.image_net {
            net.forward(image_input)
        } else {
            vec![]
        };

        // Cache outputs for backpropagation
        self.last_tabular_output = Some(tabular_features.clone());
        self.last_temporal_output = Some(temporal_features.clone());
        self.last_text_output = Some(text_features.clone());
        self.last_image_output = Some(image_features.clone());

        // Fuse features
        let fused = self.fuse_features(
            &tabular_features,
            &temporal_features,
            &text_features,
            &image_features,
        );

        self.last_fused = Some(fused.clone());

        // Final classification
        self.classifier.forward(fused)
    }

    fn backward(
        &mut self,
        grad_output: Vec<f32>,
    ) -> (Vec<f32>, Vec<Vec<f32>>, Vec<usize>, Array3<f32>) {
        // Backpropagate through classifier
        let grad_fused = self.classifier.backward(grad_output);

        // Get cached feature vectors
        let tabular_features = self
            .last_tabular_output
            .as_ref()
            .expect("No cached tabular output");
        let temporal_features = self
            .last_temporal_output
            .as_ref()
            .expect("No cached temporal output");
        let text_features = self
            .last_text_output
            .as_ref()
            .expect("No cached text output");
        let image_features = self
            .last_image_output
            .as_ref()
            .expect("No cached image output");

        // Split gradient for each modality
        let tabular_size = tabular_features.len();
        let temporal_size = temporal_features.len();
        let text_size = text_features.len();
        let image_size = image_features.len();

        // Only slice if the modality has features
        let grad_tabular = if tabular_size > 0 {
            grad_fused[0..tabular_size].to_vec()
        } else {
            vec![]
        };

        let mut offset = tabular_size;

        let grad_temporal = if temporal_size > 0 {
            let g = grad_fused[offset..(offset + temporal_size)].to_vec();
            offset += temporal_size;
            g
        } else {
            vec![]
        };

        let grad_text = if text_size > 0 {
            let g = grad_fused[offset..(offset + text_size)].to_vec();
            offset += text_size;
            g
        } else {
            vec![]
        };

        let grad_image = if image_size > 0 {
            grad_fused[offset..].to_vec()
        } else {
            vec![]
        };

        // Backpropagate through each subnetwork
        let tabular_grad = if let Some(ref mut net) = self.tabular_net {
            net.backward(grad_tabular)
        } else {
            vec![0.0; tabular_size]
        };
        let temporal_grad = if let Some(ref mut net) = self.temporal_net {
            net.backward(grad_temporal)
        } else {
            vec![vec![0.0; temporal_size]]
        };
        let text_grad = if let Some(ref mut net) = self.text_net {
            net.backward(grad_text)
        } else {
            vec![0; text_size]
        };
        let image_grad = if let Some(ref mut net) = self.image_net {
            net.backward(grad_image)
        } else {
            Array3::zeros((image_size, image_size, image_size))
        };

        (tabular_grad, temporal_grad, text_grad, image_grad)
    }

    fn gradients(&self) -> Vec<f32> {
        let mut gradients: Vec<f32> = Vec::new();

        if let Some(ref net) = self.tabular_net {
            gradients.extend(net.gradients());
        }
        if let Some(ref net) = self.temporal_net {
            gradients.extend(net.gradients());
        }
        if let Some(ref net) = self.text_net {
            gradients.extend(net.gradients());
        }
        if let Some(ref net) = self.image_net {
            gradients.extend(net.gradients());
        }
        gradients.extend(self.classifier.gradients());

        gradients
    }

    fn apply_gradients(&mut self, gradients: &[f32], learning_rate: f32) {
        let mut offset = 0;

        if let Some(ref mut net) = self.tabular_net {
            let param_count = net.get_parameters().len();
            if offset + param_count <= gradients.len() {
                net.apply_gradients(&gradients[offset..offset + param_count], learning_rate);
                offset += param_count;
            }
        }
        if let Some(ref mut net) = self.temporal_net {
            let param_count = net.get_parameters().len();
            if offset + param_count <= gradients.len() {
                net.apply_gradients(&gradients[offset..offset + param_count], learning_rate);
                offset += param_count;
            }
        }
        if let Some(ref mut net) = self.text_net {
            let param_count = net.get_parameters().len();
            if offset + param_count <= gradients.len() {
                net.apply_gradients(&gradients[offset..offset + param_count], learning_rate);
                offset += param_count;
            }
        }
        if let Some(ref mut net) = self.image_net {
            let param_count = net.get_parameters().len();
            if offset + param_count <= gradients.len() {
                net.apply_gradients(&gradients[offset..offset + param_count], learning_rate);
                offset += param_count;
            }
        }

        // Apply remaining gradients to classifier
        if offset < gradients.len() {
            self.classifier
                .apply_gradients(&gradients[offset..], learning_rate);
        }
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut net) = self.tabular_net {
            net.zero_grad();
        }
        if let Some(ref mut net) = self.temporal_net {
            net.zero_grad();
        }
        if let Some(ref mut net) = self.text_net {
            net.zero_grad();
        }
        if let Some(ref mut net) = self.image_net {
            net.zero_grad();
        }
        self.classifier.zero_grad();
    }
}
