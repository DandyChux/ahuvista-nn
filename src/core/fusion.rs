use crate::config::ModalityConfig;
use crate::core::{Activation, Layer, Module, Network};
use crate::error::{AppError, AppResult};
use crate::models::{ImageNet, PatientSequence, SequenceNet, TextNet};
use ndarray::{Array3, ArrayBase};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{error, info};

const VERSION: u32 = 2;

/// Input for modular late fusion network with optional modalities
#[derive(Debug, Clone)]
pub struct MultiModalInput {
    pub sequence: Option<PatientSequence>,
    pub text: Option<Vec<usize>>,
    pub image: Option<Array3<f32>>,
}

impl MultiModalInput {
    pub fn new() -> Self {
        Self {
            sequence: None,
            text: None,
            image: None,
        }
    }

    pub fn with_sequence(mut self, sequence: PatientSequence) -> Self {
        self.sequence = Some(sequence);
        self
    }

    pub fn with_text(mut self, data: Vec<usize>) -> Self {
        self.text = Some(data);
        self
    }

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

    sequence_net: Option<SequenceNet>,
    text_net: Option<TextNet>,
    image_net: Option<ImageNet>,

    /// Final classifier network
    classifier: Network,

    /// Cache for last outputs from each modality
    last_tabular_output: Option<Vec<f32>>,
    last_text_output: Option<Vec<f32>>,
    last_image_output: Option<Vec<f32>>,
    last_fused: Option<Vec<f32>>,

    /// Cached feature sizes for each modality
    tabular_output_size: usize,
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
    /// * `sequence_params` - (input_size, hidden_sizes, output_size) if sequence is enabled
    /// * `text_params` - (vocab, embedding_dim, hidden_size) if text is enabled
    /// * `image_params` - (channels, height, width) if image is enabled
    /// * `classifier_hidden_sizes` - Hidden layer sizes for the final classifier
    pub fn new(
        config: ModalityConfig,
        sequence_params: Option<(usize, usize, usize, usize)>,
        text_params: Option<(HashMap<String, usize>, usize, usize)>,
        image_params: Option<(usize, usize, usize)>,
        classifier_hidden_sizes: &[usize],
    ) -> AppResult<Self, String> {
        // Validate configuration
        config.validate()?;

        // Initialize enabled subnetworks and track their output sizes
        let mut tabular_output_size = 0;
        let mut text_output_size = 0;
        let mut image_output_size = 0;

        let sequence_net = if config.use_sequence {
            let (dynamic_input_size, static_input_size, hidden_size, output_size) = sequence_params
                .ok_or("Sequence parameters required when sequence modality is enabled")?;

            tabular_output_size = output_size;

            Some(SequenceNet::new(
                dynamic_input_size,
                static_input_size,
                hidden_size,
                output_size,
            ))
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
        let fused_size = tabular_output_size + text_output_size + image_output_size;

        if fused_size == 0 {
            return Err("No modalities enabled, cannot create classifier".to_string());
        }

        // Create classifier network
        let mut classifier = Network::new();
        let mut prev_size = fused_size;

        for &size in classifier_hidden_sizes {
            classifier.add_layer(Layer::new(size, prev_size, Activation::LeakyReLU));
            prev_size = size;
        }

        // Final output layer (single neuron for binary classification)
        classifier.add_layer(Layer::new_linear(1, prev_size));

        Ok(LateFusionNet {
            config,
            sequence_net,
            text_net,
            image_net,
            classifier,
            last_tabular_output: None,
            last_text_output: None,
            last_image_output: None,
            last_fused: None,
            tabular_output_size,
            text_output_size,
            image_output_size,
        })
    }

    /// Fuse feature vectors from all modalities.
    ///
    /// # Arguments
    /// * `tabular_features` - Features from tabular network
    /// * `text_features` - Features from text network
    /// * `image_features` - Features from image network
    ///
    /// # Returns
    /// Combined feature vector
    fn fuse_features(
        &self,
        tabular_features: &[f32],
        text_features: &[f32],
        image_features: &[f32],
    ) -> Vec<f32> {
        let mut fused =
            Vec::with_capacity(tabular_features.len() + text_features.len() + image_features.len());

        fused.extend_from_slice(tabular_features);
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
        if self.config.use_sequence && input.sequence.is_none() {
            return Err("Static/dynamic sequence data required but not provided".to_string());
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

        if self.config.use_sequence {
            if let (Some(net), Some(sequence)) = (&mut self.sequence_net, input.sequence) {
                let features = net.forward(sequence);

                self.last_tabular_output = Some(features.clone());
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
            "tabular" => self.config.use_sequence,
            "text" => self.config.use_text,
            "image" => self.config.use_image,
            _ => false,
        }
    }

    /// Get the list of enabled modalities
    pub fn enabled_modalities(&self) -> Vec<&str> {
        self.config.enabled_modalities()
    }

    /// Get all parameters from enabled subnetworks
    pub fn get_all_parameters(&self) -> Vec<f32> {
        let mut params = Vec::new();

        if let Some(ref net) = self.sequence_net {
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
    pub fn set_all_parameters(&mut self, params: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        let mut offset = 0;

        if let Some(ref mut net) = self.sequence_net {
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

    /// Set the bias of the final output layer to match a target base rate.
    /// This helps training converge faster by starting predictions near the data's true rate.
    ///
    /// # Arguments
    /// * `positive_rate` - The proportion of positive samples (e.g., 0.1 for 10%)
    pub fn set_output_bias_for_base_rate(&mut self, positive_rate: f32) {
        // logit(p) = ln(p / (1-p))
        let target_bias = (positive_rate / (1.0 - positive_rate)).ln();

        // Get the last layer (output layer) of the classifier
        if let Some(last_layer) = self.classifier.layers.last_mut() {
            if let Some(neuron) = last_layer.neurons.first_mut() {
                neuron.bias = target_bias;
            }
        }
    }
}

impl Module<(PatientSequence, Vec<usize>, Array3<f32>), Vec<f32>> for LateFusionNet {
    fn forward(&mut self, input: (PatientSequence, Vec<usize>, Array3<f32>)) -> Vec<f32> {
        let (sequence_input, text_input, image_input) = input;

        let tabular_features = if let Some(ref mut net) = self.sequence_net {
            net.forward(sequence_input)
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

        self.last_tabular_output = Some(tabular_features.clone());
        self.last_text_output = Some(text_features.clone());
        self.last_image_output = Some(image_features.clone());

        let fused = self.fuse_features(&tabular_features, &text_features, &image_features);

        self.last_fused = Some(fused.clone());

        self.classifier.forward(fused)
    }

    fn backward(&mut self, grad_output: Vec<f32>) -> (PatientSequence, Vec<usize>, Array3<f32>) {
        // Backpropagate through classifier
        let grad_fused = self.classifier.backward(grad_output);

        // Get cached feature vectors
        let tabular_features = self
            .last_tabular_output
            .as_ref()
            .expect("No cached tabular output");
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
        let text_size = text_features.len();
        let image_size = image_features.len();

        let mut offset = 0;

        // Only slice if the modality has features
        let grad_tabular = grad_fused[offset..offset + tabular_size].to_vec();
        offset += tabular_size;

        let grad_text = grad_fused[offset..offset + text_size].to_vec();
        offset += text_size;

        let grad_image = grad_fused[offset..offset + image_size].to_vec();

        // Backpropagate through each subnetwork
        let sequence_gradient = if let Some(ref mut net) = self.sequence_net {
            net.backward(grad_tabular)
        } else {
            PatientSequence::new(Vec::new(), Vec::new())
        };

        let text_gradient = if let Some(ref mut net) = self.text_net {
            net.backward(grad_text)
        } else {
            Vec::new()
        };

        let image_gradient = if let Some(ref mut net) = self.image_net {
            net.backward(grad_image)
        } else {
            Array3::zeros((0, 0, 0))
        };

        (sequence_gradient, text_gradient, image_gradient)
    }

    fn gradients(&self) -> Vec<f32> {
        let mut gradients: Vec<f32> = Vec::new();

        if let Some(ref net) = self.sequence_net {
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

        if let Some(ref mut net) = self.sequence_net {
            let param_count = net.get_parameters().len();

            net.apply_gradients(&gradients[offset..offset + param_count], learning_rate);

            offset += param_count;
        }
        if let Some(ref mut net) = self.text_net {
            let param_count = net.get_parameters().len();

            net.apply_gradients(&gradients[offset..offset + param_count], learning_rate);

            offset += param_count;
        }
        if let Some(ref mut net) = self.image_net {
            let param_count = net.get_parameters().len();

            net.apply_gradients(&gradients[offset..offset + param_count], learning_rate);

            offset += param_count;
        }

        let classifier_count = self.classifier.gradients().len();

        assert_eq!(
            gradients.len(),
            offset + classifier_count,
            "Fusion gradient size mismatch"
        );

        self.classifier
            .apply_gradients(&gradients[offset..], learning_rate);
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut net) = self.sequence_net {
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
