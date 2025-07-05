use crate::image::ImageNet;
use crate::layer::Layer;
use crate::module::Module;
use crate::network::Network;
use crate::tabular::TabularNet;
use crate::temporal::TemporalNet;
use crate::text::TextNet;
use ndarray::Array3;

/// Multimodal network that fuses features from different modalities.
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
    /// Network for tabular data
    tabular_net: TabularNet,

    /// Network for temporal data
    temporal_net: TemporalNet,

    /// Network for text data
    text_net: TextNet,

    /// Network for image data
    image_net: ImageNet,

    /// Final classifier network
    classifier: Network,

    /// Cache for last outputs from each modality
    last_tabular_output: Option<Vec<f32>>,
    last_temporal_output: Option<Vec<f32>>,
    last_text_output: Option<Vec<f32>>,
    last_image_output: Option<Vec<f32>>,

    /// Cache for fused vector
    last_fused: Option<Vec<f32>>,
}

impl LateFusionNet {
    /// Create a new late fusion network.
    ///
    /// # Arguments
    /// * `tabular_input_size` - Number of tabular features
    /// * `tabular_hidden_sizes` - Sizes of hidden layers in tabular network
    /// * `tabular_output_size` - Size of tabular feature vector
    /// * `temporal_input_size` - Size of each time step in temporal data
    /// * `temporal_hidden_size` - Size of RNN hidden state
    /// * `vocab` - Text vocabulary mapping
    /// * `embedding_dim` - Size of word embeddings
    /// * `text_hidden_size` - Size of text RNN hidden state
    /// * `image_channels` - Number of image channels (e.g., 3 for RGB)
    /// * `image_height` - Height of input images
    /// * `image_width` - Width of input images
    /// * `classifier_hidden_sizes` - Sizes of hidden layers in final classifier
    pub fn new(
        // Tabular network params
        tabular_input_size: usize,
        tabular_hidden_sizes: &[usize],
        tabular_output_size: usize,

        // Temporal network params
        temporal_input_size: usize,
        temporal_hidden_size: usize,

        // Text network params
        vocab: std::collections::HashMap<String, usize>,
        embedding_dim: usize,
        text_hidden_size: usize,

        // Image network params
        image_channels: usize,
        image_height: usize,
        image_width: usize,

        // Classifier params
        classifier_hidden_sizes: &[usize],
    ) -> Self {
        // Initialize subnetworks
        let tabular_net = TabularNet::new(
            tabular_input_size,
            tabular_hidden_sizes,
            tabular_output_size,
        );

        let temporal_net = TemporalNet::new(temporal_input_size, temporal_hidden_size);

        let text_net = TextNet::new(vocab, embedding_dim, text_hidden_size);

        let image_net = ImageNet::new(image_channels, image_height, image_width);

        // Calculate total feature vector size after fusion
        let fused_size =
            tabular_output_size + temporal_hidden_size + text_hidden_size + image_net.output_size();

        // Create classifier network
        let mut classifier = Network::new();

        let mut prev_size = fused_size;
        for &size in classifier_hidden_sizes {
            classifier.add_layer(Layer::new(size, prev_size));
            prev_size = size;
        }

        // Final output layer (single neuron for binary classification)
        classifier.add_layer(Layer::new(1, prev_size));

        LateFusionNet {
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
        }
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

    /// Process each modality and make a prediction.
    ///
    /// # Arguments
    /// * `tabular_input` - Tabular features
    /// * `temporal_input` - Temporal sequence
    /// * `text_input` - Text token indices
    /// * `image_input` - Image data
    ///
    /// # Returns
    /// Probability of positive outcome (0-1)
    pub fn predict(
        &mut self,
        tabular_input: Vec<f32>,
        temporal_input: Vec<Vec<f32>>,
        text_input: Vec<usize>,
        image_input: Array3<f32>,
    ) -> f32 {
        let output = self.forward((tabular_input, temporal_input, text_input, image_input));

        // Output is a single value between 0 and 1
        output[0]
    }
}

impl Module<(Vec<f32>, Vec<Vec<f32>>, Vec<usize>, Array3<f32>), Vec<f32>> for LateFusionNet {
    fn forward(&mut self, input: (Vec<f32>, Vec<Vec<f32>>, Vec<usize>, Array3<f32>)) -> Vec<f32> {
        let (tabular_input, temporal_input, text_input, image_input) = input;

        // Process each modality independently
        let tabular_features = self.tabular_net.forward(tabular_input);
        let temporal_features = self.temporal_net.forward(temporal_input);
        let text_features = self.text_net.forward(text_input);
        let image_features = self.image_net.forward(image_input);

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

        let grad_tabular = grad_fused[0..tabular_size].to_vec();
        let grad_temporal = grad_fused[tabular_size..(tabular_size + temporal_size)].to_vec();
        let grad_text = grad_fused
            [(tabular_size + temporal_size)..(tabular_size + temporal_size + text_size)]
            .to_vec();
        let grad_image = grad_fused[(tabular_size + temporal_size + text_size)..].to_vec();

        // Backpropagate through each subnetwork
        let tabular_grad = self.tabular_net.backward(grad_tabular);
        let temporal_grad = self.temporal_net.backward(grad_temporal);
        let text_grad = self.text_net.backward(grad_text);
        let image_grad = self.image_net.backward(grad_image);

        (tabular_grad, temporal_grad, text_grad, image_grad)
    }

    fn update(&mut self, lr: f32) {
        // Update all subnetworks
        self.tabular_net.update(lr);
        self.temporal_net.update(lr);
        self.text_net.update(lr);
        self.image_net.update(lr);
        self.classifier.update(lr);
    }

    fn zero_grad(&mut self) {
        // Reset gradients in all subnetworks
        self.tabular_net.zero_grad();
        self.temporal_net.zero_grad();
        self.text_net.zero_grad();
        self.image_net.zero_grad();
        self.classifier.zero_grad();
    }
}
