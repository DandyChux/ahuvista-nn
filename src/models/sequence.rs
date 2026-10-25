use serde::Deserialize;

use crate::{
    core::{Layer, Module},
    models::GruLayer,
};

/// Static patient features and longitudinal dynamic features.
///
/// Static features include values such as baseline genetics, admission
/// demographics, pregnancy history, and pre-existing conditions.
///
/// Dynamic features contain one feature vector per ordered timestep.
#[derive(Debug, Clone, Deserialize)]
pub struct PatientSequence {
    pub static_features: Vec<f32>,
    pub dynamic_features: Vec<Vec<f32>>,
}

impl PatientSequence {
    pub fn new(static_features: Vec<f32>, dynamic_features: Vec<Vec<f32>>) -> Self {
        Self {
            static_features,
            dynamic_features,
        }
    }

    pub fn static_size(&self) -> usize {
        self.static_features.len()
    }

    pub fn dynamic_size(&self) -> usize {
        self.dynamic_features
            .first()
            .map(|timestep| timestep.len())
            .unwrap_or(0)
    }

    pub fn sequence_length(&self) -> usize {
        self.dynamic_features.len()
    }

    pub fn validate(
        &self,
        expected_static_size: usize,
        expected_dynamic_size: usize,
    ) -> Result<(), String> {
        if self.static_features.len() != expected_static_size {
            return Err(format!(
                "Static feature size mismatch: expected {}, got {}",
                expected_static_size,
                self.static_features.len()
            ));
        }

        for (index, timestep) in self.dynamic_features.iter().enumerate() {
            if timestep.len() != expected_dynamic_size {
                return Err(format!(
                    "Dynamic feature size mismatch at timestep {}: expected {}, got {}",
                    index,
                    expected_dynamic_size,
                    timestep.len()
                ));
            }
        }

        Ok(())
    }
}

/// Unified static-plus-longitudinal structured-data model.
///
/// Dynamic features are processed by the GRU. The final GRU hidden state is
/// concatenated with static baseline features before the output projection.
#[derive(Debug)]
pub struct SequenceNet {
    pub gru: GruLayer,

    /// Projects [final GRU hidden state, static features] into the output
    /// feature representation.
    pub classifier: Layer,

    pub static_size: usize,
    pub dynamic_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
}

impl SequenceNet {
    pub fn new(
        dynamic_input_size: usize,
        static_input_size: usize,
        hidden_size: usize,
        output_size: usize,
    ) -> Self {
        let gru = GruLayer::new(dynamic_input_size, hidden_size);

        let classifier_input_size = hidden_size + static_input_size;

        let classifier = Layer::new_linear(output_size, classifier_input_size);

        Self {
            gru,
            classifier,
            static_size: static_input_size,
            dynamic_size: dynamic_input_size,
            hidden_size,
            output_size,
        }
    }

    pub fn new_for_binary_classification(
        dynamic_input_size: usize,
        static_input_size: usize,
        hidden_size: usize,
    ) -> Self {
        Self::new(dynamic_input_size, static_input_size, hidden_size, 1)
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn static_size(&self) -> usize {
        self.static_size
    }

    pub fn dynamic_size(&self) -> usize {
        self.dynamic_size
    }

    pub fn get_parameters(&self) -> Vec<f32> {
        let mut parameters = self.gru.get_parameters();
        parameters.extend(self.classifier.get_parameters());
        parameters
    }

    pub fn set_parameters(
        &mut self,
        parameters: &[f32],
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let mut offset = 0;

        offset += self.gru.set_parameters(&parameters[offset..])?;
        offset += self.classifier.set_parameters(&parameters[offset..])?;

        Ok(offset)
    }

    pub fn parameter_count(&self) -> usize {
        self.get_parameters().len()
    }
}

impl Module<PatientSequence, Vec<f32>> for SequenceNet {
    fn forward(&mut self, input: PatientSequence) -> Vec<f32> {
        input
            .validate(self.static_size, self.dynamic_size)
            .unwrap_or_else(|error| panic!("{error}"));

        let hidden_state = self.gru.forward_sequence(&input.dynamic_features);

        let mut classifier_input = Vec::with_capacity(self.hidden_size + self.static_size);

        classifier_input.extend_from_slice(&hidden_state);
        classifier_input.extend_from_slice(&input.static_features);

        self.classifier.forward(classifier_input)
    }

    fn backward(&mut self, grad_output: Vec<f32>) -> PatientSequence {
        assert_eq!(
            grad_output.len(),
            self.output_size,
            "Classifier gradient size mismatch: expected {}, got {}",
            self.output_size,
            grad_output.len()
        );

        let grad_classifier_input = self.classifier.backward(grad_output);

        let grad_hidden = grad_classifier_input[..self.hidden_size].to_vec();

        let grad_static = grad_classifier_input[self.hidden_size..].to_vec();

        let grad_dynamic = self.gru.backward_sequence(&grad_hidden);

        PatientSequence {
            static_features: grad_static,
            dynamic_features: grad_dynamic,
        }
    }

    fn gradients(&self) -> Vec<f32> {
        let mut gradients = self.gru.gradients();
        gradients.extend(self.classifier.gradients());
        gradients
    }

    fn apply_gradients(&mut self, gradients: &[f32], learning_rate: f32) {
        let gru_parameter_count = self.gru.parameter_count();
        let classifier_parameter_count = self.classifier.gradients().len();

        let expected_count = gru_parameter_count + classifier_parameter_count;

        assert_eq!(
            gradients.len(),
            expected_count,
            "Model gradient size mismatch: expected {}, got {}",
            expected_count,
            gradients.len()
        );

        self.gru
            .apply_gradients(&gradients[..gru_parameter_count], learning_rate);

        self.classifier
            .apply_gradients(&gradients[gru_parameter_count..], learning_rate);
    }

    fn zero_grad(&mut self) {
        self.gru.zero_grad();
        self.classifier.zero_grad();
    }
}
