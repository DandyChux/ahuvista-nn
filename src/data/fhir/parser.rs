use chrono::{NaiveDate, Utc};

use crate::{
    core::PredictionInput,
    data::{
        mapping::{FhirMappingConfig, TabularFeatureMapping},
        resources::{Bundle, CodeableConcept, ObservationResource, PatientResource},
    },
};

pub struct FhirParser {
    config: FhirMappingConfig,
}

impl FhirParser {
    pub fn new(config: FhirMappingConfig) -> Self {
        Self { config }
    }

    /// Main entry point: Bundle → PredictionInput
    pub fn bundle_to_prediction_input(
        &self,
        bundle: &Bundle,
    ) -> Result<PredictionInput, FhirError> {
        // Extract resources by type
        let patient = self.extract_patient(bundle)?;
        let observations = self.extract_observations(bundle);
        let documents = self.extract_documents(bundle);
        let conditions = self.extract_conditions(bundle);
        let imaging = self.extract_imaging(bundle);

        Ok(PredictionInput {
            patient_id: patient.id.clone(),
            tabular: Some(self.build_tabular_features(&patient, &observations, &conditions)?),
            temporal: Some(self.build_temporal_features(&observations)?),
            text: self.extract_text_content(&documents),
            image_path: self.extract_image_reference(&imaging),
        })
    }

    fn build_tabular_features(
        &self,
        patient: &PatientResource,
        observations: &[ObservationResource],
        conditions: &[ConditionResource],
    ) -> Result<Vec<f32>, FhirError> {
        let mut features = vec![0.0; self.config.expected_tabular_size];

        // Age from birthdate
        if let Some(birth_date) = &patient.birth_date {
            if let Ok(dob) = NaiveDate::parse_from_str(birth_date, "%Y-%m-%d") {
                let age = calculate_age(dob);
                // Assuming age is feature index 0
                features[0] = age as f32;
            }
        }

        // Map observations to features by LOINC code
        for obs in observations {
            if let Some(value) = self.extract_observation_value(obs) {
                if let Some(mapping) = self.find_tabular_mapping(&obs.code) {
                    let converted_value = self.convert_units(value, obs, &mapping);
                    features[mapping.feature_index] = converted_value;
                }
            }
        }

        // Map conditions to binary flags
        for condition in conditions {
            if let Some(mapping) = self.find_condition_mapping(&condition.code) {
                features[mapping.feature_index] = 1.0;
            }
        }

        Ok(features)
    }

    fn build_temporal_features(
        &self,
        observations: &[ObservationResource],
    ) -> Result<Vec<Vec<f32>>, FhirError> {
        // Group observations by timestamp
        let mut time_points: BTreeMap<String, Vec<f32>> = BTreeMap::new();

        for obs in observations {
            let timestamp = obs
                .effective_date_time
                .as_ref()
                .or(obs.effective_period.as_ref().and_then(|p| p.start.as_ref()));

            if let Some(ts) = timestamp {
                if let Some(mapping_idx) = self.find_temporal_feature_index(&obs.code) {
                    let entry = time_points
                        .entry(ts.clone())
                        .or_insert_with(|| vec![0.0; self.config.expected_temporal_feature_count]);

                    if let Some(value) = self.extract_observation_value(obs) {
                        entry[mapping_idx] = value;
                    }
                }
            }
        }

        // Convert to Vec<Vec<f32>>, sorted by time
        let mut result: Vec<Vec<f32>> = time_points.into_values().collect();

        // Truncate or pad to max sequence length
        result.truncate(self.config.temporal_features.max_sequence_length);

        Ok(result)
    }

    fn extract_observation_value(&self, obs: &ObservationResource) -> Option<f32> {
        // Handle different value types
        if let Some(qty) = &obs.value_quantity {
            return qty.value.map(|v| v as f32);
        }

        // Handle component observations (e.g., blood pressure)
        if let Some(components) = &obs.component {
            // For BP, might need special handling
            // Return systolic as primary value, handle diastolic separately
        }

        None
    }

    fn find_tabular_mapping(&self, code: &CodeableConcept) -> Option<&TabularFeatureMapping> {
        // Search by LOINC first, then SNOMED
        for coding in code.coding.as_ref()?.iter() {
            if coding.system.as_deref() == Some("http://loinc.org") {
                return self
                    .config
                    .tabular_features
                    .iter()
                    .find(|m| Some(m.loinc_code.as_str()) == coding.code.as_deref());
            }
        }
        None
    }
}
