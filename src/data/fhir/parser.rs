use chrono::{Datelike, NaiveDate, Utc};
use std::collections::BTreeMap;

use crate::models::PatientSequence;

use super::{
    mapping::{FhirMappingConfig, StaticFeatureMapping},
    resources::{
        Bundle, CodeableConcept, Condition, DocumentReference, ImagingStudy, Observation, Patient,
        Reference, Resource,
    },
};
use crate::error::FhirError;

#[derive(Debug, Clone)]
pub struct FhirPatientData {
    pub patient_id: String,
    pub sequence: PatientSequence,
    pub text: Option<String>,
    pub image_references: Vec<String>,
}

pub struct FhirParser {
    config: FhirMappingConfig,
}

impl FhirParser {
    pub fn new(config: FhirMappingConfig) -> Self {
        Self { config }
    }

    pub fn parse_bundle(&self, bundle: &Bundle) -> Result<FhirPatientData, FhirError> {
        let patient = self.extract_patient(bundle)?;

        let observations = self.extract_observations(bundle);
        let conditions = self.extract_conditions(bundle);
        let documents = self.extract_documents(bundle);
        let imaging = self.extract_imaging(bundle);

        let patient_id = patient.id.clone().ok_or(FhirError::MissingPatient)?;

        let static_features = self.build_static_features(&patient, &observations, &conditions)?;

        let dynamic_features = self.build_dynamic_features(&observations)?;

        Ok(FhirPatientData {
            patient_id,
            sequence: PatientSequence {
                static_features,
                dynamic_features,
            },
            text: self.extract_text_content(&documents),
            image_references: self.extract_image_references(&imaging),
        })
    }

    fn extract_patient(&self, bundle: &Bundle) -> Result<Patient, FhirError> {
        self.resources(bundle)
            .into_iter()
            .find_map(|resource| match resource {
                Resource::Patient(patient) => Some(patient),
                _ => None,
            })
            .ok_or(FhirError::MissingPatient)
    }

    fn extract_observations(&self, bundle: &Bundle) -> Vec<Observation> {
        self.resources(bundle)
            .into_iter()
            .filter_map(|resource| match resource {
                Resource::Observation(observation) => Some(observation),
                _ => None,
            })
            .collect()
    }

    fn extract_conditions(&self, bundle: &Bundle) -> Vec<Condition> {
        self.resources(bundle)
            .into_iter()
            .filter_map(|resource| match resource {
                Resource::Condition(condition) => Some(condition),
                _ => None,
            })
            .collect()
    }

    fn extract_documents(&self, bundle: &Bundle) -> Vec<DocumentReference> {
        self.resources(bundle)
            .into_iter()
            .filter_map(|resource| match resource {
                Resource::DocumentReference(document) => Some(document),
                _ => None,
            })
            .collect()
    }

    fn extract_imaging(&self, bundle: &Bundle) -> Vec<ImagingStudy> {
        self.resources(bundle)
            .into_iter()
            .filter_map(|resource| match resource {
                Resource::ImagingStudy(study) => Some(study),
                _ => None,
            })
            .collect()
    }

    fn resources(&self, bundle: &Bundle) -> Vec<Resource> {
        bundle
            .entry
            .as_ref()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.resource.clone())
            .collect()
    }

    fn build_static_features(
        &self,
        patient: &Patient,
        observations: &[Observation],
        conditions: &[Condition],
    ) -> Result<Vec<f32>, FhirError> {
        let mut features = vec![0.0; self.config.static_feature_count()];

        if let Some(birth_date) = &patient.birth_date {
            if let Ok(date) = NaiveDate::parse_from_str(birth_date, "%Y-%m-%d") {
                if let Some(index) = self
                    .config
                    .static_features
                    .iter()
                    .find(|mapping| mapping.feature_name == "age")
                    .map(|mapping| mapping.feature_index)
                {
                    features[index] = calculate_age(date) as f32;
                }
            }
        }

        for observation in observations {
            let Some(value) = extract_observation_value(observation) else {
                continue;
            };

            let Some(mapping) = self.find_static_mapping(&observation.code) else {
                continue;
            };

            features[mapping.feature_index] = apply_conversion(value, observation, mapping);
        }

        for condition in conditions {
            let Some((system, code)) = condition_code(condition) else {
                continue;
            };

            for mapping in &self.config.condition_flags {
                if mapping.matches(system.as_deref(), &code) {
                    features[mapping.feature_index] = 1.0;
                }
            }
        }

        Ok(features)
    }

    fn build_dynamic_features(
        &self,
        observations: &[Observation],
    ) -> Result<Vec<Vec<f32>>, FhirError> {
        let feature_count = self.config.dynamic_feature_count();

        if feature_count == 0 {
            return Ok(Vec::new());
        }

        let mut points: BTreeMap<String, Vec<f32>> = BTreeMap::new();

        for observation in observations {
            let Some(timestamp) = observation_timestamp(observation) else {
                continue;
            };

            let Some(mapping) = self.find_dynamic_mapping(&observation.code) else {
                continue;
            };

            let Some(value) = extract_observation_value(observation) else {
                continue;
            };

            let entry = points
                .entry(timestamp)
                .or_insert_with(|| vec![mapping.default_value; feature_count]);

            entry[mapping.feature_index] = apply_conversion_dynamic(value, observation, mapping);
        }

        let mut sequence: Vec<Vec<f32>> = points.into_values().collect();

        if sequence.len() > self.config.max_sequence_length {
            let start = sequence.len() - self.config.max_sequence_length;
            sequence = sequence.split_off(start);
        }

        Ok(sequence)
    }

    fn find_static_mapping(&self, code: &CodeableConcept) -> Option<&StaticFeatureMapping> {
        self.config.static_features.iter().find(|mapping| {
            code.coding.iter().any(|coding| {
                coding.system == mapping.system
                    && coding.code.as_deref() == Some(mapping.code.as_str())
            })
        })
    }

    fn find_dynamic_mapping(
        &self,
        code: &CodeableConcept,
    ) -> Option<&super::mapping::DynamicFeatureMapping> {
        self.config.dynamic_features.iter().find(|mapping| {
            code.coding.iter().any(|coding| {
                coding.system == mapping.system
                    && coding.code.as_deref() == Some(mapping.code.as_str())
            })
        })
    }

    fn extract_text_content(&self, documents: &[DocumentReference]) -> Option<String> {
        let mut text = Vec::new();

        for document in documents {
            for content in &document.content {
                if let Some(data) = &content.attachment.data {
                    text.push(data.clone());
                }
            }
        }

        if text.is_empty() {
            None
        } else {
            Some(text.join("\n"))
        }
    }

    fn extract_image_references(&self, imaging: &[ImagingStudy]) -> Vec<String> {
        imaging
            .iter()
            .flat_map(|study| study.endpoint.clone().unwrap_or_default())
            .filter_map(|reference| reference.reference)
            .collect()
    }
}

fn extract_observation_value(observation: &Observation) -> Option<f32> {
    if let Some(quantity) = &observation.value_quantity {
        return quantity.value.map(|value| value as f32);
    }

    observation
        .value_string
        .as_ref()
        .and_then(|value| value.parse::<f32>().ok())
}

fn observation_timestamp(observation: &Observation) -> Option<String> {
    observation.effective_date_time.clone().or_else(|| {
        observation
            .effective_period
            .as_ref()
            .and_then(|period| period.start.clone())
    })
}

fn condition_code(condition: &Condition) -> Option<(Option<String>, String)> {
    let coding = condition
        .code
        .as_ref()?
        .coding
        .iter()
        .find(|coding| coding.code.is_some())?;
    Some((coding.system.clone(), coding.code.clone()?))
}

fn apply_conversion(value: f32, observation: &Observation, mapping: &StaticFeatureMapping) -> f32 {
    apply_conversion_value(value, observation, mapping.unit_conversion.as_ref())
}

fn apply_conversion_dynamic(
    value: f32,
    observation: &Observation,
    mapping: &super::mapping::DynamicFeatureMapping,
) -> f32 {
    apply_conversion_value(value, observation, mapping.unit_conversion.as_ref())
}

fn apply_conversion_value(
    value: f32,
    observation: &Observation,
    conversion: Option<&super::mapping::UnitConversion>,
) -> f32 {
    let Some(conversion) = conversion else {
        return value;
    };

    let source_unit = observation
        .value_quantity
        .as_ref()
        .and_then(|quantity| quantity.code.as_deref())
        .or_else(|| {
            observation
                .value_quantity
                .as_ref()
                .and_then(|quantity| quantity.unit.as_deref())
        });

    if source_unit == Some(conversion.from_unit.as_str()) {
        value * conversion.multiplier
    } else {
        value
    }
}

fn calculate_age(birth_date: NaiveDate) -> i32 {
    let today = Utc::now().date_naive();

    let mut age = today.year() - birth_date.year();

    if (today.month(), today.day()) < (birth_date.month(), birth_date.day()) {
        age -= 1;
    }

    age.max(0)
}
