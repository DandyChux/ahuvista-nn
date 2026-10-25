use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use tracing::info;

use crate::{
    data::{
        db::DatabaseManager,
        source::{ClassifiedInput, InputFormat, InputKind},
    },
    error::{AppError, AppResult},
    models::PatientSequence,
};

/// A generic record from any CSV file
#[derive(Debug, Clone, Default)]
pub struct GenericRecord {
    pub values: HashMap<String, String>,
    pub headers: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PatientObservation {
    pub timestamp: String,
    pub feature_name: String,
    // pub code: String,
    pub value: f32,
    // pub unit: Option<String>,
}

/// Patient data aggregated from all sources
#[derive(Debug, Default, Clone)]
pub struct PatientData {
    pub patient_id: String,
    pub age: Option<f32>,
    pub race: Option<String>,
    pub sex: Option<String>,
    pub insurance: Option<String>,
    pub systolic_bp: Option<f32>,
    pub diastolic_bp: Option<f32>,
    pub glucose: Option<f32>,
    pub bmi: Option<f32>,
    pub pregnancies: Option<u32>,
    pub risk_level: Option<String>,
    pub existing_outcome: Option<f32>,
    pub cause_code: Option<u32>,
    pub extra_features: HashMap<String, f32>,
    pub observations: Vec<PatientObservation>,
    pub clinical_notes: Vec<String>,
    pub image_count: usize,
    pub has_ultrasound: bool,
    pub has_complications: bool,
    pub risk_score: f32,
}

impl PatientData {
    /// Convert patient data to a feature vector.
    /// If extra_keys is provided, uses those keys in order (with 0.0 for missing features).
    /// Otherwise, uses this patient's extra features sorted alphabetically.
    pub fn to_raw_features(&self, extra_keys: Option<&[String]>) -> Vec<f32> {
        let owned_keys: Vec<String>;
        let keys = match extra_keys {
            Some(keys) => keys,
            None => {
                owned_keys = {
                    let mut keys: Vec<_> = self.extra_features.keys().cloned().collect();
                    keys.sort();
                    keys
                };
                &owned_keys
            }
        };

        let mut features = vec![
            self.age.unwrap_or_default(),
            self.systolic_bp.unwrap_or_default(),
            self.diastolic_bp.unwrap_or_default(),
            self.glucose.unwrap_or_default(),
            self.bmi.unwrap_or_default(),
            self.pregnancies.unwrap_or_default() as f32,
            if self.has_complications { 1.0 } else { 0.0 },
            self.risk_score,
        ];

        // Add extra features in the order specified by keys
        // Use 0.0 for missing features
        for key in keys {
            features.push(*self.extra_features.get(key).unwrap_or(&0.0));
        }

        features
    }

    pub fn to_static_features(&self, feature_names: &[String]) -> Vec<f32> {
        feature_names
            .iter()
            .map(|name| match name.as_str() {
                "age" => self.age.unwrap_or_default(),
                "systolic_bp" => self.systolic_bp.unwrap_or_default(),
                "diastolic_bp" => self.diastolic_bp.unwrap_or_default(),
                "glucose" => self.glucose.unwrap_or_default(),
                "bmi" => self.bmi.unwrap_or_default(),
                "pregnancies" => self.pregnancies.unwrap_or_default() as f32,
                "has_complications" => {
                    if self.has_complications {
                        1.0
                    } else {
                        0.0
                    }
                }
                name => self.extra_features.get(name).copied().unwrap_or_default(),
            })
            .collect()
    }

    pub fn to_dynamic_features(&self, feature_names: &[String]) -> Vec<Vec<f32>> {
        self.observations
            .iter()
            .map(|observation| {
                feature_names
                    .iter()
                    .map(|feature_name| {
                        if feature_name == &observation.feature_name {
                            observation.value
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    pub fn to_patient_sequence(
        &self,
        static_feature_names: &[String],
        dynamic_feature_names: &[String],
    ) -> PatientSequence {
        PatientSequence {
            static_features: self.to_static_features(static_feature_names),
            dynamic_features: self.to_dynamic_features(dynamic_feature_names),
        }
    }

    /// Merge another PatientData into this one (for combining data from multiple CSVs)
    pub fn merge(&mut self, other: &Self) {
        if self.patient_id != other.patient_id {
            return;
        }

        if other.age.is_some() {
            self.age = other.age;
        }

        if other.sex.is_some() {
        	self.sex = other.sex.clone();
        }

        if other.race.is_some() {
            self.race = other.race.clone();
        }

        if other.insurance.is_some() {
            self.insurance = other.insurance.clone();
        }

        if other.systolic_bp.is_some() {
            self.systolic_bp = other.systolic_bp;
        }

        if other.diastolic_bp.is_some() {
            self.diastolic_bp = other.diastolic_bp;
        }

        if other.glucose.is_some() {
            self.glucose = other.glucose;
        }

        if other.bmi.is_some() {
            self.bmi = other.bmi;
        }

        if other.pregnancies.is_some() {
            self.pregnancies = other.pregnancies;
        }

        if other.risk_level.is_some() {
            self.risk_level = other.risk_level.clone();
        }

        if other.existing_outcome.is_some() {
            self.existing_outcome = other.existing_outcome;
        }

        if other.cause_code.is_some() {
            self.cause_code = other.cause_code;
        }

        for (name, value) in &other.extra_features {
            self.extra_features.insert(name.clone(), *value);
        }

        self.observations.extend(other.observations.iter().cloned());

        self.clinical_notes
            .extend(other.clinical_notes.iter().cloned());

        self.image_count += other.image_count;
        self.has_ultrasound |= other.has_ultrasound;
        self.has_complications |= other.has_complications;

        if other.risk_score != 0.0 {
            self.risk_score = other.risk_score;
        }
    }
}

pub struct IngestionService {
    database: DatabaseManager,
}

impl IngestionService {
    pub fn new(database: DatabaseManager) -> Self {
        Self { database }
    }

    pub fn database(&self) -> &DatabaseManager {
        &self.database
    }

    pub fn database_mut(&mut self) -> &mut DatabaseManager {
        &mut self.database
    }

    pub fn into_database(self) -> DatabaseManager {
        self.database
    }

    pub fn import_csv_directory(
        &mut self,
        directory: &Path,
        patient_id_column: &str,
        dataset_source: Option<&str>,
    ) -> AppResult<usize> {
        if !directory.exists() {
            return Ok(0);
        }

        let imported =
            self.database
                .import_csv_directory(directory, patient_id_column, dataset_source)?;

        info!("Imported {} records from {}", imported, directory.display());

        Ok(imported)
    }

    pub fn build_prepared_data(&mut self) -> AppResult<usize> {
        info!("Preparing: removing non-maternal patients");

        let before = self.database.count_patients()?;

        info!(
        	"Patients before maternal clenaup: {}",
         	before
        );

        let removed = self.database.cleanup_non_maternal_patients()?;

        let after = self.database.count_patients()?;

        info!("Preparing: removed {} non-maternal patients", removed);

        info!(
        	"Patients after maternal clenaup: {}",
         	after
        );

        info!("Preparing: building patient sequences");

        let sequences = self.database.build_patient_sequences()?;

        info!("Preparing: built {} patient sequences", sequences.len());

        info!("Preparing: storing patient sequences");

        if sequences.is_empty() {
            info!("No patient sequences to store");
            return Ok(0);
        }

        self.database.store_patient_sequences(&sequences)?;

        info!("Preparing: stored patient sequences");

        info!("Preparing: building text data");

        let text_data = self.database.build_text_data(10_000)?;

        info!(
            "Preparing: built text data: {} documents",
            text_data.documents.len()
        );

        info!("Preparing: storing text data");

        self.database.store_text_data(&text_data)?;

        info!("Preparing: stored text data");

        info!("Preparing: building outcomes");

        let (targets, causes) = self.database.build_outcomes()?;

        info!("Preparing: built {} outcomes", targets.len());

        info!("Preparing: storing outcomes");

        self.database.store_outcomes(&targets, &causes)?;

        info!("Preparing: stored outcomes");

        Ok(sequences.len())
    }

    /// Prepare explicitly supplied files/directories. Unsupported files are skipped with a warning;
    /// image files are reserved for image ingestion and image metadata CSVs are never imported as
    /// generic tabular data.
    pub fn prepare_inputs(
        &mut self,
        inputs: &[PathBuf],
        patient_id_column: &str,
    ) -> AppResult<usize> {
        let classified = crate::data::source::classify_inputs(inputs, None)?;
        self.prepare_classified_inputs(&classified, patient_id_column)
    }

    pub fn prepare_classified_inputs(
        &mut self,
        classified: &[ClassifiedInput],
        patient_id_column: &str,
    ) -> AppResult<usize> {
        let mut imported = 0;
        info!("Processing classified input files");

        let csv_files = classified
            .iter()
            .filter(|input| {
                matches!(input.format, InputFormat::Csv)
                    && matches!(
                        input.kind,
                        InputKind::Patient
                            | InputKind::Observation
                            | InputKind::ClinicalNote
                            | InputKind::Label
                    )
            })
            .collect::<Vec<_>>();

        for input in classified.iter().filter(|input| {
            matches!(
                input.kind,
                InputKind::FhirBundle
                    | InputKind::FhirResource
                    | InputKind::ImageMetadata
                    | InputKind::Unknown
            )
        }) {
            info!(
                "Not importing {} as generic tabular data ({:?}/{:?})",
                input.path.display(),
                input.kind,
                input.format
            );
        }

        info!("Discovered {} classified tabular files", csv_files.len());

        for input in csv_files {
            let path = &input.path;
            let dataset_source = Some(input.group.as_str());

            info!(
                "Importing CSV file: {}{}",
                path.display(),
                dataset_source
                    .as_deref()
                    .map(|source| format!("[{source}]"))
                    .unwrap_or_default()
            );

            imported +=
                self.database.import_csv_with_mapping(
                        &path,
                        patient_id_column,
                        dataset_source.as_deref(),
                        input.mapping.as_ref(),
                    )?;
        }

        self.build_prepared_data()?;

        Ok(imported)
    }

    pub fn import_fhir_bundle(
        &mut self,
        bundle: &crate::data::fhir::Bundle,
        mapping: &crate::data::fhir::FhirMappingConfig,
    ) -> AppResult<usize> {
        if bundle.resource_type != "Bundle" {
            return Err(AppError::InvalidData(format!(
                "expected a FHIR Bundle resource, got {}",
                bundle.resource_type
            )));
        }

        let parsed = crate::data::fhir::FhirParser::new(mapping.clone())
            .parse_bundle(bundle)
            .map_err(|error| {
                AppError::InvalidData(format!("FHIR bundle parsing failed: {error}"))
            })?;

        let static_features = &parsed.sequence.static_features;
        let patient = PatientData {
            patient_id: parsed.patient_id,
            age: None,
            race: None,
            sex: None,
            insurance: None,
            systolic_bp: None,
            diastolic_bp: None,
            glucose: None,
            bmi: None,
            pregnancies: None,
            risk_level: None,
            existing_outcome: None,
            cause_code: None,
            extra_features: mapping
                .static_features
                .iter()
                .filter_map(|feature| {
                    static_features
                        .get(feature.feature_index)
                        .copied()
                        .map(|value| (feature.feature_name.clone(), value))
                })
                .collect(),
            observations: Vec::new(),
            clinical_notes: parsed.text.into_iter().collect(),
            image_count: parsed.image_references.len(),
            has_ultrasound: !parsed.image_references.is_empty(),
            has_complications: mapping.condition_flags.iter().any(|condition| {
                static_features
                    .get(condition.feature_index)
                    .copied()
                    .unwrap_or_default()
                    != 0.0
            }),
            risk_score: 0.0,
        };

        let patient_id = patient.patient_id.clone();
        self.database.upsert_patient(&patient)?;

        // Preserve existing prepared sequences and replace only this patient's sequence.
        self.database
            .store_fhir_patient_sequence(&patient_id, &parsed.sequence)?;

        Ok(1)
    }
}
