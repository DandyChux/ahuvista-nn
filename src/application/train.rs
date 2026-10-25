use std::{collections::HashMap, fs, path::Path};

use rand::seq::SliceRandom;
use tracing::{info, warn};

use crate::{
    config::{ModalityConfig, Settings}, core::fusion::LateFusionNet, data::{
        DataLoader, FeatureStats, db::{DatabaseManager, PreparedLoadOptions}, loader::{PreparedDataset, create_data_loader}, preprocessing::extract_patient_strata,
    }, error::{AppError, AppResult}, model_artifact::{ModelArtifact, ModelSpec}, models::PatientSequence, training::{
        balancing::{BayesianCalibrator, CauseWeights, SmoteSynthesizer, compute_sample_weights},
        training_loop::train_model_with_weights,
    },
};

use super::args::TrainArgs;

pub fn run(args: TrainArgs) -> AppResult<()> {
    let mut settings = load_settings(&args)?;

    apply_cli_overrides(&mut settings, &args);

    settings
        .modality_config
        .validate()
        .map_err(AppError::Config)?;

    info!(
        "Enabled modalities: {:?}",
        settings.modality_config.enabled_modalities()
    );

    let database = DatabaseManager::new(&args.database)?;

    let options =
        PreparedLoadOptions {
            use_sequence:
                settings.modality_config.use_sequence,
            use_text:
                settings.modality_config.use_text,
            use_image:
                settings.modality_config.use_image,
        };

    database.validate_prepared_data(options)?;

    let records =
        database.load_prepared_records(options)?;

    let text_vocabulary = records
        .iter()
        .flat_map(|record| record.text.iter().copied())
        .max()
        .map(|max_token| {
            (0..=max_token)
                .map(|index| (format!("<TOKEN_{index}>"), index))
                .collect()
        })
        .unwrap_or_default();

    let image_metadata = records
        .iter()
        .flat_map(|record| record.images.iter())
        .next()
        .map(|image| {
            let shape = image.shape();
            (shape[0], (shape[1], shape[2]))
        });

    let dataset = PreparedDataset::from_records(
        records,
        text_vocabulary,
        image_metadata.map(|metadata| metadata.0),
        image_metadata.map(|metadata| metadata.1),
    )?;

    if settings.modality_config.use_image {
        dataset.require_images()?;
    }

    if dataset.is_empty() {
        return Err(AppError::InvalidData(
            "The prepared database contains no training samples"
                .to_string(),
        ));
    }

    let sequences =
        dataset.sequences();

    let targets = dataset.targets();

    let positive_count = targets.iter().filter(|&&target| target > 0.5).count();
    let positive_rate = positive_count as f32 / targets.len().max(1) as f32;
    if positive_count < 20 || positive_rate < 0.01 {
        warn!(
            "Only {} of {} prepared records are positive ({:.4}%); metrics and model stability may be unreliable",
            positive_count,
            targets.len(),
            positive_rate * 100.0
        );
    }


    let (mut train_dataset, mut validation_dataset) = stratified_split(&dataset, 0.8)?;

    let train_positive_count = train_dataset
        .targets()
        .iter()
        .filter(|&&target| target > 0.5)
        .count();
    let validation_positive_count = validation_dataset
        .targets()
        .iter()
        .filter(|&&target| target > 0.5)
        .count();
    info!(
        "Stratified split: train positives={}, validation positives={}",
        train_positive_count, validation_positive_count
    );

    // Fit preprocessing statistics on the training split only to avoid leaking
    // validation-distribution information into the model.
    let static_normalization = fit_static_normalization(&train_dataset.sequences());
    apply_static_normalization(&mut train_dataset, &static_normalization);
    apply_static_normalization(&mut validation_dataset, &static_normalization);

    if train_dataset.is_empty() {
        return Err(AppError::InvalidData("Training split is empty".to_string()));
    }

    if validation_dataset.is_empty() {
        return Err(AppError::InvalidData(
            "Validation split is empty".to_string(),
        ));
    }

    let cause_weights = build_cause_weights();
    let stratum_multipliers = build_stratum_multipliers();

    let train_targets = train_dataset.targets();
    let train_causes = train_dataset.causes();
    let train_strata = extract_patient_strata(&train_dataset.sequences());
    let train_base_weights = compute_sample_weights(
        &train_targets,
        Some(&train_strata),
        Some(&stratum_multipliers),
    );
    let train_weights = SmoteSynthesizer::apply_cause_weights(
        &train_base_weights,
        &train_causes,
        &cause_weights,
        &train_targets,
        Some(5.0),
    );

    let validation_targets = validation_dataset.targets();
    let validation_causes = validation_dataset.causes();
    let validation_strata = extract_patient_strata(&validation_dataset.sequences());
    let validation_base_weights = compute_sample_weights(
        &validation_targets,
        Some(&validation_strata),
        Some(&stratum_multipliers),
    );
    let validation_weights = SmoteSynthesizer::apply_cause_weights(
        &validation_base_weights,
        &validation_causes,
        &cause_weights,
        &validation_targets,
        Some(5.0),
    );

    let train_prevalence = calculate_prevalence(&train_targets);

    let calibrator = BayesianCalibrator::new(train_prevalence);

    let mut train_loader = create_data_loader(train_dataset);

    let mut validation_loader = create_data_loader(validation_dataset);

    let metadata = PreparedMetadata {
        text_vocabulary: dataset.text_vocabulary.clone(),
        image_channels: dataset.image_channels,
        image_size: dataset.image_size,
    };

    let (mut model, mut model_spec) = build_model(&settings, &sequences, &metadata)?;
    model_spec.static_normalization = static_normalization;

    model.set_output_bias_for_base_rate(train_prevalence);

    info!(
        "Training with {} samples and validating with {} samples",
        train_loader.len(),
        validation_loader.len()
    );

    let mut mutable_train_weights = train_weights;

    train_model_with_weights(
        &mut model,
        &mut train_loader,
        &mut validation_loader,
        &settings.training,
        &mut mutable_train_weights,
        &validation_weights,
        &calibrator,
    );

    fs::create_dir_all(&args.output_dir)?;

    let model_path = Path::new(&args.output_dir).join("maternal-risk.ahv");

    let artifact = ModelArtifact::from_model(model_spec, &model)?;

    artifact.save(&model_path)?;

    info!("Saved model artifact to {}", model_path.display());

    Ok(())
}

fn load_settings(args: &TrainArgs) -> AppResult<Settings> {
    match args.config.as_deref() {
        Some(path) => Settings::from_file(path),
        None => Ok(Settings::default()),
    }
}

fn apply_cli_overrides(settings: &mut Settings, args: &TrainArgs) {
    if let Some(modalities) = &args.modalities {
        let names: Vec<&str> = modalities.iter().map(String::as_str).collect();

        settings.modality_config = ModalityConfig::from_names(&names);
    }

    if let Some(epochs) = args.epochs {
        settings.training.epochs = epochs;
    }

    if let Some(batch_size) = args.batch_size {
        settings.training.batch_size = batch_size;
    }

    if let Some(learning_rate) = args.learning_rate {
        settings.training.learning_rate = learning_rate;
    }

    if args.no_privacy {
        settings.training.use_privacy = false;
        warn!("Differential privacy disabled");
    }

    if let Some(epsilon) = args.epsilon {
        settings.training.privacy_epsilon = epsilon;
    }

    if let Some(clip_norm) = args.clip_norm {
        settings.training.gradient_clip_norm = clip_norm;
    }

    if args.adaptive_lr {
        settings.training.use_adaptive_lr = true;
    }
}

fn validate_alignment(
    sequences: &[PatientSequence],
    text: &[Vec<usize>],
    images: &[ndarray::Array3<f32>],
    targets: &[f32],
) -> AppResult<()> {
    if sequences.len() != targets.len() {
        return Err(AppError::InvalidData(format!(
            "Sequence count {} does not match target count {}",
            sequences.len(),
            targets.len()
        )));
    }

    if !text.is_empty() && text.len() != targets.len() {
        return Err(AppError::InvalidData(format!(
            "Text count {} does not match target count {}",
            text.len(),
            targets.len()
        )));
    }

    if !images.is_empty() && images.len() != targets.len() {
        return Err(AppError::InvalidData(format!(
            "Image count {} does not match target count {}",
            images.len(),
            targets.len()
        )));
    }

    Ok(())
}

#[derive(Debug, Clone)]
pub struct PreparedMetadata {
	pub text_vocabulary: HashMap<String, usize>,
	pub image_channels: Option<usize>,
	pub image_size: Option<(usize, usize)>
}

fn stratified_split(
    dataset: &PreparedDataset,
    train_ratio: f32,
) -> AppResult<(PreparedDataset, PreparedDataset)> {
    if !(0.0..=1.0).contains(&train_ratio) {
        return Err(AppError::InvalidData(format!(
            "Train ratio must be between 0 and 1, got {}",
            train_ratio
        )));
    }

    let mut positive_indices = Vec::new();
    let mut negative_indices = Vec::new();

    for (index, record) in dataset.records.iter().enumerate() {
        if record.target > 0.5 {
            positive_indices.push(index);
        } else {
            negative_indices.push(index);
        }
    }

    let mut rng = rand::rng();
    positive_indices.shuffle(&mut rng);
    negative_indices.shuffle(&mut rng);

    let split_count = |count: usize| -> usize {
        if count <= 1 {
            return count;
        }

        ((count as f32 * train_ratio).round() as usize).clamp(1, count - 1)
    };

    let positive_train_count = split_count(positive_indices.len());
    let negative_train_count = split_count(negative_indices.len());

    let mut train_indices = positive_indices[..positive_train_count].to_vec();
    train_indices.extend_from_slice(&negative_indices[..negative_train_count]);

    let mut validation_indices = positive_indices[positive_train_count..].to_vec();
    validation_indices.extend_from_slice(&negative_indices[negative_train_count..]);

    train_indices.shuffle(&mut rng);
    validation_indices.shuffle(&mut rng);

    let train_records = train_indices
        .into_iter()
        .map(|index| dataset.records[index].clone())
        .collect();
    let validation_records = validation_indices
        .into_iter()
        .map(|index| dataset.records[index].clone())
        .collect();

    Ok((
        PreparedDataset::from_records(
            train_records,
            dataset.text_vocabulary.clone(),
            dataset.image_channels,
            dataset.image_size,
        )?,
        PreparedDataset::from_records(
            validation_records,
            dataset.text_vocabulary.clone(),
            dataset.image_channels,
            dataset.image_size,
        )?,
    ))
}

fn fit_static_normalization(sequences: &[PatientSequence]) -> Vec<FeatureStats> {
    let width = sequences
        .first()
        .map(PatientSequence::static_size)
        .unwrap_or(0);

    (0..width)
        .map(|index| {
            let values = sequences
                .iter()
                .filter_map(|sequence| sequence.static_features.get(index).copied())
                .collect::<Vec<_>>();
            FeatureStats::new(format!("static_feature_{index}"), &values)
        })
        .collect()
}

fn apply_static_normalization(
    dataset: &mut PreparedDataset,
    statistics: &[FeatureStats],
) {
    for record in &mut dataset.records {
        for (index, value) in record.sequence.static_features.iter_mut().enumerate() {
            if let Some(stats) = statistics.get(index) {
                // Bound standardized values so a corrupt clinical measurement
                // cannot saturate the network during training or inference.
                *value = stats.normalize(*value).clamp(-5.0, 5.0);
            }
        }
    }
}

fn build_model(
    settings: &Settings,
    sequences: &[PatientSequence],
    metadata: &PreparedMetadata,
) -> AppResult<(LateFusionNet, ModelSpec)> {
    let first_sequence = sequences.first();

    let sequence_params = if settings.modality_config.use_sequence {
        let sequence = first_sequence.ok_or_else(|| {
            AppError::InvalidData("Sequence modality enabled but no sequences exist".to_string())
        })?;

        let dynamic_input_size = sequence
            .dynamic_features
            .first()
            .map(Vec::len)
            .unwrap_or(0);

        if dynamic_input_size == 0 {
            warn!(
                "Sequence modality has no dynamic features; training will use static features only"
            );
        }

        Some((
            dynamic_input_size,
            sequence.static_size(),
            settings.model.sequence_hidden_size,
            settings.model.sequence_output_size,
        ))
    } else {
        None
    };

    let text_params = if settings.modality_config.use_text {
    	if settings.modality_config.use_text {
	        Some((
	            metadata.text_vocabulary.clone(),
	            settings.model.text_embedding_dim,
	            settings.model.text_hidden_size,
	        ))
     	} else {
      		None
      	}
    } else {
        None
    };

    let image_params = if settings.modality_config.use_image {
       	let channels = metadata.image_channels.ok_or_else(|| {
      		AppError::InvalidData(
         			"Image modality enabled but the image metadata is missing".to_string()
          		)
       	})?;

       	let (height, width) = metadata.image_size.ok_or_else(|| {
      		AppError::InvalidData(
          		"Image modaity enabled but image dimensions are missing".to_string()
           	)
       	})?;

       	Some((channels, height, width))
    } else {
       	None
    };

    let model = LateFusionNet::new(
        settings.modality_config.clone(),
        sequence_params,
        text_params.clone(),
        image_params,
        &settings.model.classifier_hidden_sizes,
    )
    .map_err(AppError::Config)?;

    let static_size = first_sequence
        .map(PatientSequence::static_size)
        .unwrap_or(0);

    let dynamic_size = first_sequence
        .map(PatientSequence::dynamic_size)
        .unwrap_or(0);

    let static_feature_names = (0..static_size)
        .map(|index| format!("static_feature_{index}"))
        .collect();

    let dynamic_feature_names = (0..dynamic_size)
        .map(|index| format!("dynamic_feature_{index}"))
        .collect();

    let spec = ModelSpec {
        model_id: "ahuvista-maternal-risk".to_string(),
        model_version: env!("CARGO_PKG_VERSION").to_string(),

        modalities: settings.modality_config.clone(),

        sequence_dynamic_input_size: sequence_params.map(|value| value.0),
        sequence_static_input_size: sequence_params.map(|value| value.1),
        sequence_hidden_size: sequence_params.map(|value| value.2),
        sequence_output_size: sequence_params.map(|value| value.3),

        text_embedding_size: text_params.as_ref().map(|value| value.1),
        text_hidden_size: text_params.as_ref().map(|value| value.2),
        text_vocab: text_params
            .as_ref()
            .map(|value| value.0.clone())
            .unwrap_or_default(),

        image_channels: image_params.map(|value| value.0),
        image_height: image_params.map(|value| value.1),
        image_width: image_params.map(|value| value.2),

        classifier_hidden_sizes: settings.model.classifier_hidden_sizes.clone(),

        static_feature_names,
        dynamic_feature_names,

        static_normalization: Vec::new(),
        dynamic_normalization: Vec::new(),

        fhir_mapping_version: None,
    };

    Ok((model, spec))
}

fn build_cause_weights() -> CauseWeights {
    let frequencies = HashMap::from([(0, 0.21), (1, 0.20), (2, 0.11), (3, 0.08), (4, 0.40)]);

    CauseWeights::from_population_frequencies(&frequencies)
}

fn build_stratum_multipliers() -> HashMap<String, f32> {
    HashMap::from([
        ("urban_young".to_string(), 1.0),
        ("urban_old".to_string(), 1.2),
        ("rural_young".to_string(), 1.5),
        ("rural_old".to_string(), 1.8),
    ])
}

fn calculate_prevalence(targets: &[f32]) -> f32 {
    if targets.is_empty() {
        return 0.01;
    }

    let positive_count = targets.iter().filter(|&&target| target > 0.5).count();

    // Preserve the observed prevalence when positives exist. A 0.5% floor
    // would substantially overstate rare outcomes such as hospital mortality.
    (positive_count as f32 / targets.len() as f32).max(0.0001)
}
