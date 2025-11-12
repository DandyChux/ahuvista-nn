//! Main training program for the multi-modal neural network.
//!
//! Demonstrates how to:
//! 1. Load and preprocess data from different modalities
//! 2. Build and configure the modular late fusion network
//! 3. Train the model with population-informed class balancing
//! 4. Evaluate performance with calibrated predictions
//! 5. Make predictions on new data for maternal mortality risk
mod audit;
mod config;
mod core;
mod data;
mod models;
mod training;
mod utils;

use chrono::Utc;
use clap::Parser;
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fmt::Debug};
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt};

use crate::{
    audit::{AuditLogger, PredictionAuditRecord},
    config::{AppConfig, ModalityConfig, Settings},
    core::fusion::{LateFusionNet, MultiModalInput},
    data::{
        loader::{
            create_data_loader, load_all_image_data, load_all_outcomes, load_all_tabular_data,
            load_all_temporal_data, load_all_text_data, DataConfig,
        },
        preprocessing::{extract_patient_strata, split_data_with_weights},
    },
    training::{
        balancing::{compute_sample_weights, BayesianCalibrator, CauseWeights, SmoteSynthesizer},
        training_loop::{
            evaluate_by_cause, evaluate_model_with_calibration, train_model_with_weights,
            TrainingConfig,
        },
    },
    utils::error::{AppError, AppResult},
};

/// Command-line arguments for the training program
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to configuration file (JSON format)
    #[arg(short, long, value_name = "FILE")]
    config: Option<String>,

    /// Comma-separated list of modalities to use (overrides config file)
    /// Valid values: tabular,temporal,text,image
    #[arg(short, long, value_delimiter = ',')]
    modalities: Option<Vec<String>>,

    /// Number of training epochs (overrides config)
    #[arg(short, long)]
    epochs: Option<usize>,

    /// Batch size (overrides config)
    #[arg(short, long)]
    batch_size: Option<usize>,

    /// Learning rate (overrides config)
    #[arg(short, long)]
    learning_rate: Option<f32>,

    /// Output directory for model and logs
    #[arg(short, long, default_value = "models")]
    output_dir: String,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> AppResult<()> {
    // Parse command-line arguments
    let args = Args::parse();

    // Initialize logging
    if args.verbose {
        tracing_subscriber::registry()
            .with(fmt::layer().pretty())
            .init();
    } else {
        tracing_subscriber::registry()
            .with(fmt::layer().json())
            .init();
    }

    info!("Ahuvista-NN Multi-Modal Network for Maternal Mortality Prediction");
    info!("====================================================================");

    // Load configuration
    let mut settings = if let Some(config_path) = &args.config {
        info!("Loading configuration from: {}", config_path);
        Settings::from_file(config_path).map_err(|e| {
            AppError::from(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to load config file: {}", e),
            ))
        })?
    } else {
        info!("Using default configuration");
        Settings::default()
    };

    // Override modalities from command line if provided
    if let Some(ref modality_names) = args.modalities {
        info!(
            "Overriding modalities from command line: {:?}",
            modality_names
        );
        let modality_strs: Vec<&str> = modality_names.iter().map(|s| s.as_str()).collect();
        settings.modality_config = ModalityConfig::from_names(&modality_strs);
    }

    // Override training parameters if provided
    if let Some(epochs) = args.epochs {
        settings.epochs = epochs;
    }
    if let Some(batch_size) = args.batch_size {
        settings.batch_size = batch_size;
    }
    if let Some(learning_rate) = args.learning_rate {
        settings.learning_rate = learning_rate;
    }

    // Validate modality configuration
    settings
        .modality_config
        .validate()
        .map_err(|e| AppError::from(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

    info!("Configuration:");
    info!(
        "  Enabled modalities: {:?}",
        settings.modality_config.enabled_modalities()
    );
    info!("  Epochs: {}", settings.epochs);
    info!("  Batch size: {}", settings.batch_size);
    info!("  Learning rate: {}", settings.learning_rate);
    info!("  Data directory: {:?}", settings.data_dir);

    // Configure data loading
    let data_config = DataConfig {
        data_dir: settings.data_dir.clone(),
        merge_files: true,
        patient_id_column: settings.patient_id_column.clone(),
        has_headers: true,
    };

    // Step 1: Load and preprocess data (only for enabled modalities)
    info!("Loading and preprocessing data...");

    // Load tabular data if enabled
    let tabular_data = if settings.modality_config.use_tabular {
        let data = load_all_tabular_data(&data_config)?;
        info!(
            "  ✓ Loaded tabular data: {} samples, {} features",
            data.data.len(),
            data.feature_dim()
        );
        Some(data)
    } else {
        info!("  ✗ Tabular modality disabled, skipping");
        None
    };

    // Load temporal data if enabled
    let temporal_data = if settings.modality_config.use_temporal {
        let data = load_all_temporal_data(&data_config)?;
        info!(
            "  ✓ Loaded temporal data: {} sequences, {} features per timestep",
            data.sequences.len(),
            data.feature_dim()
        );
        Some(data)
    } else {
        info!("  ✗ Temporal modality disabled, skipping");
        None
    };

    // Load text data if enabled
    let text_data = if settings.modality_config.use_text {
        let data = load_all_text_data(&data_config)?;
        info!(
            "  ✓ Loaded text data: {} documents, vocabulary size {}",
            data.documents.len(),
            data.vocab_size()
        );
        Some(data)
    } else {
        info!("  ✗ Text modality disabled, skipping");
        None
    };

    // Load image data if enabled
    let image_data = if settings.modality_config.use_image {
        let data = load_all_image_data(&data_config)?;
        info!(
            "  ✓ Loaded image data: {} images, size {}x{}",
            data.images.len(),
            data.image_size.0,
            data.image_size.1
        );
        Some(data)
    } else {
        info!("  ✗ Image modality disabled, skipping");
        None
    };

    // Load target values and mortality causes
    let (targets, causes) = load_all_outcomes(&data_config)?;
    info!("  Loaded outcomes: {} patients", targets.len());

    // Validate that we have data for all enabled modalities
    if settings.modality_config.use_tabular && tabular_data.is_none() {
        return Err(AppError::from(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Tabular modality enabled but no data loaded",
        )));
    }

    // Step 2: Configure population-based weighting and calibration
    info!("Configuring population-based weighting and calibration...");

    // Define population-level cause frequencies
    let mut cause_frequencies = HashMap::new();
    cause_frequencies.insert(0, 0.25); // Hemorrhage: 25% of deaths
    cause_frequencies.insert(1, 0.20); // Hypertensive disorders: 20%
    cause_frequencies.insert(2, 0.15); // Infection/sepsis: 15%
    cause_frequencies.insert(3, 0.10); // Thromboembolism: 10%
    cause_frequencies.insert(4, 0.30); // Other causes: 30%

    let cause_weights = CauseWeights::from_population_frequencies(&cause_frequencies);

    // Define demographic stratum multipliers
    let mut stratum_multipliers = HashMap::new();
    stratum_multipliers.insert("urban_young".to_string(), 1.0);
    stratum_multipliers.insert("urban_old".to_string(), 1.2);
    stratum_multipliers.insert("rural_young".to_string(), 1.5);
    stratum_multipliers.insert("rural_old".to_string(), 1.8);

    // Extract patient strata (from demographic data if tabular is enabled)
    let patient_strata: Vec<String> = if let Some(ref tab_data) = tabular_data {
        extract_patient_strata(tab_data)
    } else {
        vec![]
    };

    // Compute sample weights
    let base_sample_weights = compute_sample_weights(
        &targets,
        if patient_strata.is_empty() {
            None
        } else {
            Some(&patient_strata)
        },
        Some(&stratum_multipliers),
    );

    let sample_weights =
        SmoteSynthesizer::apply_cause_weights(&base_sample_weights, &causes, &cause_weights);

    // Create Bayesian calibrator
    let calibrator = BayesianCalibrator::new(0.001); // 0.1% prevalence

    // Step 3: Split data into training and validation sets
    let tabular_vec = tabular_data
        .as_ref()
        .map(|d| d.data.clone())
        .unwrap_or_default();
    let temporal_vec = temporal_data
        .as_ref()
        .map(|d| d.sequences.clone())
        .unwrap_or_default();
    let text_vec = text_data
        .as_ref()
        .map(|d| d.token_sequences.clone())
        .unwrap_or_default();
    let image_vec = image_data
        .as_ref()
        .map(|d| d.images.clone())
        .unwrap_or_default();

    let (train_data, val_data, train_targets, val_targets, train_weights, val_weights) =
        split_data_with_weights(
            &tabular_vec,
            &temporal_vec,
            &text_vec,
            &image_vec,
            &targets,
            &sample_weights,
            0.8, // 80% for training, 20% for validation
        );

    info!(
        "Split data: {} training samples, {} validation samples",
        train_targets.len(),
        val_targets.len()
    );

    // Step 4: Create multi-modal data loaders
    let mut train_loader = create_data_loader(
        train_data.0,
        train_data.1,
        train_data.2,
        train_data.3,
        train_targets.clone(),
    );

    let mut val_loader = create_data_loader(
        val_data.0.clone(),
        val_data.1.clone(),
        val_data.2.clone(),
        val_data.3.clone(),
        val_targets.clone(),
    );

    // Step 5: Build the modular model
    info!("Building modular multi-modal network...");

    let tabular_params = if settings.modality_config.use_tabular {
        let feature_dim = tabular_data.as_ref().unwrap().feature_dim();
        Some((feature_dim, settings.tabular_hidden_sizes.clone(), 16))
    } else {
        None
    };

    let temporal_params = if settings.modality_config.use_temporal {
        let feature_dim = temporal_data.as_ref().unwrap().feature_dim();
        Some((feature_dim, settings.temporal_hidden_size, 16))
    } else {
        None
    };

    let text_params = if settings.modality_config.use_text {
        let vocab = text_data.as_ref().unwrap().vocab.clone();
        Some((vocab, settings.text_embedding_dim, 32))
    } else {
        None
    };

    let image_params = if settings.modality_config.use_image {
        let img_data = image_data.as_ref().unwrap();
        Some((
            img_data.channels,
            img_data.image_size.0,
            img_data.image_size.1,
        ))
    } else {
        None
    };

    let mut model = LateFusionNet::new(
        settings.modality_config.clone(),
        tabular_params,
        temporal_params,
        text_params,
        image_params,
        &[64, 32],
    )
    .map_err(|e| {
        AppError::from(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to create model: {}", e),
        ))
    })?;

    info!(
        "  Model created with modalities: {:?}",
        model.enabled_modalities()
    );

    // Step 6: Create SMOTE synthesizer for minority class oversampling
    let _smote = SmoteSynthesizer::new(5);

    // Step 7: Train the model
    info!("Training model with population-weighted samples...");

    let training_config = TrainingConfig {
        epochs: settings.epochs,
        batch_size: settings.batch_size,
        learning_rate: settings.learning_rate,
        val_frequency: 1,
    };

    train_model_with_weights(
        &mut model,
        &mut train_loader,
        &mut val_loader,
        &training_config,
        &train_weights,
        &val_weights,
        &calibrator,
    );

    // Step 8: Example prediction with modular input
    info!("Example prediction for high-risk patient...");

    if !val_data.0.is_empty()
        || !val_data.1.is_empty()
        || !val_data.2.is_empty()
        || !val_data.3.is_empty()
    {
        let mut input = MultiModalInput::new();

        // Add data only for enabled modalities
        if settings.modality_config.use_tabular && !val_data.0.is_empty() {
            input = input.with_tabular(val_data.0[0].clone());
        }
        if settings.modality_config.use_temporal && !val_data.1.is_empty() {
            input = input.with_temporal(val_data.1[0].clone());
        }
        if settings.modality_config.use_text && !val_data.2.is_empty() {
            input = input.with_text(val_data.2[0].clone());
        }
        if settings.modality_config.use_image && !val_data.3.is_empty() {
            input = input.with_image(val_data.3[0].clone());
        }

        match model.predict(input) {
            Ok(raw_prediction) => {
                let calibrated_prediction = calibrator.calibrate(raw_prediction);

                info!("  Raw prediction: {:.2}%", raw_prediction * 100.0);
                info!(
                    "  Calibrated prediction: {:.2}%",
                    calibrated_prediction * 100.0
                );
                info!(
                    "  Risk assessment: {}",
                    if calibrated_prediction > 0.5 {
                        "HIGH RISK"
                    } else {
                        "LOW RISK"
                    }
                );

                // Step 9: Audit the prediction
                info!("Logging prediction to audit trail...");
                let audit_logger = AuditLogger::new("logs/prediction_audit.log");

                let mut hasher = Sha256::new();
                hasher.update(format!("{:?}", raw_prediction));
                let input_hash = format!("{:x}", hasher.finalize());

                let record = PredictionAuditRecord {
                    event_timestamp: Utc::now(),
                    model_version: "2.0.0-modular".to_string(),
                    patient_id_hash: "example_patient".to_string(),
                    input_data_hash: input_hash,
                    raw_prediction,
                    calibrated_prediction,
                };

                if let Err(e) = audit_logger.log(&record) {
                    error!("Failed to write to audit log: {}", e);
                }
            }
            Err(e) => {
                error!("Failed to make prediction: {}", e);
            }
        }
    }

    // Step 10: Save model
    info!("Saving trained model...");

    std::fs::create_dir_all(&args.output_dir)?;
    let model_path = format!("{}/maternal_mortality_model_modular.bin", args.output_dir);

    match model.save(&model_path) {
        Ok(_) => info!("  ✓ Model saved successfully to {}", model_path),
        Err(e) => {
            error!("  ✗ Failed to save model: {}", e);
            return Err(AppError::from(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to save model: {}", e),
            )));
        }
    }

    info!("Training complete!");
    info!("Model configuration:");
    info!("  Modalities: {:?}", model.enabled_modalities());
    info!("  Output: {}", model_path);

    Ok(())
}
