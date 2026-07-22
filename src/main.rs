//! Main training program for the multi-modal neural network.
//!
//! Demonstrates how to:
//! 1. Load and preprocess data from different modalities
//! 2. Build and configure the modular late fusion network
//! 3. Train the model with population-informed class balancing
//! 4. Evaluate performance with calibrated predictions
//! 5. Make predictions on new data for maternal mortality risk

use chrono::{Local, Utc};
use clap::Parser;
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fmt::Debug, fs, path::Path};
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt};

use ahuvista_nn::{
    audit::{AuditLogger, PredictionAuditRecord},
    config::{AppConfig, ModalityConfig, Settings, TrainingParams},
    core::fusion::{LateFusionNet, MultiModalInput},
    data::{
        db::DatabaseManager,
        loader::{
            DataConfig, create_data_loader, load_all_image_data, load_all_outcomes,
            load_all_tabular_data, load_all_temporal_data, load_all_text_data,
        },
        preprocessing::{extract_patient_strata, split_data_with_weights},
    },
    training::{
        balancing::{BayesianCalibrator, CauseWeights, SmoteSynthesizer, compute_sample_weights},
        training_loop::{
            evaluate_by_cause, evaluate_model_with_calibration, train_model_with_weights,
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

    /// Path to the prepared database (created by ahuvista-prepare)
    /// If not provided, will look for ahuvista.db in current directory
    #[arg(long, default_value = "ahuvista.db")]
    database: String,

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

    /// Disable differential privacy (for debugging only)
    #[arg(long)]
    no_privacy: bool,

    /// Privacy epsilon (default 10.0, smaller = more privacy)
    #[arg(long)]
    epsilon: Option<f32>,

    /// Gradient clipping norm (default 5.0)
    #[arg(long)]
    clip_norm: Option<f32>,

    /// Enable adaptive learning rate
    #[arg(long)]
    adaptive_lr: bool,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> AppResult<()> {
    // Parse command-line arguments
    let args = Args::parse();

    // Generate log filename with timestamp
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
    let log_path = format!("logs/training_{}.jsonl", timestamp);
    fs::create_dir_all("logs")?;

    // Create file appender
    let log_file = fs::File::create(&log_path).map_err(|e| {
        AppError::from(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to create log file: {}", e),
        ))
    })?;

    // Initialize logging with both console and file output
    // The file layer must use the same format type for compatibility
    if args.verbose {
        let file_layer = fmt::layer().with_writer(log_file).with_ansi(false).pretty();

        tracing_subscriber::registry()
            .with(fmt::layer().pretty())
            .with(file_layer)
            .init();
    } else {
        let file_layer = fmt::layer().with_writer(log_file).with_ansi(false).json();

        tracing_subscriber::registry()
            .with(fmt::layer().json())
            .with(file_layer)
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
        warn!("Privacy DISABLED - for debugging only!");
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

    // Validate modality configuration
    settings
        .modality_config
        .validate()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    info!("Configuration:");
    info!(
        "  Enabled modalities: {:?}",
        settings.modality_config.enabled_modalities()
    );
    info!("  Epochs: {}", settings.training.epochs);
    info!("  Batch size: {}", settings.training.batch_size);
    info!("  Learning rate: {}", settings.training.learning_rate);
    info!("  Database: {:?}", args.database);

    // =========================================================
    // Step 1: LOAD DATA FROM DATABASE (built with prepare binary)
    // =========================================================

    info!("Loading data from database: {}", args.database);

    let db = DatabaseManager::new(&args.database).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Failed to open database '{}'. Did you run 'ahuvista-prepare' first? Error: {}",
                args.database, e
            ),
        )
    })?;

    // Check what data is available in the database
    let counts = db.get_sample_counts().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to get sample counts: {}", e),
        )
    })?;

    info!("Database contents:");
    info!("  Tabular samples:  {}", counts.tabular);
    info!("  Temporal sequences: {}", counts.temporal);
    info!("  Text documents:   {}", counts.text);
    info!("  Images:           {}", counts.image);
    info!("  Outcomes:         {}", counts.outcomes);

    if counts.outcomes == 0 {
        return Err(AppError::from(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "No data in database. Run 'ahuvista-prepare --data-dir datasets/' first.",
        )));
    }

    // Load tabular data if enabled
    let mut tabular_data = if settings.modality_config.use_tabular && counts.tabular > 0 {
        let data = db.load_tabular_data().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to load tabular data: {}", e),
            )
        })?;
        info!(
            "  ✓ Loaded tabular data: {} samples, {} features",
            data.data.len(),
            data.feature_dim()
        );
        Some(data)
    } else if settings.modality_config.use_tabular {
        warn!("  ✗ Tabular modality enabled but no data in database");
        None
    } else {
        info!("  ✗ Tabular modality disabled, skipping");
        None
    };

    // Load temporal data if enabled
    let mut temporal_data = if settings.modality_config.use_temporal && counts.temporal > 0 {
        let data = db.load_temporal_data().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to load temporal data: {}", e),
            )
        })?;
        info!(
            "  ✓ Loaded temporal data: {} sequences, {} features per timestep",
            data.sequences.len(),
            data.feature_dim()
        );
        Some(data)
    } else if settings.modality_config.use_temporal {
        warn!("  ✗ Temporal modality enabled but no data in database");
        None
    } else {
        info!("  ✗ Temporal modality disabled, skipping");
        None
    };

    // Load text data if enabled
    let mut text_data = if settings.modality_config.use_text && counts.text > 0 {
        let data = db.load_text_data().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to load text data: {}", e),
            )
        })?;
        info!(
            "  ✓ Loaded text data: {} documents, vocabulary size {}",
            data.documents.len(),
            data.vocab_size()
        );
        Some(data)
    } else if settings.modality_config.use_text {
        warn!("  ✗ Text modality enabled but no data in database");
        None
    } else {
        info!("  ✗ Text modality disabled, skipping");
        None
    };

    // Load image data if enabled
    let mut image_data = if settings.modality_config.use_image && counts.image > 0 {
        let data = db.load_image_data().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to load image data: {}", e),
            )
        })?;
        info!(
            "  ✓ Loaded image data: {} images, size {}x{}",
            data.images.len(),
            data.image_size.0,
            data.image_size.1
        );
        Some(data)
    } else if settings.modality_config.use_image {
        warn!("  ✗ Image modality enabled but no data in database");
        None
    } else {
        info!("  ✗ Image modality disabled, skipping");
        None
    };

    // Load outcomes from database
    let (all_targets, all_causes) = db.load_outcomes().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to load outcomes: {}", e),
        )
    })?;
    info!("  Loaded outcomes: {} patients", all_targets.len());

    // Determine the actual number of samples based on enabled modalities
    let mut sample_counts: Vec<usize> = vec![];
    if let Some(ref mut tab_data) = tabular_data {
        sample_counts.push(tab_data.data.len());
    }
    if let Some(ref mut temp_data) = temporal_data {
        sample_counts.push(temp_data.sequences.len());
    }
    if let Some(ref mut txt_data) = text_data {
        sample_counts.push(txt_data.documents.len());
    }
    if let Some(ref mut img_data) = image_data {
        sample_counts.push(img_data.images.len());
    }
    let num_samples = if sample_counts.is_empty() {
        return Err(AppError::from(std::io::Error::new(
            std::io::ErrorKind::Other,
            "No data loaded for any enabled modality",
        )));
    } else {
        *sample_counts.iter().min().unwrap()
    };

    // Truncate targets and causes to match the data size (assuming patient ordering is consistent across files)
    // Truncate all modality data to ensure consistent sizes
    if let Some(ref mut tab_data) = tabular_data {
        tab_data.data.truncate(num_samples);
        tab_data.patient_ids.truncate(num_samples);
    }
    if let Some(ref mut temp_data) = temporal_data {
        temp_data.sequences.truncate(num_samples);
    }
    if let Some(ref mut txt_data) = text_data {
        txt_data.documents.truncate(num_samples);
        txt_data.token_sequences.truncate(num_samples);
    }
    if let Some(ref mut img_data) = image_data {
        img_data.images.truncate(num_samples);
    }

    let targets = all_targets
        .into_iter()
        .take(num_samples)
        .collect::<Vec<_>>();
    let causes = all_causes.into_iter().take(num_samples).collect::<Vec<_>>();

    info!("  Using {} samples with matching targets", num_samples);

    // Validate that we have data for all enabled modalities
    if settings.modality_config.use_tabular && tabular_data.is_none() {
        return Err(AppError::from(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Tabular modality enabled but no data loaded",
        )));
    }
    if settings.modality_config.use_image && image_data.is_none() {
        return Err(AppError::from(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Image modality enabled but no data loaded",
        )));
    }
    if settings.modality_config.use_text && text_data.is_none() {
        return Err(AppError::from(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Text modality enabled but no data loaded",
        )));
    }
    if settings.modality_config.use_temporal && temporal_data.is_none() {
        return Err(AppError::from(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Temporal modality enabled but no data loaded",
        )));
    }

    // Step 2: Configure population-based weighting and calibration
    info!("Configuring population-based weighting and calibration...");

    // Define population-level cause frequencies
    let mut cause_frequencies = HashMap::new();
    cause_frequencies.insert(0, 0.21); // Hemorrhage: 21% of deaths
    cause_frequencies.insert(1, 0.20); // Hypertensive disorders: 20%
    cause_frequencies.insert(2, 0.11); // Infection/sepsis: 11%
    cause_frequencies.insert(3, 0.08); // Abortion + Miscarriage: 8%
    cause_frequencies.insert(4, 0.40); // Other causes: 40%

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

    let sample_weights = SmoteSynthesizer::apply_cause_weights(
        &base_sample_weights,
        &causes,
        &cause_weights,
        &targets,
        Some(5.0), // Cap to prevent extreme gradients
    );

    // Calculate actual prevalence from the data
    let positive_samples = targets.iter().filter(|&&t| t > 0.5).count();
    let total_samples = targets.len();
    let actual_prevalence = positive_samples as f32 / total_samples as f32;

    info!(
        "Dataset statistics: {} positive samples out of {} total ({:.2}% prevalence)",
        positive_samples,
        total_samples,
        actual_prevalence * 100.0
    );

    // Create Bayesian calibrator with actual prevalence or a reasonable estimate
    // For maternal mortality, typical rates are 0.5-2% in developed countries
    let calibration_prevalence = if actual_prevalence > 0.0 {
        actual_prevalence.max(0.005) // Use at least 0.5% to avoid over-calibration
    } else {
        0.01 // Default to 1% if no positive samples (shouldn't happen but just in case)
    };

    // Create Bayesian calibrator
    let calibrator = BayesianCalibrator::new(calibration_prevalence);
    info!(
        "Using calibration prevalence: {:.2}%",
        calibration_prevalence * 100.0
    );

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

    let (train_data, val_data, train_targets, val_targets, mut train_weights, val_weights) =
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

    // Initialize output bias to match base rate for faster convergence
    model.set_output_bias_for_base_rate(actual_prevalence);
    info!(
        "  Output bias initialized for {:.2}% base rate",
        actual_prevalence * 100.0
    );

    // Step 6: Create SMOTE synthesizer for minority class oversampling
    let _smote = SmoteSynthesizer::new(5);

    // Step 7: Train the model
    info!("Training model with population-weighted samples...");

    train_model_with_weights(
        &mut model,
        &mut train_loader,
        &mut val_loader,
        &settings.training,
        &mut train_weights,
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
    let model_path = format!("{}/ahuvista-nn.bin", args.output_dir);
    model.save(&model_path)?;
    info!("  Model saved to: {}", model_path);

    Ok(())
}
