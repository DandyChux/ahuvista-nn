//! Lightweight inference binary for making predictions with trained models.
//!
//! This binary is designed to be small and fast, suitable for deployment
//! on edge devices, embedded systems, and low-resource environments.
//!
//! Supports modular models with selective modality processing.

use ahuvista_nn::config::{ModalityConfig, Settings};
use ahuvista_nn::core::PredictionInput;
use ahuvista_nn::core::fusion::{LateFusionNet, MultiModalInput};
use ahuvista_nn::training::balancing::BayesianCalibrator;
use ahuvista_nn::utils::error::AppResult;
use clap::Parser;
use ndarray::Array3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt};

/// Command-line arguments for the prediction program
#[derive(Parser, Debug)]
#[command(author, version, about = "Make predictions with trained Ahuvista-NN models", long_about = None)]
struct Args {
    /// Path to the trained model file
    #[arg(short, long, value_name = "FILE")]
    model: String,

    /// Path to input data (JSON format)
    #[arg(short, long, value_name = "FILE")]
    input: Option<String>,

    /// Path to model configuration file (JSON format)
    /// This should match the configuration used during training
    #[arg(short, long, value_name = "FILE")]
    config: Option<String>,

    /// Comma-separated list of modalities in the model
    /// Valid values: tabular,temporal,text,image
    #[arg(long, value_delimiter = ',')]
    modalities: Option<Vec<String>>,

    /// Output file for prediction results (JSON format)
    #[arg(short, long, default_value = "prediction_output.json")]
    output: String,

    /// Apply Bayesian calibration with specified prevalence
    #[arg(long)]
    calibrate: Option<f32>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Batch prediction mode: input is a directory of JSON files
    #[arg(short, long)]
    batch: bool,
}

/// Metadata about the model (saved alongside model weights)
#[derive(Debug, Serialize, Deserialize)]
struct ModelMetadata {
    modality_config: ModalityConfig,
    tabular_input_size: Option<usize>,
    tabular_hidden_sizes: Option<Vec<usize>>,
    tabular_output_size: Option<usize>,
    temporal_input_size: Option<usize>,
    temporal_hidden_size: Option<usize>,
    temporal_output_size: Option<usize>,
    text_embedding_dim: Option<usize>,
    text_hidden_size: Option<usize>,
    image_channels: Option<usize>,
    image_height: Option<usize>,
    image_width: Option<usize>,
    classifier_hidden_sizes: Vec<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    info!("Ahuvista-NN Prediction System");
    info!("==============================");

    // Load model configuration
    let (model, metadata) = load_model(&args)?;

    info!("Model loaded successfully!");
    info!("  Modalities: {:?}", model.enabled_modalities());

    // Setup calibrator if requested
    let calibrator = args.calibrate.map(BayesianCalibrator::new);
    if let Some(ref cal) = calibrator {
        info!(
            "  Bayesian calibration enabled (prevalence: {})",
            args.calibrate.unwrap()
        );
    }

    // Run prediction(s)
    if args.batch {
        run_batch_predictions(&args, model, &metadata, calibrator.as_ref())?;
    } else {
        run_single_prediction(&args, model, &metadata, calibrator.as_ref())?;
    }

    Ok(())
}

/// Run a single prediction
fn run_single_prediction(
    args: &Args,
    mut model: LateFusionNet,
    metadata: &ModelMetadata,
    calibrator: Option<&BayesianCalibrator>,
) -> Result<(), Box<dyn std::error::Error>> {
    let input_path = args
        .input
        .as_ref()
        .ok_or("Input file required for single prediction")?;

    info!("Loading input from: {}", input_path);
    let input_json = std::fs::read_to_string(input_path)?;
    let input: PredictionInput = serde_json::from_str(&input_json)?;

    // Process input and make prediction
    let output = make_prediction(&mut model, metadata, &input, calibrator)?;

    // Print results
    info!("\n=== Prediction Results ===");
    if let Some(ref pid) = output.patient_id {
        info!("Patient ID: {}", pid);
    }
    info!("Risk Score: {:.2}%", output.risk_score * 100.0);
    if let Some(calibrated) = output.calibrated_score {
        info!("Calibrated Score: {:.2}%", calibrated * 100.0);
    }
    info!("Risk Category: {}", output.risk_category);
    info!("Confidence: {:.2}%", output.confidence * 100.0);
    info!("Modalities Used: {:?}", output.modalities_used);

    // Save output
    let output_json = serde_json::to_string_pretty(&output)?;
    std::fs::write(&args.output, output_json)?;
    info!("\nResults saved to: {}", args.output);

    Ok(())
}

/// Run batch predictions on multiple input files
fn run_batch_predictions(
    args: &Args,
    mut model: LateFusionNet,
    metadata: &ModelMetadata,
    calibrator: Option<&BayesianCalibrator>,
) -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = args
        .input
        .as_ref()
        .ok_or("Input directory required for batch mode")?;

    info!("Running batch predictions from directory: {}", input_dir);

    let mut results = Vec::new();
    let mut success_count = 0;
    let mut error_count = 0;

    // Read all JSON files from directory
    for entry in std::fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            info!("Processing: {:?}", path);

            match process_single_file(&mut model, metadata, &path, calibrator) {
                Ok(output) => {
                    info!(
                        "  ✓ Risk: {:.2}% ({})",
                        output.risk_score * 100.0,
                        output.risk_category
                    );
                    results.push(output);
                    success_count += 1;
                }
                Err(e) => {
                    error!("  ✗ Error: {}", e);
                    error_count += 1;
                }
            }
        }
    }

    // Save batch results
    let batch_output = serde_json::to_string_pretty(&results)?;
    std::fs::write(&args.output, batch_output)?;

    info!("\n=== Batch Prediction Summary ===");
    info!("Total processed: {}", success_count + error_count);
    info!("Successful: {}", success_count);
    info!("Errors: {}", error_count);
    info!("Results saved to: {}", args.output);

    Ok(())
}

/// Process a single input file
fn process_single_file(
    model: &mut LateFusionNet,
    metadata: &ModelMetadata,
    path: &PathBuf,
    calibrator: Option<&BayesianCalibrator>,
) -> Result<PredictionOutput, Box<dyn std::error::Error>> {
    let input_json = std::fs::read_to_string(path)?;
    let input: PredictionInput = serde_json::from_str(&input_json)?;
    make_prediction(model, metadata, &input, calibrator)
}

/// Make a prediction from input data
fn make_prediction(
    model: &mut LateFusionNet,
    metadata: &ModelMetadata,
    input: &PredictionInput,
    calibrator: Option<&BayesianCalibrator>,
) -> Result<PredictionOutput, Box<dyn std::error::Error>> {
    info!("Processing input data...");

    // Build multi-modal input based on what's available and enabled
    let mut multi_modal_input = MultiModalInput::new();

    // Add tabular data if enabled and provided
    if model.is_enabled("tabular") {
        if let Some(ref tabular) = input.tabular {
            multi_modal_input = multi_modal_input.with_tabular(tabular.clone());
            info!("  ✓ Tabular data: {} features", tabular.len());
        } else {
            return Err("Tabular data required but not provided".into());
        }
    }

    // Add temporal data if enabled and provided
    if model.is_enabled("temporal") {
        if let Some(ref temporal) = input.temporal {
            multi_modal_input = multi_modal_input.with_temporal(temporal.clone());
            info!("  ✓ Temporal data: {} timesteps", temporal.len());
        } else {
            return Err("Temporal data required but not provided".into());
        }
    }

    // Add text data if enabled and provided
    if model.is_enabled("text") {
        if let Some(ref text) = input.text {
            // Tokenize text (you'll need to load vocab with the model)
            let tokens = tokenize_text(text, model)?;
            multi_modal_input = multi_modal_input.with_text(tokens);
            info!("  ✓ Text data: {} tokens", text.split_whitespace().count());
        } else {
            return Err("Text data required but not provided".into());
        }
    }

    // Add image data if enabled and provided
    if model.is_enabled("image") {
        if let Some(ref image_path) = input.image_path {
            let image_data = load_and_process_image(
                image_path,
                metadata.image_height.unwrap_or(224),
                metadata.image_width.unwrap_or(224),
            )?;
            multi_modal_input = multi_modal_input.with_image(image_data);
            info!("  ✓ Image data: {}", image_path);
        } else {
            return Err("Image data required but not provided".into());
        }
    }

    // Make prediction
    info!("Making prediction...");
    let risk_score = model.predict(multi_modal_input)?;

    // Apply calibration if requested
    let calibrated_score = calibrator.map(|cal| cal.calibrate(risk_score));

    // Use calibrated score for categorization if available
    let score_for_category = calibrated_score.unwrap_or(risk_score);
    let (risk_category, confidence) = categorize_risk(score_for_category);

    // Create output
    Ok(PredictionOutput {
        risk_score,
        calibrated_score,
        risk_category,
        confidence,
        modalities_used: model
            .enabled_modalities()
            .iter()
            .map(|&s| s.to_string())
            .collect(),
        patient_id: input.patient_id.clone(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}

/// Load a trained model and its metadata
fn load_model(args: &Args) -> AppResult<(LateFusionNet, ModelMetadata)> {
    info!("Loading model from: {}", args.model);

    // Load metadata from config file or derive from command-line args
    let metadata = load_model_metadata(args)?;

    // Create model with the right architecture
    let tabular_params = if metadata.modality_config.use_tabular {
        Some((
            metadata
                .tabular_input_size
                .ok_or("Missing tabular_input_size")?,
            metadata
                .tabular_hidden_sizes
                .clone()
                .ok_or("Missing tabular_hidden_sizes")?,
            metadata
                .tabular_output_size
                .ok_or("Missing tabular_output_size")?,
        ))
    } else {
        None
    };

    let temporal_params = if metadata.modality_config.use_temporal {
        Some((
            metadata
                .temporal_input_size
                .ok_or("Missing temporal_input_size")?,
            metadata
                .temporal_hidden_size
                .ok_or("Missing temporal_hidden_size")?,
            metadata
                .temporal_output_size
                .ok_or("Missing temporal_output_size")?,
        ))
    } else {
        None
    };

    let text_params = if metadata.modality_config.use_text {
        // Vocab will be loaded from model file
        Some((
            HashMap::new(), // Placeholder, will be loaded
            metadata
                .text_embedding_dim
                .ok_or("Missing text_embedding_dim")?,
            metadata
                .text_hidden_size
                .ok_or("Missing text_hidden_size")?,
        ))
    } else {
        None
    };

    let image_params = if metadata.modality_config.use_image {
        Some((
            metadata.image_channels.ok_or("Missing image_channels")?,
            metadata.image_height.ok_or("Missing image_height")?,
            metadata.image_width.ok_or("Missing image_width")?,
        ))
    } else {
        None
    };

    let mut model = LateFusionNet::new(
        metadata.modality_config.clone(),
        tabular_params,
        temporal_params,
        text_params,
        image_params,
        &metadata.classifier_hidden_sizes,
    )?;

    // Load trained weights
    model.load(&args.model)?;

    Ok((model, metadata))
}

/// Load model metadata from config file
fn load_model_metadata(args: &Args) -> AppResult<ModelMetadata> {
    if let Some(ref config_path) = args.config {
        info!("Loading model configuration from: {}", config_path);
        let settings = Settings::from_file(config_path)?;

        // Convert settings to metadata
        Ok(ModelMetadata {
            modality_config: settings.modality_config,
            tabular_input_size: Some(10), // You should save this with the model
            tabular_hidden_sizes: Some(settings.tabular_hidden_sizes),
            tabular_output_size: Some(16),
            temporal_input_size: Some(5),
            temporal_hidden_size: Some(settings.temporal_hidden_size),
            temporal_output_size: Some(16),
            text_embedding_dim: Some(settings.text_embedding_dim),
            text_hidden_size: Some(32),
            image_channels: Some(3),
            image_height: Some(224),
            image_width: Some(224),
            classifier_hidden_sizes: vec![64, 32],
        })
    } else if let Some(ref modalities) = args.modalities {
        info!("Using modalities from command line: {:?}", modalities);
        let modality_strs: Vec<&str> = modalities.iter().map(|s| s.as_str()).collect();

        Ok(ModelMetadata {
            modality_config: ModalityConfig::from_names(&modality_strs),
            tabular_input_size: Some(10),
            tabular_hidden_sizes: Some(vec![64, 32]),
            tabular_output_size: Some(16),
            temporal_input_size: Some(5),
            temporal_hidden_size: Some(32),
            temporal_output_size: Some(16),
            text_embedding_dim: Some(50),
            text_hidden_size: Some(32),
            image_channels: Some(3),
            image_height: Some(224),
            image_width: Some(224),
            classifier_hidden_sizes: vec![64, 32],
        })
    } else {
        Err("Either --config or --modalities must be specified".into())
    }
}

/// Load and preprocess an image
fn load_and_process_image(
    path: &str,
    target_height: usize,
    target_width: usize,
) -> AppResult<Array3<f32>> {
    let img = image::open(path)?;
    let img = img.resize_exact(
        target_width as u32,
        target_height as u32,
        image::imageops::FilterType::Lanczos3,
    );

    let rgb = img.to_rgb8();
    let mut array = Array3::<f32>::zeros((3, target_height, target_width));

    for (x, y, pixel) in rgb.enumerate_pixels() {
        array[[0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
        array[[1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
        array[[2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
    }

    Ok(array)
}

/// Tokenize text using model's vocabulary
fn tokenize_text(
    text: &str,
    model: &LateFusionNet,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    // TODO: Access model's text vocabulary
    // For now, this is a placeholder that just splits on whitespace
    warn!("Using placeholder tokenization - implement proper vocab lookup");

    let tokens: Vec<usize> = text
        .to_lowercase()
        .split_whitespace()
        .enumerate()
        .map(|(i, _)| i % 1000) // Placeholder
        .collect();

    Ok(tokens)
}

/// Categorize risk score into discrete categories
fn categorize_risk(score: f32) -> (String, f32) {
    let (category, confidence) = if score > 0.8 {
        ("CRITICAL", 1.0 - (1.0 - score) / 0.2)
    } else if score > 0.6 {
        ("HIGH", 1.0 - (0.8 - score) / 0.2)
    } else if score > 0.4 {
        ("MODERATE", 1.0 - (0.6 - score) / 0.2)
    } else if score > 0.2 {
        ("LOW", 1.0 - (0.4 - score) / 0.2)
    } else {
        ("MINIMAL", 1.0 - score / 0.2)
    };

    (category.to_string(), confidence.max(0.5).min(1.0))
}
