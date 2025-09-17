//! Main training program for the multi-modal neural network.
//!
//! Demonstrates how to:
//! 1. Load and preprocess data from different modalities
//! 2. Build and configure the late fusion network
//! 3. Train the model with population-informed class balancing
//! 4. Evaluate performance with calibrated predictions
//! 5. Make predictions on new data for maternal mortality risk

use ndarray::Array3;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use rusty_nn::{
    balancing::{
        compute_sample_weights, weighted_binary_cross_entropy, BayesianCalibrator, CauseWeights,
        SmoteSynthesizer,
    },
    data::{
        DataLoader, ImageData, MultiModalBatch, MultiModalDataLoader, TabularData, TemporalData,
        TextData,
    },
    fusion::LateFusionNet,
    loss::{binary_accuracy, binary_cross_entropy, binary_cross_entropy_gradient},
    module::Module,
};

fn main() -> std::io::Result<()> {
    println!("Ahuvista-NN Multi-Modal Network for Maternal Mortality Prediction");
    println!("------------------------------------------------------------");

    // Step 1: Load and preprocess data
    println!("Loading and preprocessing data...");

    // 1.1: Load tabular data
    let tabular_data = load_tabular_data("data/patient_features.csv")?;
    println!(
        "  Loaded tabular data: {} samples, {} features",
        tabular_data.data.len(),
        tabular_data.feature_dim()
    );

    // 1.2: Load temporal data
    let temporal_data = load_temporal_data("data/patient_timeseries.csv")?;
    println!(
        "  Loaded temporal data: {} sequences, {} features per timestep, max length {}",
        temporal_data.sequences.len(),
        temporal_data.feature_dim(),
        temporal_data.max_seq_length
    );

    // 1.3: Load text data
    let text_data = load_text_data("data/patient_notes.csv")?;
    println!(
        "  Loaded text data: {} documents, vocabulary size {}, max length {}",
        text_data.documents.len(),
        text_data.vocab_size(),
        text_data.max_seq_length
    );

    // 1.4: Load image data
    let image_data = load_image_data("data/patient_images")?;
    println!(
        "  Loaded image data: {} images, size {}x{}, {} channels",
        image_data.images.len(),
        image_data.image_size.0,
        image_data.image_size.1,
        image_data.channels
    );

    // 1.5: Load target values and mortality causes
    let (targets, causes) = load_targets_and_causes("data/patient_outcomes.csv")?;
    println!("  Loaded target outcomes: {} values", targets.len());

    // Step 2: Configure population-based weighting and calibration
    println!("Configuring population-based weighting and calibration...");

    // 2.1: Define population-level cause frequencies
    let mut cause_frequencies = HashMap::new();
    cause_frequencies.insert(0, 0.25); // Hemorrhage: 25% of deaths
    cause_frequencies.insert(1, 0.20); // Hypertensive disorders: 20%
    cause_frequencies.insert(2, 0.15); // Infection/sepsis: 15%
    cause_frequencies.insert(3, 0.10); // Thromboembolism: 10%
    cause_frequencies.insert(4, 0.30); // Other causes: 30%

    // 2.2: Create cause-specific weights
    let cause_weights = CauseWeights::from_population_frequencies(&cause_frequencies);

    // 2.3: Define demographic stratum multipliers
    let mut stratum_multipliers = HashMap::new();
    stratum_multipliers.insert("urban_young".to_string(), 1.0);
    stratum_multipliers.insert("urban_old".to_string(), 1.2);
    stratum_multipliers.insert("rural_young".to_string(), 1.5);
    stratum_multipliers.insert("rural_old".to_string(), 1.8);

    // 2.4: Extract patient strata (from demographic data)
    let patient_strata: Vec<String> = extract_patient_strata(&tabular_data);

    // 2.5: Compute sample weights
    let base_sample_weights =
        compute_sample_weights(&targets, Some(&patient_strata), Some(&stratum_multipliers));

    // 2.6: Apply cause-specific weights
    let sample_weights = apply_cause_weights(&base_sample_weights, &causes, &cause_weights);

    // 2.7: Create Bayesian calibrator with global maternal mortality prevalence
    let calibrator = BayesianCalibrator::new(0.001); // 0.1% prevalence

    // Step 3: Split data into training and validation sets
    let (train_data, val_data, train_targets, val_targets, train_weights, val_weights) =
        split_data_with_weights(
            &tabular_data.data,
            &temporal_data.sequences,
            &text_data.token_sequences,
            &image_data.images,
            &targets,
            &sample_weights,
            0.8, // 80% for training, 20% for validation
        );

    println!(
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
        val_data.0,
        val_data.1,
        val_data.2,
        val_data.3,
        val_targets.clone(),
    );

    // Step 5: Build the model
    println!("Building multi-modal network...");

    let mut model = LateFusionNet::new(
        // Tabular network
        tabular_data.feature_dim(), // input size
        &[64, 32],                  // hidden sizes
        16,                         // output size
        // Temporal network
        temporal_data.feature_dim(), // input size per timestep
        32,                          // hidden size
        // Text network
        text_data.vocab.clone(), // vocabulary
        50,                      // embedding dimension
        32,                      // hidden size
        // Image network
        image_data.channels,     // number of channels
        image_data.image_size.0, // height
        image_data.image_size.1, // width
        // Classifier
        &[64, 32], // hidden sizes
    );

    // Step 6: Create SMOTE synthesizer for minority class oversampling
    let smote = SmoteSynthesizer::new(5); // 5 nearest neighbors

    // Optional: Generate synthetic samples for training
    let (
        synthetic_tabular,
        synthetic_temporal,
        synthetic_text,
        synthetic_images,
        synthetic_targets,
    ) = generate_synthetic_samples(
        &train_data.0,
        &train_data.1,
        &train_data.2,
        &train_data.3,
        &train_targets,
        &smote,
        100, // Number of synthetic samples to create
    );

    println!(
        "Generated {} synthetic samples for minority class",
        synthetic_targets.len()
    );

    // Add synthetic samples to training data if needed

    // Step 7: Train the model
    println!("Training model with population-weighted samples...");

    let config = TrainingConfig {
        epochs: 10,
        batch_size: 16,
        learning_rate: 0.001,
        val_frequency: 1,
    };

    train_model_with_weights(
        &mut model,
        &mut train_loader,
        &mut val_loader,
        &config,
        &train_weights,
        &val_weights,
        &calibrator,
    );

    // Step 8: Final evaluation on validation set
    println!("Final evaluation on validation set...");

    let val_metrics =
        evaluate_model_with_calibration(&mut model, &mut val_loader, &val_weights, &calibrator);

    println!(
        "  Validation loss: {:.4}, accuracy: {:.2}%",
        val_metrics.0,
        val_metrics.1 * 100.0
    );

    // Step 9: Print cause-specific performance
    println!("Cause-specific performance:");
    let cause_metrics = evaluate_by_cause(
        &mut model,
        &mut val_loader,
        &causes[train_targets.len()..],
        &calibrator,
    );

    for (cause_id, (loss, accuracy)) in cause_metrics {
        let cause_name = match cause_id {
            0 => "Hemorrhage",
            1 => "Hypertensive disorders",
            2 => "Infection/sepsis",
            3 => "Thromboembolism",
            _ => "Other causes",
        };

        println!(
            "  {}: loss={:.4}, accuracy={:.2}%",
            cause_name,
            loss,
            accuracy * 100.0
        );
    }

    // Step 10: Example prediction
    println!("Example prediction for high-risk patient...");

    // Load a new patient's data
    let tabular_features = tabular_data.data[0].clone(); // Just using first sample as example
    let temporal_sequence = temporal_data.sequences[0].clone();
    let text_tokens = text_data.token_sequences[0].clone();
    let image = image_data.images[0].clone();

    let raw_prediction = model.predict(tabular_features, temporal_sequence, text_tokens, image);

    let calibrated_prediction = calibrator.calibrate(raw_prediction);

    println!("  Raw prediction: {:.2}%", raw_prediction * 100.0);
    println!(
        "  Calibrated prediction: {:.2}%",
        calibrated_prediction * 100.0
    );
    println!(
        "  Risk assessment: {}",
        if calibrated_prediction > 0.5 {
            "HIGH RISK"
        } else {
            "LOW RISK"
        }
    );

    // Step 11: Save model (placeholder - actual implementation would depend on serialization approach)
    println!("Model training complete!");

    Ok(())
}

/// Configuration for model training
struct TrainingConfig {
    /// Number of training epochs
    epochs: usize,

    /// Batch size
    batch_size: usize,

    /// Learning rate
    learning_rate: f32,

    /// Validation frequency (every N epochs)
    val_frequency: usize,
}

/// Train the model with sample weights and calibration
fn train_model_with_weights(
    model: &mut LateFusionNet,
    train_loader: &mut impl DataLoader<MultiModalBatch>,
    val_loader: &mut impl DataLoader<MultiModalBatch>,
    config: &TrainingConfig,
    train_weights: &[f32],
    val_weights: &[f32],
    calibrator: &BayesianCalibrator,
) {
    for epoch in 0..config.epochs {
        // Training phase
        let mut epoch_loss = 0.0;
        let mut correct_predictions = 0;
        let mut total_predictions = 0;

        train_loader.reset();
        let mut sample_idx = 0;

        // Process batches
        while let Some(batch) = train_loader.next_batch(config.batch_size) {
            let batch_size = batch.tabular.len();

            // Process each sample in the batch
            for i in 0..batch_size {
                // Get sample weight
                let sample_weight = if sample_idx < train_weights.len() {
                    train_weights[sample_idx]
                } else {
                    1.0 // Default weight
                };

                sample_idx += 1;

                // Zero gradients before forward pass
                model.zero_grad();

                // Forward pass
                let prediction = model.forward((
                    batch.tabular[i].clone(),
                    batch.temporal[i].clone(),
                    batch.text[i].clone(),
                    batch.images[i].clone(),
                ))[0];

                // Compute weighted loss
                let target = batch.targets[i];
                let (loss, grad) =
                    weighted_binary_cross_entropy(&[prediction], &[target], &[sample_weight]);

                epoch_loss += loss;

                // Calibrate prediction for accuracy tracking
                let calibrated_pred = calibrator.calibrate(prediction);
                let predicted_class = if calibrated_pred > 0.5 { 1.0 } else { 0.0 };

                if (predicted_class - target).abs() < 1e-6 {
                    correct_predictions += 1;
                }
                total_predictions += 1;

                // Backward pass with weighted gradient
                model.backward(vec![grad[0]]);

                // Update weights
                model.update(config.learning_rate);
            }
        }

        // Calculate training metrics
        let train_loss = epoch_loss / total_predictions as f32;
        let train_accuracy = correct_predictions as f32 / total_predictions as f32;

        // Validation phase (if it's time)
        if (epoch + 1) % config.val_frequency == 0 || epoch == config.epochs - 1 {
            let (val_loss, val_accuracy) =
                evaluate_model_with_calibration(model, val_loader, val_weights, calibrator);

            println!(
                "Epoch {}/{}: Train loss={:.4}, accuracy={:.2}% | Val loss={:.4}, accuracy={:.2}%",
                epoch + 1,
                config.epochs,
                train_loss,
                train_accuracy * 100.0,
                val_loss,
                val_accuracy * 100.0
            );
        } else {
            println!(
                "Epoch {}/{}: Train loss={:.4}, accuracy={:.2}%",
                epoch + 1,
                config.epochs,
                train_loss,
                train_accuracy * 100.0
            );
        }
    }
}

/// Evaluate the model with sample weights and calibration
fn evaluate_model_with_calibration(
    model: &mut LateFusionNet,
    data_loader: &mut impl DataLoader<MultiModalBatch>,
    sample_weights: &[f32],
    calibrator: &BayesianCalibrator,
) -> (f32, f32) {
    let mut total_loss = 0.0;
    let mut correct_predictions = 0;
    let mut total_predictions = 0;

    data_loader.reset();
    let mut sample_idx = 0;

    while let Some(batch) = data_loader.next_batch(32) {
        // Use a fixed batch size for evaluation
        let batch_size = batch.tabular.len();

        for i in 0..batch_size {
            // Get sample weight
            let sample_weight = if sample_idx < sample_weights.len() {
                sample_weights[sample_idx]
            } else {
                1.0 // Default weight
            };

            sample_idx += 1;

            // Forward pass
            let prediction = model.forward((
                batch.tabular[i].clone(),
                batch.temporal[i].clone(),
                batch.text[i].clone(),
                batch.images[i].clone(),
            ))[0];

            // Compute weighted loss
            let target = batch.targets[i];
            let (loss, _) =
                weighted_binary_cross_entropy(&[prediction], &[target], &[sample_weight]);

            total_loss += loss;

            // Calibrate prediction for accuracy evaluation
            let calibrated_pred = calibrator.calibrate(prediction);
            let predicted_class = if calibrated_pred > 0.5 { 1.0 } else { 0.0 };

            if (predicted_class - target).abs() < 1e-6 {
                correct_predictions += 1;
            }
            total_predictions += 1;
        }
    }

    let avg_loss = total_loss / total_predictions as f32;
    let accuracy = correct_predictions as f32 / total_predictions as f32;

    (avg_loss, accuracy)
}

/// Evaluate performance for each cause of mortality
fn evaluate_by_cause(
    model: &mut LateFusionNet,
    data_loader: &mut impl DataLoader<MultiModalBatch>,
    causes: &[usize],
    calibrator: &BayesianCalibrator,
) -> HashMap<usize, (f32, f32)> {
    let mut cause_losses: HashMap<usize, f32> = HashMap::new();
    let mut cause_correct: HashMap<usize, usize> = HashMap::new();
    let mut cause_total: HashMap<usize, usize> = HashMap::new();

    data_loader.reset();
    let mut sample_idx = 0;

    while let Some(batch) = data_loader.next_batch(32) {
        let batch_size = batch.tabular.len();

        for i in 0..batch_size {
            // Get the cause of mortality
            let cause = if sample_idx < causes.len() {
                causes[sample_idx]
            } else {
                4 // Default to "other causes"
            };

            sample_idx += 1;

            // Forward pass
            let prediction = model.forward((
                batch.tabular[i].clone(),
                batch.temporal[i].clone(),
                batch.text[i].clone(),
                batch.images[i].clone(),
            ))[0];

            // Compute loss
            let target = batch.targets[i];
            let loss = binary_cross_entropy(prediction, target);

            // Calibrate prediction
            let calibrated_pred = calibrator.calibrate(prediction);
            let predicted_class = if calibrated_pred > 0.5 { 1.0 } else { 0.0 };

            // Update metrics for this cause
            *cause_losses.entry(cause).or_insert(0.0) += loss;
            *cause_total.entry(cause).or_insert(0) += 1;

            if (predicted_class - target).abs() < 1e-6 {
                *cause_correct.entry(cause).or_insert(0) += 1;
            }
        }
    }

    // Calculate average metrics for each cause
    let mut cause_metrics = HashMap::new();

    for cause in cause_total.keys() {
        let total = *cause_total.get(cause).unwrap_or(&1) as f32;
        let correct = *cause_correct.get(cause).unwrap_or(&0) as f32;
        let loss = *cause_losses.get(cause).unwrap_or(&0.0) / total;
        let accuracy = correct / total;

        cause_metrics.insert(*cause, (loss, accuracy));
    }

    cause_metrics
}

/// Apply cause-specific weights to base sample weights
fn apply_cause_weights(
    base_weights: &[f32],
    causes: &[usize],
    cause_weights: &CauseWeights,
) -> Vec<f32> {
    base_weights
        .iter()
        .zip(causes.iter())
        .map(|(&base_weight, &cause)| cause_weights.apply(base_weight, cause))
        .collect()
}

/// Extract patient strata from tabular data
fn extract_patient_strata(tabular_data: &TabularData) -> Vec<String> {
    // This is a placeholder function - in a real implementation,
    // you would extract demographic features from the tabular data
    // to determine each patient's stratum

    let mut strata = Vec::with_capacity(tabular_data.data.len());

    for (i, _) in tabular_data.data.iter().enumerate() {
        // This is just a dummy implementation
        let stratum = if i % 4 == 0 {
            "urban_young".to_string()
        } else if i % 4 == 1 {
            "urban_old".to_string()
        } else if i % 4 == 2 {
            "rural_young".to_string()
        } else {
            "rural_old".to_string()
        };

        strata.push(stratum);
    }

    strata
}

/// Generate synthetic samples for minority class using SMOTE
fn generate_synthetic_samples(
    tabular_data: &[Vec<f32>],
    temporal_data: &[Vec<Vec<f32>>],
    text_data: &[Vec<usize>],
    image_data: &[Array3<f32>],
    targets: &[f32],
    smote: &SmoteSynthesizer,
    n_synthetic: usize,
) -> (
    Vec<Vec<f32>>,
    Vec<Vec<Vec<f32>>>,
    Vec<Vec<usize>>,
    Vec<Array3<f32>>,
    Vec<f32>,
) {
    // Identify minority class samples (assuming 1.0 is the minority class)
    let minority_indices: Vec<usize> = targets
        .iter()
        .enumerate()
        .filter(|(_, &target)| target > 0.5)
        .map(|(i, _)| i)
        .collect();

    if minority_indices.is_empty() {
        // No minority samples to synthesize from
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    // Extract minority class tabular data
    let minority_tabular: Vec<Vec<f32>> = minority_indices
        .iter()
        .map(|&i| tabular_data[i].clone())
        .collect();

    // Generate synthetic tabular samples
    let synthetic_tabular = smote.generate_synthetic_samples(&minority_tabular, n_synthetic);

    // For simplicity, we'll just duplicate existing temporal, text and image data
    // In a real implementation, you would use modality-specific generation methods
    let mut synthetic_temporal = Vec::with_capacity(n_synthetic);
    let mut synthetic_text = Vec::with_capacity(n_synthetic);
    let mut synthetic_images = Vec::with_capacity(n_synthetic);

    for i in 0..n_synthetic {
        // Pick a random minority sample to duplicate
        let source_idx = minority_indices[i % minority_indices.len()];

        synthetic_temporal.push(temporal_data[source_idx].clone());
        synthetic_text.push(text_data[source_idx].clone());
        synthetic_images.push(image_data[source_idx].clone());
    }

    // All synthetic samples have target = 1.0
    let synthetic_targets = vec![1.0; n_synthetic];

    (
        synthetic_tabular,
        synthetic_temporal,
        synthetic_text,
        synthetic_images,
        synthetic_targets,
    )
}

/// Create a multi-modal data loader from separate data sources
fn create_data_loader(
    tabular_data: Vec<Vec<f32>>,
    temporal_data: Vec<Vec<Vec<f32>>>,
    text_data: Vec<Vec<usize>>,
    image_data: Vec<Array3<f32>>,
    targets: Vec<f32>,
) -> MultiModalDataLoader {
    // Create data containers
    let tabular_container = TabularData {
        data: tabular_data,
        feature_names: Vec::new(),
        numeric_stats: HashMap::new(),
        categorical_mappings: HashMap::new(),
        categorical_features: Vec::new(),
        current_idx: 0,
    };

    let temporal_container = TemporalData {
        sequences: temporal_data,
        feature_names: Vec::new(),
        feature_stats: HashMap::new(),
        max_seq_length: 0,
        current_idx: 0,
    };

    let text_container = TextData {
        documents: Vec::new(),
        token_sequences: text_data,
        vocab: HashMap::new(),
        reverse_vocab: HashMap::new(),
        max_seq_length: 0,
        current_idx: 0,
    };

    let image_container = ImageData {
        images: image_data,
        image_size: (224, 224), // Standard image size
        channels: 3,            // RGB images
        channel_means: Vec::new(),
        channel_stds: Vec::new(),
        current_idx: 0,
    };

    MultiModalDataLoader::new(
        tabular_container,
        temporal_container,
        text_container,
        image_container,
        targets,
    )
}

/// Split data into training and validation sets, including sample weights
fn split_data_with_weights(
    tabular_data: &[Vec<f32>],
    temporal_data: &[Vec<Vec<f32>>],
    text_data: &[Vec<usize>],
    image_data: &[Array3<f32>],
    targets: &[f32],
    weights: &[f32],
    train_ratio: f32,
) -> (
    (
        Vec<Vec<f32>>,
        Vec<Vec<Vec<f32>>>,
        Vec<Vec<usize>>,
        Vec<Array3<f32>>,
    ),
    (
        Vec<Vec<f32>>,
        Vec<Vec<Vec<f32>>>,
        Vec<Vec<usize>>,
        Vec<Array3<f32>>,
    ),
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
) {
    let total_samples = targets.len();
    let train_size = (total_samples as f32 * train_ratio) as usize;

    // Split tabular data
    let train_tabular = tabular_data[0..train_size].to_vec();
    let val_tabular = tabular_data[train_size..].to_vec();

    // Split temporal data
    let train_temporal = temporal_data[0..train_size].to_vec();
    let val_temporal = temporal_data[train_size..].to_vec();

    // Split text data
    let train_text = text_data[0..train_size].to_vec();
    let val_text = text_data[train_size..].to_vec();

    // Split image data
    let train_image = image_data[0..train_size].to_vec();
    let val_image = image_data[train_size..].to_vec();

    // Split targets
    let train_targets = targets[0..train_size].to_vec();
    let val_targets = targets[train_size..].to_vec();

    // Split weights
    let train_weights = weights[0..train_size].to_vec();
    let val_weights = weights[train_size..].to_vec();

    (
        (train_tabular, train_temporal, train_text, train_image),
        (val_tabular, val_temporal, val_text, val_image),
        train_targets,
        val_targets,
        train_weights,
        val_weights,
    )
}

/// Load tabular data from a CSV file
fn load_tabular_data<P: AsRef<Path>>(path: P) -> io::Result<TabularData> {
    // Define which columns should be treated as categorical
    let categorical_features = vec![
        "ethnicity".to_string(),
        "education_level".to_string(),
        "insurance_type".to_string(),
        "delivery_type".to_string(),
        "prenatal_care_adequacy".to_string(),
        "location_type".to_string(), // urban/rural
    ];

    TabularData::from_csv(path, true, &categorical_features)
}

/// Load temporal data from a CSV file
fn load_temporal_data<P: AsRef<Path>>(path: P) -> io::Result<TemporalData> {
    // Load from a single CSV file where each patient has multiple rows
    // Expects columns: PatientID, Timestamp, and various vital signs/lab values
    TemporalData::from_single_csv(
        path,
        true, // has header
        "PatientID".to_string(),
        Some("Timestamp".to_string()),
    )
}

/// Load text data from a CSV file
fn load_text_data<P: AsRef<Path>>(path: P) -> io::Result<TextData> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut documents = Vec::new();

    // Skip header and read documents
    for (i, line) in reader.lines().enumerate() {
        if i == 0 {
            continue;
        } // Skip header

        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();

        if parts.len() < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Line {} has insufficient columns", i + 1),
            ));
        }

        // Assume format: PatientID,NoteText
        documents.push(parts[1..].join(",").trim_matches('"').to_string());
    }

    // Create text data with a reasonable vocabulary size
    TextData::from_documents(documents, 10000)
}

/// Load image data from a directory
fn load_image_data<P: AsRef<Path>>(dir_path: P) -> io::Result<ImageData> {
    // Collect all image file paths
    let paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir_path)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();

            // Check if it's an image file
            if path.is_file() {
                let ext = path.extension()?.to_str()?.to_lowercase();
                if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "bmp") {
                    Some(path)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    if paths.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No image files found in directory",
        ));
    }

    // Load images with standard size for medical imaging
    // Using 224x224 for compatibility with standard models
    ImageData::from_files(&paths, (224, 224), false) // false = keep RGB
}

/// Load target values and mortality causes from a CSV file
fn load_targets_and_causes<P: AsRef<Path>>(path: P) -> io::Result<(Vec<f32>, Vec<usize>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut targets = Vec::new();
    let mut causes = Vec::new();

    // Skip header
    let mut lines = reader.lines();
    if let Some(header) = lines.next() {
        let _ = header?; // Consume header
    }

    // Read each line
    for (line_num, line) in lines.enumerate() {
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();

        if parts.len() < 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Line {} has insufficient columns", line_num + 2),
            ));
        }

        // Expected format: PatientID,Mortality(0/1),CauseCode
        let mortality = parts[1].trim().parse::<f32>().map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid mortality value on line {}: {}", line_num + 2, e),
            )
        })?;

        // Validate binary target
        if mortality != 0.0 && mortality != 1.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Mortality must be 0 or 1, found {} on line {}",
                    mortality,
                    line_num + 2
                ),
            ));
        }

        targets.push(mortality);

        // Parse cause code (0-4, where 4 is "other/none")
        let cause = if mortality > 0.5 {
            parts[2].trim().parse::<usize>().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid cause code on line {}: {}", line_num + 2, e),
                )
            })?
        } else {
            4 // No cause for survivors
        };

        causes.push(cause);
    }

    if targets.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No data found in outcomes file",
        ));
    }

    Ok((targets, causes))
}
