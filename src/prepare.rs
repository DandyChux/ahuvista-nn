//! Data preparation binary for loading data from datasets directory into database.
//!
//! This binary handles:
//! 1. Loading data from the datasets directory (tabular, temporal, text, images)
//! 2. Normalizing and preparing the data
//! 3. Storing everything in the SQLite database for training

use clap::Parser;
use std::path::PathBuf;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt};

use ahuvista_nn::{
    data::{db::DatabaseManager, loader::DataConfig},
    utils::error::AppResult,
};

/// Command-line arguments for the data preparation program
#[derive(Parser, Debug)]
#[command(author, version, about = "Prepare data for Ahuvista-NN training")]
struct Args {
    /// Path to datasets directory
    #[arg(short, long, default_value = "datasets")]
    data_dir: String,
    /// Path to database file
    #[arg(long, default_value = "ahuvista.db")]
    database: String,
    /// Patient ID column name
    #[arg(long, default_value = "PatientID")]
    patient_id_column: String,
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
    /// Clear existing data in database before importing
    #[arg(long)]
    clear: bool,
}

/// Statistics from data preparation
#[derive(Default)]
struct PrepareStats {
    tabular_samples: usize,
    tabular_features: usize,
    temporal_sequences: usize,
    temporal_features: usize,
    text_documents: usize,
    vocab_size: usize,
    image_count: usize,
    image_size: (usize, usize),
    outcomes: usize,
}

fn main() -> AppResult<()> {
    let args = Args::parse();

    // Initialize logging
    if args.verbose {
        tracing_subscriber::registry()
            .with(fmt::layer().pretty())
            .init();
    } else {
        tracing_subscriber::registry()
            .with(fmt::layer().compact())
            .init();
    }

    info!("Ahuvista-NN Data Preparation");
    info!("============================");
    info!("Loading from: {}", args.data_dir);
    info!("Storing to:   {}", args.database);

    // Configure FILE-BASED loading (reads from datasets directory)
    let data_config = DataConfig {
        data_dir: PathBuf::from(&args.data_dir),
        merge_files: true,
        patient_id_column: args.patient_id_column.clone(),
        has_headers: true,
    };

    // Open database for WRITING
    let mut db = DatabaseManager::new(&args.database)?;

    if args.clear {
        info!("Clearing existing data...");
        db.clear_all_data()?;
    }

    // =========================================================================
    // LOAD FROM FILES → STORE TO DATABASE
    // =========================================================================

    // Track statistics
    let mut stats = PrepareStats::default();

    // Import CSV files into relational tables (patients, patient_features)
    info!("Importing patient data from CSV files...");
    let tabular_dir = PathBuf::from(&args.data_dir).join("tabular");
    let patient_count = db.import_csv_directory(&tabular_dir, &args.patient_id_column, None)?;
    info!("  ✓ Imported {} patients", patient_count);

    // Import temporal data
    let temporal_dir = PathBuf::from(&args.data_dir).join("temporal");
    db.import_csv_directory(&temporal_dir, &args.patient_id_column, None)?;

    // Import text data
    let text_dir = PathBuf::from(&args.data_dir).join("text");
    db.import_csv_directory(&text_dir, &args.patient_id_column, None)?;

    // Import image data
    let image_dir = PathBuf::from(&args.data_dir).join("image");
    db.import_csv_directory(&image_dir, &args.patient_id_column, None)?;

    info!("Cleaning up non-maternal records...");
    let removed = db.cleanup_non_maternal_patients()?;
    info!("  ✓ Removed {} non-pregnant patients", removed);

    // Build TabularData from the database
    info!("Building tabular features from database...");
    let tabular_data = db.build_tabular_data()?;
    info!(
        "  ✓ Built {} samples, {} features",
        tabular_data.data.len(),
        tabular_data.feature_dim()
    );
    stats.tabular_samples = tabular_data.data.len();
    stats.tabular_features = tabular_data.feature_dim();
    db.store_tabular_data(&tabular_data)?;

    // Build and store temporal data from database
    info!("Building temporal data from database...");
    let temporal_data = db.build_temporal_data()?;
    info!(
        "  ✓ Built {} temporal sequences, {} features per timestep",
        temporal_data.sequences.len(),
        temporal_data.feature_dim()
    );
    stats.temporal_sequences = temporal_data.sequences.len();
    stats.temporal_features = temporal_data.feature_dim();
    db.store_temporal_data(&temporal_data)?;

    // Build and store text data from database
    info!("Building text data from database...");
    let text_data = db.build_text_data(10000)?; // max 10k vocab
    info!(
        "  ✓ Built {} documents, vocabulary size {}",
        text_data.documents.len(),
        text_data.vocab_size()
    );
    stats.text_documents = text_data.documents.len();
    stats.vocab_size = text_data.vocab_size();
    db.store_text_data(&text_data)?;

    // Build and store image data from files
    info!("Building image data from files...");
    let image_data = db.build_image_data(&image_dir, (224, 224), false)?;
    info!(
        "  ✓ Built {} images, size {}x{}",
        image_data.images.len(),
        image_data.image_size.0,
        image_data.image_size.1
    );
    stats.image_count = image_data.images.len();
    stats.image_size = image_data.image_size;
    db.store_image_data(&image_data)?;

    // Load and store outcomes
    info!("Building outcomes from database...");
    let (targets, causes) = db.build_outcomes()?;
    info!("  ✓ Built {} outcomes", targets.len());
    stats.outcomes = targets.len();
    db.store_outcomes(&targets, &causes)?;

    // Build and save normalizer from database
    info!("Building normalizer from database...");
    let normalizer = db.build_normalizer()?;
    db.save_normalizer(&normalizer)?;

    // Print summary
    info!("");
    info!("Data Preparation Complete!");
    info!("==========================");
    info!("Summary:");
    if stats.tabular_samples > 0 {
        info!(
            "  Tabular: {} samples, {} features",
            stats.tabular_samples, stats.tabular_features
        );
    }
    if stats.temporal_sequences > 0 {
        info!(
            "  Temporal: {} sequences, {} features",
            stats.temporal_sequences, stats.temporal_features
        );
    }
    if stats.text_documents > 0 {
        info!(
            "  Text: {} documents, {} vocabulary",
            stats.text_documents, stats.vocab_size
        );
    }
    if stats.image_count > 0 {
        info!(
            "  Images: {} images, {}x{}",
            stats.image_count, stats.image_size.0, stats.image_size.1
        );
    }
    if stats.outcomes > 0 {
        info!("  Outcomes: {} records", stats.outcomes);
    }
    info!("");
    info!(
        "Ready for training with: ahuvista-train --database {}",
        args.database
    );

    Ok(())
}
