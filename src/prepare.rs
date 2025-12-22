//! Data preparation binary for loading data from datasets directory into database.
//!
//! This binary handles:
//! 1. Loading data from the datasets directory (tabular, temporal, text, images)
//! 2. Normalizing and preparing the data
//! 3. Storing everything in the SQLite database for training

mod config;
mod data;
mod utils;

use clap::Parser;
use std::path::PathBuf;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt};

use crate::{
    config::Settings,
    data::{
        db::DatabaseManager,
        loader::{
            DataConfig, Normalizer, load_all_image_data, load_all_outcomes, load_all_tabular_data,
            load_all_temporal_data, load_all_text_data,
        },
    },
    utils::error::{AppError, AppResult},
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
            .with(fmt::layer().json())
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

    // Load and store tabular data
    info!("Loading tabular data from files...");
    let tabular_data = load_all_tabular_data(&data_config)?;
    info!(
        "  ✓ Loaded {} samples from CSV, {} features",
        tabular_data.data.len(),
        tabular_data.feature_dim()
    );
    stats.tabular_samples = tabular_data.data.len();
    stats.tabular_features = tabular_data.feature_dim();
    db.store_tabular_data(&tabular_data)?;

    // Load and store temporal data
    info!("Loading temporal data...");
    let temporal_data = load_all_temporal_data(&data_config)?;
    info!(
        "  ✓ Loaded {} temporal sequences, {} features per timestep",
        temporal_data.sequences.len(),
        temporal_data.feature_dim()
    );
    stats.temporal_sequences = temporal_data.sequences.len();
    stats.temporal_features = temporal_data.feature_dim();
    db.store_temporal_data(&temporal_data)?;

    // Load and store text data
    info!("Loading text data...");
    let text_data = load_all_text_data(&data_config)?;
    info!(
        "  ✓ Loaded {} documents, vocabulary size {}",
        text_data.documents.len(),
        text_data.vocab_size()
    );
    stats.text_documents = text_data.documents.len();
    stats.vocab_size = text_data.vocab_size();
    db.store_text_data(&text_data)?;

    // Load and store image data
    info!("Loading image data...");
    let image_data = load_all_image_data(&data_config)?;
    info!(
        "  ✓ Loaded {} images, size {}x{}",
        image_data.images.len(),
        image_data.image_size.0,
        image_data.image_size.1
    );
    stats.image_count = image_data.images.len();
    stats.image_size = image_data.image_size;
    db.store_image_data(&image_data)?;

    // Load and store outcomes
    info!("Loading outcomes...");
    let (targets, causes) = load_all_outcomes(&data_config)?;
    info!("  ✓ Loaded {} outcomes", targets.len());
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
