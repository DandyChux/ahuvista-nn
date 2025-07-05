# Ahuvista-NN: Multi-Modal Neural Network for Healthcare

A lightweight, Rust-based neural network library optimized for multi-modal data processing in low-compute environments, with a focus on maternal health outcome predictions.

## Features

- **Multi-modal architecture**: Process tabular, temporal, text, and image data simultaneously
- **Late fusion strategy**: Extracts specialized features from each modality before combination
- **Population-informed weighting**: Addresses class imbalance with demographic stratification
- **Optimized for efficiency**: Designed to run on low-compute resources for accessibility
- **Complete data pipeline**: Includes preprocessors and transformers for all data types

## Usage

```rust
use rusty_nn::fusion::LateFusionNet;
use rusty_nn::data::{TabularData, TemporalData, TextData, ImageData};

fn main() {
    // Load and preprocess data
    let tabular_data = TabularData::from_csv("patient_data.csv", true, &[])?;
    let temporal_data = TemporalData::from_single_csv("vitals.csv", true, "PatientID", Some("Timestamp"))?;
    let text_data = TextData::from_files(&["notes.txt"], 5000)?;
    let image_data = ImageData::from_files(&["scan.jpg"], (32, 32), false)?;

    // Build model
    let mut model = LateFusionNet::new(
        // Configuration parameters...
    );

    // Make predictions
    let prediction = model.predict(
        tabular_features,
        temporal_sequence,
        text_tokens,
        image_data
    );

    println!("Risk assessment: {}", if prediction > 0.5 { "HIGH RISK" } else { "LOW RISK" });
}
