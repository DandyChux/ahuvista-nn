# Ahuvista-NN: Modular Multi-Modal Neural Network for Healthcare

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Static Badge](https://img.shields.io/badge/license-Custom-blue)](LICENSE.md)

A lightweight, Rust-based neural network library optimized for multi-modal data processing in low-compute environments, with a focus on maternal health outcome predictions.

## 🌟 Key Features

- **🧩 Modular Multi-Modal Architecture**: Selectively enable/disable tabular, temporal, text, and image modalities based on available data
- **🔗 Late Fusion Strategy**: Extracts specialized features from each modality before combination for optimal performance
- **⚖️ Population-Informed Weighting**: Addresses class imbalance with demographic stratification and cause-specific weights
- **⚡ Optimized for Efficiency**: Designed to run on low-compute resources (edge devices, embedded systems, limited infrastructure)
- **🔄 Complete Data Pipeline**: Includes preprocessors and transformers for all supported data types
- **⚙️ Flexible Configuration**: Configure via JSON files, programmatic API, or command-line arguments
- **🚀 Production-Ready Inference**: Lightweight prediction binary optimized for deployment
- **📊 Bayesian Calibration**: Probability calibration based on population prevalence for reliable risk assessment

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Loading & Automatic Preparation](#data-loading--automatic-preparation)
- [Architecture Overview](#architecture-overview)
- [Training Models](#training-models)
  - [Configuration Files](#1-configuration-file-approach-recommended)
  - [Command-Line Interface](#2-command-line-interface)
  - [Programmatic API](#3-programmatic-api)
- [Making Predictions](#making-predictions)
  - [Single Predictions](#single-prediction)
  - [Batch Predictions](#batch-predictions)
  - [Calibrated Predictions](#calibrated-predictions)
- [Modality Selection Guide](#modality-selection-guide)
- [Data Format Specifications](#data-format-specifications)
- [Configuration Reference](#configuration-reference)
- [Advanced Usage](#advanced-usage)
- [Performance & Benchmarks](#performance--benchmarks)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- **Rust**: 1.70 or higher ([Install Rust](https://www.rust-lang.org/tools/install))
- **Cargo**: Comes bundled with Rust
- **Operating Systems**: Linux, macOS, Windows

### Build from Source

```bash
# Clone the repository
git clone https://github.com/dandychux/ahuvista-nn.git
cd ahuvista-nn

# Build the project (debug mode)
cargo build

# Build optimized release version
cargo build --release

# Run tests to verify installation
cargo test --release
```

### Compiled Binaries

After building, you'll find three binaries in `target/release/`:

- **`ahuvista-prepare`**: Data preparation binary - loads CSV files and stores in SQLite database
- **`ahuvista-train`**: Training binary - loads data from database and trains models
- **`ahuvista-predict`**: Lightweight inference binary for production deployment

### Verify Installation

```bash
# Check data preparation binary
cargo run --release --bin ahuvista-prepare -- --help

# Check training binary
cargo run --release --bin ahuvista-train -- --help

# Check prediction binary
cargo run --release --bin ahuvista-predict -- --help

```

## 🏃 Quick Start

### 1. Prepare Your Data

First, prepare your CSV data and store it in the database:

```bash
# Prepare all modalities from the datasets directory
cargo run --release --bin ahuvista-prepare -- \
  --data-dir datasets \
  --database ahuvista.db \
  --verbose
```

### 2. Train a Simple Model (Tabular Data Only)

```bash
# Train with minimal configuration
cargo run --release --bin ahuvista-train -- \
  --modalities tabular \
  --epochs 20 \
  --batch-size 32 \
  --verbose
```

### 3. Train with Multiple Modalities

```bash
# Train with tabular and temporal data
cargo run --release --bin ahuvista-train -- \
  --modalities tabular,temporal \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --output-dir ./models
```

### 4. Make a Prediction

First, create an input file `patient_data.json`:

```json
{
  "patient_id": "PAT-001",
  "tabular": [0.5, 0.3, 0.8, 0.1, 0.9, 0.4, 0.7, 0.2, 0.6, 0.5],
  "temporal": [
    [120.0, 80.0, 98.6, 72.0, 16.0],
    [125.0, 82.0, 98.8, 75.0, 18.0]
  ]
}
```

Then run prediction:

```bash
cargo run --release --bin ahuvista-predict -- \
  --model models/maternal_mortality_model_modular.bin \
  --modalities tabular,temporal \
  --input patient_data.json \
  --output result.json
```

## 📊 Data Pipeline: Prepare → Train → Predict

AHuVista-NN uses a **three-stage pipeline** for data handling:

### Pipeline Overview

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  ahuvista-      │     │  ahuvista-      │     │  ahuvista-      │
│  prepare        │ ──▶ │  train          │ ──▶ │  predict        │
│                 │     │                 │     │                 │
│  CSV files      │     │  SQLite DB      │     │  Trained model  │
│  ──────────▶    │     │  ──────────▶    │     │  ──────────▶    │
│  SQLite DB      │     │  Model file     │     │  Predictions    │
└─────────────────┘     └─────────────────┘     └─────────────────┘

### Stage 1: Data Preparation (`ahuvista-prepare`)

Loads CSV files from the datasets directory, normalizes data, and stores in SQLite:

```bash
cargo run --release --bin ahuvista-prepare -- \
  --data-dir datasets \
  --database ahuvista.db \
  --verbose
```

Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--data-dir` | Path to datasets directory | `datasets` |
| `--database` | Output SQLite database path | `ahuvista.db` |
| `--patient-id-column` | Name of patient ID column | `PatientID` |
| `--modalities` | Which modalities to prepare | all |
| `--clear` | Clear existing data before import | false |
| `--verbose` | Enable verbose logging | false |

### Stage 2: Training (`ahuvista-train`)

Loads prepared data from database and trains the model:
```bash
cargo run --release --bin ahuvista-train -- \
  --database ahuvista.db \
  --epochs 50 \
  --verbose
```

### Stage 3: Prediction (`ahuvista-predict`)

Uses trained model for inference (unchanged from before).

### Directory Structure

Place your CSV files in these directories before running `ahuvista-prepare`:

```
datasets/
├── tabular/     # Patient demographics, lab results, risk assessments
├── temporal/    # Time-series measurements (vitals over time)
├── text/        # Clinical notes and reports
├── image/       # Medical images (PNG, JPG)
└── outcomes/    # Outcome labels (auto-generated if not present)
```

### Benefits of Database-Backed Training

- **Faster startup**: Data is pre-processed once, training starts immediately
- **Consistent data**: Same normalized data across multiple training runs
- **Scalability**: SQLite handles large datasets efficiently
- **Reproducibility**: Database captures exact data state for each experiment
- **Separation of concerns**: Data preparation is decoupled from training
```

### Supported Column Formats

The system automatically recognizes various column naming conventions:

#### Tabular Data
- **Patient ID**: `PatientID`, `patient_id`, `ID`, `id`
- **Age**: `Age`, `age`, `AGE`
- **Blood Pressure**: `BloodPressure`, `BP`, `SystolicBP`, `DiastolicBP`
- **Glucose**: `Glucose`, `glucose`, `BS`, `Blood Sugar`
- **BMI**: `BMI`, `bmi`, `BodyMassIndex`
- **Pregnancies**: `Pregnancies`, `pregnancies`, `Gravida`
- **Risk Level**: `RiskLevel`, `Risk Level`, `risk_level`
- **Outcome**: `Outcome`, `outcome`, `Mortality`

#### Example CSV Files

**tabular/patient_data.csv:**
```csv
PatientID,Age,BloodPressure,Glucose,BMI,Pregnancies
P001,28,120,95,22.5,2
P002,35,145,110,28.0,4
```

**temporal/measurements.csv:**
```csv
PatientID,Timestamp,HeartRate,Temperature
P001,2024-01-01T08:00,72,98.6
P001,2024-01-01T12:00,75,98.7
```

**text/clinical_notes.csv:**
```csv
PatientID,Notes
P001,"Mild hypertension noted. Previous pregnancy uncomplicated."
```

### Automatic Outcome Generation

If outcome files don't exist, the system generates them using:

1. **Risk Scoring**: Analyzes patient characteristics (age, BP, glucose, BMI, etc.)
2. **Mortality Assignment**: Higher risk scores → higher mortality probability
3. **Cause Mapping**: Risk factors mapped to likely causes:
   - High BP → Hypertensive disorders (code 1)
   - Multiple pregnancies + age → Hemorrhage (code 0)
   - High glucose → Infection/sepsis (code 2)
   - High BMI + age → Abortion + Miscarriage (code 3)
   - Other/None → code 4 (includes all survivors)

### Key Benefits

✅ **No separate data preparation step** - just drop files and train  
✅ **Handles inconsistent formats** - flexible column name matching  
✅ **Automatic merging** - combines multi-modal data by patient ID  
✅ **Smart outcome generation** - creates realistic training targets  
✅ **Incremental updates** - add new files anytime, just re-run training  

## 🏗️ Architecture Overview

### System Design

Ahuvista-NN uses a **late fusion architecture** where each modality is processed by a specialized neural network before features are combined:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Modular Late Fusion Network                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────┐│
│  │   Tabular    │  │   Temporal   │  │     Text     │  │ Image  ││
│  │   Network    │  │   Network    │  │   Network    │  │Network ││
│  │ (Feed-Fwd)   │  │    (RNN)     │  │   (RNN +     │  │ (CNN)  ││
│  │              │  │              │  │  Embedding)  │  │        ││
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └───┬────┘│
│         │                 │                 │              │      │
│         │    Feature      │    Feature      │   Feature    │      │
│         │    Vector       │    Vector       │   Vector     │      │
│         │    (16-32d)     │    (16-32d)     │   (32-64d)   │      │
│         │                 │                 │              │      │
│         └─────────────────┴─────────────────┴──────────────┘      │
│                                  │                                 │
│                          ┌───────▼────────┐                        │
│                          │  Concatenation │                        │
│                          │  Fusion Layer  │                        │
│                          └───────┬────────┘                        │
│                                  │                                 │
│                          ┌───────▼────────┐                        │
│                          │   Classifier   │                        │
│                          │    Network     │                        │
│                          │   (64→32→1)    │                        │
│                          └───────┬────────┘                        │
│                                  │                                 │
│                          ┌───────▼────────┐                        │
│                          │   Prediction   │                        │
│                          │  Risk Score    │                        │
│                          │    (0.0-1.0)   │                        │
│                          └────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Supported Modalities

| Modality | Description | Network Type | Use Cases |
|----------|-------------|--------------|-----------|
| **Tabular** | Structured clinical data | Feed-forward MLP | Demographics, lab results, vital signs |
| **Temporal** | Time-series sequences | Recurrent (RNN) | Continuous monitoring, longitudinal data |
| **Text** | Unstructured text | RNN + Embeddings | Clinical notes, reports, documentation |
| **Image** | Visual data | Convolutional (CNN) | Medical imaging, ultrasounds, photographs |

### Why Late Fusion?

1. **Modality-Specific Learning**: Each network learns patterns specific to its data type
2. **Flexible Architecture**: Easy to add/remove modalities without retraining everything
3. **Computational Efficiency**: Only processes enabled modalities
4. **Interpretability**: Can analyze contribution of each modality independently

## 🎓 Training Models

Ahuvista-NN offers three methods for training models, each suited for different use cases.

### 1. Configuration File Approach (Recommended)

Best for: Production deployments, reproducible experiments, complex configurations

#### Step 1: Create Configuration File

Create `config.json`:

```json
{
  "modalities": {
    "use_tabular": true,
    "use_temporal": true,
    "use_text": false,
    "use_image": false
  },
  "data_paths": {
    "data_dir": "datasets",
    "patient_id_column": "PatientID",
    "tabular_files": ["patient_demographics.csv", "lab_results.csv"],
    "temporal_files": ["vitals_timeseries.csv"],
    "text_files": null,
    "image_files": null
  },
  "training": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "model": {
    "tabular_hidden_sizes": [128, 64],
    "tabular_output_size": 32,
    "temporal_hidden_size": 64,
    "temporal_output_size": 32,
    "text_embedding_dim": 100,
    "text_hidden_size": 64,
    "text_max_vocab": 10000,
    "image_channels": 3,
    "image_height": 224,
    "image_width": 224,
    "classifier_hidden_sizes": [128, 64]
  }
}
```

#### Step 2: Train with Configuration

```bash
cargo run --release --bin ahuvista-train -- --config config.json
```

#### Step 3: Override Specific Parameters

```bash
# Use config but override training parameters
cargo run --release --bin ahuvista-train -- \
  --config config.json \
  --epochs 200 \
  --learning-rate 0.0005 \
  --batch-size 64
```

### 2. Command-Line Interface

Best for: Quick experiments, testing, simple configurations

#### Basic Training

```bash
cargo run --release --bin ahuvista-train -- \
  --modalities tabular,temporal \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --output-dir ./trained_models \
  --verbose
```

#### All CLI Options

| Option | Short | Description | Example |
|--------|-------|-------------|---------|
| `--database` | | Path to prepared SQLite database | `--database ahuvista.db` |
| `--config` | `-c` | Path to JSON config file | `--config config.json` |
| `--modalities` | `-m` | Comma-separated modalities | `--modalities tabular,text` |
| `--epochs` | `-e` | Number of training epochs | `--epochs 100` |
| `--batch-size` | `-b` | Training batch size | `--batch-size 64` |
| `--learning-rate` | `-l` | Learning rate | `--learning-rate 0.001` |
| `--output-dir` | `-o` | Model output directory | `--output-dir ./models` |
| `--no-privacy` | | Disable differential privacy | `--no-privacy` |
| `--epsilon` | | Privacy epsilon value | `--epsilon 10.0` |
| `--clip-norm` | | Gradient clipping norm | `--clip-norm 5.0` |
| `--adaptive-lr` | | Enable adaptive learning rate | `--adaptive-lr` |
| `--verbose` | `-v` | Enable verbose logging | `--verbose` |

#### Examples

```bash
# Quick test with verbose output
cargo run --release --bin ahuvista-train -- \
  --modalities tabular \
  --epochs 10 \
  --verbose

# Production training with all options
cargo run --release --bin ahuvista-train -- \
  --config production.json \
  --epochs 200 \
  --batch-size 64 \
  --learning-rate 0.0001 \
  --output-dir ./production_models

# Multi-modal with custom output
cargo run --release --bin ahuvista-train -- \
  --modalities tabular,temporal,text,image \
  --epochs 150 \
  --batch-size 32 \
  --output-dir ./multimodal_models
```

### 3. Programmatic API

Best for: Custom training pipelines, research, integration with other systems

```rust
use ahuvista_nn::{
    config::{ModalityConfig, Settings},
    core::fusion_modular::{LateFusionNet, MultiModalInput},
    training::training_loop::TrainingConfig,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configure modalities
    let modality_config = ModalityConfig {
        use_tabular: true,
        use_temporal: true,
        use_text: false,
        use_image: false,
    };

    // 2. Build model
    let mut model = LateFusionNet::new(
        modality_config,
        Some((10, vec![64, 32], 16)),  // tabular: (input_size, hidden_layers, output_size)
        Some((5, 32, 16)),              // temporal: (input_size, hidden_size, output_size)
        None,                            // text: disabled
        None,                            // image: disabled
        &[64, 32],                      // classifier hidden sizes
    )?;

    println!("Model created with modalities: {:?}", model.enabled_modalities());

    // 3. Load and prepare data
    // ... your data loading code ...

    // 4. Configure training
    let training_config = TrainingConfig {
        epochs: 50,
        batch_size: 32,
        learning_rate: 0.001,
        val_frequency: 1,
    };

    // 5. Train model
    // ... your training loop ...

    // 6. Save model
    model.save("models/my_model.bin")?;

    Ok(())
}
```

#### Advanced: Dynamic Modality Selection

```rust
fn create_model_from_available_data(
    has_tabular: bool,
    has_temporal: bool,
    has_text: bool,
    has_image: bool,
) -> Result<LateFusionNet, Box<dyn std::error::Error>> {
    // Build configuration from available data
    let config = ModalityConfig {
        use_tabular: has_tabular,
        use_temporal: has_temporal,
        use_text: has_text,
        use_image: has_image,
    };

    // Validate at least one modality
    config.validate()?;

    // Build model with appropriate parameters
    let model = LateFusionNet::new(
        config,
        if has_tabular { Some((10, vec![64, 32], 16)) } else { None },
        if has_temporal { Some((5, 32, 16)) } else { None },
        if has_text { Some((HashMap::new(), 50, 32)) } else { None },
        if has_image { Some((3, 224, 224)) } else { None },
        &[64, 32],
    )?;

    Ok(model)
}
```

## 🔮 Making Predictions

### Single Prediction

#### Step 1: Prepare Input Data

Create `patient_input.json`:

```json
{
  "patient_id": "PAT-12345",
  "tabular": [0.5, 0.3, 0.8, 0.1, 0.9, 0.4, 0.7, 0.2, 0.6, 0.5],
  "temporal": [
    [120.0, 80.0, 98.6, 72.0, 16.0],
    [125.0, 82.0, 98.8, 75.0, 18.0],
    [130.0, 85.0, 99.0, 78.0, 20.0]
  ],
  "text": null,
  "image_path": null
}
```

#### Step 2: Run Prediction

```bash
cargo run --release --bin ahuvista-predict -- \
  --model models/maternal_mortality_model_modular.bin \
  --config config.json \
  --input patient_input.json \
  --output prediction_result.json
```

Or using modality specification:

```bash
cargo run --release --bin ahuvista-predict -- \
  --model models/model.bin \
  --modalities tabular,temporal \
  --input patient_input.json \
  --output result.json
```

#### Step 3: View Results

Output file `prediction_result.json`:

```json
{
  "risk_score": 0.73,
  "calibrated_score": null,
  "risk_category": "HIGH",
  "confidence": 0.89,
  "modalities_used": ["tabular", "temporal"],
  "patient_id": "PAT-12345",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Batch Predictions

Process multiple patients efficiently:

#### Step 1: Organize Input Files

```bash
mkdir patient_inputs
# Add patient_001.json, patient_002.json, patient_003.json, etc.
```

#### Step 2: Run Batch Prediction

```bash
cargo run --release --bin ahuvista-predict -- \
  --model models/model.bin \
  --config config.json \
  --input patient_inputs \
  --output batch_results.json \
  --batch
```

#### Step 3: Review Batch Results

Output `batch_results.json` contains an array of all predictions:

```json
[
  {
    "risk_score": 0.73,
    "risk_category": "HIGH",
    "patient_id": "PAT-001",
    ...
  },
  {
    "risk_score": 0.24,
    "risk_category": "LOW",
    "patient_id": "PAT-002",
    ...
  }
]
```

### Calibrated Predictions

Apply Bayesian calibration for more accurate probability estimates:

```bash
cargo run --release --bin ahuvista-predict -- \
  --model models/model.bin \
  --config config.json \
  --input patient.json \
  --calibrate 0.001 \
  --output result.json
```

The `--calibrate` parameter specifies the population prevalence (e.g., 0.001 = 0.1% prevalence).

**Why Use Calibration?**

- Adjusts predictions based on known population prevalence
- Provides more reliable probability estimates
- Crucial for imbalanced datasets
- Recommended for clinical decision support

## 🎛️ Modality Selection Guide

### Understanding Modalities

Each modality processes a different type of healthcare data:

| Modality | Data Type | Examples | When to Use |
|----------|-----------|----------|-------------|
| **Tabular** | Structured | Demographics, lab results, vitals | Always include if available |
| **Temporal** | Time-series | Continuous monitoring, trends | When tracking changes over time |
| **Text** | Unstructured | Clinical notes, reports | Rich qualitative information available |
| **Image** | Visual | Ultrasounds, X-rays, photos | Visual diagnosis required |

### Common Configuration Patterns

#### Pattern 1: Clinical Data Only (Tabular + Temporal)

**Use Case**: EMR/EHR systems with structured data

```bash
cargo run --release --bin ahuvista-train -- \
  --modalities tabular,temporal \
  --epochs 50
```

**Configuration**:
```json
{
  "modalities": {
    "use_tabular": true,
    "use_temporal": true,
    "use_text": false,
    "use_image": false
  }
}
```

#### Pattern 2: Medical Imaging Focus

**Use Case**: Radiology, ultrasound analysis

```bash
cargo run --release --bin ahuvista-train -- \
  --modalities image \
  --epochs 100
```

#### Pattern 3: Documentation-Rich System

**Use Case**: Systems with extensive clinical notes

```bash
cargo run --release --bin ahuvista-train -- \
  --modalities tabular,temporal,text \
  --epochs 75
```

#### Pattern 4: Full Multi-Modal (All Data Types)

**Use Case**: Comprehensive healthcare systems with all data types

```bash
cargo run --release --bin ahuvista-train -- \
  --modalities tabular,temporal,text,image \
  --epochs 150
```

### Modality Selection Decision Tree

```
Do you have structured patient data?
├─ YES → Enable TABULAR
│   │
│   └─ Do you have time-series data?
│       ├─ YES → Enable TEMPORAL
│       └─ NO → Continue
│
└─ Do you have clinical notes?
    ├─ YES → Enable TEXT
    └─ NO → Continue

Do you have medical images?
├─ YES → Enable IMAGE
└─ NO → Complete configuration
```

### Performance Considerations

| Configuration | Memory | Speed | Accuracy Potential |
|--------------|---------|-------|-------------------|
| Tabular only | ~50MB | Very Fast | Good |
| Tabular + Temporal | ~75MB | Fast | Better |
| Tabular + Text | ~100MB | Moderate | Better |
| Tabular + Image | ~150MB | Moderate | Better |
| All modalities | ~200MB | Slower | Best |

## 📊 Data Format Specifications

### Tabular Data

**Format**: CSV files with headers

**Requirements**:
- Must include `PatientID` column (or configured identifier)
- Numeric features only
- Missing values should be imputed before training

**Example** (`patient_data.csv`):
```csv
PatientID,Age,BMI,BloodPressure,HeartRate,Temperature,Hemoglobin,Glucose,RiskFactor1,RiskFactor2
PAT-001,28,24.5,120,75,98.6,12.5,95,0.8,0.3
PAT-002,35,28.1,130,82,99.1,11.8,110,0.6,0.5
PAT-003,22,22.3,115,70,98.4,13.2,88,0.2,0.1
```

### Temporal Data

**Format**: CSV with timestamp column

**Requirements**:
- Must include `PatientID` column
- Must include `Timestamp` or time-ordered rows
- Each patient can have multiple time points

**Example** (`vitals_timeseries.csv`):
```csv
PatientID,Timestamp,SystolicBP,DiastolicBP,Temperature,HeartRate,RespiratoryRate
PAT-001,2024-01-01 08:00,120,80,98.6,72,16
PAT-001,2024-01-01 12:00,125,82,98.8,75,18
PAT-001,2024-01-01 16:00,130,85,99.0,78,20
PAT-002,2024-01-01 08:00,135,88,99.2,85,22
```

### Text Data

**Format**: Plain text files or CSV with text column

**Requirements**:
- One document per patient
- UTF-8 encoding
- Vocabulary will be built automatically

**Example** (`clinical_notes.txt`):
```
PAT-001: Patient presents with mild hypertension. Blood pressure elevated but stable...
PAT-002: Routine prenatal visit. Patient reports feeling well. No complications noted...
```

### Image Data

**Format**: Standard image formats (JPG, PNG, TIFF)

**Requirements**:
- Images will be resized to configured dimensions (default 224x224)
- RGB or grayscale
- One image per patient (or multiple images in directory structure)

**Directory Structure**:
```
images/
├── PAT-001.jpg
├── PAT-002.png
├── PAT-003.jpg
```

### Prediction Input Format

**JSON Structure**:

```json
{
  "patient_id": "string (optional)",
  "tabular": [array of numbers] or null,
  "temporal": [[array of numbers per timestep]] or null,
  "text": "string" or null,
  "image_path": "path/to/image.jpg" or null
}
```

**Complete Example**:

```json
{
  "patient_id": "PAT-12345",
  "tabular": [28.0, 24.5, 120.0, 75.0, 98.6, 12.5, 95.0, 0.8, 0.3, 0.1],
  "temporal": [
    [120.0, 80.0, 98.6, 72.0, 16.0],
    [125.0, 82.0, 98.8, 75.0, 18.0],
    [130.0, 85.0, 99.0, 78.0, 20.0]
  ],
  "text": "Patient presents with mild hypertension during routine prenatal visit.",
  "image_path": "images/ultrasound_001.jpg"
}
```

## ⚙️ Configuration Reference

### Complete Configuration Schema

```json
{
  "modalities": {
    "use_tabular": true,      // Enable/disable tabular data processing
    "use_temporal": true,     // Enable/disable temporal/time-series processing
    "use_text": false,        // Enable/disable text/NLP processing
    "use_image": false        // Enable/disable image processing
  },
  "data_paths": {
    "data_dir": "datasets",             // Root directory for all data files
    "patient_id_column": "PatientID",   // Column name for patient identifiers
    "tabular_files": ["file1.csv", "file2.csv"],    // List of tabular CSV files
    "temporal_files": ["timeseries.csv"],            // List of temporal CSV files
    "text_files": ["notes.txt", "reports.txt"],      // List of text files
    "image_files": ["images/", "scans/"]             // Image directories or file lists
  },
  "training": {
    "epochs": 50,              // Number of complete passes through training data
    "batch_size": 32,          // Number of samples per training batch
    "learning_rate": 0.001     // Learning rate for gradient descent
  },
  "model": {
    // Tabular network architecture
    "tabular_hidden_sizes": [64, 32],   // Hidden layer sizes (2 layers: 64→32 neurons)
    "tabular_output_size": 16,           // Size of tabular feature vector

    // Temporal network architecture
    "temporal_hidden_size": 32,          // RNN hidden state size
    "temporal_output_size": 16,          // Size of temporal feature vector

    // Text network architecture
    "text_embedding_dim": 50,            // Word embedding dimension
    "text_hidden_size": 32,              // Text RNN hidden state size
    "text_max_vocab": 5000,              // Maximum vocabulary size

    // Image network architecture
    "image_channels": 3,                 // Number of image channels (3=RGB, 1=grayscale)
    "image_height": 224,                 // Target image height in pixels
    "image_width": 224,                  // Target image width in pixels

    // Classifier network
    "classifier_hidden_sizes": [64, 32]  // Final classifier hidden layers
  }
}
```

### Configuration Parameter Guide

#### Modalities Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_tabular` | boolean | `true` | Process structured tabular data |
| `use_temporal` | boolean | `true` | Process time-series sequences |
| `use_text` | boolean | `false` | Process unstructured text |
| `use_image` | boolean | `false` | Process image data |

**Note**: At least one modality must be enabled.

#### Training Section

| Parameter | Type | Range | Recommended | Description |
|-----------|------|-------|-------------|-------------|
| `epochs` | integer | 1-1000 | 50-100 | Full training passes |
| `batch_size` | integer | 1-256 | 16-64 | Samples per batch |
| `learning_rate` | float | 0.00001-0.1 | 0.001-0.01 | Gradient step size |

**Tuning Tips**:
- **Small datasets** (< 1000 samples): Lower batch size (8-16), more epochs (100-200)
- **Medium datasets** (1000-10000): Standard settings (batch=32, epochs=50-100)
- **Large datasets** (> 10000): Larger batch size (64-128), fewer epochs (20-50)

#### Model Architecture Section

**Tabular Network**:
- `tabular_hidden_sizes`: List of hidden layer sizes. Example: `[128, 64]` creates two layers
- `tabular_output_size`: Feature vector dimension (typically 16-32)

**Temporal Network**:
- `temporal_hidden_size`: RNN memory size (typically 32-64)
- `temporal_output_size`: Feature vector dimension (typically 16-32)

**Text Network**:
- `text_embedding_dim`: Word vector size (typically 50-300)
- `text_hidden_size`: RNN memory size (typically 32-128)
- `text_max_vocab`: Vocabulary limit (typically 5000-20000)

**Image Network**:
- `image_channels`: 3 for RGB, 1 for grayscale
- `image_height`/`image_width`: Target dimensions (commonly 224x224 or 256x256)

**Classifier**:
- `classifier_hidden_sizes`: Final layers that combine modality features

## 🎯 Advanced Usage

### Custom Training with Population Weighting

```rust
use ahuvista_nn::training::balancing::{
    compute_sample_weights,
    BayesianCalibrator,
    CauseWeights
};
use std::collections::HashMap;

// Define population-level cause frequencies
let mut cause_frequencies = HashMap::new();
cause_frequencies.insert(0, 0.25); // Hemorrhage: 25%
cause_frequencies.insert(1, 0.20); // Hypertensive disorders: 20%
cause_frequencies.insert(2, 0.15); // Infection: 15%
cause_frequencies.insert(3, 0.10); // Abortion + Miscarriage: 8%
cause_frequencies.insert(4, 0.30); // Other: 30%

let cause_weights = CauseWeights::from_population_frequencies(&cause_frequencies);

// Define demographic strata
let mut stratum_multipliers = HashMap::new();
stratum_multipliers.insert("urban_young".to_string(), 1.0);
stratum_multipliers.insert("urban_old".to_string(), 1.2);
stratum_multipliers.insert("rural_young".to_string(), 1.5);
stratum_multipliers.insert("rural_old".to_string(), 1.8);

// Compute sample weights
let sample_weights = compute_sample_weights(
    &targets,
    Some(&patient_strata),
    Some(&stratum_multipliers)
);
```

### Model Inspection and Debugging

```rust
// Check enabled modalities
let modalities = model.enabled_modalities();
println!("Active modalities: {:?}", modalities);

// Verify specific modality
if model.is_enabled("tabular") {
    println!("Tabular processing enabled");
}

// Get model configuration
let config = model.config();
println!("Modality config: {:?}", config);
```

### Save and Load Models

```rust
// Save trained model
model.save("models/my_model.bin")?;

// Load model for inference
let mut loaded_model = LateFusionNet::new(
    config,
    tabular_params,
    temporal_params,
    text_params,
    image_params,
    &classifier_sizes,
)?;

loaded_model.load("models/my_model.bin")?;
```

### Custom Prediction Pipeline

```rust
use ahuvista_nn::core::fusion_modular::MultiModalInput;

// Build input programmatically
let mut input = MultiModalInput::new();

if has_tabular_data {
    input = input.with_tabular(tabular_vector);
}

if has_temporal_data {
    input = input.with_temporal(temporal_sequences);
}

// Make prediction
let risk_score = model.predict(input)?;

// Apply custom thresholds
let risk_level = if risk_score > 0.8 {
    "CRITICAL"
} else if risk_score > 0.5 {
    "HIGH"
} else {
    "MODERATE"
};
```

## 📈 Performance & Benchmarks

### Computational Requirements

| Configuration | RAM Usage | Training Time* | Inference Time** |
|--------------|-----------|----------------|------------------|
| Tabular only | ~50 MB | 2-5 min | < 10 ms |
| Tabular + Temporal | ~75 MB | 5-10 min | < 20 ms |
| Tabular + Text | ~100 MB | 10-15 min | < 30 ms |
| Tabular + Image | ~150 MB | 15-25 min | < 50 ms |
| All modalities | ~200 MB | 30-45 min | < 100 ms |

\* Training time for 10,000 samples, 50 epochs on CPU (Intel i7)
\** Single prediction on CPU

### Optimization Tips

**For Faster Training**:
1. Use release mode: `cargo build --release`
2. Increase batch size on machines with more RAM
3. Disable unused modalities
4. Use smaller network architectures for prototyping

**For Lower Memory**:
1. Reduce batch size
2. Use smaller hidden layer sizes
3. Limit vocabulary size for text
4. Reduce image dimensions

**For Faster Inference**:
1. Compile with optimizations: `--release`
2. Use minimal modalities needed
3. Batch predictions when possible
4. Consider model quantization (future feature)

### Scalability

- **Small datasets** (< 1,000 samples): Runs on laptops, embedded systems
- **Medium datasets** (1,000-50,000 samples): Standard workstations
- **Large datasets** (> 50,000 samples): Server-grade hardware recommended

## 📚 Examples

### Example 1: Quick Prototype with Tabular Data

```bash
# Step 1: Create minimal config
cat > quick_config.json << 'EOF'
{
  "modalities": {
    "use_tabular": true,
    "use_temporal": false,
    "use_text": false,
    "use_image": false
  },
  "training": {
    "epochs": 20,
    "batch_size": 16,
    "learning_rate": 0.001
  }
}
EOF

# Step 2: Train
cargo run --release --bin ahuvista-train -- \
  --config quick_config.json \
  --verbose

# Step 3: Test prediction
cat > test_input.json << 'EOF'
{
  "tabular": [0.5, 0.3, 0.8, 0.1, 0.9, 0.4, 0.7, 0.2, 0.6, 0.5]
}
EOF

cargo run --release --bin ahuvista-predict -- \
  --model models/maternal_mortality_model_modular.bin \
  --modalities tabular \
  --input test_input.json
```

### Example 2: Production Multi-Modal System

```bash
# Step 1: Create production configuration
cat > production_config.json << 'EOF'
{
  "modalities": {
    "use_tabular": true,
    "use_temporal": true,
    "use_text": true,
    "use_image": false
  },
  "data_paths": {
    "data_dir": "/data/healthcare",
    "patient_id_column": "MRN",
    "tabular_files": ["demographics.csv", "labs.csv"],
    "temporal_files": ["vitals_monitoring.csv"],
    "text_files": ["clinical_notes.txt"]
  },
  "training": {
    "epochs": 150,
    "batch_size": 64,
    "learning_rate": 0.0005
  },
  "model": {
    "tabular_hidden_sizes": [256, 128, 64],
    "tabular_output_size": 64,
    "temporal_hidden_size": 128,
    "temporal_output_size": 64,
    "text_embedding_dim": 200,
    "text_hidden_size": 128,
    "text_max_vocab": 15000,
    "classifier_hidden_sizes": [256, 128]
  }
}
EOF

# Step 2: Train with production settings
cargo run --release --bin ahuvista-train -- \
  --config production_config.json \
  --output-dir /models/production

# Step 3: Deploy for inference with calibration
cargo run --release --bin ahuvista-predict -- \
  --model /models/production/maternal_mortality_model_modular.bin \
  --config production_config.json \
  --input /data/inference/patient.json \
  --calibrate 0.0015 \
  --output /results/prediction.json
```

### Example 3: Batch Processing for Research

```bash
# Process 1000 patients
mkdir -p batch_inputs batch_outputs

# Generate input files (your data preparation script)
# ...

# Run batch prediction
cargo run --release --bin ahuvista-predict -- \
  --model models/research_model.bin \
  --config research_config.json \
  --input batch_inputs \
  --output batch_outputs/results.json \
  --batch \
  --calibrate 0.001
```

### Example 4: Experimentation with Different Architectures

```bash
# Test different modality combinations
for combo in "tabular" "temporal" "tabular,temporal" "tabular,text" "all"; do
  echo "Testing: $combo"

  if [ "$combo" = "all" ]; then
    modalities="tabular,temporal,text,image"
  else
    modalities="$combo"
  fi

  cargo run --release --bin ahuvista-train -- \
    --modalities $modalities \
    --epochs 30 \
    --output-dir experiments/$combo
done
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue: "At least one modality must be enabled"

**Cause**: No modalities selected in configuration

**Solution**:
```bash
# Ensure at least one modality is enabled
cargo run --release --bin ahuvista-train -- \
  --modalities tabular \
  --epochs 10
```

#### Issue: "Tabular data required but not provided"

**Cause**: Model expects tabular data but input doesn't include it

**Solution**: Ensure input JSON includes all required modalities:
```json
{
  "tabular": [/* your data */],
  "temporal": null
}
```

#### Issue: Model training is too slow

**Solutions**:
1. Reduce batch size: `--batch-size 16`
2. Use fewer epochs: `--epochs 20`
3. Disable unnecessary modalities
4. Ensure using release mode: `--release`

#### Issue: Out of memory during training

**Solutions**:
1. Reduce batch size: `--batch-size 8`
2. Use smaller network architectures
3. Process fewer modalities simultaneously
4. Reduce image dimensions in config

#### Issue: Predictions seem uncalibrated

**Solution**: Use Bayesian calibration:
```bash
cargo run --release --bin ahuvista-predict -- \
  --model model.bin \
  --config config.json \
  --input patient.json \
  --calibrate 0.001
```

#### Issue: "Failed to load config file"

**Causes & Solutions**:
- Invalid JSON syntax → Validate JSON with `jsonlint`
- File not found → Check path is correct
- Wrong permissions → Ensure read permissions on file

### Getting Help

1. **Check Logs**: Use `--verbose` flag for detailed output
2. **Review Examples**: See the `examples/` directory
3. **Read Documentation**: Check inline code documentation
4. **Open an Issue**: [GitHub Issues](https://github.com/dandychux/ahuvista-nn/issues)

## 🤝 Contributing

Contributions are welcome! We appreciate:

- 🐛 Bug reports and fixes
- 📚 Documentation improvements
- ✨ New features and enhancements
- 🧪 Additional tests and benchmarks

### Development Setup

```bash
# Clone repository
git clone https://github.com/dandychux/ahuvista-nn.git
cd ahuvista-nn

# Create feature branch
git checkout -b feature/my-new-feature

# Make changes and test
cargo test
cargo fmt
cargo clippy

# Submit pull request
```

### Code Style

- Follow Rust standard formatting (`cargo fmt`)
- Pass all clippy lints (`cargo clippy`)
- Add tests for new features
- Update documentation as needed

## 📄 License

This project is dual-licensed under:

- **MIT License** ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- **Apache License 2.0** ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

You may choose either license for your use.

## 📧 Contact & Support

- **Author**: Chukwuma Okoroji
- **Email**: dandychux@gmail.com
- **GitHub**: [@dandychux](https://github.com/dandychux)
- **Issues**: [GitHub Issues](https://github.com/dandychux/ahuvista-nn/issues)

## 🙏 Acknowledgments

This project was developed to improve maternal health outcomes through accessible, efficient AI systems that can run in resource-constrained environments.

Special thanks to the open-source Rust community and healthcare professionals who provided domain expertise.

## ⚠️ Important Notices

### Clinical Use Disclaimer

**This is a research and development tool. It is NOT approved for clinical use.**

- Always consult qualified healthcare professionals for medical decisions
- Do not use as the sole basis for diagnosis or treatment
- Predictions should be validated in clinical context
- Follow all applicable regulations and guidelines

### Data Privacy

When using this system with patient data:

- Comply with HIPAA, GDPR, and other relevant regulations
- Implement appropriate data security measures
- De-identify data when possible
- Document all data handling procedures
- Obtain necessary approvals and consents

### Research Use

If using for research:

- Cite this project appropriately
- Follow ethical research guidelines
- Obtain IRB approval when required
- Report limitations and biases honestly

---

## 🚀 Quick Reference Card

### Data Preparation

```bash
# Prepare all modalities
cargo run --release --bin ahuvista-prepare -- \
  --data-dir datasets \
  --database ahuvista.db

# Prepare specific modalities only
cargo run --release --bin ahuvista-prepare -- \
  --data-dir datasets \
  --database ahuvista.db \
  --modalities tabular,temporal

# Clear and re-import
cargo run --release --bin ahuvista-prepare -- \
  --data-dir datasets \
  --database ahuvista.db \
  --clear \
  --verbose
```

### Training

```bash
# Basic (requires prepared database)
cargo run --release --bin ahuvista-train -- \
  --database ahuvista.db \
  --modalities tabular \
  --epochs 20

# With config
cargo run --release --bin ahuvista-train -- \
  --database ahuvista.db \
  --config config.json

# Full options
cargo run --release --bin ahuvista-train -- \
  --database ahuvista.db \
  --config config.json \
  --modalities tabular,temporal \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --output-dir ./models \
  --verbose
```

### Prediction

```bash
# Single
cargo run --release --bin ahuvista-predict -- \
  --model model.bin \
  --config config.json \
  --input patient.json

# Batch
cargo run --release --bin ahuvista-predict -- \
  --model model.bin \
  --config config.json \
  --input patients_dir \
  --batch

# Calibrated
cargo run --release --bin ahuvista-predict -- \
  --model model.bin \
  --config config.json \
  --input patient.json \
  --calibrate 0.001
```

### Modalities

- `tabular` - Structured data
- `temporal` - Time-series
- `text` - Clinical notes
- `image` - Medical images

Combine with commas: `--modalities tabular,temporal,text`

### Typical Workflow

1. Prepare data (once, or when data changes)
ahuvista-prepare --data-dir datasets --database ahuvista.db

2. Train model (can run multiple times with different configs)
ahuvista-train --database ahuvista.db --epochs 100 --output-dir models

3. Make predictions
ahuvista-predict --model models/model.bin --input patient.json

---

**Version**: 2.0.1
**Last Updated**: 2025
**Status**: Active Development
