use std::{fs, path::Path};

use chrono::Utc;
use image::imageops::FilterType;
use ndarray::Array3;
use tracing::info;

use crate::{
    core::fusion::{LateFusionNet, MultiModalInput},
    error::{AppError, AppResult},
    model_artifact::ModelArtifact,
    prediction::{PredictionInput, PredictionOutput},
    training::balancing::BayesianCalibrator,
};

use super::args::PredictArgs;

pub fn run(args: PredictArgs) -> AppResult<()> {
    let artifact = ModelArtifact::load(&args.model)?;

    let mut model = artifact.instantiate()?;

    let input_json = fs::read_to_string(&args.input)?;

    let input: PredictionInput = serde_json::from_str(&input_json)?;

    let prediction = predict_one(&mut model, &artifact, input, None)?;

    let output_path = args
        .output
        .unwrap_or_else(|| "prediction_output.json".to_string());

    let output_value = match args.output_format.as_str() {
        "json" => serde_json::to_value(&prediction)?,
        "fhir-risk-assessment" => serde_json::to_value(prediction.to_fhir_risk_assessment(
            "https://ahuvista.example.org/outcomes",
            "maternal-risk",
            "Maternal health risk",
            "https://ahuvista.example.org/models",
            "ahuvista-model",
            "AhuVista maternal risk model",
        ))?,
        "fhir-bundle" => serde_json::json!({
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{ "resource": prediction.to_fhir_risk_assessment(
                "https://ahuvista.example.org/outcomes",
                "maternal-risk",
                "Maternal health risk",
                "https://ahuvista.example.org/models",
                "ahuvista-model",
                "AhuVista maternal risk model",
            ) }]
        }),
        format => {
            return Err(AppError::InvalidData(format!(
                "unsupported output format: {format}"
            )));
        }
    };

    if args.publish {
        let url = args.fhir_url.as_deref().ok_or_else(|| {
            AppError::InvalidData("--fhir-url is required with --publish".to_string())
        })?;
        let token = args
            .fhir_token_env
            .as_deref()
            .and_then(std::env::var_os)
            .and_then(|value| value.into_string().ok());
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|error| AppError::DomainError(error.to_string()))?;
        runtime.block_on(
            crate::data::fhir::FhirClient::new(url, token).publish_resource(&output_value),
        )?;
        info!("Prediction published to FHIR server");
    }

    let output = serde_json::to_vec_pretty(&output_value)?;
    fs::write(&output_path, output)?;

    info!("Prediction written to {}", output_path);

    Ok(())
}

pub fn predict_one(
    model: &mut LateFusionNet,
    artifact: &ModelArtifact,
    input: PredictionInput,
    calibrator: Option<&BayesianCalibrator>,
) -> AppResult<PredictionOutput> {
    let mut model_input = MultiModalInput::new();

    if artifact.spec.modalities.use_sequence {
        let mut sequence = input.sequence.ok_or_else(|| {
            AppError::InvalidData("Sequence input is required by this model".to_string())
        })?;

        for (index, value) in sequence.static_features.iter_mut().enumerate() {
            if let Some(stats) = artifact.spec.static_normalization.get(index) {
                *value = stats.normalize(*value).clamp(-5.0, 5.0);
            }
        }

        for timestep in &mut sequence.dynamic_features {
            for (index, value) in timestep.iter_mut().enumerate() {
                if let Some(stats) = artifact.spec.dynamic_normalization.get(index) {
                    *value = stats.normalize(*value).clamp(-5.0, 5.0);
                }
            }
        }

        validate_sequence(
            &sequence,
            artifact.spec.sequence_static_input_size.unwrap_or(0),
            artifact.spec.sequence_dynamic_input_size.unwrap_or(0),
        )?;

        model_input = model_input.with_sequence(sequence);
    }

    if artifact.spec.modalities.use_text {
        let text = input.text.ok_or_else(|| {
            AppError::InvalidData("Text input is required by this model".to_string())
        })?;

        let tokens = tokenize_text(&text, &artifact.spec.text_vocab);

        model_input = model_input.with_text(tokens);
    }

    if artifact.spec.modalities.use_image {
        let image_path = input.image_path.ok_or_else(|| {
            AppError::InvalidData("Image input is required by this model".to_string())
        })?;

        let image =
            load_image(
                &image_path,
                artifact.spec.image_height.ok_or_else(|| {
                    AppError::ModelError("Model image height is missing".to_string())
                })?,
                artifact.spec.image_width.ok_or_else(|| {
                    AppError::ModelError("Model image width is missing".to_string())
                })?,
            )?;

        model_input = model_input.with_image(image);
    }

    let raw_score = model.predict(model_input).map_err(AppError::Config)?;

    let calibrated_score = calibrator.map(|calibrator| calibrator.calibrate(raw_score));

    let score = calibrated_score.unwrap_or(raw_score);

    let (risk_category, confidence) = categorize_risk(score);

    Ok(PredictionOutput {
        risk_score: raw_score,
        calibrated_score,
        risk_category,
        confidence,
        modalities_used: model
            .enabled_modalities()
            .into_iter()
            .map(str::to_string)
            .collect(),
        patient_id: input.patient_id,
        timestamp: Utc::now().to_rfc3339(),
    })
}

fn validate_sequence(
    sequence: &crate::models::PatientSequence,
    expected_static_size: usize,
    expected_dynamic_size: usize,
) -> AppResult<()> {
    sequence
        .validate(expected_static_size, expected_dynamic_size)
        .map_err(AppError::InvalidData)
}

fn tokenize_text(text: &str, vocabulary: &std::collections::HashMap<String, usize>) -> Vec<usize> {
    text.split_whitespace()
        .map(|token| vocabulary.get(&token.to_lowercase()).copied().unwrap_or(0))
        .collect()
}

fn load_image(path: impl AsRef<Path>, height: usize, width: usize) -> AppResult<Array3<f32>> {
    let image = image::open(path)?;

    let image = image.resize_exact(width as u32, height as u32, FilterType::Lanczos3);

    let rgb = image.to_rgb8();

    let mut output = Array3::<f32>::zeros((3, height, width));

    for (x, y, pixel) in rgb.enumerate_pixels() {
        output[[0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;

        output[[1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;

        output[[2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
    }

    Ok(output)
}

fn categorize_risk(score: f32) -> (String, f32) {
    let score = score.clamp(0.0, 1.0);

    if score >= 0.8 {
        ("CRITICAL".to_string(), score)
    } else if score >= 0.6 {
        ("HIGH".to_string(), score)
    } else if score >= 0.4 {
        ("MODERATE".to_string(), score)
    } else if score >= 0.2 {
        ("LOW".to_string(), 1.0 - score)
    } else {
        ("MINIMAL".to_string(), 1.0 - score)
    }
}
