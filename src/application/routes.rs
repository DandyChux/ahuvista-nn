use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::{
    core::fusion::LateFusionNet,
    data::fhir::RiskAssessment,
    error::AppError,
    model_artifact::ModelArtifact,
    prediction::{PredictionInput, PredictionOutput},
};

use super::predict::predict_one;

#[derive(Clone)]
pub struct AppState {
    pub model: Arc<Mutex<LateFusionNet>>,
    pub artifact: Arc<ModelArtifact>,
}

#[derive(Debug, Deserialize)]
pub struct PredictionRequest {
    #[serde(flatten)]
    pub input: PredictionInput,

    #[serde(default)]
    pub source_reference: Option<String>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    model_id: String,
    model_version: String,
}

#[derive(Debug, Serialize)]
struct MetadataResponse {
    model_id: String,
    model_version: String,
    modalities: Vec<String>,
    sequence_static_input_size: Option<usize>,
    sequence_dynamic_input_size: Option<usize>,
    sequence_hidden_size: Option<usize>,
    sequence_output_size: Option<usize>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug)]
pub struct ApiError(pub AppError);

impl From<AppError> for ApiError {
    fn from(error: AppError) -> Self {
        Self(error)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self.0 {
            AppError::Config(_) | AppError::InvalidData(_) | AppError::FhirError(_) => {
                StatusCode::BAD_REQUEST
            }

            AppError::ModelError(_) => StatusCode::INTERNAL_SERVER_ERROR,

            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };

        (
            status,
            Json(ErrorResponse {
                error: self.0.to_string(),
            }),
        )
            .into_response()
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metadata", get(metadata))
        .route("/predict", post(predict))
        .route("/fhir/RiskAssessment", post(fhir_risk_assessment))
        .with_state(state)
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        model_id: state.artifact.spec.model_id.clone(),
        model_version: state.artifact.spec.model_version.clone(),
    })
}

async fn metadata(State(state): State<AppState>) -> Json<MetadataResponse> {
    let spec = &state.artifact.spec;

    Json(MetadataResponse {
        model_id: spec.model_id.clone(),
        model_version: spec.model_version.clone(),
        modalities: spec
            .modalities
            .enabled_modalities()
            .into_iter()
            .map(str::to_string)
            .collect(),
        sequence_static_input_size: spec.sequence_static_input_size,
        sequence_dynamic_input_size: spec.sequence_dynamic_input_size,
        sequence_hidden_size: spec.sequence_hidden_size,
        sequence_output_size: spec.sequence_output_size,
    })
}

async fn predict(
    State(state): State<AppState>,
    Json(request): Json<PredictionRequest>,
) -> Result<Json<PredictionOutput>, ApiError> {
    let mut model = state.model.lock().await;

    let prediction = predict_one(&mut model, &state.artifact, request.input, None)?;

    Ok(Json(prediction))
}

async fn fhir_risk_assessment(
    State(state): State<AppState>,
    Json(request): Json<PredictionRequest>,
) -> Result<Json<RiskAssessment>, ApiError> {
    let mut model = state.model.lock().await;

    let prediction = predict_one(&mut model, &state.artifact, request.input, None)?;

    let risk_assessment = prediction.to_fhir_risk_assessment(
        "https://ahuvista.example.org/fhir/CodeSystem/maternal-outcome",
        "adverse-maternal-outcome",
        "Adverse maternal outcome",
        "https://ahuvista.example.org/fhir/CodeSystem/model",
        &state.artifact.spec.model_id,
        &state.artifact.spec.model_version,
    );

    Ok(Json(risk_assessment))
}
