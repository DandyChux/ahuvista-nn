//! SMART on FHIR Service for Ahuvista-NN
//!
//! This binary provides:
//! - SMART App Launch (EHR-embedded launch)
//! - Standalone FHIR API endpoints
//! - RiskAssessment write-back to EHR

use ahuvista_nn::config::Settings;
use ahuvista_nn::core::fusion::{LateFusionNet, MultiModalInput};
use ahuvista_nn::data::fhir::{
    Bundle, FhirError, FhirMappingConfig, FhirParser, RiskAssessmentBuilder, RiskAssessmentOutput,
};
use ahuvista_nn::data::resources::Bundle;
use ahuvista_nn::training::balancing::BayesianCalibrator;

use axum::{
    Router,
    extract::{Json, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Redirect},
    routing::{get, post},
};
use clap::Parser;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "SMART on FHIR service for Ahuvista-NN predictions"
)]
struct Args {
    /// Path to trained model file
    #[arg(short, long)]
    model: PathBuf,

    /// Path to model metadata file
    #[arg(long)]
    model_metadata: PathBuf,

    /// Path to FHIR mapping configuration
    #[arg(long)]
    fhir_mapping: PathBuf,

    /// Path to OAuth/SMART configuration
    #[arg(long)]
    oauth_config: PathBuf,

    /// Host to bind to
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Enable write-back of RiskAssessment to EHR
    #[arg(long)]
    enable_writeback: bool,

    /// Apply Bayesian calibration with specified prevalence
    #[arg(long)]
    calibrate: Option<f32>,
}

// ============================================================================
// Configuration Types
// ============================================================================

#[derive(Debug, Deserialize, Clone)]
pub struct OAuthConfig {
    /// Your app's client ID (registered with EHR vendor)
    pub client_id: String,

    /// Client secret (only for confidential apps - server-side)
    pub client_secret: Option<String>,

    /// Redirect URI registered with EHR
    pub redirect_uri: String,

    /// Requested scopes
    pub scopes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SmartConfiguration {
    authorization_endpoint: String,
    token_endpoint: String,
    capabilities: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct LaunchParams {
    iss: String,    // FHIR server base URL
    launch: String, // Opaque launch context token
}

#[derive(Debug, Deserialize)]
struct CallbackParams {
    code: String,
    state: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct TokenResponse {
    access_token: String,
    token_type: String,
    expires_in: Option<u64>,
    scope: String,
    patient: Option<String>,   // Patient ID from launch context
    encounter: Option<String>, // Encounter ID if launched from encounter
    id_token: Option<String>,
    refresh_token: Option<String>,
}

// ============================================================================
// Application State
// ============================================================================

#[derive(Clone)]
struct AppState {
    model: Arc<Mutex<LateFusionNet>>,
    model_metadata: Arc<ModelMetadata>,
    fhir_parser: Arc<FhirParser>,
    risk_assessment_builder: Arc<RiskAssessmentBuilder>,
    oauth_config: Arc<OAuthConfig>,
    http_client: Client,
    calibrator: Option<Arc<BayesianCalibrator>>,
    enable_writeback: bool,
    // In-memory session store (use Redis/DB in production)
    sessions: Arc<Mutex<HashMap<String, SessionData>>>,
}

#[derive(Debug, Clone)]
struct SessionData {
    fhir_base_url: String,
    state_token: String,
    created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelMetadata {
    modality_config: ahuvista_nn::config::ModalityConfig,
    tabular_input_size: Option<usize>,
    // ... other fields from predict.rs
}

// ============================================================================
// Main Entry Point
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("ahuvista_fhir_service=debug,tower_http=debug")
        .init();

    let args = Args::parse();

    info!("Loading model from {:?}", args.model);
    let model = load_model(&args.model)?;
    let model_metadata = load_model_metadata(&args.model_metadata)?;

    info!("Loading FHIR mapping from {:?}", args.fhir_mapping);
    let fhir_mapping = FhirMappingConfig::load(&args.fhir_mapping)?;
    let fhir_parser = FhirParser::new(fhir_mapping);

    info!("Loading OAuth configuration from {:?}", args.oauth_config);
    let oauth_config: OAuthConfig =
        serde_json::from_str(&std::fs::read_to_string(&args.oauth_config)?)?;

    let calibrator = args
        .calibrate
        .map(|prevalence| Arc::new(BayesianCalibrator::new(prevalence)));

    let state = AppState {
        model: Arc::new(Mutex::new(model)),
        model_metadata: Arc::new(model_metadata),
        fhir_parser: Arc::new(fhir_parser),
        risk_assessment_builder: Arc::new(RiskAssessmentBuilder::default()),
        oauth_config: Arc::new(oauth_config),
        http_client: Client::new(),
        calibrator,
        enable_writeback: args.enable_writeback,
        sessions: Arc::new(Mutex::new(HashMap::new())),
    };

    let app = Router::new()
        // SMART App Launch endpoints
        .route("/launch", get(smart_launch))
        .route("/callback", get(oauth_callback))
        // Standalone API endpoints
        .route("/api/predict", post(predict_from_bundle))
        .route("/api/predict/batch", post(batch_predict))
        // Health & discovery
        .route("/health", get(health_check))
        .route("/.well-known/smart-configuration", get(smart_configuration))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    info!("Starting FHIR service on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ============================================================================
// SMART App Launch Handlers
// ============================================================================

/// Step 1: EHR redirects here when user launches the app
async fn smart_launch(
    Query(params): Query<LaunchParams>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, AppError> {
    info!("SMART launch initiated from {}", params.iss);

    // 1. Discover the EHR's SMART configuration
    let smart_config = discover_smart_configuration(&state.http_client, &params.iss).await?;

    // 2. Generate state token for CSRF protection
    let state_token = generate_state_token();

    // 3. Store session data
    {
        let mut sessions = state.sessions.lock().await;
        sessions.insert(
            state_token.clone(),
            SessionData {
                fhir_base_url: params.iss.clone(),
                state_token: state_token.clone(),
                created_at: chrono::Utc::now(),
            },
        );
    }

    // 4. Build authorization URL
    let scopes = state.oauth_config.scopes.join(" ");
    let auth_url = format!(
        "{}?response_type=code&client_id={}&redirect_uri={}&scope={}&launch={}&aud={}&state={}",
        smart_config.authorization_endpoint,
        urlencoding::encode(&state.oauth_config.client_id),
        urlencoding::encode(&state.oauth_config.redirect_uri),
        urlencoding::encode(&scopes),
        urlencoding::encode(&params.launch),
        urlencoding::encode(&params.iss),
        urlencoding::encode(&state_token)
    );

    info!("Redirecting to authorization endpoint");
    Ok(Redirect::temporary(&auth_url))
}

/// Step 2: EHR redirects back here after user authorizes
async fn oauth_callback(
    Query(params): Query<CallbackParams>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, AppError> {
    info!("OAuth callback received");

    // 1. Validate state token
    let session = {
        let sessions = state.sessions.lock().await;
        sessions
            .get(&params.state)
            .cloned()
            .ok_or(AppError::InvalidSession)?
    };

    // 2. Discover token endpoint
    let smart_config =
        discover_smart_configuration(&state.http_client, &session.fhir_base_url).await?;

    // 3. Exchange code for access token
    let token = exchange_code_for_token(
        &state.http_client,
        &smart_config.token_endpoint,
        &params.code,
        &state.oauth_config,
    )
    .await?;

    info!(
        "Access token obtained, patient context: {:?}",
        token.patient
    );

    // 4. Get patient ID from launch context
    let patient_id = token
        .patient
        .as_ref()
        .ok_or(AppError::MissingPatientContext)?;

    // 5. Fetch patient data from FHIR server
    let bundle = fetch_patient_data(
        &state.http_client,
        &session.fhir_base_url,
        patient_id,
        &token.access_token,
    )
    .await?;

    // 6. Parse FHIR data into prediction input
    let prediction_input = state.fhir_parser.bundle_to_prediction_input(&bundle)?;

    // 7. Run prediction
    let prediction = {
        let mut model = state.model.lock().await;
        make_prediction(
            &mut model,
            &state.model_metadata,
            &prediction_input,
            state.calibrator.as_deref(),
        )?
    };

    // 8. Generate FHIR RiskAssessment
    let patient_reference = format!("Patient/{}", patient_id);
    let encounter_reference = token.encounter.as_ref().map(|e| format!("Encounter/{}", e));

    let risk_assessment = state.risk_assessment_builder.build(
        &prediction,
        &patient_reference,
        encounter_reference.as_deref(),
        &[], // basis references - could populate from bundle
    );

    // 9. Optionally write RiskAssessment back to EHR
    if state.enable_writeback {
        match write_risk_assessment(
            &state.http_client,
            &session.fhir_base_url,
            &risk_assessment,
            &token.access_token,
        )
        .await
        {
            Ok(location) => info!("RiskAssessment created: {}", location),
            Err(e) => warn!("Failed to write RiskAssessment: {}", e),
        }
    }

    // 10. Render results page
    let html = render_results_html(&prediction, &risk_assessment);
    Ok(Html(html))
}

// ============================================================================
// Standalone API Endpoints
// ============================================================================

/// Direct prediction from FHIR Bundle (no OAuth required)
/// Use for backend integrations, batch processing, research
async fn predict_from_bundle(
    State(state): State<AppState>,
    Json(bundle): Json<Bundle>,
) -> Result<Json<RiskAssessmentOutput>, AppError> {
    let prediction_input = state.fhir_parser.bundle_to_prediction_input(&bundle)?;

    let prediction = {
        let mut model = state.model.lock().await;
        make_prediction(
            &mut model,
            &state.model_metadata,
            &prediction_input,
            state.calibrator.as_deref(),
        )?
    };

    let risk_assessment = state.risk_assessment_builder.build(
        &prediction,
        &prediction_input.patient_id.unwrap_or_default(),
        None,
        &[],
    );

    Ok(Json(risk_assessment))
}

/// Batch prediction from multiple bundles
#[derive(Debug, Deserialize)]
struct BatchRequest {
    bundles: Vec<Bundle>,
}

#[derive(Debug, Serialize)]
struct BatchResponse {
    results: Vec<BatchResultItem>,
    summary: BatchSummary,
}

#[derive(Debug, Serialize)]
struct BatchResultItem {
    patient_id: Option<String>,
    risk_assessment: Option<RiskAssessmentOutput>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct BatchSummary {
    total: usize,
    successful: usize,
    failed: usize,
}

async fn batch_predict(
    State(state): State<AppState>,
    Json(request): Json<BatchRequest>,
) -> Result<Json<BatchResponse>, AppError> {
    let mut results = Vec::with_capacity(request.bundles.len());
    let mut successful = 0;
    let mut failed = 0;

    for bundle in request.bundles {
        match process_single_bundle(&state, &bundle).await {
            Ok(risk_assessment) => {
                successful += 1;
                results.push(BatchResultItem {
                    patient_id: extract_patient_id(&bundle),
                    risk_assessment: Some(risk_assessment),
                    error: None,
                });
            }
            Err(e) => {
                failed += 1;
                results.push(BatchResultItem {
                    patient_id: extract_patient_id(&bundle),
                    risk_assessment: None,
                    error: Some(e.to_string()),
                });
            }
        }
    }

    Ok(Json(BatchResponse {
        results,
        summary: BatchSummary {
            total: successful + failed,
            successful,
            failed,
        },
    }))
}

async fn process_single_bundle(
    state: &AppState,
    bundle: &Bundle,
) -> Result<RiskAssessmentOutput, AppError> {
    let prediction_input = state.fhir_parser.bundle_to_prediction_input(bundle)?;

    let prediction = {
        let mut model = state.model.lock().await;
        make_prediction(
            &mut model,
            &state.model_metadata,
            &prediction_input,
            state.calibrator.as_deref(),
        )?
    };

    Ok(state.risk_assessment_builder.build(
        &prediction,
        &prediction_input.patient_id.unwrap_or_default(),
        None,
        &[],
    ))
}

// ============================================================================
// Discovery & Health Endpoints
// ============================================================================

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "ahuvista-fhir-service",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// SMART configuration for apps that want to discover our capabilities
async fn smart_configuration() -> impl IntoResponse {
    Json(serde_json::json!({
        "authorization_endpoint": "/launch",
        "token_endpoint": "/callback",
        "capabilities": [
            "launch-ehr",
            "client-public",
            "client-confidential-symmetric",
            "context-ehr-patient",
            "sso-openid-connect"
        ],
        "scopes_supported": [
            "patient/Patient.read",
            "patient/Observation.read",
            "patient/Condition.read",
            "patient/DocumentReference.read",
            "launch"
        ]
    }))
}

// ============================================================================
// Helper Functions
// ============================================================================

async fn discover_smart_configuration(
    client: &Client,
    fhir_base: &str,
) -> Result<SmartConfiguration, AppError> {
    let url = format!(
        "{}/.well-known/smart-configuration",
        fhir_base.trim_end_matches('/')
    );

    let response = client
        .get(&url)
        .header("Accept", "application/json")
        .send()
        .await
        .map_err(|e| AppError::HttpError(e.to_string()))?;

    response
        .json()
        .await
        .map_err(|e| AppError::HttpError(e.to_string()))
}

async fn exchange_code_for_token(
    client: &Client,
    token_endpoint: &str,
    code: &str,
    oauth_config: &OAuthConfig,
) -> Result<TokenResponse, AppError> {
    let mut params = HashMap::new();
    params.insert("grant_type", "authorization_code");
    params.insert("code", code);
    params.insert("redirect_uri", &oauth_config.redirect_uri);
    params.insert("client_id", &oauth_config.client_id);

    let mut request = client.post(token_endpoint).form(&params);

    // Add client secret for confidential apps
    if let Some(secret) = &oauth_config.client_secret {
        request = request.basic_auth(&oauth_config.client_id, Some(secret));
    }

    let response = request
        .send()
        .await
        .map_err(|e| AppError::HttpError(e.to_string()))?;

    if !response.status().is_success() {
        let error_text = response.text().await.unwrap_or_default();
        return Err(AppError::HttpError(format!(
            "Token exchange failed: {}",
            error_text
        )));
    }

    response
        .json()
        .await
        .map_err(|e| AppError::HttpError(e.to_string()))
}

async fn fetch_patient_data(
    client: &Client,
    fhir_base: &str,
    patient_id: &str,
    access_token: &str,
) -> Result<Bundle, AppError> {
    // Fetch everything we need using $everything or individual queries
    // Using individual queries for more control:

    let base = fhir_base.trim_end_matches('/');

    // Build a search bundle with all relevant data
    let queries = vec![
        format!("{}/Patient/{}", base, patient_id),
        format!(
            "{}/Observation?patient={}&category=vital-signs&_count=100",
            base, patient_id
        ),
        format!(
            "{}/Observation?patient={}&category=laboratory&_count=100",
            base, patient_id
        ),
        format!(
            "{}/Condition?patient={}&clinical-status=active",
            base, patient_id
        ),
        format!(
            "{}/DocumentReference?patient={}&_count=10",
            base, patient_id
        ),
    ];

    let mut entries = Vec::new();

    for url in queries {
        let response = client
            .get(&url)
            .header("Authorization", format!("Bearer {}", access_token))
            .header("Accept", "application/fhir+json")
            .send()
            .await
            .map_err(|e| AppError::HttpError(e.to_string()))?;

        if response.status().is_success() {
            let bundle_or_resource: serde_json::Value = response
                .json()
                .await
                .map_err(|e| AppError::HttpError(e.to_string()))?;

            // Handle both Bundle responses and single resources
            if bundle_or_resource
                .get("resourceType")
                .and_then(|v| v.as_str())
                == Some("Bundle")
            {
                if let Some(bundle_entries) =
                    bundle_or_resource.get("entry").and_then(|e| e.as_array())
                {
                    for entry in bundle_entries {
                        entries.push(entry.clone());
                    }
                }
            } else {
                // Single resource
                entries.push(serde_json::json!({ "resource": bundle_or_resource }));
            }
        }
    }

    // Construct combined bundle
    let bundle_json = serde_json::json!({
        "resourceType": "Bundle",
        "type": "searchset",
        "entry": entries
    });

    serde_json::from_value(bundle_json).map_err(|e| AppError::FhirError(FhirError::JsonError(e)))
}

async fn write_risk_assessment(
    client: &Client,
    fhir_base: &str,
    risk_assessment: &RiskAssessmentOutput,
    access_token: &str,
) -> Result<String, AppError> {
    let url = format!("{}/RiskAssessment", fhir_base.trim_end_matches('/'));

    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", access_token))
        .header("Content-Type", "application/fhir+json")
        .json(risk_assessment)
        .send()
        .await
        .map_err(|e| AppError::HttpError(e.to_string()))?;

    if response.status().is_success() || response.status() == StatusCode::CREATED {
        let location = response
            .headers()
            .get("Location")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("unknown")
            .to_string();
        Ok(location)
    } else {
        let error = response.text().await.unwrap_or_default();
        Err(AppError::HttpError(format!(
            "Failed to create RiskAssessment: {}",
            error
        )))
    }
}

fn generate_state_token() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:x}", timestamp)
}

fn extract_patient_id(bundle: &Bundle) -> Option<String> {
    // Extract patient ID from bundle
    bundle
        .entry
        .as_ref()?
        .iter()
        .filter_map(|e| e.resource.as_ref())
        .find_map(|r| r.id.clone())
}
