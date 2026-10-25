use std::{path::Path, time::Duration};

use reqwest::{
    Client, Method, StatusCode,
    header::{AUTHORIZATION, CONTENT_TYPE, HeaderName, IF_NONE_MATCH},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::resources::{Bundle, BundleEntry};
use crate::error::{AppError, AppResult};

#[derive(Debug, Clone, Default)]
pub struct FhirClientConfig {
    pub token: Option<String>,
    pub retry_attempts: usize,
    pub retry_backoff_ms: u64,
    pub profile_url: Option<String>,
    pub idempotency_key: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SmartClientCredentials {
    pub token_url: String,
    pub client_id: String,
    pub client_secret: Option<String>,
    pub scope: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilitySummary {
    pub resource_types: Vec<String>,
    pub fhir_version: Option<String>,
}

#[derive(Clone)]
pub struct FhirClient {
    base_url: String,
    config: FhirClientConfig,
    client: Client,
}

impl FhirClient {
    pub fn new(base_url: impl Into<String>, token: Option<String>) -> Self {
        Self::with_config(
            base_url,
            FhirClientConfig {
                token,
                retry_attempts: 3,
                retry_backoff_ms: 250,
                ..Default::default()
            },
        )
    }

    pub fn with_config(base_url: impl Into<String>, config: FhirClientConfig) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            config,
            client: Client::new(),
        }
    }

    pub async fn from_smart_credentials(
        base_url: impl Into<String>,
        credentials: SmartClientCredentials,
        mut config: FhirClientConfig,
    ) -> AppResult<Self> {
        let client = Client::new();
        let mut form = vec![
            ("grant_type", "client_credentials".to_string()),
            ("client_id", credentials.client_id.clone()),
        ];
        if let Some(scope) = credentials.scope {
            form.push(("scope", scope));
        }

        let form_body = form
            .iter()
            .map(|(key, value)| format!("{}={}", urlencoding(key), urlencoding(value)))
            .collect::<Vec<_>>()
            .join("&");
        let mut request = client
            .post(credentials.token_url)
            .header(CONTENT_TYPE, "application/x-www-form-urlencoded")
            .body(form_body);
        if let Some(secret) = credentials.client_secret {
            request = request.basic_auth(credentials.client_id, Some(secret));
        }
        let response = request.send().await.map_err(network_error)?;
        let status = response.status();
        let body = response.text().await.map_err(network_error)?;
        if !status.is_success() {
            return Err(AppError::DomainError(format!(
                "SMART token endpoint returned {status}: {body}"
            )));
        }
        let token: TokenResponse = serde_json::from_str(&body).map_err(AppError::JsonError)?;
        config.token = Some(token.access_token);
        Ok(Self::with_config(base_url, config))
    }

    pub async fn capability_summary(&self) -> AppResult<CapabilitySummary> {
        let value = self
            .get_json(&format!("{}/metadata", self.base_url))
            .await?;
        let resources = value["rest"][0]["resource"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|item| item["type"].as_str().map(str::to_owned))
            .collect();
        Ok(CapabilitySummary {
            resource_types: resources,
            fhir_version: value["fhirVersion"].as_str().map(str::to_owned),
        })
    }

    pub async fn fetch_patient_bundle(&self, patient_id: &str) -> AppResult<Bundle> {
        self.fetch_resources_for_patient(
            patient_id,
            &[
                "Patient",
                "Observation",
                "Condition",
                "DocumentReference",
                "DiagnosticReport",
                "ImagingStudy",
            ],
        )
        .await
    }

    pub async fn fetch_cohort_bundle(&self, query: &[(String, String)]) -> AppResult<Bundle> {
        let mut entries = Vec::new();
        for resource in [
            "Patient",
            "Observation",
            "Condition",
            "DocumentReference",
            "DiagnosticReport",
            "ImagingStudy",
        ] {
            let mut parameters = query.to_vec();
            let url = format!(
                "{}/{}?{}",
                self.base_url,
                resource,
                encode_query(&parameters)
            );
            entries.extend(self.fetch_paginated_entries(&url).await?);
        }
        Ok(Bundle {
            resource_type: "Bundle".into(),
            r#type: "collection".into(),
            id: None,
            entry: Some(entries),
        })
    }

    async fn fetch_resources_for_patient(
        &self,
        patient_id: &str,
        resources: &[&str],
    ) -> AppResult<Bundle> {
        let mut entries = Vec::new();
        for resource in resources {
            let query_key = if *resource == "Patient" {
                "_id"
            } else {
                "patient"
            };
            let url = format!(
                "{}/{resource}?{}={}",
                self.base_url,
                query_key,
                urlencoding(patient_id)
            );
            entries.extend(self.fetch_paginated_entries(&url).await?);
        }
        Ok(Bundle {
            resource_type: "Bundle".into(),
            r#type: "collection".into(),
            id: None,
            entry: Some(entries),
        })
    }

    async fn fetch_paginated_entries(&self, initial_url: &str) -> AppResult<Vec<BundleEntry>> {
        let mut url = Some(initial_url.to_string());
        let mut entries = Vec::new();
        while let Some(next_url) = url.take() {
            let value = self.get_json(&next_url).await?;
            let bundle: Bundle =
                serde_json::from_value(value.clone()).map_err(AppError::JsonError)?;
            entries.extend(bundle.entry.unwrap_or_default());
            url = value["link"]
                .as_array()
                .and_then(|links| links.iter().find(|link| link["relation"] == "next"))
                .and_then(|link| link["url"].as_str())
                .map(str::to_owned);
        }
        Ok(entries)
    }

    pub async fn publish_resource(&self, resource: &Value) -> AppResult<Value> {
        self.send_json(Method::POST, &self.base_url, resource).await
    }

    pub async fn publish_bundle(&self, resources: Vec<Value>) -> AppResult<Value> {
        let bundle = serde_json::json!({
            "resourceType": "Bundle", "type": "transaction",
            "entry": resources.into_iter().map(|resource| serde_json::json!({
                "resource": resource,
                "request": { "method": "POST", "url": resource.get("resourceType").and_then(Value::as_str).unwrap_or("Resource") }
            })).collect::<Vec<_>>()
        });
        self.send_json(Method::POST, &self.base_url, &bundle).await
    }

    async fn get_json(&self, url: &str) -> AppResult<Value> {
        self.send_json(Method::GET, url, &Value::Null).await
    }

    async fn send_json(&self, method: Method, url: &str, body: &Value) -> AppResult<Value> {
        let attempts = self.config.retry_attempts.max(1);
        for attempt in 0..attempts {
            let mut request = self
                .client
                .request(method.clone(), url)
                .header(CONTENT_TYPE, "application/fhir+json");
            if let Some(token) = &self.config.token {
                request = request.header(AUTHORIZATION, format!("Bearer {token}"));
            }
            if let Some(key) = &self.config.idempotency_key {
                request = request.header(HeaderName::from_static("idempotency-key"), key);
            }
            if method == Method::GET {
                request = request.header(IF_NONE_MATCH, "*");
            }
            if self.config.profile_url.is_some() {
                request = request.header(
                    "Accept-Profile",
                    self.config.profile_url.as_deref().unwrap_or_default(),
                );
            }
            if method != Method::GET {
                request = request.json(body);
            }
            match request.send().await {
                Ok(response) => {
                    let status = response.status();
                    let text = response.text().await.map_err(network_error)?;
                    if status.is_success() {
                        return serde_json::from_str(&text).map_err(AppError::JsonError);
                    }
                    if !is_retryable(status) || attempt + 1 == attempts {
                        return Err(AppError::DomainError(format!(
                            "FHIR server returned {status}: {text}"
                        )));
                    }
                }
                Err(error) if attempt + 1 < attempts => {
                    tokio::time::sleep(Duration::from_millis(
                        self.config
                            .retry_backoff_ms
                            .saturating_mul((attempt + 1) as u64),
                    ))
                    .await;
                    let _ = error;
                }
                Err(error) => return Err(network_error(error)),
            }
        }
        Err(AppError::DomainError(
            "FHIR request exhausted retry policy".into(),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
}

fn is_retryable(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}
fn network_error(error: reqwest::Error) -> AppError {
    AppError::DomainError(format!("FHIR network error: {error}"))
}

pub fn read_bundle(path: impl AsRef<Path>) -> AppResult<Bundle> {
    let content = std::fs::read_to_string(path).map_err(AppError::Io)?;
    serde_json::from_str(&content).map_err(AppError::JsonError)
}

fn encode_query(query: &[(String, String)]) -> String {
    query
        .iter()
        .map(|(key, value)| format!("{}={}", urlencoding(key), urlencoding(value)))
        .collect::<Vec<_>>()
        .join("&")
}
fn urlencoding(value: &str) -> String {
    value
        .bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (byte as char).to_string()
            }
            _ => format!("%{byte:02X}"),
        })
        .collect()
}
