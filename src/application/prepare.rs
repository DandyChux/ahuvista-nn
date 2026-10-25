use std::path::{Path, PathBuf};

use tracing::info;

use crate::{
    data::{
        db::DatabaseManager,
        fhir::{
            FhirMappingConfig,
            client::{FhirClient, FhirClientConfig, SmartClientCredentials, read_bundle},
        },
        ingestion::IngestionService,
        source::{
            classify_inputs, discover_associated_images, load_manifest, manifest_image_roots,
        },
    },
    error::AppResult,
};

use super::args::PrepareArgs;

pub fn run(args: PrepareArgs) -> AppResult<()> {
    let data_dir = Path::new(&args.data_dir);
    let mut inputs = args.inputs.iter().map(PathBuf::from).collect::<Vec<_>>();
    if args.inputs.is_empty() && args.fhir_bundle.is_none() && args.fhir_url.is_none() {
        if !data_dir.exists() {
            return Err(crate::error::AppError::InvalidData(format!(
                "Data directory does not exist: {}",
                data_dir.display()
            )));
        }

        inputs.push(data_dir.to_path_buf());
    }

    let database = DatabaseManager::new(&args.database)?;

    let mut ingestion = IngestionService::new(database);

    if args.clear {
        info!("Clearing prepared data");
        ingestion.database_mut().clear_all_data()?;
    }

    let mapping_path = Path::new("fhir_mapping.json");
    let mapping: FhirMappingConfig = serde_json::from_str(
        &std::fs::read_to_string(mapping_path).map_err(crate::error::AppError::Io)?,
    )?;

    let imported = if let Some(bundle_path) = args.fhir_bundle {
        let bundle = read_bundle(&bundle_path)?;
        let imported = ingestion.import_fhir_bundle(&bundle, &mapping)?;

        ingestion
            .database()
            .record_ingestion_run("fhir-bundle", &bundle_path, imported)?;

        imported
    } else if let Some(url) = args.fhir_url {
        let token = args
            .fhir_token_env
            .as_deref()
            .and_then(std::env::var_os)
            .and_then(|value| value.into_string().ok());

        let config = FhirClientConfig {
            token,
            retry_attempts: args.fhir_retries,
            profile_url: args.fhir_profile.clone(),
            idempotency_key: Some(format!(
                "ahuvista-{}",
                chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
            )),
            ..Default::default()
        };

        let runtime = tokio::runtime::Runtime::new()
            .map_err(|error| crate::error::AppError::DomainError(error.to_string()))?;

        let client = if let Some(credentials_path) = args.smart_credentials {
            let credentials: SmartClientCredentials = serde_json::from_str(
                &std::fs::read_to_string(credentials_path).map_err(crate::error::AppError::Io)?,
            )?;
            runtime.block_on(FhirClient::from_smart_credentials(
                &url,
                credentials,
                config,
            ))?
        } else {
            FhirClient::with_config(&url, config)
        };

        let bundle = if let Some(patient) = args.patient {
            runtime.block_on(client.fetch_patient_bundle(&patient))?
        } else {
            let query = args
                .fhir_queries
                .iter()
                .filter_map(|item| {
                    item.split_once('=')
                        .map(|(key, value)| (key.to_string(), value.to_string()))
                })
                .collect::<Vec<_>>();

            runtime.block_on(client.fetch_cohort_bundle(&query))?
        };

        let imported = ingestion.import_fhir_bundle(&bundle, &mapping)?;

        ingestion
            .database()
            .record_ingestion_run("fhir-server", &url, imported)?;

        imported
    } else {
        let manifest = args
            .manifest
            .as_deref()
            .map(|path| load_manifest(Path::new(path)))
            .transpose()?;

        let classified = classify_inputs(&inputs, manifest.as_ref())?;

        if let Some(manifest) = manifest.as_ref() {
        	let manifest_roots = manifest_image_roots(manifest, data_dir);

            for (_group_name, image_root, image_config) in manifest_roots
            {
                let records = discover_associated_images(&image_root, &image_config)?;

                if !records.is_empty() {
                    let image_data = ingestion.database().build_image_data_for_records(
                        &records,
                        (224, 224),
                        false,
                    )?;

                    ingestion.database_mut().store_image_data(&image_data)?;
                }
            }
        }

        let imported = ingestion.prepare_classified_inputs(&classified, &args.patient_id_column)?;

        ingestion.database().record_ingestion_run(
            "file",
            &inputs
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(","),
            imported,
        )?;

        imported
    };

    let counts = ingestion.database().get_sample_counts()?;

    info!("Preparation complete: imported {} raw records", imported);

    info!(
        "Prepared samples: sequences={}, text={}, images={}, outcomes={}",
        counts.sequences, counts.text, counts.image, counts.outcomes
    );

    if counts.sequences == 0 {
        return Err(crate::error::AppError::InvalidData(
            "Preparation produced no patient sequences".to_string(),
        ));
    }

    if counts.outcomes == 0 {
        return Err(crate::error::AppError::InvalidData(
            "Preparation produced no outcomes".to_string(),
        ));
    }

    Ok(())
}
