use std::net::SocketAddr;

use tokio::net::TcpListener;
use tracing::info;

use crate::{
    error::{AppError, AppResult},
    model_artifact::ModelArtifact,
};

use super::{
    args::ServeArgs,
    routes::{AppState, router},
};

pub fn run(args: ServeArgs) -> AppResult<()> {
    let runtime = tokio::runtime::Runtime::new().map_err(|error| {
        crate::error::AppError::InvalidData(format!("Failed to create Tokio runtime: {}", error))
    })?;
    runtime.block_on(run_async(args))
}

async fn run_async(args: ServeArgs) -> AppResult<()> {
    let artifact = ModelArtifact::load(&args.model)?;

    let model = artifact.instantiate()?;

    let state = AppState {
        model: std::sync::Arc::new(tokio::sync::Mutex::new(model)),
        artifact: std::sync::Arc::new(artifact),
    };

    let router = router(state);

    let address: SocketAddr = args.bind.parse().map_err(|error| {
        AppError::Config(format!("Invalid bind address '{}': {}", args.bind, error))
    })?;

    let listener = TcpListener::bind(address).await?;

    info!("AhuVista service listening on {}", address);

    axum::serve(listener, router)
        .await
        .map_err(|error| AppError::Io(std::io::Error::other(error)))
}
