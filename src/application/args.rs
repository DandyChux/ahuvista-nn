use clap::Args;

#[derive(Debug, Args)]
pub struct PrepareArgs {
    /// Directory or file inputs. A directory is scanned recursively.
    #[arg(long, default_value = "datasets")]
    pub data_dir: String,

    /// Additional files or directories to ingest without requiring modality folders.
    #[arg(long = "input", short = 'i')]
    pub inputs: Vec<String>,

    /// Optional manifest describing ambiguous files in a flat input directory.
    #[arg(long)]
    pub manifest: Option<String>,

    /// Local FHIR server base URL. The response is snapshotted into the database.
    #[arg(long)]
    pub fhir_url: Option<String>,

    /// Read an exported FHIR Bundle from a JSON file.
    #[arg(long)]
    pub fhir_bundle: Option<String>,

    /// Patient id to use when querying a FHIR server.
    #[arg(long)]
    pub patient: Option<String>,

    /// Cohort search parameters, for example `--fhir-query gender=female --fhir-query _lastUpdated=ge2025-01-01`.
    #[arg(long = "fhir-query")]
    pub fhir_queries: Vec<String>,

    /// OAuth2 client credentials JSON file for SMART-style server authentication.
    #[arg(long)]
    pub smart_credentials: Option<String>,

    /// FHIR profile URL sent as Accept-Profile and recorded in provenance.
    #[arg(long)]
    pub fhir_profile: Option<String>,

    /// Number of network retries for transient FHIR failures.
    #[arg(long, default_value_t = 3)]
    pub fhir_retries: usize,

    /// Bearer token, preferably supplied through an environment variable name.
    #[arg(long)]
    pub fhir_token_env: Option<String>,

    #[arg(long, default_value = "ahuvista.db")]
    pub database: String,

    #[arg(long, default_value = "subject_id")]
    pub patient_id_column: String,

    #[arg(long)]
    pub clear: bool,

    #[arg(short, long)]
    pub verbose: bool,
}

#[derive(Debug, Args)]
pub struct TrainArgs {
    #[arg(short, long)]
    pub config: Option<String>,

    #[arg(long, default_value = "ahuvista.db")]
    pub database: String,

    #[arg(long, value_delimiter = ',')]
    pub modalities: Option<Vec<String>>,

    #[arg(short, long)]
    pub epochs: Option<usize>,

    #[arg(short, long)]
    pub batch_size: Option<usize>,

    #[arg(short, long)]
    pub learning_rate: Option<f32>,

    #[arg(short, long, default_value = "models")]
    pub output_dir: String,

    #[arg(long)]
    pub no_privacy: bool,

    #[arg(long)]
    pub epsilon: Option<f32>,

    #[arg(long)]
    pub clip_norm: Option<f32>,

    #[arg(long)]
    pub adaptive_lr: bool,

    #[arg(short, long)]
    pub verbose: bool,
}

#[derive(Debug, Args)]
pub struct PredictArgs {
    #[arg(short, long)]
    pub model: String,

    #[arg(short, long)]
    pub input: String,

    #[arg(short, long)]
    pub output: Option<String>,

    /// json (default), fhir-risk-assessment, or fhir-bundle.
    #[arg(long, default_value = "json")]
    pub output_format: String,

    #[arg(long)]
    pub database: Option<String>,

    /// FHIR server URL for an explicit publish operation.
    #[arg(long)]
    pub fhir_url: Option<String>,

    #[arg(long)]
    pub fhir_token_env: Option<String>,

    /// Required before sending a prediction to a FHIR server.
    #[arg(long)]
    pub publish: bool,

    #[arg(short, long)]
    pub verbose: bool,
}

#[cfg(feature = "serve")]
#[derive(Debug, Args)]
pub struct ServeArgs {
    #[arg(long)]
    pub model: String,

    #[arg(long)]
    pub database: Option<String>,

    #[arg(long, default_value = "127.0.0.1:8080")]
    pub bind: String,
}
