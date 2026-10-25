use ahuvista_nn::{application::args::PrepareArgs, error::AppResult};
use clap::{Parser, Subcommand};
use tracing_subscriber::{EnvFilter, fmt};

#[cfg(feature = "predict")]
use ahuvista_nn::application::args::PredictArgs;

#[cfg(feature = "train")]
use ahuvista_nn::application::args::TrainArgs;

#[cfg(feature = "serve")]
use ahuvista_nn::application::args::ServeArgs;

#[derive(Debug, Parser)]
#[command(
    name = "ahuvista",
    version,
    about = "AhuVista maternal-risk prediction system"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    #[cfg(feature = "prepare")]
    Prepare(PrepareArgs),

    #[cfg(feature = "train")]
    Train(TrainArgs),

    #[cfg(feature = "predict")]
    Predict(PredictArgs),

    #[cfg(feature = "serve")]
    Serve(ServeArgs),
}

fn main() -> AppResult<()> {
    let cli = Cli::parse();

    let verbose = match &cli.command {
        #[cfg(feature = "prepare")]
        Command::Prepare(args) => args.verbose,

        #[cfg(feature = "train")]
        Command::Train(args) => args.verbose,

        #[cfg(feature = "predict")]
        Command::Predict(args) => args.verbose,

        #[cfg(feature = "serve")]
        Command::Serve(_) => false,
    };

    let default_filter = if verbose { "debug" } else { "info" };

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter)),
        )
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .compact()
        .init();

    tracing::info!("Starting AhuVista");

    match cli.command {
        #[cfg(feature = "prepare")]
        Command::Prepare(args) => ahuvista_nn::application::prepare::run(args),

        #[cfg(feature = "train")]
        Command::Train(args) => ahuvista_nn::application::train::run(args),

        #[cfg(feature = "predict")]
        Command::Predict(args) => ahuvista_nn::application::predict::run(args),

        #[cfg(feature = "serve")]
        Command::Serve(args) => ahuvista_nn::application::serve::run(args),
    }
}
