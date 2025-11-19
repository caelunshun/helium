use clap::{Parser, Subcommand};
use mimalloc::MiMalloc;

mod model;
mod train;
mod training_data;

#[global_allocator]
static ALLOCATOR: MiMalloc = MiMalloc;

#[derive(Debug, Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    DownloadData,
    Compact,
    Train(train::Cli),
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    match cli.command {
        Command::DownloadData => training_data::download::download_data()?,
        Command::Compact => training_data::download::compact()?,
        Command::Train(cli) => train::train(cli)?,
    }
    Ok(())
}
