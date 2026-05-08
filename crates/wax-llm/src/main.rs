use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};
use wax_core::{
    stats::{BenchStats, CANDLE_VERSION},
    DTypeChoice, DeviceChoice, Engine, EngineConfig, GenerateRequest, Result as WaxResult,
    SamplingConfig,
};

#[derive(Debug, Parser)]
#[command(
    name = "wax",
    version,
    about = "Small Candle-based local LLM inference CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Run(RunArgs),
    Bench(BenchArgs),
}

#[derive(Debug, Parser)]
struct RunArgs {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    prompt: String,

    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    #[arg(long)]
    top_k: Option<usize>,

    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long, default_value_t = 1.0)]
    repetition_penalty: f32,

    #[arg(long, default_value_t = 128)]
    repeat_last_n: usize,

    #[arg(long, default_value_t = 299_792_458)]
    seed: u64,

    #[arg(long, default_value_t = true)]
    stream: bool,

    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    #[arg(long, value_enum, default_value_t = DTypeArg::Auto)]
    dtype: DTypeArg,
}

#[derive(Debug, Parser)]
struct BenchArgs {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    prompt_file: PathBuf,

    #[arg(long, default_value_t = 5)]
    runs: usize,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    #[arg(long)]
    top_k: Option<usize>,

    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long, default_value_t = 1.0)]
    repetition_penalty: f32,

    #[arg(long, default_value_t = 128)]
    repeat_last_n: usize,

    #[arg(long, default_value_t = 299_792_458)]
    seed: u64,

    #[arg(long)]
    json: bool,

    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    #[arg(long, value_enum, default_value_t = DTypeArg::Auto)]
    dtype: DTypeArg,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceArg {
    Auto,
    Cpu,
    Cuda,
    Metal,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DTypeArg {
    Auto,
    F32,
    F16,
    Bf16,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => run(args),
        Commands::Bench(args) => bench(args),
    }
}

fn run(args: RunArgs) -> anyhow::Result<()> {
    let mut engine = Engine::load(engine_config(&args.model, args.device, args.dtype))
        .with_context(|| format!("failed to load model from {}", args.model.display()))?;
    let request = GenerateRequest {
        prompt: args.prompt,
        max_new_tokens: args.max_new_tokens,
        sampling: sampling_config(
            args.temperature,
            args.top_k,
            args.top_p,
            args.repetition_penalty,
            args.repeat_last_n,
            args.seed,
        ),
        stream: args.stream,
    };

    let stdout = io::stdout();
    let mut handle = stdout.lock();
    let stats = engine.generate(request, |delta: &str| -> WaxResult<()> {
        write!(handle, "{delta}")?;
        handle.flush()?;
        Ok(())
    })?;
    writeln!(handle)?;
    eprintln!("{}", serde_json::to_string_pretty(&stats)?);
    Ok(())
}

fn bench(args: BenchArgs) -> anyhow::Result<()> {
    if args.runs == 0 {
        anyhow::bail!("--runs must be > 0");
    }

    let prompt = fs::read_to_string(&args.prompt_file)
        .with_context(|| format!("failed to read {}", args.prompt_file.display()))?;

    let load_start = Instant::now();
    let mut engine = Engine::load(engine_config(&args.model, args.device, args.dtype))
        .with_context(|| format!("failed to load model from {}", args.model.display()))?;
    let model_load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let mut results = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let stats = engine.generate(
            GenerateRequest {
                prompt: prompt.clone(),
                max_new_tokens: args.max_new_tokens,
                sampling: sampling_config(
                    args.temperature,
                    args.top_k,
                    args.top_p,
                    args.repetition_penalty,
                    args.repeat_last_n,
                    args.seed,
                ),
                stream: false,
            },
            noop_stream,
        )?;
        results.push(stats);
    }

    let first = results
        .first()
        .expect("runs is checked to be greater than zero");
    let avg_prefill_ms = average(results.iter().map(|stats| stats.prefill_ms));
    let avg_total_ms = average(results.iter().map(|stats| stats.total_ms));
    let avg_decode_tok_s = average_option(results.iter().filter_map(|stats| stats.decode_tok_s));
    let avg_ttft_ms = average_option(results.iter().filter_map(|stats| stats.ttft_ms));

    let stats = BenchStats {
        model: first.model.clone(),
        device: first.device.clone(),
        dtype: first.dtype.clone(),
        prompt_tokens: first.prompt_tokens,
        generated_tokens: first.generated_tokens,
        model_load_ms,
        prefill_ms: avg_prefill_ms,
        prefill_tok_s: if avg_prefill_ms > 0.0 {
            Some(first.prompt_tokens as f64 / (avg_prefill_ms / 1000.0))
        } else {
            None
        },
        ttft_ms: avg_ttft_ms,
        decode_tok_s: avg_decode_tok_s,
        total_generation_ms: avg_total_ms,
        peak_memory_mb: current_process_memory_mb(),
        candle_version: CANDLE_VERSION,
        rust_version: rust_version(),
        git_commit: git_commit(),
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
    } else {
        print_bench_summary(&stats);
    }

    Ok(())
}

fn engine_config(model: &Path, device: DeviceArg, dtype: DTypeArg) -> EngineConfig {
    EngineConfig {
        model_dir: model.to_path_buf(),
        device: device.into(),
        dtype: dtype.into(),
    }
}

fn sampling_config(
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
    repetition_penalty: f32,
    repeat_last_n: usize,
    seed: u64,
) -> SamplingConfig {
    SamplingConfig {
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        repeat_last_n,
        seed,
    }
}

fn average(values: impl Iterator<Item = f64>) -> f64 {
    let mut count = 0usize;
    let mut sum = 0.0;
    for value in values {
        count += 1;
        sum += value;
    }
    sum / count as f64
}

fn average_option(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut count = 0usize;
    let mut sum = 0.0;
    for value in values {
        count += 1;
        sum += value;
    }
    (count > 0).then_some(sum / count as f64)
}

fn print_bench_summary(stats: &BenchStats) {
    println!("model: {}", stats.model);
    println!("device: {}", stats.device);
    println!("dtype: {}", stats.dtype);
    println!("prompt tokens: {}", stats.prompt_tokens);
    println!("generated tokens: {}", stats.generated_tokens);
    println!("model load ms: {:.2}", stats.model_load_ms);
    println!("prefill ms: {:.2}", stats.prefill_ms);
    if let Some(value) = stats.prefill_tok_s {
        println!("prefill tok/s: {value:.2}");
    }
    if let Some(value) = stats.ttft_ms {
        println!("ttft ms: {value:.2}");
    }
    if let Some(value) = stats.decode_tok_s {
        println!("decode tok/s: {value:.2}");
    }
    println!("total generation ms: {:.2}", stats.total_generation_ms);
}

fn current_process_memory_mb() -> Option<u64> {
    let mut system = sysinfo::System::new_all();
    system.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
    let pid = sysinfo::get_current_pid().ok()?;
    system
        .process(pid)
        .map(|process| process.memory() / 1024 / 1024)
}

fn noop_stream(_: &str) -> WaxResult<()> {
    Ok(())
}

fn rust_version() -> String {
    command_stdout("rustc", &["--version"]).unwrap_or_else(|| "unknown".to_string())
}

fn git_commit() -> Option<String> {
    command_stdout("git", &["rev-parse", "--short", "HEAD"])
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

impl From<DeviceArg> for DeviceChoice {
    fn from(value: DeviceArg) -> Self {
        match value {
            DeviceArg::Auto => Self::Auto,
            DeviceArg::Cpu => Self::Cpu,
            DeviceArg::Cuda => Self::Cuda,
            DeviceArg::Metal => Self::Metal,
        }
    }
}

impl From<DTypeArg> for DTypeChoice {
    fn from(value: DTypeArg) -> Self {
        match value {
            DTypeArg::Auto => Self::Auto,
            DTypeArg::F32 => Self::F32,
            DTypeArg::F16 => Self::F16,
            DTypeArg::Bf16 => Self::BF16,
        }
    }
}
