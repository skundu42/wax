use std::{
    fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;
use wax_core::{
    stats::{BenchStats, CANDLE_VERSION},
    ChatMessage, ChatTemplate, DTypeChoice, DeviceChoice, Engine, EngineConfig, GenerateOutput,
    GenerateRequest, Result as WaxResult, SamplingConfig,
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
    Chat(ChatArgs),
    Bench(BenchArgs),
}

#[derive(Debug, Parser)]
struct RunArgs {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    prompt_file: Option<PathBuf>,

    #[arg(long)]
    stdin: bool,

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

    #[arg(long)]
    json: bool,

    #[arg(long)]
    output_file: Option<PathBuf>,

    #[arg(long)]
    stop: Vec<String>,

    #[arg(long = "eos-token-id")]
    eos_token_ids: Vec<u32>,

    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    #[arg(long, value_enum, default_value_t = DTypeArg::Auto)]
    dtype: DTypeArg,
}

#[derive(Debug, Parser)]
struct ChatArgs {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    system: Option<String>,

    #[arg(long = "message")]
    messages: Vec<String>,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = 0.7)]
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

    #[arg(long)]
    json: bool,

    #[arg(long)]
    output_file: Option<PathBuf>,

    #[arg(long)]
    stop: Vec<String>,

    #[arg(long = "eos-token-id")]
    eos_token_ids: Vec<u32>,

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
        Commands::Chat(args) => chat(args),
        Commands::Bench(args) => bench(args),
    }
}

fn run(args: RunArgs) -> anyhow::Result<()> {
    let prompt = prompt_from_args(args.prompt, args.prompt_file, args.stdin)?;
    let mut engine = Engine::load(engine_config(&args.model, args.device, args.dtype))
        .with_context(|| format!("failed to load model from {}", args.model.display()))?;
    let request = GenerateRequest {
        prompt,
        max_new_tokens: args.max_new_tokens,
        sampling: sampling_config(
            args.temperature,
            args.top_k,
            args.top_p,
            args.repetition_penalty,
            args.repeat_last_n,
            args.seed,
        ),
        stream: args.stream && !args.json,
        stop: args.stop,
        eos_token_ids: args.eos_token_ids,
        add_special_tokens: true,
    };

    let printed_stream = request.stream;
    let output = generate_to_stdout(&mut engine, request, args.json)?;
    write_output_file(args.output_file, &output.text)?;
    print_run_output(&output, args.json, printed_stream)?;
    Ok(())
}

fn chat(args: ChatArgs) -> anyhow::Result<()> {
    if args.messages.is_empty() {
        anyhow::bail!("at least one --message is required");
    }

    let mut messages = Vec::with_capacity(args.messages.len() + usize::from(args.system.is_some()));
    if let Some(system) = args.system {
        messages.push(ChatMessage::new("system", system));
    }
    messages.extend(args.messages.into_iter().map(parse_chat_message));

    let template = ChatTemplate::load_for_model_path(&args.model)
        .with_context(|| format!("failed to load chat template for {}", args.model.display()))?;
    let prompt = template.render(&messages, true)?;

    let mut engine = Engine::load(engine_config(&args.model, args.device, args.dtype))
        .with_context(|| format!("failed to load model from {}", args.model.display()))?;
    let request = GenerateRequest {
        prompt,
        max_new_tokens: args.max_new_tokens,
        sampling: sampling_config(
            args.temperature,
            args.top_k,
            args.top_p,
            args.repetition_penalty,
            args.repeat_last_n,
            args.seed,
        ),
        stream: args.stream && !args.json,
        stop: args.stop,
        eos_token_ids: args.eos_token_ids,
        add_special_tokens: false,
    };

    let printed_stream = request.stream;
    let output = generate_to_stdout(&mut engine, request, args.json)?;
    write_output_file(args.output_file, &output.text)?;
    print_run_output(&output, args.json, printed_stream)?;
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
        let output = engine.generate(
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
                stop: Vec::new(),
                eos_token_ids: Vec::new(),
                add_special_tokens: true,
            },
            noop_stream,
        )?;
        results.push(output.stats);
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

fn prompt_from_args(
    prompt: Option<String>,
    prompt_file: Option<PathBuf>,
    read_stdin: bool,
) -> anyhow::Result<String> {
    let selected = usize::from(prompt.is_some())
        + usize::from(prompt_file.is_some())
        + usize::from(read_stdin);
    if selected > 1 {
        anyhow::bail!("provide only one of --prompt, --prompt-file, or --stdin");
    }

    if let Some(prompt) = prompt {
        return Ok(prompt);
    }
    if let Some(path) = prompt_file {
        return fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()));
    }
    if read_stdin {
        let mut prompt = String::new();
        io::stdin().read_to_string(&mut prompt)?;
        return Ok(prompt);
    }

    anyhow::bail!("provide --prompt, --prompt-file, or --stdin")
}

fn parse_chat_message(message: String) -> ChatMessage {
    if let Some((role, content)) = message.split_once(':') {
        if matches!(role, "system" | "user" | "assistant" | "tool") {
            return ChatMessage::new(role, content);
        }
    }
    ChatMessage::new("user", message)
}

fn generate_to_stdout(
    engine: &mut Engine,
    request: GenerateRequest,
    suppress_stream: bool,
) -> anyhow::Result<GenerateOutput> {
    let should_stream = request.stream && !suppress_stream;
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    let output = engine.generate(request, |delta: &str| -> WaxResult<()> {
        if should_stream {
            write!(handle, "{delta}")?;
            handle.flush()?;
        }
        Ok(())
    })?;
    if should_stream {
        writeln!(handle)?;
    }
    Ok(output)
}

fn write_output_file(path: Option<PathBuf>, text: &str) -> anyhow::Result<()> {
    if let Some(path) = path {
        fs::write(&path, text).with_context(|| format!("failed to write {}", path.display()))?;
    }
    Ok(())
}

fn print_run_output(
    output: &GenerateOutput,
    json: bool,
    already_streamed: bool,
) -> anyhow::Result<()> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&RunOutputJson::from(output))?
        );
    } else if !already_streamed {
        println!("{}", output.text);
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct RunOutputJson<'a> {
    text: &'a str,
    stats: &'a wax_core::stats::GenerateStats,
}

impl<'a> From<&'a GenerateOutput> for RunOutputJson<'a> {
    fn from(output: &'a GenerateOutput) -> Self {
        Self {
            text: &output.text,
            stats: &output.stats,
        }
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
