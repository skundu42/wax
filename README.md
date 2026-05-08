# wax

`wax` is a small Rust-native LLM inference engine built on
[Candle](https://github.com/huggingface/candle).

It is intentionally narrow: load a local model, run a decoder-only Llama-like
causal LM, stream tokens, measure performance, and keep the implementation easy
to read.

## Features

- Local inference from the command line.
- Safetensors model folders with `config.json` and `tokenizer.json`.
- Direct `.gguf` model files through Candle's quantized Llama backend.
- Token streaming to stdout.
- Greedy, temperature, top-k, top-p, and repetition-penalty sampling.
- EOS and max-token stopping.
- Basic timing and throughput stats.
- JSON benchmark output.
- CPU, Metal, CUDA, and Accelerate feature flags.
- MLX model folder detection with a clear conversion error.

## Status

This project is early and intentionally limited.

| Area | Status |
| --- | --- |
| Safetensors Llama-like causal LM | Supported |
| GGUF Llama-family models | Supported |
| MLX model folders | Detected, not directly executable |
| OpenAI-compatible HTTP server | Not implemented |
| GGUF conversion | Not implemented |
| Quantization beyond GGUF backend | Not implemented |
| Batching / PagedAttention | Not implemented |
| Multimodal models | Not implemented |

MLX note: Candle does not directly execute MLX weight folders. Convert MLX
models to Hugging Face safetensors or GGUF before using them with `wax`.

## Requirements

- Rust 1.94 or newer.
- A local model in one of the supported formats.
- macOS with Apple Silicon for the `metal` feature, or a CUDA environment for
  the `cuda` feature.

## Build

CPU build:

```bash
cargo build -p wax-llm --release
```

Metal build on macOS:

```bash
cargo build -p wax-llm --release --features metal
```

Install the `wax` binary from this checkout:

```bash
cargo install --path crates/wax-llm --features metal
```

The package name is `wax-llm`; the installed binary is `wax`.

## Quickstart

Download a small safetensors model:

```bash
mkdir -p models/TinyLlama-1.1B-Chat-v1.0

hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  config.json \
  tokenizer.json \
  tokenizer_config.json \
  generation_config.json \
  model.safetensors \
  --local-dir models/TinyLlama-1.1B-Chat-v1.0
```

Run generation with Metal:

```bash
cargo run -p wax-llm --features metal -- run \
  --model ./models/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "Explain Rust ownership simply" \
  --max-new-tokens 128 \
  --temperature 0.7 \
  --top-p 0.9 \
  --stream
```

After `cargo install`, the same command is:

```bash
wax run \
  --model ./models/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "Explain Rust ownership simply" \
  --max-new-tokens 128 \
  --temperature 0.7 \
  --top-p 0.9 \
  --stream
```

## GGUF

Download a small GGUF model:

```bash
mkdir -p models/gguf-smollm2-360m

hf download HuggingFaceTB/SmolLM2-360M-Instruct-GGUF \
  smollm2-360m-instruct-q8_0.gguf \
  --local-dir models/gguf-smollm2-360m
```

Run it directly:

```bash
cargo run -p wax-llm --features metal -- run \
  --model ./models/gguf-smollm2-360m/smollm2-360m-instruct-q8_0.gguf \
  --prompt "Say hello" \
  --max-new-tokens 64 \
  --temperature 0 \
  --stream
```

For GGUF, `wax` uses `tokenizer.json` next to the model if present. If not, it
tries to build a tokenizer from GGUF metadata.

## CLI

Run text generation:

```bash
wax run \
  --model ./models/my-model \
  --prompt "Hello" \
  --max-new-tokens 64 \
  --temperature 0.7 \
  --top-k 40 \
  --top-p 0.9 \
  --repetition-penalty 1.1 \
  --seed 42 \
  --device auto \
  --dtype auto \
  --stream
```

Benchmark a prompt:

```bash
wax bench \
  --model ./models/my-model \
  --prompt-file prompts/short.txt \
  --runs 5 \
  --max-new-tokens 128 \
  --json
```

Device options:

```text
auto | cpu | cuda | metal
```

DType options:

```text
auto | f32 | f16 | bf16
```

For GGUF models, stats report `dtype: "gguf"` because the model's quantized
weight format is determined by the GGUF file.

## Model Layouts

Safetensors folder:

```text
model/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── generation_config.json
├── model.safetensors
└── model.safetensors.index.json
```

Only one of `model.safetensors` or `model.safetensors.index.json` is required.

GGUF:

```text
model.gguf
```

or:

```text
model/
├── model.gguf
└── tokenizer.json
```

If a folder contains multiple `.gguf` files, rename the intended file to
`model.gguf` or pass the exact `.gguf` path.

## Architecture

```text
wax
├── wax-core   # loading, tokenization, generation, sampling, stats
├── wax-llm    # CLI package, installs the `wax` binary
└── wax-bench  # shared benchmark types/helpers
```

The core crate is intentionally independent of HTTP/server dependencies.

## Development

Run the default test suite:

```bash
cargo test --workspace --no-default-features
```

Run the Metal feature build:

```bash
cargo test --workspace --features metal
```

Run formatting and lint checks:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --no-default-features -- -D warnings
```

Current tests cover loader format detection, safetensors index handling, MLX
detection, CLI argument behavior, sampling, stats serialization, device/dtype
selection, and token streaming.

## Releasing

Crates are published to crates.io by the `Publish crates` GitHub Actions
workflow when a GitHub Release is published from `main`.

Release requirements:

- Set the repository secret `CARGO_REGISTRY_TOKEN` to a crates.io API token.
- Bump the workspace version in `Cargo.toml`.
- Create a GitHub Release whose tag matches the version, for example `v0.1.0`.
- Create the release from `main`.

The workflow publishes crates in dependency order:

1. `wax-core`
2. `wax-bench`
3. `wax-llm`

The package name is `wax-llm`, and the installed binary is `wax`.

## Contributing

Small, focused changes are preferred. Please keep the core inference path simple
and measurable.

Before opening a PR, run:

```bash
cargo fmt --check
cargo test --workspace --no-default-features
cargo clippy --workspace --all-targets --no-default-features -- -D warnings
```

If your change touches GPU-specific behavior, also run the relevant feature
build, for example:

```bash
cargo test --workspace --features metal
```

## License

Checkout the full license [here](LICENSE.md).
