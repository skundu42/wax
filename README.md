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
- EOS, max-token, explicit EOS token, and stop-string stopping.
- Prompt input from CLI arguments, prompt files, or stdin.
- Chat-template rendering from `tokenizer_config.json`.
- Basic timing and throughput stats.
- JSON benchmark output.
- Browser chat UI built with Next.js.
- CPU, Metal, CUDA, and Accelerate feature flags.
- MLX model folder detection with a clear conversion error.

## Status

This project is early and intentionally limited.

| Area | Status |
| --- | --- |
| Safetensors Llama-like causal LM | Supported |
| GGUF Llama-family models | Supported |
| MLX model folders | Detected, not directly executable |
| Browser chat UI | Supported through `apps/wax-chat` |
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

## Install And Run

Run directly from this workspace using the `wax-llm` crate:

```bash
cargo run -p wax-llm -- run \
  --model ./models/my-model \
  --prompt "Hello" \
  --max-new-tokens 64
```

Enable an accelerator feature when running from source:

```bash
cargo run -p wax-llm --features metal -- run \
  --model ./models/my-model \
  --prompt "Hello" \
  --max-new-tokens 64
```

Install the published CLI crate and run the installed `wax` binary:

```bash
cargo install wax-llm --features metal

wax run \
  --model ./models/my-model \
  --prompt "Hello" \
  --max-new-tokens 64
```

Install from this checkout while developing:

```bash
cargo install --path crates/wax-llm --features metal
```

The published package name is `wax-llm`; the installed binary is `wax`.
See [docs/build.md](docs/build.md) for detailed build and feature-flag
instructions.

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
  --stop "</s>" \
  --eos-token-id 2 \
  --seed 42 \
  --device auto \
  --dtype auto \
  --stream \
  --output-file output.txt
```

Use exactly one of `--prompt`, `--prompt-file`, or `--stdin`. Add `--json`
to emit a JSON object containing generated `text` and generation `stats`.

Prompt files and stdin are supported:

```bash
wax run --model ./models/my-model --prompt-file prompts/short.txt
cat prompts/short.txt | wax run --model ./models/my-model --stdin
```

Run chat generation using the model's Hugging Face chat template:

```bash
wax chat \
  --model ./models/my-chat-model \
  --system "You are concise." \
  --message "Hello" \
  --max-new-tokens 128 \
  --temperature 0.7 \
  --top-p 0.9 \
  --stop "</s>" \
  --stream
```

Repeated `--message` values are treated as user turns by default. To pass a
specific role, prefix the value with `system:`, `user:`, `assistant:`, or
`tool:`.

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
├── wax-bench  # shared benchmark types/helpers
└── apps/wax-chat  # local Next.js browser chat UI
```

The core crate is intentionally independent of HTTP/server dependencies.
See [docs/architecture.md](docs/architecture.md) for the detailed crate layout
and generation flow.

## Browser Chat UI

Install and run the local web UI:

```bash
cd apps/wax-chat
npm install
npm run dev
```

Open the printed localhost URL, enter a model path, and send a message. By
default the API route shells out to:

```bash
cargo run -q -p wax-llm -- chat ...
```

For faster repeated use, build or install `wax` and point the UI at it:

```bash
WAX_BIN=/path/to/wax npm run dev
```

If the app is launched from somewhere other than `apps/wax-chat`, set
`WAX_WORKSPACE_ROOT` to this repository root.

## Contributing

Small, focused changes are preferred. Please keep the core inference path simple
and measurable.

See [docs/development.md](docs/development.md) for local validation and
[docs/releasing.md](docs/releasing.md) for the crate release workflow.

## License

Checkout the full license [here](LICENSE.md).
