# Build

`wax` is a Rust workspace with a CLI binary exposed by the `wax-llm` crate.
The package name is `wax-llm`; the installed binary is `wax`.

## Requirements

- Rust 1.94 or newer.
- A local model in one of the supported formats.
- macOS with Apple Silicon for the `metal` feature, or a CUDA environment for
  the `cuda` feature.

## Run From Source

Run the CLI package directly from the workspace:

```bash
cargo run -p wax-llm -- run \
  --model ./models/my-model \
  --prompt "Hello" \
  --max-new-tokens 64
```

Run with Metal on macOS:

```bash
cargo run -p wax-llm --features metal -- run \
  --model ./models/my-model \
  --prompt "Hello" \
  --max-new-tokens 64
```

Run chat generation from source:

```bash
cargo run -p wax-llm --features metal -- chat \
  --model ./models/my-chat-model \
  --message "Hello" \
  --max-new-tokens 128
```

## Build From Source

CPU build:

```bash
cargo build -p wax-llm --release
```

Metal build on macOS:

```bash
cargo build -p wax-llm --release --features metal
```

CUDA build:

```bash
cargo build -p wax-llm --release --features cuda
```

Accelerate build:

```bash
cargo build -p wax-llm --release --features accelerate
```

## Install

Install the published CLI crate from crates.io:

```bash
cargo install wax-llm --features metal
```

Install the CLI binary from this checkout:

```bash
cargo install --path crates/wax-llm --features metal
```

After install, run:

```bash
wax run \
  --model ./models/my-model \
  --prompt "Hello" \
  --max-new-tokens 64
```

## Feature Flags

The workspace crates use no default accelerator feature. Enable the relevant
feature at build or run time:

```text
accelerate
cuda
metal
```

Device selection at runtime is controlled by:

```text
auto | cpu | cuda | metal
```

DType selection at runtime is controlled by:

```text
auto | f32 | f16 | bf16
```
