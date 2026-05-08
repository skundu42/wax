# Architecture

`wax` is intentionally narrow: load a local model, run a decoder-only
Llama-like causal LM, stream tokens, measure performance, and keep the
implementation easy to read.

## Workspace Layout

```text
wax
├── crates/wax-core   # loading, tokenization, generation, sampling, stats
├── crates/wax-llm    # CLI package, installs the `wax` binary
├── crates/wax-bench  # shared benchmark types/helpers
└── apps/wax-chat     # local Next.js browser chat UI
```

## Crates

`wax-core` owns the inference path. It handles model source detection,
tokenizer loading, device and dtype selection, generation, sampling,
chat-template rendering, stop handling, streaming callbacks, and stats.

`wax-llm` owns the command-line interface. It maps CLI arguments into
`wax-core` requests, prints text or JSON output, reads prompt files or stdin,
and exposes `run`, `chat`, and `bench` commands.

`wax-bench` is intentionally small. It re-exports benchmark and generation
stats types for consumers that want stable benchmark output types without
depending on the CLI crate.

`apps/wax-chat` is a local browser UI. Its API route shells out to the `wax`
CLI or to `cargo run -p wax-llm`, keeping HTTP and UI dependencies out of
`wax-core`.

## Model Loading

Model source detection lives in `wax-core::loader`.

Supported model sources:

- Safetensors model folders with `config.json`, `tokenizer.json`, and either
  `model.safetensors` or `model.safetensors.index.json`.
- Direct `.gguf` files or folders containing a single GGUF file.
- MLX folders are detected and rejected with a conversion-focused error.

Safetensors loading uses Candle's Llama model implementation. GGUF loading uses
Candle's quantized Llama backend.

## Generation Flow

Generation starts with `Engine::load`, which:

1. Validates the model path.
2. Resolves the model source.
3. Selects the runtime device.
4. Selects the dtype.
5. Loads the model backend and tokenizer.
6. Discovers EOS token ids from config and tokenizer metadata.

`Engine::generate` then:

1. Tokenizes the prompt.
2. Creates the backend KV cache when needed.
3. Runs the prompt prefill pass.
4. Samples one token at a time.
5. Applies EOS, explicit EOS id, max-token, and stop-string stopping.
6. Streams decoded text deltas through the caller's `StreamSink`.
7. Returns generated text plus `GenerateStats`.

Stop strings are applied against decoded generated text. Streaming keeps a
small hold-back window so a partial stop prefix is not emitted before the
engine knows whether it will become a full stop sequence.

## Chat Templates

`wax-core::chat` reads `tokenizer_config.json` and renders the Hugging Face
`chat_template` field with MiniJinja. The CLI `chat` command renders messages
into a prompt before calling the same generation path used by `run`.

If a model does not provide `tokenizer_config.json` or a `chat_template`, the
chat path fails clearly instead of guessing a model-specific prompt format.

## Dependency Boundaries

`wax-core` intentionally avoids HTTP and browser UI dependencies. Server and UI
surfaces should sit outside the core crate and call its public API or the CLI.

This keeps the core inference code testable, reusable, and easier to benchmark.
