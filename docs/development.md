# Development

Small, focused changes are preferred. Keep the core inference path simple and
measurable.

## Validation

Run the default test suite:

```bash
cargo test --workspace --no-default-features
```

Run formatting and lint checks:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --no-default-features -- -D warnings
```

If a change touches GPU-specific behavior, also run the relevant feature build,
for example:

```bash
cargo test --workspace --features metal
```

For the browser chat UI:

```bash
cd apps/wax-chat
npm install
npm run build
npm audit --audit-level=moderate
```

## Test Coverage

Current tests cover:

- Loader format detection.
- Safetensors index handling.
- MLX model detection.
- CLI argument behavior.
- Chat-template rendering.
- Sampling behavior.
- Stats serialization.
- Device and dtype selection.
- Token streaming and stop-sequence handling.

## Local Browser UI

Run the local web UI:

```bash
cd apps/wax-chat
npm install
npm run dev
```

By default the API route shells out to:

```bash
cargo run -q -p wax-llm -- chat ...
```

For faster repeated use, build or install `wax` and point the UI at it:

```bash
WAX_BIN=/path/to/wax npm run dev
```

If the app is launched from somewhere other than `apps/wax-chat`, set
`WAX_WORKSPACE_ROOT` to this repository root.
