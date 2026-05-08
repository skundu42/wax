# Releasing

Crates are published to crates.io by the `Publish crates` GitHub Actions
workflow when a GitHub Release is published from `main`.

## Requirements

- Set the repository secret `CARGO_REGISTRY_TOKEN` to a crates.io API token.
- Bump the workspace version in `Cargo.toml`.
- Create a GitHub Release whose tag matches the version, for example `v0.1.0`.
- Create the release from `main`.

## Publish Order

The workflow publishes crates in dependency order:

1. `wax-core`
2. `wax-bench`
3. `wax-llm`

The package name is `wax-llm`, and the installed binary is `wax`.

## Pre-Release Checks

Before publishing, run:

```bash
cargo fmt --check
cargo test --workspace --no-default-features
cargo clippy --workspace --all-targets --no-default-features -- -D warnings
```

If the release changes GPU-specific behavior, also run the relevant feature
build, for example:

```bash
cargo test --workspace --features metal
```
