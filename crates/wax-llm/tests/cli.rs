use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn help_lists_supported_commands() {
    let mut cmd = Command::cargo_bin("wax").unwrap();

    cmd.arg("--help").assert().success().stdout(
        predicate::str::contains("Small Candle-based local LLM inference CLI")
            .and(predicate::str::contains("run"))
            .and(predicate::str::contains("chat"))
            .and(predicate::str::contains("bench")),
    );
}

#[test]
fn run_requires_prompt_source() {
    let dir = tempfile::tempdir().unwrap();
    let mut cmd = Command::cargo_bin("wax").unwrap();

    cmd.args(["run", "--model"])
        .arg(dir.path())
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--prompt, --prompt-file, or --stdin",
        ));
}

#[test]
fn run_rejects_multiple_prompt_sources_before_loading_model() {
    let dir = tempfile::tempdir().unwrap();
    let prompt = dir.path().join("prompt.txt");
    std::fs::write(&prompt, "hello").unwrap();

    let mut cmd = Command::cargo_bin("wax").unwrap();
    cmd.args(["run", "--model"])
        .arg(dir.path().join("missing-model"))
        .args(["--prompt", "hello", "--prompt-file"])
        .arg(&prompt)
        .assert()
        .failure()
        .stderr(predicate::str::contains("provide only one"));
}

#[test]
fn chat_requires_message_before_loading_model() {
    let dir = tempfile::tempdir().unwrap();
    let mut cmd = Command::cargo_bin("wax").unwrap();

    cmd.args(["chat", "--model"])
        .arg(dir.path().join("missing-model"))
        .assert()
        .failure()
        .stderr(predicate::str::contains("at least one --message"));
}

#[test]
fn bench_rejects_zero_runs_before_loading_model() {
    let dir = tempfile::tempdir().unwrap();
    let prompt = dir.path().join("prompt.txt");
    std::fs::write(&prompt, "hello").unwrap();

    let mut cmd = Command::cargo_bin("wax").unwrap();
    cmd.args(["bench", "--model"])
        .arg(dir.path().join("missing-model"))
        .args(["--prompt-file"])
        .arg(&prompt)
        .args(["--runs", "0"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--runs must be > 0"));
}

#[test]
fn run_reports_mlx_conversion_requirement() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.json"), r#"{"model_type":"mlx"}"#).unwrap();
    std::fs::write(dir.path().join("model.safetensors.index.json"), "{}").unwrap();

    let mut cmd = Command::cargo_bin("wax").unwrap();
    cmd.args(["run", "--model"])
        .arg(dir.path())
        .args(["--prompt", "hello"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("unsupported model format: mlx")
                .and(predicate::str::contains("Convert the model")),
        );
}
