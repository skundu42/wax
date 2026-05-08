use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct GenerateStats {
    pub model: String,
    pub device: String,
    pub dtype: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_ms: f64,
    pub ttft_ms: Option<f64>,
    pub decode_tok_s: Option<f64>,
    pub total_ms: f64,
    pub stop_reason: StopReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    Eos,
    MaxTokens,
    StopSequence,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchStats {
    pub model: String,
    pub device: String,
    pub dtype: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub model_load_ms: f64,
    pub prefill_ms: f64,
    pub prefill_tok_s: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub decode_tok_s: Option<f64>,
    pub total_generation_ms: f64,
    pub peak_memory_mb: Option<u64>,
    pub candle_version: &'static str,
    pub rust_version: String,
    pub git_commit: Option<String>,
}

pub const CANDLE_VERSION: &str = "0.10.2";

#[cfg(test)]
mod tests {
    use super::{BenchStats, GenerateStats, StopReason};

    #[test]
    fn generate_stats_serialize_stop_reason_as_snake_case() {
        let json = serde_json::to_value(GenerateStats {
            model: "local".to_string(),
            device: "cpu".to_string(),
            dtype: "f32".to_string(),
            prompt_tokens: 2,
            generated_tokens: 3,
            prefill_ms: 1.0,
            ttft_ms: Some(2.0),
            decode_tok_s: Some(3.0),
            total_ms: 4.0,
            stop_reason: StopReason::MaxTokens,
        })
        .unwrap();

        assert_eq!(json["stop_reason"], "max_tokens");
    }

    #[test]
    fn bench_stats_allow_unknown_peak_memory_and_git_commit() {
        let json = serde_json::to_value(BenchStats {
            model: "local".to_string(),
            device: "metal:0".to_string(),
            dtype: "f16".to_string(),
            prompt_tokens: 9,
            generated_tokens: 4,
            model_load_ms: 10.0,
            prefill_ms: 2.0,
            prefill_tok_s: Some(4.5),
            ttft_ms: None,
            decode_tok_s: None,
            total_generation_ms: 12.0,
            peak_memory_mb: None,
            candle_version: "0.10.2",
            rust_version: "rustc test".to_string(),
            git_commit: None,
        })
        .unwrap();

        assert!(json["peak_memory_mb"].is_null());
        assert!(json["git_commit"].is_null());
    }
}
