use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, WaxError>;

#[derive(Debug, thiserror::Error)]
pub enum WaxError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("chat template error: {0}")]
    Template(String),

    #[error("unsupported architecture: {architecture}. This engine currently supports only Llama-like causal LM models.")]
    UnsupportedArchitecture { architecture: String },

    #[error("missing required model file: {0}")]
    MissingModelFile(PathBuf),

    #[error("invalid model folder {path}: {reason}")]
    InvalidModelFolder { path: PathBuf, reason: String },

    #[error("unsupported model format: {format}. {message}")]
    UnsupportedModelFormat {
        format: &'static str,
        message: String,
    },

    #[error("invalid generation request: {0}")]
    InvalidRequest(String),
}

impl WaxError {
    pub(crate) fn tokenizer(err: impl std::fmt::Display) -> Self {
        Self::Tokenizer(err.to_string())
    }

    pub(crate) fn template(err: impl std::fmt::Display) -> Self {
        Self::Template(err.to_string())
    }
}
