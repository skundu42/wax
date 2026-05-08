use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use candle_core::{
    quantized::{gguf_file, tokenizer::TokenizerFromGguf},
    DType, Device, Tensor,
};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    llama::{self, Llama},
    quantized_llama,
};
use tokenizers::Tokenizer;

use crate::{
    device::{device_label, dtype_label, select_device, select_dtype},
    loader::{resolve_model_source, ModelConfig, ModelSource},
    sampler::{Sampler, SamplingConfig},
    stats::{GenerateStats, StopReason},
    DTypeChoice, DeviceChoice, Result, WaxError,
};

pub trait StreamSink {
    fn token(&mut self, text: &str) -> Result<()>;
}

impl<F> StreamSink for F
where
    F: FnMut(&str) -> Result<()>,
{
    fn token(&mut self, text: &str) -> Result<()> {
        self(text)
    }
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model_dir: PathBuf,
    pub device: DeviceChoice,
    pub dtype: DTypeChoice,
}

impl EngineConfig {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            device: DeviceChoice::Auto,
            dtype: DTypeChoice::Auto,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
    pub stream: bool,
    pub stop: Vec<String>,
    pub eos_token_ids: Vec<u32>,
    pub add_special_tokens: bool,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_new_tokens: 64,
            sampling: SamplingConfig::default(),
            stream: true,
            stop: Vec::new(),
            eos_token_ids: Vec::new(),
            add_special_tokens: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateOutput {
    pub text: String,
    pub stats: GenerateStats,
}

pub struct Engine {
    model_dir: PathBuf,
    model_name: String,
    backend: ModelBackend,
    tokenizer: Tokenizer,
    eos_token_ids: Vec<u32>,
    device: Device,
    dtype: DType,
    dtype_label: String,
}

enum ModelBackend {
    Safetensors {
        model: Llama,
        llama_config: llama::Config,
    },
    Gguf {
        model: quantized_llama::ModelWeights,
    },
}

impl Engine {
    pub fn load(config: EngineConfig) -> Result<Self> {
        let model_dir = config.model_dir;
        validate_model_path(&model_dir)?;

        let source = resolve_model_source(&model_dir)?;
        let device = select_device(config.device)?;
        let dtype = select_dtype(config.dtype, &device);
        let model_name = model_display_name(&model_dir);
        let (backend, tokenizer, eos_token_ids, dtype_label) =
            load_backend(&model_dir, source, &device, dtype)?;

        Ok(Self {
            model_dir,
            model_name,
            backend,
            tokenizer,
            eos_token_ids,
            device,
            dtype,
            dtype_label,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn device_label(&self) -> String {
        device_label(&self.device)
    }

    pub fn dtype_label(&self) -> String {
        self.dtype_label.clone()
    }

    pub fn generate<S: StreamSink>(
        &mut self,
        request: GenerateRequest,
        mut stream: S,
    ) -> Result<GenerateOutput> {
        validate_generate_request(&request)?;

        let mut all_tokens = self
            .tokenizer
            .encode(request.prompt.as_str(), request.add_special_tokens)
            .map_err(WaxError::tokenizer)?
            .get_ids()
            .to_vec();
        if all_tokens.is_empty() {
            return Err(WaxError::InvalidRequest(
                "prompt produced no tokens".to_string(),
            ));
        }

        let prompt_tokens = all_tokens.len();
        let mut cache = self.backend.new_cache(self.dtype, &self.device)?;
        let mut sampler = Sampler::new(request.sampling)?;
        let mut generated_token_ids = Vec::with_capacity(request.max_new_tokens);
        let mut generated_text = String::new();
        let mut emitted_len = 0usize;

        let total_start = Instant::now();
        let prefill_start = Instant::now();
        let input = Tensor::new(all_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = self
            .backend
            .forward(&input, 0, cache.as_mut())?
            .squeeze(0)?;
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        let mut generated_tokens = 0usize;
        let mut ttft_ms = None;
        let mut decode_forward_secs = 0.0f64;
        let mut stop_reason = StopReason::MaxTokens;

        for (step, index_pos) in (0..request.max_new_tokens).zip(prompt_tokens..) {
            let next_token = sampler.sample(&logits, &all_tokens)?;
            generated_tokens += 1;

            if ttft_ms.is_none() {
                ttft_ms = Some(total_start.elapsed().as_secs_f64() * 1000.0);
            }

            all_tokens.push(next_token);
            if self.is_eos(next_token, &request.eos_token_ids) {
                stop_reason = StopReason::Eos;
                break;
            }

            generated_token_ids.push(next_token);
            let decoded = self
                .tokenizer
                .decode(&generated_token_ids, true)
                .map_err(WaxError::tokenizer)?;
            let (next_text, hit_stop) = apply_stop_sequences(&decoded, &request.stop);
            generated_text = next_text;

            let target_emit_len = if hit_stop {
                generated_text.len()
            } else {
                streamable_len(&generated_text, &request.stop)
            };
            if request.stream && target_emit_len > emitted_len {
                let delta = generated_text
                    .get(emitted_len..target_emit_len)
                    .ok_or_else(|| {
                        WaxError::InvalidRequest("invalid UTF-8 stream boundary".to_string())
                    })?;
                stream.token(delta)?;
                emitted_len = target_emit_len;
            }

            if hit_stop {
                stop_reason = StopReason::StopSequence;
                break;
            }

            if step + 1 == request.max_new_tokens {
                break;
            }

            let decode_start = Instant::now();
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            logits = self
                .backend
                .forward(&input, index_pos, cache.as_mut())?
                .squeeze(0)?;
            decode_forward_secs += decode_start.elapsed().as_secs_f64();
        }

        if request.stream
            && stop_reason != StopReason::StopSequence
            && generated_text.len() > emitted_len
        {
            let delta = generated_text.get(emitted_len..).ok_or_else(|| {
                WaxError::InvalidRequest("invalid UTF-8 stream boundary".to_string())
            })?;
            stream.token(delta)?;
        }

        let decode_tok_s = if generated_tokens > 1 && decode_forward_secs > 0.0 {
            Some((generated_tokens - 1) as f64 / decode_forward_secs)
        } else {
            None
        };

        Ok(GenerateOutput {
            text: generated_text,
            stats: GenerateStats {
                model: self.model_name.clone(),
                device: self.device_label(),
                dtype: self.dtype_label(),
                prompt_tokens,
                generated_tokens,
                prefill_ms,
                ttft_ms,
                decode_tok_s,
                total_ms: total_start.elapsed().as_secs_f64() * 1000.0,
                stop_reason,
            },
        })
    }

    fn is_eos(&self, token: u32, extra_eos_token_ids: &[u32]) -> bool {
        self.eos_token_ids.contains(&token) || extra_eos_token_ids.contains(&token)
    }
}

fn apply_stop_sequences(text: &str, stops: &[String]) -> (String, bool) {
    let stop_at = stops.iter().filter_map(|stop| text.find(stop)).min();
    match stop_at {
        Some(index) => (text[..index].to_string(), true),
        None => (text.to_string(), false),
    }
}

fn streamable_len(text: &str, stops: &[String]) -> usize {
    let mut hold_len = 0usize;
    for stop in stops {
        for (prefix_len, _) in stop.char_indices().skip(1) {
            if text.ends_with(&stop[..prefix_len]) {
                hold_len = hold_len.max(prefix_len);
            }
        }
    }
    text.len().saturating_sub(hold_len)
}

impl ModelBackend {
    fn new_cache(&self, dtype: DType, device: &Device) -> Result<Option<llama::Cache>> {
        match self {
            Self::Safetensors { llama_config, .. } => {
                Ok(Some(llama::Cache::new(true, dtype, llama_config, device)?))
            }
            Self::Gguf { .. } => Ok(None),
        }
    }

    fn forward(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        cache: Option<&mut llama::Cache>,
    ) -> Result<Tensor> {
        match self {
            Self::Safetensors { model, .. } => {
                let cache = cache.ok_or_else(|| {
                    WaxError::InvalidRequest("missing safetensors KV cache".to_string())
                })?;
                Ok(model.forward(input, index_pos, cache)?)
            }
            Self::Gguf { model } => Ok(model.forward(input, index_pos)?),
        }
    }
}

fn load_backend(
    model_dir: &Path,
    source: ModelSource,
    device: &Device,
    dtype: DType,
) -> Result<(ModelBackend, Tokenizer, Vec<u32>, String)> {
    match source {
        ModelSource::Safetensors { files } => {
            let tokenizer = load_tokenizer_json(model_dir)?;
            let model_config = ModelConfig::load(model_dir)?;
            let eos_token_ids = eos_token_ids(&tokenizer, model_config.llama.eos_token_id.as_ref());
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? };
            let model = Llama::load(vb, &model_config.llama)?;
            Ok((
                ModelBackend::Safetensors {
                    model,
                    llama_config: model_config.llama,
                },
                tokenizer,
                eos_token_ids,
                dtype_label(dtype),
            ))
        }
        ModelSource::Gguf { file } => {
            let mut reader = fs::File::open(&file)?;
            let content = gguf_file::Content::read(&mut reader)
                .map_err(|err| err.with_path(file.clone()))?;
            let tokenizer_base = if model_dir.is_file() {
                file.parent().unwrap_or_else(|| Path::new("."))
            } else {
                model_dir
            };
            let tokenizer = match load_tokenizer_json(tokenizer_base) {
                Ok(tokenizer) => tokenizer,
                Err(WaxError::MissingModelFile(_)) => {
                    Tokenizer::from_gguf(&content).map_err(WaxError::tokenizer)?
                }
                Err(err) => return Err(err),
            };
            let eos_token_ids = eos_token_ids(&tokenizer, None);
            let model = quantized_llama::ModelWeights::from_gguf(content, &mut reader, device)?;
            Ok((
                ModelBackend::Gguf { model },
                tokenizer,
                eos_token_ids,
                "gguf".to_string(),
            ))
        }
        ModelSource::Mlx { .. } => Err(WaxError::UnsupportedModelFormat {
            format: "mlx",
            message: "MLX model folders are not directly executable by Candle. Convert the model to Hugging Face safetensors or GGUF, then load that converted folder/file with wax.".to_string(),
        }),
    }
}

fn load_tokenizer_json(model_dir: &Path) -> Result<Tokenizer> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.is_file() {
        return Err(WaxError::MissingModelFile(tokenizer_path));
    }
    Tokenizer::from_file(&tokenizer_path).map_err(WaxError::tokenizer)
}

fn eos_token_ids(tokenizer: &Tokenizer, config_eos: Option<&llama::LlamaEosToks>) -> Vec<u32> {
    let mut ids = match config_eos {
        Some(llama::LlamaEosToks::Single(id)) => vec![*id],
        Some(llama::LlamaEosToks::Multiple(ids)) => ids.clone(),
        None => Vec::new(),
    };

    for token in ["</s>", "<|end_of_text|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(token) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids
}

fn validate_model_path(model_dir: &Path) -> Result<()> {
    if !model_dir.is_dir() && !model_dir.is_file() {
        return Err(WaxError::InvalidModelFolder {
            path: model_dir.to_path_buf(),
            reason: "path is not a directory or .gguf file".to_string(),
        });
    }
    if model_dir.is_file() && model_dir.extension().is_none_or(|ext| ext != "gguf") {
        return Err(WaxError::InvalidModelFolder {
            path: model_dir.to_path_buf(),
            reason: "file model paths must have a .gguf extension".to_string(),
        });
    }
    Ok(())
}

fn model_display_name(model_path: &Path) -> String {
    let name = if model_path.is_file() {
        model_path.file_stem()
    } else {
        model_path.file_name()
    };
    name.and_then(|name| name.to_str())
        .unwrap_or("local")
        .to_string()
}

fn validate_generate_request(request: &GenerateRequest) -> Result<()> {
    if request.prompt.is_empty() {
        return Err(WaxError::InvalidRequest(
            "prompt must not be empty".to_string(),
        ));
    }
    if request.max_new_tokens == 0 {
        return Err(WaxError::InvalidRequest(
            "max-new-tokens must be > 0".to_string(),
        ));
    }
    if request.stop.iter().any(|stop| stop.is_empty()) {
        return Err(WaxError::InvalidRequest(
            "stop strings must not be empty".to_string(),
        ));
    }
    request.sampling.validate()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{GenerateRequest, SamplingConfig};

    #[test]
    fn default_request_streams_sixty_four_tokens_max() {
        let request = GenerateRequest {
            prompt: "hello".to_string(),
            ..GenerateRequest::default()
        };

        assert!(request.stream);
        assert_eq!(request.max_new_tokens, 64);
    }

    #[test]
    fn request_validation_rejects_empty_prompt() {
        let err = super::validate_generate_request(&GenerateRequest {
            prompt: String::new(),
            max_new_tokens: 1,
            sampling: SamplingConfig::default(),
            stream: true,
            stop: Vec::new(),
            eos_token_ids: Vec::new(),
            add_special_tokens: true,
        })
        .unwrap_err();

        assert!(err.to_string().contains("prompt"));
    }

    #[test]
    fn directory_model_name_preserves_version_suffix() {
        let path = Path::new("/tmp/TinyLlama-1.1B-Chat-v1.0");

        assert_eq!(super::model_display_name(path), "TinyLlama-1.1B-Chat-v1.0");
    }

    #[test]
    fn gguf_file_model_name_removes_extension() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model-q8_0.gguf");
        std::fs::write(&file, b"").unwrap();

        assert_eq!(super::model_display_name(&file), "model-q8_0");
    }

    #[test]
    fn stop_sequence_truncates_at_earliest_match() {
        let (text, stopped) = super::apply_stop_sequences(
            "hello <stop> ignored",
            &["ignored".to_string(), "<stop>".to_string()],
        );

        assert!(stopped);
        assert_eq!(text, "hello ");
    }

    #[test]
    fn streamable_len_holds_possible_stop_prefix() {
        assert_eq!(
            super::streamable_len("hello <st", &["<stop>".to_string()]),
            6
        );
        assert_eq!(super::streamable_len("hello <", &["<stop>".to_string()]), 6);
        assert_eq!(
            super::streamable_len("hello world", &["<stop>".to_string()]),
            "hello world".len()
        );
    }
}
