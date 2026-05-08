use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use candle_transformers::models::llama::{Config, LlamaConfig};
use serde::Deserialize;

use crate::{Result, WaxError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    Safetensors { files: Vec<PathBuf> },
    Gguf { file: PathBuf },
    Mlx { path: PathBuf },
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub llama: Config,
}

#[derive(Debug, Deserialize)]
struct ConfigFile {
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(flatten)]
    llama: LlamaConfig,
}

impl ModelConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        if !config_path.is_file() {
            return Err(WaxError::MissingModelFile(config_path));
        }

        let bytes = fs::read(&config_path)?;
        let parsed: ConfigFile = serde_json::from_slice(&bytes)?;
        validate_architectures(&parsed.architectures)?;

        Ok(Self {
            architectures: parsed.architectures,
            llama: parsed.llama.into_config(false),
        })
    }
}

fn validate_architectures(architectures: &[String]) -> Result<()> {
    if architectures.is_empty() {
        return Ok(());
    }

    if architectures.iter().any(|arch| {
        arch == "LlamaForCausalLM"
            || arch == "MistralForCausalLM"
            || arch.contains("Llama")
            || arch.contains("TinyLlama")
    }) {
        return Ok(());
    }

    Err(WaxError::UnsupportedArchitecture {
        architecture: architectures.join(", "),
    })
}

pub fn resolve_safetensors_files(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.is_file() {
        return resolve_indexed_safetensors(model_dir, &index_path);
    }

    let single_file = model_dir.join("model.safetensors");
    if single_file.is_file() {
        return Ok(vec![single_file]);
    }

    Err(WaxError::InvalidModelFolder {
        path: model_dir.to_path_buf(),
        reason: "expected model.safetensors or model.safetensors.index.json".to_string(),
    })
}

pub fn resolve_model_source(model_dir: &Path) -> Result<ModelSource> {
    if model_dir.is_file() && model_dir.extension().is_some_and(|ext| ext == "gguf") {
        return Ok(ModelSource::Gguf {
            file: model_dir.to_path_buf(),
        });
    }

    if looks_like_mlx_model(model_dir) {
        return Ok(ModelSource::Mlx {
            path: model_dir.to_path_buf(),
        });
    }

    if let Some(file) = resolve_gguf_file(model_dir)? {
        return Ok(ModelSource::Gguf { file });
    }

    resolve_safetensors_files(model_dir).map(|files| ModelSource::Safetensors { files })
}

fn resolve_gguf_file(model_dir: &Path) -> Result<Option<PathBuf>> {
    let direct = model_dir.join("model.gguf");
    if direct.is_file() {
        return Ok(Some(direct));
    }

    let mut files = fs::read_dir(model_dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "gguf"))
        .collect::<Vec<_>>();
    files.sort();

    match files.len() {
        0 => Ok(None),
        1 => Ok(files.pop()),
        _ => Err(WaxError::InvalidModelFolder {
            path: model_dir.to_path_buf(),
            reason: "multiple .gguf files found; keep exactly one GGUF file or name it model.gguf"
                .to_string(),
        }),
    }
}

fn looks_like_mlx_model(model_dir: &Path) -> bool {
    model_dir.join("weights.npz").is_file()
        || has_mlx_weight_shards(model_dir)
        || model_dir.join("model.safetensors.index.json").is_file()
            && fs::read_to_string(model_dir.join("config.json"))
                .map(|config| config.contains("\"model_type\"") && config.contains("mlx"))
                .unwrap_or(false)
}

fn has_mlx_weight_shards(model_dir: &Path) -> bool {
    fs::read_dir(model_dir)
        .map(|entries| {
            entries.filter_map(|entry| entry.ok()).any(|entry| {
                let path = entry.path();
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    return false;
                };
                name.starts_with("weights.")
                    && path
                        .extension()
                        .is_some_and(|ext| ext == "safetensors" || ext == "npz")
            })
        })
        .unwrap_or(false)
}

fn resolve_indexed_safetensors(model_dir: &Path, index_path: &Path) -> Result<Vec<PathBuf>> {
    let file = fs::File::open(index_path)?;
    let json: serde_json::Value = serde_json::from_reader(file)?;
    let weight_map = json
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| WaxError::InvalidModelFolder {
            path: model_dir.to_path_buf(),
            reason: format!(
                "{} does not contain a weight_map object",
                index_path.display()
            ),
        })?;

    let mut files = BTreeSet::new();
    for value in weight_map.values() {
        let Some(filename) = value.as_str() else {
            return Err(WaxError::InvalidModelFolder {
                path: model_dir.to_path_buf(),
                reason: "weight_map values must be safetensors filenames".to_string(),
            });
        };
        files.insert(filename.to_string());
    }

    let files = files
        .into_iter()
        .map(|filename| model_dir.join(filename))
        .collect::<Vec<_>>();
    for file in &files {
        if !file.is_file() {
            return Err(WaxError::MissingModelFile(file.clone()));
        }
    }

    Ok(files)
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::{resolve_model_source, resolve_safetensors_files, ModelConfig, ModelSource};
    use crate::WaxError;

    fn write_min_llama_config(path: &Path, architectures: &[&str]) {
        let architectures = serde_json::to_string(architectures).unwrap();
        fs::write(
            path,
            format!(
                r#"{{
                    "architectures": {architectures},
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "vocab_size": 128,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "rms_norm_eps": 0.000001,
                    "rope_theta": 10000.0,
                    "max_position_embeddings": 64
                }}"#
            ),
        )
        .unwrap();
    }

    #[test]
    fn accepts_llama_architecture() {
        let dir = tempfile::tempdir().unwrap();
        write_min_llama_config(&dir.path().join("config.json"), &["LlamaForCausalLM"]);

        let config = ModelConfig::load(dir.path()).unwrap();

        assert_eq!(config.architectures, vec!["LlamaForCausalLM"]);
    }

    #[test]
    fn accepts_missing_architecture_for_hf_compatible_llama_configs() {
        let dir = tempfile::tempdir().unwrap();
        write_min_llama_config(&dir.path().join("config.json"), &[]);

        let config = ModelConfig::load(dir.path()).unwrap();

        assert!(config.architectures.is_empty());
    }

    #[test]
    fn reports_missing_config_path() {
        let dir = tempfile::tempdir().unwrap();

        let err = ModelConfig::load(dir.path()).unwrap_err();

        assert!(matches!(err, WaxError::MissingModelFile(path) if path.ends_with("config.json")));
    }

    #[test]
    fn rejects_unsupported_architecture() {
        let dir = tempfile::tempdir().unwrap();
        write_min_llama_config(&dir.path().join("config.json"), &["Qwen2ForCausalLM"]);

        let err = ModelConfig::load(dir.path()).unwrap_err();

        assert!(matches!(err, WaxError::UnsupportedArchitecture { .. }));
        assert!(err.to_string().contains("Qwen2ForCausalLM"));
    }

    #[test]
    fn resolves_single_safetensors_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors"), b"").unwrap();

        let files = resolve_safetensors_files(dir.path()).unwrap();

        assert_eq!(files, vec![dir.path().join("model.safetensors")]);
    }

    #[test]
    fn resolves_single_gguf_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("tiny.gguf"), b"GGUF").unwrap();

        let source = resolve_model_source(dir.path()).unwrap();

        assert_eq!(
            source,
            ModelSource::Gguf {
                file: dir.path().join("tiny.gguf")
            }
        );
    }

    #[test]
    fn resolves_direct_gguf_file_path() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("tiny.gguf");
        fs::write(&file, b"GGUF").unwrap();

        let source = resolve_model_source(&file).unwrap();

        assert_eq!(source, ModelSource::Gguf { file });
    }

    #[test]
    fn rejects_ambiguous_gguf_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.gguf"), b"GGUF").unwrap();
        fs::write(dir.path().join("b.gguf"), b"GGUF").unwrap();

        let err = resolve_model_source(dir.path()).unwrap_err();

        assert!(err.to_string().contains("multiple .gguf"));
    }

    #[test]
    fn detects_mlx_model_folder_before_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("config.json"), r#"{"model_type":"mlx"}"#).unwrap();
        fs::write(dir.path().join("model.safetensors.index.json"), "{}").unwrap();

        let source = resolve_model_source(dir.path()).unwrap();

        assert_eq!(
            source,
            ModelSource::Mlx {
                path: dir.path().to_path_buf()
            }
        );
    }

    #[test]
    fn detects_mlx_weight_npz_folder() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("weights.npz"), b"").unwrap();

        let source = resolve_model_source(dir.path()).unwrap();

        assert_eq!(
            source,
            ModelSource::Mlx {
                path: dir.path().to_path_buf()
            }
        );
    }

    #[test]
    fn detects_mlx_safetensors_weight_shards() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("weights.00.safetensors"), b"").unwrap();

        let source = resolve_model_source(dir.path()).unwrap();

        assert_eq!(
            source,
            ModelSource::Mlx {
                path: dir.path().to_path_buf()
            }
        );
    }

    #[test]
    fn resolves_indexed_safetensors_files_in_stable_order() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.safetensors"), b"").unwrap();
        fs::write(dir.path().join("b.safetensors"), b"").unwrap();
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"z":"b.safetensors","a":"a.safetensors","b":"b.safetensors"}}"#,
        )
        .unwrap();

        let files = resolve_safetensors_files(dir.path()).unwrap();

        assert_eq!(
            files,
            vec![
                dir.path().join("a.safetensors"),
                dir.path().join("b.safetensors")
            ]
        );
    }

    #[test]
    fn indexed_safetensors_reports_missing_shard() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"model.embed_tokens.weight":"missing.safetensors"}}"#,
        )
        .unwrap();

        let err = resolve_safetensors_files(dir.path()).unwrap_err();

        assert!(
            matches!(err, WaxError::MissingModelFile(path) if path.ends_with("missing.safetensors"))
        );
    }

    #[test]
    fn indexed_safetensors_requires_weight_map_object() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"metadata":{}}"#,
        )
        .unwrap();

        let err = resolve_safetensors_files(dir.path()).unwrap_err();

        assert!(err.to_string().contains("weight_map"));
    }
}
