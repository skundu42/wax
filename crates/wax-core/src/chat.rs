use std::{
    fs,
    path::{Path, PathBuf},
};

use minijinja::{Environment, Error, ErrorKind};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{Result, WaxError};

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatTemplate {
    template: String,
    bos_token: Option<String>,
    eos_token: Option<String>,
    unk_token: Option<String>,
    pad_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    chat_template: Option<ChatTemplateValue>,
    bos_token: Option<TokenValue>,
    eos_token: Option<TokenValue>,
    unk_token: Option<TokenValue>,
    pad_token: Option<TokenValue>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ChatTemplateValue {
    String(String),
    Named(Vec<NamedChatTemplate>),
}

#[derive(Debug, Deserialize)]
struct NamedChatTemplate {
    name: Option<String>,
    template: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenValue {
    String(String),
    Object { content: String },
}

impl ChatTemplate {
    pub fn load_for_model_path(model_path: &Path) -> Result<Self> {
        let tokenizer_base = if model_path.is_file() {
            model_path.parent().unwrap_or_else(|| Path::new("."))
        } else {
            model_path
        };
        Self::load(tokenizer_base.join("tokenizer_config.json"))
    }

    pub fn load(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if !path.is_file() {
            return Err(WaxError::MissingModelFile(path));
        }

        let bytes = fs::read(&path)?;
        let config: TokenizerConfig = serde_json::from_slice(&bytes)?;
        let template = config
            .chat_template
            .and_then(select_chat_template)
            .ok_or_else(|| WaxError::InvalidModelFolder {
                path,
                reason: "tokenizer_config.json does not define a chat_template".to_string(),
            })?;

        Ok(Self {
            template,
            bos_token: config.bos_token.and_then(token_content),
            eos_token: config.eos_token.and_then(token_content),
            unk_token: config.unk_token.and_then(token_content),
            pad_token: config.pad_token.and_then(token_content),
        })
    }

    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        if messages.is_empty() {
            return Err(WaxError::InvalidRequest(
                "chat messages must not be empty".to_string(),
            ));
        }

        let mut env = Environment::new();
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);

        let template = normalize_chat_template(&self.template);
        env.render_str(
            &template,
            json!({
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "bos_token": self.bos_token.as_deref(),
                "eos_token": self.eos_token.as_deref(),
                "unk_token": self.unk_token.as_deref(),
                "pad_token": self.pad_token.as_deref(),
            }),
        )
        .map_err(WaxError::template)
    }
}

fn select_chat_template(value: ChatTemplateValue) -> Option<String> {
    match value {
        ChatTemplateValue::String(template) => Some(template),
        ChatTemplateValue::Named(templates) => templates
            .iter()
            .find(|template| template.name.as_deref() == Some("default"))
            .or_else(|| templates.first())
            .map(|template| template.template.clone()),
    }
}

fn token_content(value: TokenValue) -> Option<String> {
    match value {
        TokenValue::String(value) | TokenValue::Object { content: value } => Some(value),
    }
}

fn normalize_chat_template(template: &str) -> String {
    template
        .replace("{% generation %}", "")
        .replace("{%- generation %}", "")
        .replace("{% generation -%}", "")
        .replace("{%- generation -%}", "")
        .replace("{% endgeneration %}", "")
        .replace("{%- endgeneration %}", "")
        .replace("{% endgeneration -%}", "")
        .replace("{%- endgeneration -%}", "")
}

fn raise_exception(message: String) -> std::result::Result<String, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

fn strftime_now(_format: String) -> String {
    String::new()
}

#[cfg(test)]
mod tests {
    use super::{ChatMessage, ChatTemplate};

    #[test]
    fn renders_basic_hf_chat_template() {
        let template = ChatTemplate {
            template: r#"{% for message in messages %}{{ '<|' + message['role'] + '|>\n' + message['content'] + eos_token }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"#.to_string(),
            bos_token: Some("<s>".to_string()),
            eos_token: Some("</s>".to_string()),
            unk_token: None,
            pad_token: None,
        };

        let rendered = template
            .render(
                &[
                    ChatMessage::new("system", "Be brief."),
                    ChatMessage::new("user", "Hello"),
                ],
                true,
            )
            .unwrap();

        assert_eq!(
            rendered,
            "<|system|>\nBe brief.</s><|user|>\nHello</s><|assistant|>\n"
        );
    }
}
