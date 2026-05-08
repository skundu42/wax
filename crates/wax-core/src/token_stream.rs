use crate::{Result, WaxError};

pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            self.decode(&self.tokens[self.prev_index..self.current_index])?
        };

        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().is_some_and(char::is_alphanumeric) {
            let (_, delta) = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(delta.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            self.decode(&self.tokens[self.prev_index..self.current_index])?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let (_, delta) = text.split_at(prev_text.len());
            Ok(Some(delta.to_string()))
        } else {
            Ok(None)
        }
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(WaxError::tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use ahash::AHashMap;
    use tokenizers::{models::wordlevel::WordLevel, Tokenizer};

    use super::TokenOutputStream;

    fn tokenizer() -> Tokenizer {
        let vocab = AHashMap::from([
            ("Hello".to_string(), 0),
            ("world".to_string(), 1),
            ("!".to_string(), 2),
            ("[UNK]".to_string(), 3),
        ]);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        Tokenizer::new(model)
    }

    #[test]
    fn streams_alphanumeric_tokens_and_flushes_punctuation_at_end() {
        let mut stream = TokenOutputStream::new(tokenizer());

        assert_eq!(stream.next_token(0).unwrap(), Some("Hello".to_string()));
        assert_eq!(stream.next_token(1).unwrap(), Some(" world".to_string()));
        assert_eq!(stream.next_token(2).unwrap(), None);
        assert_eq!(stream.decode_rest().unwrap(), Some(" !".to_string()));
    }
}
