use candle_core::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};

use crate::{Result, WaxError};

#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub repetition_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            repeat_last_n: 128,
            seed: 299_792_458,
        }
    }
}

impl SamplingConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(WaxError::InvalidRequest(
                "temperature must be finite and >= 0".to_string(),
            ));
        }
        if matches!(self.top_k, Some(0)) {
            return Err(WaxError::InvalidRequest("top-k must be > 0".to_string()));
        }
        if let Some(top_p) = self.top_p {
            if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
                return Err(WaxError::InvalidRequest(
                    "top-p must be finite and between 0 and 1".to_string(),
                ));
            }
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(WaxError::InvalidRequest(
                "repetition penalty must be finite and > 0".to_string(),
            ));
        }
        Ok(())
    }

    pub fn processor(&self) -> Result<LogitsProcessor> {
        self.validate()?;
        Ok(LogitsProcessor::from_sampling(self.seed, self.sampling()))
    }

    fn sampling(&self) -> Sampling {
        if self.temperature <= 0.0 {
            return Sampling::ArgMax;
        }

        match (self.top_k, self.top_p) {
            (None, None) => Sampling::All {
                temperature: self.temperature,
            },
            (Some(k), None) => Sampling::TopK {
                k,
                temperature: self.temperature,
            },
            (None, Some(p)) => Sampling::TopP {
                p,
                temperature: self.temperature,
            },
            (Some(k), Some(p)) => Sampling::TopKThenTopP {
                k,
                p,
                temperature: self.temperature,
            },
        }
    }
}

pub struct Sampler {
    config: SamplingConfig,
    processor: LogitsProcessor,
}

impl Sampler {
    pub fn new(config: SamplingConfig) -> Result<Self> {
        Ok(Self {
            config,
            processor: config.processor()?,
        })
    }

    pub fn sample(&mut self, logits: &Tensor, tokens: &[u32]) -> Result<u32> {
        let logits = if self.config.repetition_penalty == 1.0 {
            logits.clone()
        } else {
            let start_at = tokens.len().saturating_sub(self.config.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                logits,
                self.config.repetition_penalty,
                &tokens[start_at..],
            )?
        };

        Ok(self.processor.sample(&logits)?)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{Sampler, SamplingConfig};

    #[test]
    fn greedy_selects_argmax() {
        let logits = Tensor::new(&[0.1f32, 4.0, 0.2], &Device::Cpu).unwrap();
        let mut sampler = Sampler::new(SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        })
        .unwrap();

        let token = sampler.sample(&logits, &[]).unwrap();

        assert_eq!(token, 1);
    }

    #[test]
    fn seeded_sampling_is_deterministic() {
        let logits = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu).unwrap();
        let config = SamplingConfig {
            temperature: 0.8,
            top_k: Some(3),
            top_p: Some(0.9),
            seed: 42,
            ..SamplingConfig::default()
        };
        let mut left = Sampler::new(config).unwrap();
        let mut right = Sampler::new(config).unwrap();

        let left_token = left.sample(&logits, &[]).unwrap();
        let right_token = right.sample(&logits, &[]).unwrap();

        assert_eq!(left_token, right_token);
    }

    #[test]
    fn rejects_invalid_top_k() {
        let err = SamplingConfig {
            top_k: Some(0),
            ..SamplingConfig::default()
        }
        .validate()
        .unwrap_err();

        assert!(err.to_string().contains("top-k"));
    }
}
