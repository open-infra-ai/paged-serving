//! 引擎配置类型与验证
//!
//! 提供 [`EngineConfig`] 的定义、参数校验、JSON 序列化与文件加载/保存。
//!
//! # 示例
//!
//! ```rust
//! use paged_serving::EngineConfig;
//!
//! let config = EngineConfig {
//!     block_size: 16,
//!     max_num_blocks: 1024,
//!     ..Default::default()
//! };
//! config.validate()?;
//! # Ok::<(), paged_serving::ConfigError>(())
//! ```

use crate::error::ConfigError;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// 特殊 Token ID 配置（BOS / EOS / PAD / UNK）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpecialTokenIds {
    pub bos: u32,
    pub eos: u32,
    pub pad: u32,
    pub unk: u32,
}

impl Default for SpecialTokenIds {
    fn default() -> Self {
        Self {
            bos: 1,
            eos: 2,
            pad: 0,
            unk: 3,
        }
    }
}

/// Tokenizer 类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerKind {
    #[default]
    Simple,
    HuggingFace,
}

/// Tokenizer 配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenizerConfig {
    pub kind: TokenizerKind,
    /// HuggingFace tokenizer 文件路径（kind=HuggingFace 时必填）
    pub path: Option<PathBuf>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            kind: TokenizerKind::Simple,
            path: None,
        }
    }
}

/// Serving 配置（OpenAI 兼容 HTTP 服务）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServingConfig {
    pub host: String,
    pub port: u16,
    pub model_name: String,
}

impl Default for ServingConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            model_name: "paged-serving".to_string(),
        }
    }
}

/// 引擎配置
///
/// 所有字段均带 `pub`，推荐用 `EngineConfig { ... ..Default::default() }` 构造。
///
/// # 示例
///
/// ```rust
/// use paged_serving::EngineConfig;
///
/// let config = EngineConfig {
///     block_size: 16,
///     max_num_blocks: 1024,
///     ..Default::default()
/// };
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// 每个 KV Cache 块容纳的 token 数（常见 16 或 32）
    pub block_size: u32,
    /// 物理块最大数量（总 token 容量 = max_num_blocks * block_size）
    pub max_num_blocks: u32,
    /// 单次调度最大序列数
    pub max_batch_size: u32,
    /// 系统最大并发序列数（含 pending/prefill/decode 各阶段）
    pub max_num_seqs: u32,
    /// 最大序列长度（输入 + 输出）
    pub max_model_len: u32,
    /// 单批次最大 token 总数
    pub max_total_tokens: u32,
    /// 内存压力阈值 (0.0, 1.0]，超过时拒绝新 prefill 请求
    pub memory_threshold: f32,
    /// GPU 执行超时的最大重试次数
    pub max_retry_attempts: u32,
    /// 每个流式候选可缓冲的文本事件数；满队列时取消慢消费者。
    #[serde(default = "default_event_channel_capacity")]
    pub event_channel_capacity: u32,
    pub special_tokens: SpecialTokenIds,
    #[serde(default)]
    pub tokenizer: TokenizerConfig,
    #[serde(default)]
    pub serving: ServingConfig,
}

fn default_event_channel_capacity() -> u32 {
    64
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            max_num_blocks: 1024,
            max_batch_size: 32,
            max_num_seqs: 256,
            max_model_len: 2048,
            max_total_tokens: 4096,
            memory_threshold: 0.9,
            max_retry_attempts: 2,
            event_channel_capacity: default_event_channel_capacity(),
            special_tokens: SpecialTokenIds::default(),
            tokenizer: TokenizerConfig::default(),
            serving: ServingConfig::default(),
        }
    }
}

impl EngineConfig {
    /// 验证所有参数在有效范围内。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use paged_serving::EngineConfig;
    ///
    /// let config = EngineConfig::default();
    /// assert!(config.validate().is_ok());
    ///
    /// let invalid = EngineConfig { block_size: 0, ..Default::default() };
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.block_size == 0 {
            return Err(ConfigError::InvalidBlockSize(self.block_size));
        }
        if self.max_num_blocks == 0 {
            return Err(ConfigError::InvalidMaxNumBlocks(self.max_num_blocks));
        }
        if self.max_batch_size == 0 {
            return Err(ConfigError::InvalidMaxBatchSize(self.max_batch_size));
        }
        if self.max_num_seqs == 0 {
            return Err(ConfigError::InvalidMaxNumSeqs(self.max_num_seqs));
        }
        if self.max_model_len == 0 {
            return Err(ConfigError::InvalidMaxModelLen(self.max_model_len));
        }
        if self.max_total_tokens == 0 {
            return Err(ConfigError::InvalidMaxTotalTokens(self.max_total_tokens));
        }
        if self.event_channel_capacity == 0 {
            return Err(ConfigError::InvalidEventChannelCapacity(0));
        }
        if !self.memory_threshold.is_finite()
            || self.memory_threshold <= 0.0
            || self.memory_threshold > 1.0
        {
            return Err(ConfigError::InvalidMemoryThreshold(self.memory_threshold));
        }
        if matches!(self.tokenizer.kind, TokenizerKind::HuggingFace)
            && self.tokenizer.path.is_none()
        {
            return Err(ConfigError::MissingTokenizerPath);
        }
        if self.serving.port == 0 {
            return Err(ConfigError::InvalidServerPort(self.serving.port));
        }
        if self.serving.model_name.trim().is_empty() {
            return Err(ConfigError::InvalidModelName);
        }

        // 关系校验（B17）：拦截明显不一致的组合配置。
        // 单批次大小不可能超过系统并发上限，这种组合直接判定为配置错误。
        // 注意：不校验 max_total_tokens / max_model_len 与 KV 容量的关系——
        // 池容量小于配置上限是「资源紧张」的正常场景（调度器会拒绝/排队），
        // 并非配置错误，测试与内存压力路径均依赖该行为。
        if self.max_batch_size > self.max_num_seqs {
            return Err(ConfigError::BatchSizeExceedNumSeqs(
                self.max_batch_size,
                self.max_num_seqs,
            ));
        }
        Ok(())
    }

    /// 从 JSON 文件加载配置并验证。
    ///
    /// # 示例
    ///
    /// ```rust,no_run
    /// use paged_serving::EngineConfig;
    /// use std::path::Path;
    ///
    /// let config = EngineConfig::from_file(Path::new("config.json"))?;
    /// # Ok::<(), paged_serving::ConfigError>(())
    /// ```
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let content =
            std::fs::read_to_string(path).map_err(|e| ConfigError::FileLoadError(e.to_string()))?;
        let config: Self =
            serde_json::from_str(&content).map_err(|e| ConfigError::ParseError(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// 保存配置到 JSON 文件。
    ///
    /// # 示例
    ///
    /// ```rust,no_run
    /// use paged_serving::EngineConfig;
    /// use std::path::Path;
    ///
    /// EngineConfig::default().to_file(Path::new("config.json"))?;
    /// # Ok::<(), paged_serving::ConfigError>(())
    /// ```
    pub fn to_file(&self, path: &Path) -> Result<(), ConfigError> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| ConfigError::ParseError(e.to_string()))?;
        std::fs::write(path, content).map_err(|e| ConfigError::FileSaveError(e.to_string()))?;
        Ok(())
    }

    /// `ceil(num_tokens / block_size)`
    ///
    /// ```rust
    /// use paged_serving::EngineConfig;
    /// let config = EngineConfig { block_size: 16, ..Default::default() };
    /// assert_eq!(config.blocks_for_tokens(0), 0);
    /// assert_eq!(config.blocks_for_tokens(17), 2);
    /// ```
    pub fn blocks_for_tokens(&self, num_tokens: u32) -> u32 {
        crate::kv_cache::blocks_for_tokens(num_tokens, self.block_size)
    }

    /// `num_blocks * block_size`
    ///
    /// ```rust
    /// use paged_serving::EngineConfig;
    /// let config = EngineConfig { block_size: 16, ..Default::default() };
    /// assert_eq!(config.tokens_in_blocks(2), 32);
    /// ```
    pub fn tokens_in_blocks(&self, num_blocks: u32) -> u32 {
        num_blocks * self.block_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        assert!(EngineConfig::default().validate().is_ok());
    }

    #[test]
    fn event_channel_capacity_is_validated_and_defaults_for_old_json() {
        let mut json = serde_json::to_value(EngineConfig::default()).unwrap();
        json.as_object_mut()
            .unwrap()
            .remove("event_channel_capacity");
        let mut config: EngineConfig = serde_json::from_value(json).unwrap();
        assert_eq!(config.event_channel_capacity, 64);
        assert!(config.validate().is_ok());
        config.event_channel_capacity = 0;
        assert_eq!(
            config.validate(),
            Err(ConfigError::InvalidEventChannelCapacity(0))
        );
        config.event_channel_capacity = 1;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_zero_fields() {
        for (field, expected) in [
            ("block_size", ConfigError::InvalidBlockSize(0)),
            ("max_num_blocks", ConfigError::InvalidMaxNumBlocks(0)),
            ("max_batch_size", ConfigError::InvalidMaxBatchSize(0)),
            ("max_num_seqs", ConfigError::InvalidMaxNumSeqs(0)),
            ("max_model_len", ConfigError::InvalidMaxModelLen(0)),
            ("max_total_tokens", ConfigError::InvalidMaxTotalTokens(0)),
        ] {
            let mut config = EngineConfig::default();
            match field {
                "block_size" => config.block_size = 0,
                "max_num_blocks" => config.max_num_blocks = 0,
                "max_batch_size" => config.max_batch_size = 0,
                "max_num_seqs" => config.max_num_seqs = 0,
                "max_model_len" => config.max_model_len = 0,
                "max_total_tokens" => config.max_total_tokens = 0,
                _ => unreachable!(),
            }
            assert_eq!(config.validate(), Err(expected));
        }
    }

    #[test]
    fn test_invalid_memory_threshold() {
        for &threshold in &[
            0.0f32,
            -0.1,
            1.5,
            2.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let config = EngineConfig {
                memory_threshold: threshold,
                ..Default::default()
            };
            assert!(matches!(
                config.validate(),
                Err(ConfigError::InvalidMemoryThreshold(_))
            ));
        }
    }

    #[test]
    fn test_invalid_memory_threshold_nan_rejected() {
        // NaN 会绕过普通范围比较，必须显式 is_finite 拒绝，否则内存保护永久失效。
        let config = EngineConfig {
            memory_threshold: f32::NAN,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidMemoryThreshold(_))
        ));

        let config = EngineConfig {
            memory_threshold: f32::INFINITY,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidMemoryThreshold(_))
        ));
    }

    #[test]
    fn test_blocks_for_tokens() {
        let config = EngineConfig {
            block_size: 16,
            ..Default::default()
        };
        assert_eq!(config.blocks_for_tokens(0), 0);
        assert_eq!(config.blocks_for_tokens(1), 1);
        assert_eq!(config.blocks_for_tokens(16), 1);
        assert_eq!(config.blocks_for_tokens(17), 2);
        assert_eq!(config.blocks_for_tokens(33), 3);
    }

    #[test]
    fn test_tokens_in_blocks() {
        let config = EngineConfig {
            block_size: 16,
            ..Default::default()
        };
        assert_eq!(config.tokens_in_blocks(0), 0);
        assert_eq!(config.tokens_in_blocks(1), 16);
        assert_eq!(config.tokens_in_blocks(3), 48);
    }

    #[test]
    fn test_default_config_includes_serving_and_tokenizer_defaults() {
        let config = EngineConfig::default();
        assert_eq!(config.tokenizer.kind, TokenizerKind::Simple);
        assert_eq!(config.tokenizer.path, None);
        assert_eq!(config.serving.host, "127.0.0.1");
        assert_eq!(config.serving.port, 3000);
        assert_eq!(config.serving.model_name, "paged-serving");
    }

    #[test]
    fn test_huggingface_tokenizer_requires_path() {
        let config = EngineConfig {
            tokenizer: TokenizerConfig {
                kind: TokenizerKind::HuggingFace,
                path: None,
            },
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::MissingTokenizerPath)
        ));
    }

    #[test]
    fn test_config_round_trip_preserves_serving_and_tokenizer_settings() {
        let config = EngineConfig {
            tokenizer: TokenizerConfig {
                kind: TokenizerKind::HuggingFace,
                path: Some("fixtures/tokenizer.json".into()),
            },
            serving: ServingConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
                model_name: "demo-model".to_string(),
            },
            ..Default::default()
        };

        let json = serde_json::to_string(&config).unwrap();
        let decoded: EngineConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.tokenizer.kind, TokenizerKind::HuggingFace);
        assert_eq!(
            decoded.tokenizer.path,
            Some("fixtures/tokenizer.json".into())
        );
        assert_eq!(decoded.serving.host, "0.0.0.0");
        assert_eq!(decoded.serving.port, 8080);
        assert_eq!(decoded.serving.model_name, "demo-model");
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_config_validation(
            block_size in 0u32..100,
            max_num_blocks in 0u32..2000,
            max_batch_size in 0u32..100,
            max_num_seqs in 0u32..500,
            max_model_len in 0u32..10000,
            max_total_tokens in 0u32..10000,
            memory_threshold in -0.5f32..1.5,
        ) {
            let config = EngineConfig {
                block_size,
                max_num_blocks,
                max_batch_size,
                max_num_seqs,
                max_model_len,
                max_total_tokens,
                memory_threshold,
                ..Default::default()
            };

            let expected_valid = block_size > 0
                && max_num_blocks > 0
                && max_batch_size > 0
                && max_num_seqs > 0
                && max_model_len > 0
                && max_total_tokens > 0
                && max_batch_size <= max_num_seqs
                && memory_threshold > 0.0
                && memory_threshold <= 1.0;

            prop_assert_eq!(
                config.validate().is_ok(),
                expected_valid,
                "验证不匹配，配置: {:?}",
                config
            );
        }

        #[test]
        fn prop_invalid_configs_rejected(
            valid_block_size in 1u32..100,
            valid_max_num_blocks in 1u32..2000,
            valid_max_batch_size in 1u32..100,
            valid_max_num_seqs in 1u32..500,
            valid_max_model_len in 1u32..10000,
            valid_max_total_tokens in 1u32..10000,
            valid_memory_threshold in 0.01f32..1.0,
        ) {
            for (field, zero_value) in [
                ("block_size", 0u32),
                ("max_num_blocks", 0u32),
                ("max_batch_size", 0u32),
                ("max_num_seqs", 0u32),
                ("max_model_len", 0u32),
                ("max_total_tokens", 0u32),
            ] {
                let mut config = EngineConfig {
                    block_size: valid_block_size,
                    max_num_blocks: valid_max_num_blocks,
                    max_batch_size: valid_max_batch_size,
                    max_num_seqs: valid_max_num_seqs,
                    max_model_len: valid_max_model_len,
                    max_total_tokens: valid_max_total_tokens,
                    memory_threshold: valid_memory_threshold,
                    ..Default::default()
                };
                match field {
                    "block_size" => config.block_size = zero_value,
                    "max_num_blocks" => config.max_num_blocks = zero_value,
                    "max_batch_size" => config.max_batch_size = zero_value,
                    "max_num_seqs" => config.max_num_seqs = zero_value,
                    "max_model_len" => config.max_model_len = zero_value,
                    "max_total_tokens" => config.max_total_tokens = zero_value,
                    _ => unreachable!(),
                }
                prop_assert!(config.validate().is_err());
            }
        }
    }
}
