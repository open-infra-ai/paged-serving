//! 请求类型

use super::{CancellationReason, RequestId, RequestState, TokenId};

/// 生成参数
///
/// 控制文本生成的采样参数。
///
/// # 参数范围
///
/// | 参数 | 有效范围 |
/// |------|----------|
/// | `max_tokens` | > 0 |
/// | `temperature` | [0.0, 2.0]（0.0 表示贪心解码） |
/// | `top_p` | (0.0, 1.0] |
///
/// # 采样支持
///
/// [`validate`](Self::validate) 只做范围校验。当前 CPU 参考执行器只实现
/// greedy 解码（[`is_greedy`](Self::is_greedy)），非 greedy 参数会在
/// submit 阶段以 [`crate::EngineError::UnsupportedGenerationMode`] 拒绝，
/// 而不是在执行时被静默忽略。
#[derive(Debug, Clone)]
pub struct GenerationParams {
    /// 最大生成 token 数
    pub max_tokens: u32,

    /// 采样温度 [0.0, 2.0]，0.0 表示贪心解码
    pub temperature: f32,

    /// Top-p（核采样）参数 (0.0, 1.0]
    pub top_p: f32,

    /// 停止序列（最多 4 个，OpenAI 限制）
    ///
    /// 生成过程中一旦输出文本出现任一序列，请求立即停止（不继续生成），
    /// 输出文本中不包含该序列，且 `finish_reason` 为 `"stop"`。
    /// 空列表表示不启用停止序列。
    pub stop: Vec<String>,

    /// logprobs 输出（OpenAI `logprobs`）
    ///
    /// `Some(k)`：为每个生成 token 返回其对数概率及前 k 个候选
    /// （`k <= 5`，OpenAI 上限）；`None` 表示不返回 logprobs。
    pub logprobs: Option<usize>,

    /// 调度优先级（PSERV-112）：数值越大越优先调度，0 为默认。
    ///
    /// 仅影响调度顺序（prefill 启动与在途 prefill 的批次组合），
    /// 不改变采样、能力声明或后端行为。同级优先级保持 FCFS（先提交先调度）。
    pub priority: u8,
}

impl Default for GenerationParams {
    /// 默认参数即当前后端唯一支持的组合：greedy 解码。
    fn default() -> Self {
        Self {
            max_tokens: 100,
            temperature: 0.0,
            top_p: 1.0,
            stop: Vec::new(),
            logprobs: None,
            priority: 0,
        }
    }
}

impl GenerationParams {
    /// 验证生成参数（仅范围校验，不表达后端能力）
    pub fn validate(&self) -> Result<(), crate::EngineError> {
        if self.max_tokens == 0 {
            return Err(crate::EngineError::InvalidMaxTokens(self.max_tokens));
        }
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) {
            return Err(crate::EngineError::InvalidTemperature(self.temperature));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(crate::EngineError::InvalidTopP(self.top_p));
        }
        if self.stop.len() > 4 {
            return Err(crate::EngineError::TooManyStopSequences(self.stop.len()));
        }
        if self.logprobs.is_some_and(|k| k > 5) {
            return Err(crate::EngineError::InvalidLogprobs(self.logprobs.unwrap()));
        }
        Ok(())
    }

    /// 是否为 greedy 解码（argmax，采样参数不生效）
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0 && self.top_p == 1.0
    }
}

/// 推理请求
///
/// 表示单个推理请求，包含输入 tokens 和生成参数。
#[derive(Debug, Clone)]
pub struct Request {
    /// 请求唯一标识符
    pub id: RequestId,

    /// 输入 tokens（分词后）
    pub input_tokens: Vec<TokenId>,

    /// 生成的输出 tokens
    pub output_tokens: Vec<TokenId>,

    /// 生成参数
    pub params: GenerationParams,

    /// 是否因命中 stop 序列而提前停止（用于 finish_reason 判定）
    pub stop_sequence_hit: bool,

    /// 生成过程中每个 token 的 logprob 信息（启用 logprobs 时累积）
    pub logprobs: Vec<crate::types::TokenLogprobs>,

    /// 当前状态
    pub state: RequestState,
}

impl Request {
    /// 创建新请求
    pub fn new(id: RequestId, input_tokens: Vec<TokenId>, params: GenerationParams) -> Self {
        Self {
            id,
            input_tokens,
            output_tokens: Vec::new(),
            params,
            stop_sequence_hit: false,
            logprobs: Vec::new(),
            state: RequestState::Pending,
        }
    }

    /// 计算总 token 数（输入 + 输出）
    pub fn total_tokens(&self) -> usize {
        self.input_tokens.len() + self.output_tokens.len()
    }

    /// 检查生成是否完成
    pub fn is_complete(&self, eos_token_id: TokenId) -> bool {
        if self.output_tokens.len() >= self.params.max_tokens as usize {
            return true;
        }
        if let Some(&last_token) = self.output_tokens.last() {
            if last_token == eos_token_id {
                return true;
            }
        }
        false
    }
}

/// 生成结束原因（OpenAI `finish_reason` 语义）
///
/// 成功完成的请求一定携带其中一个值；失败/取消的请求为 `None`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// 模型生成 EOS 自然停止
    Stop,
    /// 达到 `max_tokens` 上限被截断
    Length,
}

impl FinishReason {
    /// OpenAI 兼容的字符串表示
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

/// 完成的请求
///
/// 包含已完成请求的输出结果。
#[derive(Debug, Clone)]
pub struct CompletedRequest {
    /// 原始请求 ID
    pub request_id: RequestId,

    /// 输入文本（可选）
    pub input_text: Option<String>,

    /// 生成的输出文本
    pub output_text: String,

    /// 生成的 tokens
    pub output_tokens: Vec<TokenId>,

    /// 是否成功完成
    pub success: bool,

    /// 错误信息（失败时）
    pub error: Option<String>,

    /// 主动取消时的类型化原因；失败和成功时为 None
    pub cancellation: Option<CancellationReason>,

    /// 生成结束原因（成功时为 Some，失败/取消时为 None）
    pub finish_reason: Option<FinishReason>,

    /// 每个生成 token 的 logprob 信息（请求启用且后端提供时）；否则为 None
    pub logprobs: Option<Vec<crate::types::TokenLogprobs>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_params_validation() {
        let valid = GenerationParams {
            max_tokens: 100,
            temperature: 1.0,
            top_p: 0.9,
            stop: Vec::new(),
            logprobs: None,
            priority: 0,
        };
        assert!(valid.validate().is_ok());
        assert!(valid.validate().is_ok());

        let invalid_max_tokens = GenerationParams {
            max_tokens: 0,
            temperature: 1.0,
            top_p: 0.9,
            stop: Vec::new(),
            logprobs: None,
            priority: 0,
        };
        assert!(invalid_max_tokens.validate().is_err());
        assert!(invalid_max_tokens.validate().is_err());

        let invalid_temp = GenerationParams {
            max_tokens: 100,
            temperature: 2.1,
            top_p: 0.9,
            stop: Vec::new(),
            logprobs: None,
            priority: 0,
        };
        assert!(invalid_temp.validate().is_err());

        // temperature == 0.0 表示贪心解码，是合法值
        let greedy = GenerationParams {
            max_tokens: 100,
            temperature: 0.0,
            top_p: 0.9,
            stop: Vec::new(),
            logprobs: None,
            priority: 0,
        };
        assert!(greedy.validate().is_ok());

        let invalid_top_p = GenerationParams {
            max_tokens: 100,
            temperature: 1.0,
            top_p: 1.5,
            stop: Vec::new(),
            logprobs: None,
            priority: 0,
        };
        assert!(invalid_top_p.validate().is_err());
    }

    #[test]
    fn test_nan_parameters_rejected() {
        // NaN 会绕过普通范围比较，必须显式 is_finite 拒绝。
        let nan_temp = GenerationParams {
            max_tokens: 100,
            temperature: f32::NAN,
            top_p: 1.0,
            stop: Vec::new(),
            logprobs: None,
            priority: 0,
        };
        assert!(matches!(
            nan_temp.validate(),
            Err(crate::EngineError::InvalidTemperature(_))
        ));

        let inf_temp = GenerationParams {
            max_tokens: 100,
            temperature: f32::INFINITY,
            top_p: 1.0,
            stop: Vec::new(),
            logprobs: None,
            priority: 0,
        };
        assert!(matches!(
            inf_temp.validate(),
            Err(crate::EngineError::InvalidTemperature(_))
        ));

        let nan_top_p = GenerationParams {
            max_tokens: 100,
            temperature: 1.0,
            top_p: f32::NAN,
            stop: Vec::new(),
            logprobs: None,
            priority: 0,
        };
        assert!(matches!(
            nan_top_p.validate(),
            Err(crate::EngineError::InvalidTopP(_))
        ));
    }

    #[test]
    fn test_request_is_complete() {
        let mut request = Request::new(
            1,
            vec![1, 2, 3],
            GenerationParams {
                max_tokens: 5,
                temperature: 1.0,
                top_p: 1.0,
                stop: Vec::new(),
                logprobs: None,
                priority: 0,
            },
        );

        let eos_token = 0;

        assert!(!request.is_complete(eos_token));

        request.output_tokens = vec![10, 11, 12];
        assert!(!request.is_complete(eos_token));

        request.output_tokens.push(eos_token);
        assert!(request.is_complete(eos_token));

        request.output_tokens = vec![10, 11, 12, 13, 14];
        assert!(request.is_complete(eos_token));
    }

    #[test]
    fn test_finish_reason_as_str() {
        assert_eq!(FinishReason::Stop.as_str(), "stop");
        assert_eq!(FinishReason::Length.as_str(), "length");
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_parameter_validation(
            max_tokens in 0u32..1000,
            temperature in -1.0f32..3.0,
            top_p in -0.5f32..1.5,
        ) {
            let params = GenerationParams {
                max_tokens,
                temperature,
                top_p,
                stop: Vec::new(),
                logprobs: None, priority: 0,
        };

            let validation_result = params.validate();

            let expected_valid = max_tokens > 0
                && (0.0..=2.0).contains(&temperature)
                && top_p > 0.0
                && top_p <= 1.0;

            prop_assert_eq!(
                validation_result.is_ok(),
                expected_valid,
                "验证不匹配，参数: max_tokens={}, temp={}, top_p={}",
                max_tokens, temperature, top_p
            );
        }

        #[test]
        fn prop_parameter_boundaries(
            valid_max_tokens in 1u32..1000,
        ) {
            let params_temp_boundary = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: 2.0,
                top_p: 0.5,
                stop: Vec::new(),
                logprobs: None, priority: 0,
        };
            prop_assert!(params_temp_boundary.validate().is_ok());

            let params_top_p_boundary = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: 1.0,
                top_p: 1.0,
                stop: Vec::new(),
                logprobs: None, priority: 0,
        };
            prop_assert!(params_top_p_boundary.validate().is_ok());

            let params_temp_over = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: 2.001,
                top_p: 0.5,
                stop: Vec::new(),
                logprobs: None, priority: 0,
        };
            prop_assert!(params_temp_over.validate().is_err());

            let params_top_p_over = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: 1.0,
                top_p: 1.001,
                stop: Vec::new(),
                logprobs: None, priority: 0,
        };
            prop_assert!(params_top_p_over.validate().is_err());

            // temperature == 0.0 表示贪心解码，是合法值
            let params_greedy_temp = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: 0.0,
                top_p: 0.5,
                stop: Vec::new(),
                logprobs: None, priority: 0,
        };
            prop_assert!(params_greedy_temp.validate().is_ok());

            let params_negative_temp = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: -0.1,
                top_p: 0.5,
                stop: Vec::new(),
                logprobs: None, priority: 0,
        };
            prop_assert!(params_negative_temp.validate().is_err());

            let params_zero_top_p = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: 1.0,
                top_p: 0.0,
                stop: Vec::new(),
                logprobs: None, priority: 0,
        };
            prop_assert!(params_zero_top_p.validate().is_err());

            let params_zero_max_tokens = GenerationParams {
                max_tokens: 0,
                temperature: 1.0,
                top_p: 0.5,
                stop: Vec::new(),
                logprobs: None, priority: 0,
        };
            prop_assert!(params_zero_max_tokens.validate().is_err());
        }
    }
}
