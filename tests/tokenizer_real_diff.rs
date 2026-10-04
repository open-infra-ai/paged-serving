//! Qwen2 真实 tokenizer 差分验证（paged-serving 侧）。
//!
//! 默认显式 ignored；配置以下输入后用 `--ignored` 执行，缺输入即失败：
//! - `PSERV_TOKENIZER_JSON`：Qwen2.5 tokenizer.json 路径
//! - `PSERV_TOKENIZER_FIXTURE`：tiny-llm 的 `tokenizer_fixture.json`（HF 权威基准）
//!
//! 验证内容：
//! - 编码与 HF 权威 fixture 逐 id 对齐（tiny-llm 自研 BPE 已与该 fixture 对齐，
//!   故本测试证明 paged-serving ↔ tiny-llm 词表一致）
//! - 真实特殊 token ID（BOS/PAD=151643、EOS=151645）与有效词表大小（151665）

use paged_serving::{HuggingFaceTokenizer, TokenizerTrait};
use std::path::Path;

#[test]
#[ignore = "需要真实 Qwen tokenizer 与 tiny-llm fixture；配置输入后使用 --ignored"]
fn qwen2_tokenizer_matches_tiny_llm_fixture() {
    let path = std::env::var("PSERV_TOKENIZER_JSON")
        .expect("真实 tokenizer 测试必须设置 PSERV_TOKENIZER_JSON");
    let fixture_path = std::env::var("PSERV_TOKENIZER_FIXTURE")
        .expect("真实 tokenizer 测试必须设置 PSERV_TOKENIZER_FIXTURE");
    let tokenizer =
        HuggingFaceTokenizer::from_file(Path::new(&path)).expect("failed to load tokenizer.json");
    let raw = std::fs::read_to_string(&fixture_path).expect("read fixture");
    let fixture: serde_json::Value = serde_json::from_str(&raw).expect("parse fixture");

    let cases = fixture["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty(), "tokenizer fixture 必须包含差分用例");
    let mut mismatches = 0usize;
    for (i, case) in cases.iter().enumerate() {
        let text = case["text"].as_str().unwrap();
        let want: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got = tokenizer.try_encode(text).expect("encode");
        if got != want {
            mismatches += 1;
            if mismatches <= 5 {
                eprintln!("case {i} mismatch: {text:?}\n  got = {got:?}\n  want= {want:?}");
            }
        }
    }
    assert_eq!(mismatches, 0, "{mismatches}/{} cases mismatch", cases.len());

    // 词表与特殊 token（Qwen2.5-0.5B）。
    // vocab_size=151665 是 tokenizer 的完整有效词表；GGUF/模型 embedding 为
    // 151936，多出的 271 个是 llama.cpp 填充的 [PADnnnn] 占位符（未训练行），
    // 不影响编码一致性（tiny-llm 侧已与同一 fixture 对齐）。
    assert_eq!(tokenizer.vocab_size(), 151665);
    assert_eq!(tokenizer.bos_token_id(), 151643);
    assert_eq!(tokenizer.eos_token_id(), 151645);
    assert_eq!(tokenizer.pad_token_id(), 151643);
}
