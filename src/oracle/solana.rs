// Solana oracle for Solana SDK documentation and examples
use super::{
    CodePattern, CodeSymbolInfo, KnowledgeAdapter, KnowledgeResult, PatternType, VerifiedFact,
};
pub struct SolanaOracle;

impl Default for SolanaOracle {
    fn default() -> Self {
        Self::new()
    }
}

impl SolanaOracle {
    pub fn new() -> Self {
        SolanaOracle
    }

    /// Fetch Solana SDK documentation from solana.dev
    async fn fetch_solana_docs(&self, symbol: &str) -> Option<serde_json::Value> {
        let url = format!(
            "https://docs.solana.com/developing/clients/rust/{}?hl=en",
            symbol
        );
        // Bundled-roots HTTPS: no platform verifier, no JNI abort risk.
        match crate::http::get_text(&url).await {
            Ok((status, html)) if (200..300).contains(&status) => Some(serde_json::json!({
                "type": "solana_docs",
                "content": html,
                "url": url
            })),
            _ => None,
        }
    }

    /// Fetch Solana examples repository
    async fn fetch_solana_examples(&self, example_name: &str) -> Option<serde_json::Value> {
        let url = format!(
            "https://github.com/solana-labs/solana-program-library/tree/master/{}?raw=true",
            example_name
        );
        match crate::http::get_text(&url).await {
            Ok((status, content)) if (200..300).contains(&status) => Some(serde_json::json!({
                "type": "solana_examples",
                "content": content,
                "url": url
            })),
            _ => None,
        }
    }

    /// Parse Solana documentation into knowledge
    fn parse_solana_docs(&self, docs: &serde_json::Value) -> Option<KnowledgeResult> {
        let content = docs.get("content").and_then(|v| v.as_str())?;
        let url = docs.get("url").and_then(|v| v.as_str())?;

        // Simple parsing - in real implementation would use proper HTML parsing
        let mut symbols = Vec::new();
        let mut facts = Vec::new();
        let patterns = Vec::new();
        let source_urls = vec![url.to_string()];

        // Extract Rust functions and types from documentation (simplified)
        let rust_fn_re = regex::Regex::new(r"pub fn\s+([^{]*)\{[^}]*\}").unwrap();
        for cap in rust_fn_re.captures_iter(content) {
            if let Some(func_sig) = cap.get(1) {
                let func_name = func_sig.as_str().split('(').next().unwrap_or("");
                symbols.push(CodeSymbolInfo {
                    qname: format!("solana::{}", func_name),
                    language: "rust".to_string(),
                    kind: "function".to_string(),
                    module: "solana".to_string(),
                    signature: format!("pub fn {}", func_name),
                    source_urls: vec![url.to_string()],
                    compiler_verified: false,
                    api_level: None,
                    patterns: vec![format!("{:?}", PatternType::SolanaRpc)],
                });
            }
        }

        let struct_re = regex::Regex::new(r"pub struct\s+(\w+)").unwrap();
        for cap in struct_re.captures_iter(content) {
            if let Some(struct_name) = cap.get(1) {
                symbols.push(CodeSymbolInfo {
                    qname: format!("solana::{}", struct_name.as_str()),
                    language: "rust".to_string(),
                    kind: "struct".to_string(),
                    module: "solana".to_string(),
                    signature: format!("pub struct {}", struct_name.as_str()),
                    source_urls: vec![url.to_string()],
                    compiler_verified: false,
                    api_level: None,
                    patterns: vec![format!("{:?}", PatternType::SolanaRpc)],
                });
            }
        }

        // Create Solana-specific facts
        facts.push(VerifiedFact {
            content: "Solana RPC client provides account/block/transaction access via REST API"
                .to_string(),
            confidence: 0.9,
            source_urls: vec![url.to_string()],
            compiler_verified: false,
            last_used: chrono::Utc::now().timestamp() as u64,
            utility: 0.9,
            usage_count: 0,
            dependency_count: 0,
        });

        facts.push(VerifiedFact {
            content:
                "Solana transaction simulation can be performed before submission to estimate fees"
                    .to_string(),
            confidence: 0.8,
            source_urls: vec![url.to_string()],
            compiler_verified: false,
            last_used: chrono::Utc::now().timestamp() as u64,
            utility: 0.8,
            usage_count: 0,
            dependency_count: 0,
        });

        Some(KnowledgeResult {
            symbols,
            facts,
            patterns,
            source_urls,
            compiler_verified: false,
            retrieval_time: chrono::Utc::now().timestamp() as u64,
        })
    }

    /// Parse Solana examples into knowledge
    fn parse_solana_examples(&self, examples: &serde_json::Value) -> Option<KnowledgeResult> {
        let content = examples.get("content").and_then(|v| v.as_str())?;
        let url = examples.get("url").and_then(|v| v.as_str())?;

        let symbols = Vec::new();
        let facts = Vec::new();
        let mut patterns = Vec::new();
        let source_urls = vec![url.to_string()];

        // Extract example code (simplified parsing)
        let code_block_re = regex::Regex::new(r"```rust\s*([^`]*)```").unwrap();
        for cap in code_block_re.captures_iter(content) {
            if let Some(_code) = cap.get(1) {
                // Create pattern for this example
                patterns.push(CodePattern {
                    name: format!("example_{}", chrono::Utc::now().timestamp()),
                    description: format!("Solana trading example from {}", url),
                    language: "rust".to_string(),
                    pattern_type: PatternType::TradingStrategy,
                    confidence: 0.7,
                    usage_count: 0,
                    source_urls: vec![url.to_string()],
                });
            }
        }

        Some(KnowledgeResult {
            symbols,
            facts,
            patterns,
            source_urls,
            compiler_verified: false,
            retrieval_time: chrono::Utc::now().timestamp() as u64,
        })
    }
}

impl KnowledgeAdapter for SolanaOracle {
    fn name(&self) -> &str {
        "SolanaOracle"
    }

    fn priority(&self) -> u32 {
        60
    }

    fn language(&self) -> &str {
        "rust"
    }

    fn research<'a>(
        &'a self,
        symbol: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<KnowledgeResult>> + Send + 'a>>
    {
        Box::pin(async move {
            // Try Solana docs first
            if let Some(docs) = self.fetch_solana_docs(symbol).await {
                return self.parse_solana_docs(&docs);
            }

            // Then try examples
            if let Some(examples) = self.fetch_solana_examples(symbol).await {
                return self.parse_solana_examples(&examples);
            }

            None
        })
    }

    fn can_handle(&self, symbol: &str) -> bool {
        // Handle Solana SDK symbols
        symbol.starts_with("solana.") || symbol.starts_with("solana_") || symbol.contains(".")
    }
}
