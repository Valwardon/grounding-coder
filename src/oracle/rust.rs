// Rust-specific oracle for crates.io and docs.rs
use super::{CodeSymbolInfo, KnowledgeAdapter, KnowledgeResult, VerifiedFact};

pub struct RustOracle;

impl Default for RustOracle {
    fn default() -> Self {
        Self::new()
    }
}

impl RustOracle {
    pub fn new() -> Self {
        RustOracle
    }

    /// Fetch crate metadata from crates.io
    async fn fetch_crate_metadata(&self, crate_name: &str) -> Option<serde_json::Value> {
        let url = format!("https://crates.io/api/v1/crates/{}", crate_name);
        // Bundled-roots HTTPS: no platform verifier, no JNI abort risk.
        crate::http::get_json(&url, None).await.ok()
    }

    /// Fetch documentation from docs.rs
    async fn fetch_docs_rs(&self, symbol: &str) -> Option<serde_json::Value> {
        let url = format!("https://docs.rs/{}/latest/{}", symbol, symbol);
        match crate::http::get_text(&url).await {
            Ok((status, html)) if (200..300).contains(&status) => {
                // Parse HTML to extract symbols and facts
                Some(serde_json::json!({
                    "type": "documentation",
                    "content": html,
                    "url": url
                }))
            }
            _ => None,
        }
    }

    /// Parse crate metadata into knowledge
    fn parse_crate_metadata(&self, metadata: &serde_json::Value) -> KnowledgeResult {
        let mut symbols = Vec::new();
        let mut facts = Vec::new();
        let patterns = Vec::new();
        let mut source_urls = Vec::new();

        if let Some(crates) = metadata.get("crates")
            && let Some(crate_data) = crates.get(0)
        {
            let name = crate_data.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let description = crate_data
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            // Create a symbol for the crate
            symbols.push(CodeSymbolInfo {
                qname: name.to_string(),
                language: "rust".to_string(),
                kind: "crate".to_string(),
                module: name.to_string(),
                signature: format!("// Crate {} - {}", name, description),
                source_urls: vec![format!("https://crates.io/crates/{}", name)],
                compiler_verified: false,
                api_level: None,
                patterns: vec![],
            });

            facts.push(VerifiedFact {
                content: format!("Rust crate {} - {} (v{})", name, description, "1.0.0"),
                confidence: 0.8,
                source_urls: vec![format!("https://crates.io/crates/{}", name)],
                compiler_verified: false,
                last_used: chrono::Utc::now().timestamp() as u64,
                utility: 0.9,
                usage_count: 0,
                dependency_count: 0,
            });

            source_urls.push(format!("https://crates.io/crates/{}", name));
        }

        KnowledgeResult {
            symbols,
            facts,
            patterns,
            source_urls,
            compiler_verified: false,
            retrieval_time: chrono::Utc::now().timestamp() as u64,
        }
    }

    /// Parse docs.rs content into knowledge
    fn parse_docs_rs(&self, docs: &serde_json::Value) -> Option<KnowledgeResult> {
        let content = docs.get("content").and_then(|v| v.as_str())?;
        let url = docs.get("url").and_then(|v| v.as_str())?;

        // Simple parsing - in real implementation would use proper HTML parsing
        let mut symbols = Vec::new();
        let mut facts = Vec::new();
        let patterns = Vec::new();
        let source_urls = vec![url.to_string()];

        // Extract function signatures from HTML (simplified)
        let function_re = regex::Regex::new(r"fn\s+(\w+)\s*\([^)]*\)").unwrap();
        for cap in function_re.captures_iter(content) {
            if let Some(func_name) = cap.get(1) {
                symbols.push(CodeSymbolInfo {
                    qname: format!("crate::{}", func_name.as_str()),
                    language: "rust".to_string(),
                    kind: "function".to_string(),
                    module: "crate".to_string(),
                    signature: format!("fn {}", func_name.as_str()),
                    source_urls: vec![url.to_string()],
                    compiler_verified: false,
                    api_level: None,
                    patterns: vec![],
                });
            }
        }

        // Create a fact about the documentation
        facts.push(VerifiedFact {
            content: format!("Rust documentation for {}", url),
            confidence: 0.9,
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
}

impl KnowledgeAdapter for RustOracle {
    fn name(&self) -> &str {
        "RustOracle"
    }

    fn priority(&self) -> u32 {
        100
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
            // Try crates.io first
            if let Some(metadata) = self.fetch_crate_metadata(symbol).await {
                return Some(self.parse_crate_metadata(&metadata));
            }

            // Then try docs.rs
            if let Some(docs) = self.fetch_docs_rs(symbol).await {
                return self.parse_docs_rs(&docs);
            }

            None
        })
    }

    fn can_handle(&self, symbol: &str) -> bool {
        // Handle Rust crates and standard library symbols
        !symbol.contains('.') || symbol.starts_with("crate::") || symbol.starts_with("std::")
    }
}
