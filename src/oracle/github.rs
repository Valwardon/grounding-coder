// GitHub oracle for repository analysis and code examples
use super::{CodeSymbolInfo, KnowledgeAdapter, KnowledgeResult, VerifiedFact};

pub struct GitHubOracle {
    max_results: usize,
}

impl Default for GitHubOracle {
    fn default() -> Self {
        Self::new()
    }
}

impl GitHubOracle {
    pub fn new() -> Self {
        GitHubOracle { max_results: 10 }
    }

    /// Search GitHub repositories for a given query
    async fn search_repositories(&self, query: &str) -> Option<serde_json::Value> {
        let url = format!(
            "https://api.github.com/search/repositories?q={}&per_page={}",
            query, self.max_results
        );
        // Bundled-roots HTTPS: no platform verifier, no JNI abort risk.
        crate::http::get_json(&url, None, Some("application/vnd.github.v3+json"))
            .await
            .ok()
    }

    /// Get repository contents (simplified)
    async fn get_repo_contents(&self, owner: &str, repo: &str) -> Option<serde_json::Value> {
        let url = format!("https://api.github.com/repos/{}/{}?raw=true", owner, repo);
        crate::http::get_json(&url, None, Some("application/vnd.github.v3+json"))
            .await
            .ok()
    }

    /// Parse GitHub repository data into knowledge
    fn parse_repository(&self, repo_data: &serde_json::Value) -> Option<KnowledgeResult> {
        let name = repo_data.get("full_name")?.as_str()?;
        let description = repo_data.get("description")?.as_str()?.to_string();
        let languages = repo_data.get("language")?.as_str()?.to_string();

        let mut symbols = Vec::new();
        let mut facts = Vec::new();
        let patterns = Vec::new();
        let source_urls = vec![format!("https://github.com/{}", name)];

        // Create symbols based on repository name and description
        symbols.push(CodeSymbolInfo {
            qname: name.replace("/", "::"),
            language: languages.clone(),
            kind: "repository".to_string(),
            module: "github".to_string(),
            signature: format!("# {} - {}", name, description),
            source_urls: source_urls.clone(),
            compiler_verified: false,
            api_level: None,
            patterns: vec![],
        });

        facts.push(VerifiedFact {
            content: format!("GitHub repository {} - {}", name, description),
            confidence: 0.7,
            source_urls: source_urls.clone(),
            compiler_verified: false,
            last_used: chrono::Utc::now().timestamp() as u64,
            utility: 0.6,
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

    /// Parse repository file contents into knowledge
    fn parse_repo_files(&self, file_data: &serde_json::Value) -> Option<KnowledgeResult> {
        let name = file_data.get("name")?.as_str()?;
        let path = file_data.get("path")?.as_str()?;
        let content_type = file_data.get("type")?.as_str()?;

        let mut symbols = Vec::new();
        let mut facts = Vec::new();
        let patterns = Vec::new();
        let source_urls = vec![format!("https://github.com/{}?raw=true", path)];

        if content_type == "file" {
            // Check file extension to determine content type
            if name.ends_with(".rs") {
                // Parse Rust source file
                symbols.push(CodeSymbolInfo {
                    qname: format!("repo::{}", name),
                    language: "rust".to_string(),
                    kind: "file".to_string(),
                    module: "repo".to_string(),
                    signature: "// Rust file in repository".to_string(),
                    source_urls: source_urls.clone(),
                    compiler_verified: false,
                    api_level: None,
                    patterns: vec![],
                });

                facts.push(VerifiedFact {
                    content: format!("Rust source file {} in repository", name),
                    confidence: 0.8,
                    source_urls: source_urls.clone(),
                    compiler_verified: false,
                    last_used: chrono::Utc::now().timestamp() as u64,
                    utility: 0.7,
                    usage_count: 0,
                    dependency_count: 0,
                });
            } else if name.ends_with(".java") || name.ends_with(".kt") {
                // Parse Kotlin/Java source file
                symbols.push(CodeSymbolInfo {
                    qname: format!("repo::{}", name),
                    language: "kotlin".to_string(),
                    kind: "file".to_string(),
                    module: "repo".to_string(),
                    signature: "// Kotlin/Java file in repository".to_string(),
                    source_urls: source_urls.clone(),
                    compiler_verified: false,
                    api_level: None,
                    patterns: vec![],
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

impl KnowledgeAdapter for GitHubOracle {
    fn name(&self) -> &str {
        "GitHubOracle"
    }

    fn priority(&self) -> u32 {
        40
    }

    fn language(&self) -> &str {
        "multi"
    }

    fn research<'a>(
        &'a self,
        symbol: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<KnowledgeResult>> + Send + 'a>>
    {
        Box::pin(async move {
            // Search for repositories
            if let Some(repo_data) = self.search_repositories(symbol).await
                && let Some(items) = repo_data.get("items")
            {
                for repo in items.as_array()? {
                    if let Some(result) = self.parse_repository(repo) {
                        return Some(result);
                    }
                }
            }

            // If no exact match, try repository contents
            let parts = symbol.split('/').collect::<Vec<_>>();
            if let Some(parts) = parts.get(1..4) {
                let owner = parts[0];
                let repo = parts[1];

                if let Some(file_data) = self.get_repo_contents(owner, repo).await {
                    return self.parse_repo_files(&file_data);
                }
            }

            None
        })
    }

    fn can_handle(&self, symbol: &str) -> bool {
        // Handle GitHub repository and file references
        symbol.contains("/")
            && (symbol.contains(".rs")
                || symbol.contains(".java")
                || symbol.contains(".kt")
                || symbol.contains("/tree/"))
    }
}
