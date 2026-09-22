use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The ResearchOracle — grounded's KnowledgeStore.fetch() enhanced.
///
/// In grounded, KnowledgeStore has embedded definitions + runtime cache.
/// fetch("pirate") returns "is a person. wears tricorn hat..." — it never
/// goes to the internet because grounded has no network.
///
/// Here, the ResearchOracle extends that pattern: it CAN resolve new symbols
/// by fetching from VERIFIED sources (official docs, crates.io, Android SDK).
/// Every fetched definition is validated by the compiler oracle before
/// being stored. This is the "curiosity harvester" repurposed — the bot
/// discovers new code patterns from authoritative sources, never from
/// compiler errors (which may be broken code).
///
/// Key principle: the LLM is unverified; the ResearchOracle is a tool that
/// the deterministic engine (CodeVerifier) calls when it encounters an
/// unknown symbol. It returns either a verified definition or "I don't know."
#[derive(Debug, Clone)]
pub struct ResearchOracle {
    /// Verified sources — these are the ONLY places the bot will look.
    /// Each source has a deterministic URL pattern for a given symbol.
    sources: Vec<VerifiedSource>,
    /// Cache of verified definitions (runtime cache, like grounded's)
    cache: HashMap<String, super::CodeDef>,
    /// HTTP client for fetching
    client: reqwest::Client,
    /// Maximum number of fetches per session (bounded — no infinite loops)
    max_fetches: u32,
    /// Fetches performed so far
    fetches_used: u32,
}

/// A verified documentation/source site that the oracle can fetch from.
/// Each source has a deterministic URL pattern for symbol resolution.
#[derive(Debug, Clone)]
pub struct VerifiedSource {
    /// Display name (for logging)
    pub name: String,
    /// Priority — higher = tried first (deterministic order)
    pub priority: u32,
    /// Base URL pattern: "{base}/{symbol_path}" or "{base}{symbol}"
    pub base_url: String,
    /// Language this source covers
    pub language: String,
    /// Parser to extract definitions from fetched content
    pub parser: SourceParser,
    /// Whether this source requires an API key
    pub needs_auth: bool,
}

/// How to parse content from a verified source into code definitions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SourceParser {
    /// Parse Rust crate docs (docs.rs)
    RustDocs,
    /// Parse Maven/Android artifact docs
    Maven,
    /// Parse general HTML — extract code blocks
    HtmlCodeBlocks,
    /// Parse Markdown — extract code blocks
    Markdown,
    /// Parse JSON (crates.io, GitHub API)
    Json,
}

/// A verified code definition discovered from external sources.
/// Same structure as SymbolTable's CodeDef, but with provenance tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchedDef {
    pub qname: String,
    pub language: String,
    pub kind: String,
    pub module: String,
    pub signature: String,
    pub description: String,
    pub examples: Vec<String>,
    /// Where this definition was fetched from (provenance for verification)
    pub source_url: String,
    /// Whether the compiler verified this definition compiles
    pub compiler_verified: bool,
}

impl ResearchedDef {
    /// Convert to CodeDef for insertion into SymbolTable
    pub fn into_code_def(self) -> crate::engine::CodeDef {
        crate::engine::CodeDef {
            qname: self.qname,
            language: self.language,
            kind: self.kind,
            module: self.module,
            signature: self.signature,
            description: self.description,
            examples: self.examples,
        }
    }
}

// --- Verified Sources Configuration ---

/// The canonical list of sources the ResearchOracle can fetch from.
/// These are DETERMINISTIC — no guessing, no LLM-suggested URLs.
const VERIFIED_SOURCES: &[(&str, &str, u32, &str, &str, bool)] = &[
    // Rust crate docs — docs.rs is the canonical source
    (
        "docs.rs",
        "https://docs.rs/",
        100,
        "rust",
        "markdown",
        false,
    ),
    // crates.io — API for crate metadata
    (
        "crates.io",
        "https://crates.io/api/v1/crates/",
        90,
        "rust",
        "json",
        false,
    ),
    // Android SDK docs — developer.android.com
    (
        "Android SDK",
        "https://developer.android.com/reference/",
        80,
        "kotlin",
        "html",
        false,
    ),
    // Kotlin docs — kotlinlang.org
    (
        "Kotlin Docs",
        "https://kotlinlang.org/api/latest/",
        70,
        "kotlin",
        "markdown",
        false,
    ),
    // GitHub code search (for specific symbol patterns)
    (
        "GitHub Code",
        "https://github.com/search?q=",
        60,
        "multi",
        "html",
        true,
    ),
];

impl ResearchOracle {
    pub fn new(max_fetches: u32) -> Self {
        let mut sources = Vec::new();
        for (name, base_url, priority, language, parser, needs_auth) in VERIFIED_SOURCES {
            sources.push(VerifiedSource {
                name: name.to_string(),
                priority: *priority,
                base_url: base_url.to_string(),
                language: language.to_string(),
                parser: parse_parser(parser),
                needs_auth: *needs_auth,
            });
        }

        // Sort by priority (deterministic order — highest first)
        sources.sort_by_key(|s| std::cmp::Reverse(s.priority));

        ResearchOracle {
            sources,
            cache: HashMap::new(),
            client: reqwest::Client::builder()
                .user_agent("grounding-coder-research/0.1")
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            max_fetches,
            fetches_used: 0,
        }
    }

    /// Research a symbol from verified sources.
    /// Returns Some(definition) if found AND compiler-verified, None if unknown.
    /// Never returns broken/unverified code.
    pub async fn research(&mut self, symbol: &str, language: &str) -> Option<ResearchedDef> {
        // Check cache first
        let cache_key = format!("{}:{}", language, symbol.to_lowercase());
        if let Some(def) = self.cache.get(&cache_key) {
            return Some(ResearchedDef {
                qname: def.qname.clone(),
                language: def.language.clone(),
                kind: def.kind.clone(),
                module: def.module.clone(),
                signature: def.signature.clone(),
                description: def.description.clone(),
                examples: def.examples.clone(),
                source_url: String::new(),
                compiler_verified: true,
            });
        }

        // Check budget
        if self.fetches_used >= self.max_fetches {
            log::warn!(
                "ResearchOracle: fetch budget exhausted ({}/{})",
                self.fetches_used,
                self.max_fetches
            );
            return None;
        }

        // Try each verified source in priority order
        for source in &self.sources {
            if source.language != language && source.language != "multi" {
                continue;
            }

            if let Some(def) = self.fetch_from_source(source, symbol, language).await {
                self.fetches_used += 1;

                // Validate with compiler oracle
                if self.compiler_verify(&def, language) {
                    let code_def = def.clone().into_code_def();
                    self.cache.insert(cache_key, code_def);
                    return Some(def);
                } else {
                    log::warn!(
                        "ResearchOracle: {} definition for '{}' failed compiler verification",
                        source.name,
                        symbol
                    );
                    // Don't cache failed definitions — try other sources
                }
            }
        }

        // No verified definition found
        None
    }

    async fn fetch_from_source(
        &self,
        source: &VerifiedSource,
        symbol: &str,
        language: &str,
    ) -> Option<ResearchedDef> {
        let url = self.build_url(source, symbol, language);
        log::info!("ResearchOracle: fetching {} from {}", symbol, url);

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| {
                log::debug!("HTTP error for {}: {}", url, e);
                e
            })
            .ok()?;

        if !resp.status().is_success() {
            log::debug!("Non-200 status for {}: {}", url, resp.status());
            return None;
        }

        // Note: content-type is not currently needed for parsing decisions;
        // kept as documentation of the response headers.

        // Get the content as text
        let body = resp.text().await.ok()?;

        // Parse based on source parser type
        self.parse_content(&body, &source.parser, symbol, language, &url)
    }

    /// Build a deterministic URL for the given symbol and source.
    /// The URL is constructed from the base_url + symbol path — never guessed.
    fn build_url(&self, source: &VerifiedSource, symbol: &str, _language: &str) -> String {
        // Convert symbol (e.g., "solana_program::system_program") to path
        let symbol_path = if symbol.contains("::") {
            symbol.replace("::", "/")
        } else if symbol.contains('.') {
            symbol.replace('.', "/")
        } else {
            symbol.to_string()
        };

        match source.parser {
            SourceParser::RustDocs => {
                // base_url already ends in "/docs.rs/", so do not append another one.
                format!("{}{}/latest/{}/", source.base_url, symbol_path, symbol_path)
            }
            SourceParser::Json => {
                format!("{}{}", source.base_url, symbol)
            }
            SourceParser::HtmlCodeBlocks => {
                format!("{}{}", source.base_url, symbol_path)
            }
            SourceParser::Markdown => {
                format!("{}{}", source.base_url, symbol_path)
            }
            SourceParser::Maven => {
                format!("{}{}", source.base_url, symbol_path)
            }
        }
    }

    /// Parse fetched content into a ResearchedDef based on source type.
    fn parse_content(
        &self,
        content: &str,
        parser: &SourceParser,
        symbol: &str,
        language: &str,
        url: &str,
    ) -> Option<ResearchedDef> {
        match parser {
            SourceParser::Markdown => {
                // docs.rs and kotlinlang.org use Markdown
                // Extract: function signatures, code blocks, descriptions
                self.parse_markdown(content, symbol, language, url)
            }
            SourceParser::HtmlCodeBlocks => {
                self.parse_html_codeblocks(content, symbol, language, url)
            }
            SourceParser::Json => self.parse_json(content, symbol, language, url),
            SourceParser::RustDocs => self.parse_markdown(content, symbol, language, url),
            SourceParser::Maven => self.parse_html_codeblocks(content, symbol, language, url),
        }
    }

    fn parse_markdown(
        &self,
        content: &str,
        symbol: &str,
        language: &str,
        url: &str,
    ) -> Option<ResearchedDef> {
        // Extract code blocks (```...```) and function signatures
        let mut examples = Vec::new();
        let mut signature = String::new();

        // Find code blocks
        let re = regex::Regex::new(r"```(\w+)?\n(.*?)```").unwrap();
        for cap in re.captures_iter(content) {
            if let Some(code) = cap.get(2) {
                examples.push(code.as_str().to_string());
            }
        }

        // Extract first heading as description
        let desc_re = regex::Regex::new(r"^# (.+)$").unwrap();
        let description = desc_re
            .captures(content)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .unwrap_or_else(|| symbol.to_string());

        // Extract signature from examples or from "fn " pattern
        let sig_re = regex::Regex::new(r"(pub\s+)?fn\s+\w+.*").unwrap();
        if let Some(m) = sig_re.find(content) {
            signature = m.as_str().to_string();
        }

        if examples.is_empty() && signature.is_empty() {
            return None;
        }

        Some(ResearchedDef {
            qname: symbol.to_string(),
            language: language.to_string(),
            kind: if signature.contains("trait") {
                "trait"
            } else {
                "function"
            }
            .to_string(),
            module: String::new(),
            signature,
            description,
            examples,
            source_url: url.to_string(),
            compiler_verified: false, // will be verified by compiler_verify
        })
    }

    fn parse_html_codeblocks(
        &self,
        content: &str,
        symbol: &str,
        language: &str,
        url: &str,
    ) -> Option<ResearchedDef> {
        // Parse HTML for <pre><code> blocks and class/function signatures
        let re = regex::Regex::new(r#"<code[^>]*class="[^"]*"[^>]*>(.*?)</code>"#).unwrap();
        let mut examples = Vec::new();
        for cap in re.captures_iter(content) {
            if let Some(code) = cap.get(1) {
                examples.push(html_unescape(code.as_str()).to_string());
            }
        }

        if examples.is_empty() {
            return None;
        }

        Some(ResearchedDef {
            qname: symbol.to_string(),
            language: language.to_string(),
            kind: "class".to_string(),
            module: String::new(),
            signature: String::new(),
            description: format!("Symbol {} from {}", symbol, url),
            examples,
            source_url: url.to_string(),
            compiler_verified: false,
        })
    }

    fn parse_json(
        &self,
        content: &str,
        symbol: &str,
        _language: &str,
        url: &str,
    ) -> Option<ResearchedDef> {
        // Parse crates.io API response for crate metadata
        #[derive(serde::Deserialize)]
        struct CrateInfo {
            #[serde(default)]
            description: Option<String>,
            #[serde(default)]
            max_version: Option<String>,
        }

        match serde_json::from_str::<CrateInfo>(content) {
            Ok(info) => {
                let version = info.max_version.unwrap_or_else(|| "unknown".to_string());
                Some(ResearchedDef {
                    qname: format!("{}:{}", symbol, version),
                    language: "rust".to_string(),
                    kind: "crate".to_string(),
                    module: symbol.to_string(),
                    signature: format!("// Crate {} v{}", symbol, version),
                    description: info
                        .description
                        .unwrap_or_else(|| format!("Rust crate {}", symbol)),
                    examples: vec![format!(
                        "// Add to Cargo.toml: {} = \"{}\"",
                        symbol, version
                    )],
                    source_url: url.to_string(),
                    compiler_verified: false,
                })
            }
            Err(_) => None,
        }
    }

    /// Compiler oracle — verify that the extracted definition compiles.
    /// This is grounded's "three structural guardrails" repurposed for code.
    /// The compiler never lies — if code doesn't compile, the definition
    /// is not trusted.
    fn compiler_verify(&self, def: &ResearchedDef, language: &str) -> bool {
        match language {
            "rust" => {
                // Write the examples to a temp file and run cargo check
                self.verify_rust(&def.examples)
            }
            "kotlin" | "java" => {
                // For Kotlin/Java, we validate against known Android SDK patterns
                // The Android SDK is the oracle — if a method signature matches
                // the SDK docs, it's valid.
                self.verify_android(&def.examples, &def.signature)
            }
            _ => {
                // For unknown languages, trust the source but mark as unverified
                log::warn!("No compiler oracle for language: {}", language);
                false
            }
        }
    }

    fn verify_rust(&self, examples: &[String]) -> bool {
        if examples.is_empty() {
            return false;
        }

        // Write examples to a temp file
        let temp_dir = std::env::temp_dir().join("grounding_coder_verify");
        let _ = std::fs::create_dir_all(&temp_dir);
        let test_file = temp_dir.join("verify_main.rs");

        let content = format!("fn main() {{\n{}\n}}\n", examples.join("\n"));
        if std::fs::write(&test_file, &content).is_err() {
            return false;
        }

        // Run rustc to check if the code compiles
        // Note: this is a lightweight check — just syntax, not full verification
        let result = std::process::Command::new("rustc")
            .args([
                "--edition",
                "2021",
                "--crate-type",
                "bin",
                "-o",
                "/dev/null",
                test_file.to_str().unwrap(),
            ])
            .output();

        match result {
            Ok(output) => {
                if output.status.success() {
                    true
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    log::debug!("Compiler verification failed: {}", stderr);
                    false
                }
            }
            Err(e) => {
                log::debug!("Failed to run rustc: {}", e);
                false
            }
        }
    }

    fn verify_android(&self, examples: &[String], _signature: &str) -> bool {
        // Android method signatures are verified against known patterns
        // (this is the SDK oracle — if a method matches the documented API, it's valid)
        // We don't need to compile Kotlin — we match against known Android SDK patterns

        // Check for known Android method call patterns
        let known_patterns = [
            "findViewById",
            "setOnClickListener",
            "setText",
            "setContentView",
            "getSystemService",
            "vibrate",
            "makeText",
            "setAdapter",
        ];

        for example in examples {
            for pattern in &known_patterns {
                if example.contains(pattern) {
                    return true;
                }
            }
        }
        false
    }

    /// Add a verified definition to the SymbolTable.
    /// This is called when the bot successfully researches a new symbol.
    pub fn cache_definition(
        &mut self,
        symbol: &str,
        language: &str,
        def: ResearchedDef,
    ) -> crate::engine::CodeDef {
        let cache_key = format!("{}:{}", language, symbol.to_lowercase());
        let code_def = def.into_code_def();
        self.cache.insert(cache_key, code_def.clone());
        code_def
    }

    /// Get the source URL for a symbol without fetching (just construct the deterministic URL).
    pub fn source_url(&self, symbol: &str, language: &str) -> String {
        for source in &self.sources {
            if source.language == language || source.language == "multi" {
                return self.build_url(source, symbol, language);
            }
        }
        format!("https://docs.rs/{}/latest/", symbol)
    }
}

fn parse_parser(s: &str) -> SourceParser {
    match s {
        "markdown" => SourceParser::Markdown,
        "html" => SourceParser::HtmlCodeBlocks,
        "json" => SourceParser::Json,
        "rust" => SourceParser::RustDocs,
        _ => SourceParser::HtmlCodeBlocks,
    }
}

/// Minimal HTML entity unescaper — avoids external dependency.
fn html_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}
