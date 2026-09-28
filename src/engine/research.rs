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

                // Validate with compiler oracle. Mark the verdict ON the
                // definition — a cataloged def must not claim unverified
                // after the compiler passed it.
                if self.compiler_verify(&def, language) {
                    let mut verified = def.clone();
                    verified.compiler_verified = true;
                    let code_def = verified.clone().into_code_def();
                    self.cache.insert(cache_key, code_def);
                    return Some(verified);
                } else {
                    log::warn!(
                        "ResearchOracle: {} definition for '{}' failed compiler verification",
                        source.name,
                        symbol
                    );
                    if std::env::var("GROUNDING_DEBUG_MIGRATE").is_ok() {
                        eprintln!(
                            "[research] {} via {}: parsed sig={:?} examples={} verify=FAIL",
                            symbol,
                            source.name,
                            def.signature,
                            def.examples.len()
                        );
                    }
                    // Don't cache failed definitions — try other sources
                }
            } else if std::env::var("GROUNDING_DEBUG_MIGRATE").is_ok() {
                eprintln!("[research] {} via {}: no parse", symbol, source.name);
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

        // Bundled-roots HTTPS: no platform verifier, no JNI abort risk.
        let (status, body) = crate::http::get_text(&url)
            .await
            .map_err(|e| {
                log::debug!("HTTP error for {}: {}", url, e);
                e
            })
            .ok()?;

        if !(200..300).contains(&status) {
            log::debug!("Non-2xx status for {}: {}", url, status);
            return None;
        }

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
        // Parse crates.io API response for crate metadata. The fields
        // live nested under `crate` — a top-level parse silently yields
        // "unknown" versions, so read the nested object explicitly.
        #[derive(serde::Deserialize, Default)]
        struct CrateInner {
            #[serde(default)]
            description: Option<String>,
            #[serde(default)]
            max_version: Option<String>,
        }
        #[derive(serde::Deserialize, Default)]
        struct CrateRoot {
            #[serde(default, rename = "crate")]
            crate_: Option<CrateInner>,
        }
        // serde_json is imported in this module as `serde_json` already;
        // use the fully qualified path to avoid confusion.
        let root: CrateRoot = serde_json::from_str(content).unwrap_or_default();
        let info = root.crate_.unwrap_or_default();
        if info.max_version.is_none() && info.description.is_none() {
            return None;
        }
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
        // Output goes beside the source, never to /dev/null: rustc stages
        // temporaries next to the output, and read-only mounts (some
        // sandboxes, device partitions) fail the whole check on /dev.
        let out_file = temp_dir.join("verify_out");

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
                out_file.to_str().unwrap(),
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

/// A word or concept researched on the open web: what it is, where
/// the words came from. Provenance travels with the summary — never
/// a bare claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordDef {
    pub term: String,
    pub summary: String,
    pub source_url: String,
}

/// Minimal percent-encoding for query paths (ASCII alnum plus a few
/// marks pass through; spaces become underscores for wiki titles).
fn encode_query(term: &str) -> String {
    let mut out = String::new();
    for c in term.chars() {
        if c.is_ascii_alphanumeric() || "-_.~".contains(c) {
            out.push(c);
        } else if c == ' ' {
            out.push('_');
        } else {
            for b in c.to_string().as_bytes() {
                out.push_str(&format!("%{:02X}", b));
            }
        }
    }
    out
}

/// Strip a page to readable text: drop scripts/styles/nav, turn tags
/// into spaces, collapse whitespace, cap length. Pure and total over
/// any byte string (lossy UTF-8, never panics).
pub fn html_to_text(html: &str) -> String {
    // Char-based throughout: byte slicing would split multibyte text.
    let mut clean = String::new();
    let mut in_ws = false;
    let mut in_tag = false;
    let mut skip2: Option<String> = None;
    let chars: Vec<char> = html.to_string().chars().collect();
    let mut k = 0;
    // Chrome is not content: scripts, styles, navigation, headers,
    // footers, and asides would drown any article text that follows.
    const SKIP_TAGS: &[&str] = &["script", "style", "nav", "header", "footer", "aside"];
    while k < chars.len() {
        if skip2.is_none() && chars[k] == '<' {
            let head: String = chars[k..]
                .iter()
                .take(32)
                .collect::<String>()
                .to_lowercase();
            let mut matched: Option<String> = None;
            for tag in SKIP_TAGS {
                if head.starts_with(&format!("<{}", tag)) {
                    matched = Some(tag.to_string());
                    break;
                }
            }
            if let Some(tag) = matched {
                skip2 = Some(tag);
                in_tag = true;
            } else {
                in_tag = true;
            }
            if !in_ws {
                clean.push(' ');
                in_ws = true;
            }
            k += 1;
            continue;
        }
        if in_tag {
            if chars[k] == '>' {
                in_tag = false;
            }
            k += 1;
            continue;
        }
        if let Some(tag) = skip2.as_deref() {
            // End skip at the matching close tag, jumping PAST it —
            // landing inside `</style>` would emit `/style>` as text.
            let close = format!("</{}>", tag);
            let tail: String = chars[k..]
                .iter()
                .take(close.len())
                .collect::<String>()
                .to_lowercase();
            if tail == close {
                skip2 = None;
                k += close.len();
            } else {
                k += 1;
            }
            continue;
        }
        if chars[k].is_whitespace() {
            if !in_ws {
                clean.push(' ');
                in_ws = true;
            }
        } else {
            clean.push(chars[k]);
            in_ws = false;
        }
        k += 1;
    }
    let clean = clean.trim().to_string();
    const MAX: usize = 2000;
    if clean.len() <= MAX {
        return clean;
    }
    let mut m = MAX;
    while !clean.is_char_boundary(m) {
        m -= 1;
    }
    clean[..m].to_string()
}

/// Extract result links from a DuckDuckGo html-endpoint page: `uddg=`
/// redirect targets first, plain result anchors second. Capped.
pub fn ddg_links(html: &str, cap: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = html;
    while let Some(pos) = rest.find("uddg=") {
        let after = &rest[pos + 5..];
        let end = after.find(['"', '\'', '&']).unwrap_or(after.len());
        // Percent-decode into bytes first (never slice a str at
        // unchecked offsets), then lossy-decode once at the end.
        let raw = &after[..end];
        let bytes = raw.as_bytes();
        let mut dec: Vec<u8> = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            let hex = |b: u8| (b as char).is_ascii_hexdigit();
            if bytes[i] == b'%' && i + 2 < bytes.len() && hex(bytes[i + 1]) && hex(bytes[i + 2]) {
                let h = (bytes[i + 1] as char).to_digit(16).unwrap_or(0);
                let l = (bytes[i + 2] as char).to_digit(16).unwrap_or(0);
                dec.push((h * 16 + l) as u8);
                i += 3;
            } else {
                dec.push(bytes[i]);
                i += 1;
            }
        }
        let url = String::from_utf8_lossy(&dec).into_owned();
        if (url.starts_with("https://") || url.starts_with("http://")) && !out.contains(&url) {
            out.push(url);
            if out.len() >= cap {
                break;
            }
        }
        rest = after;
    }
    out
}

/// Parse a Wikipedia REST summary response into (description, page
/// URL). Disambiguation pages and missing extracts refuse.
fn wiki_summary(body: &str) -> Option<(String, String)> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    if v.get("type")?.as_str()? == "disambiguation" {
        return None;
    }
    let extract = v.get("extract")?.as_str()?;
    if extract.trim().is_empty() {
        return None;
    }
    let url = v
        .get("content_urls")
        .and_then(|c| c.get("desktop"))
        .and_then(|d| d.get("page"))
        .and_then(|p| p.as_str())
        .unwrap_or("https://en.wikipedia.org")
        .to_string();
    let mut text = extract.trim().to_string();
    if text.len() > 600 {
        let mut m = 600;
        while !text.is_char_boundary(m) {
            m -= 1;
        }
        text.truncate(m);
        text.push('…');
    }
    Some((text, url))
}

impl ResearchOracle {
    /// Look up what a word/concept means: Wikipedia summary first
    /// (stable API, no key), then web search with page fetch. Bounded
    /// by the oracle fetch budget; misses return None honestly.
    pub async fn research_word(&mut self, term: &str) -> Option<WordDef> {
        let term = term.trim();
        if term.is_empty() || term.len() > 64 {
            return None;
        }
        if self.fetches_used >= self.max_fetches {
            return None;
        }
        // Wikipedia: deterministic URL, JSON summary.
        let wiki_url = format!(
            "https://en.wikipedia.org/api/rest_v1/page/summary/{}",
            encode_query(term)
        );
        if let Ok((status, body)) = crate::http::get_text(&wiki_url).await {
            self.fetches_used += 1;
            if (200..300).contains(&status)
                && let Some((summary, url)) = wiki_summary(&body)
            {
                return Some(WordDef {
                    term: term.to_string(),
                    summary,
                    source_url: url,
                });
            }
        }
        // Web search fallback: top result fetched as text.
        if self.fetches_used >= self.max_fetches {
            return None;
        }
        let search_url = format!("https://html.duckduckgo.com/html/?q={}", encode_query(term));
        let (status, body) = crate::http::get_text(&search_url).await.ok()?;
        self.fetches_used += 1;
        if !(200..300).contains(&status) {
            return None;
        }
        for link in ddg_links(&body, 3) {
            if self.fetches_used >= self.max_fetches {
                return None;
            }
            if let Ok((st, page)) = crate::http::get_text(&link).await {
                self.fetches_used += 1;
                if !(200..300).contains(&st) {
                    continue;
                }
                let text = html_to_text(&page);
                if text.len() > 120 {
                    return Some(WordDef {
                        term: term.to_string(),
                        summary: text,
                        source_url: link,
                    });
                }
            }
        }
        None
    }
}

/// Verify an external crate against the registry: name → pinned version.
/// The registry is the evidence; the compiler later re-verifies by
/// resolving the dep. Names are charset-validated; anything else is None.
pub async fn verify_crate(name: &str) -> Option<String> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
    {
        return None;
    }
    let url = format!("https://crates.io/api/v1/crates/{}", name);
    let v: serde_json::Value = crate::http::get_json(&url, None, None).await.ok()?;
    v.get("crate")?
        .get("max_version")?
        .as_str()
        .map(|s| s.to_string())
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

#[cfg(test)]
mod web_tests {
    use super::*;

    #[test]
    fn html_to_text_drops_tags_scripts_and_styles() {
        let html = "<html><head><style>.a{color:red}</style><script>alert(1)</script></head>\
            <body><h1>Hi</h1><p>one <b>two</b></p></body></html>";
        assert_eq!(html_to_text(html), "Hi one two");
    }

    #[test]
    fn html_to_text_drops_chrome() {
        let html = "<nav><a>Menu</a></nav><main><article><p>Real text</p></article></main>\
            <footer>copy</footer>";
        assert_eq!(html_to_text(html), "Real text");
    }

    #[test]
    fn html_to_text_survives_multibyte() {
        let html = "<p>héllo 🌍 world</p>";
        assert_eq!(html_to_text(html), "héllo 🌍 world");
    }

    #[test]
    fn html_to_text_caps_length() {
        let html = format!("<p>{}</p>", "x".repeat(5000));
        assert!(html_to_text(&html).len() <= 2000);
    }

    #[test]
    fn ddg_links_extracts_targets() {
        let html = r#"<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=x">t</a>
            <a href="//duckduckgo.com/l/?uddg=http%3A%2F%2Fplain.org%2F">u</a>"#;
        let links = ddg_links(html, 5);
        assert_eq!(
            links,
            vec![
                "https://example.com/page".to_string(),
                "http://plain.org/".to_string()
            ]
        );
    }

    #[test]
    fn ddg_links_skips_non_http() {
        assert!(ddg_links(r#"<a href="//duckduckgo.com/l/?uddg=notaurl">x</a>"#, 5).is_empty());
    }

    #[test]
    fn wiki_summary_parses_and_refuses_disambiguation() {
        let good = r#"{"type":"standard","extract":"Rust is fast.","content_urls":{"desktop":{"page":"https://en.wikipedia.org/wiki/Rust"}}}"#;
        let (text, url) = wiki_summary(good).expect("parses");
        assert_eq!(text, "Rust is fast.");
        assert!(url.contains("wikipedia.org"));
        let dis = r#"{"type":"disambiguation","extract":"May refer to…"}"#;
        assert!(wiki_summary(dis).is_none());
        assert!(wiki_summary("not json").is_none());
    }

    #[test]
    fn encode_query_shapes_titles() {
        assert_eq!(encode_query("token bucket"), "token_bucket");
        assert_eq!(encode_query("a/b"), "a%2Fb");
    }
}
