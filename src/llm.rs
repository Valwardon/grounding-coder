//! LLM translator module — the unverified layer.
//!
//! In the grounded-coder architecture, the LLM is ONLY an unverified
//! NL→structured-intent translator. It never writes code directly.
//! All code generation, verification, and correction is driven by
//! the deterministic engine (TaskDecomposer, SymbolTable, CorrectionPipeline,
//! CodeVerifier, CodeWriter).
//!
//! This module will contain the OpenRouter API client for sending
//! natural language prompts and receiving structured intent JSON.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

fn default_project_path() -> String {
    ".".to_string()
}

fn default_max_retries() -> u32 {
    5
}

fn default_model() -> String {
    "google/gemini-2.0-flash-001".to_string()
}

fn default_base_url() -> String {
    "https://openrouter.ai/api/v1".to_string()
}

/// API key storage for the LLM translator plus app settings.
/// Extra fields carry `serde(default)` so older config files keep loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    #[serde(default)]
    pub openrouter_key: Option<String>,
    #[serde(default)]
    pub github_key: Option<String>,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_base_url")]
    pub base_url: String,
    /// Project directory the engine works in (device-local path).
    #[serde(default = "default_project_path")]
    pub project_path: String,
    /// Max correction attempts per task (RetryBudget).
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

impl Default for ApiConfig {
    fn default() -> Self {
        ApiConfig {
            openrouter_key: None,
            github_key: None,
            model: default_model(),
            base_url: default_base_url(),
            project_path: default_project_path(),
            max_retries: default_max_retries(),
        }
    }
}

/// The structured intent JSON — output of the LLM translator only.
/// The engine takes over from here: all code is generated deterministically.
pub use crate::engine::StructuredIntent;

/// Free models probed working against OpenRouter, in fallback order.
/// The user's configured model is always tried first; these only cover
/// shared-pool 429s. Probed 2026-09: liquid + nemotron answer, others
/// rate-limit or gate.
const FREE_FALLBACK_MODELS: &[&str] = &[
    "liquid/lfm-2.5-2.6b:free",
    "nvidia/nemotron-3.5-lightning:free",
];

/// True when the error is upstream throttling worth retrying/falling back.
fn is_rate_limited(e: &str) -> bool {
    e.contains("429") || e.to_lowercase().contains("rate-limit")
}

/// OpenRouter client — sends NL prompt, receives structured intent.
/// UNVERIFIED: the LLM's output is never trusted. It's validated as JSON
/// and then decomposed into deterministic sub-tasks.
pub struct LlmClient {
    config: Arc<ApiConfig>,
}

impl LlmClient {
    pub fn new(config: ApiConfig) -> Self {
        LlmClient {
            config: Arc::new(config),
        }
    }

    /// Send a natural language prompt and receive a structured intent.
    /// NEVER writes code — only translates NL → intent JSON.
    pub async fn translate(&self, prompt: &str) -> Result<StructuredIntent, String> {
        let system = concat!(
            "You translate natural language into a structured JSON intent. ",
            "You are an unverified layer ONLY. Do NOT write code. ",
            "NEVER include `code` fields — only metadata. ",
            "Output ONLY a JSON object (no markdown fences). ",
            "Required fields and types:\n",
            "- goal: string\n",
            "- file: string | null (target file relative to project root, e.g. \"src/lib.rs\")\n",
            "- language: \"rust\" | \"kotlin\" | etc.\n",
            "- actions: array of {\"Action\": {\"action\": string, \"params\": [string], \"references\": [string]}} or {\"Research\": {\"topic\": string, \"reason\": string}}\n",
            "- references: array of strings fully qualified (e.g. \"std::collections::HashMap\")\n",
            "- define: array of null or { name: string, kind: string, references: [string], signature?: string, cases?: [{input: string, expected: string}], fields?: [{name: string, type: string}], methods?: [{name: string, self: \"none\"|\"ref\"|\"mut\", params: [\"n: Type\"], ret?: string, op: \"new\"|\"add_assign\"|\"get\", field?: string, amount?: string}], title?: string, sections?: [{heading: string, body: string}], footer?: string } — NO code; cases/sections are literal values only; kind \"struct\" with fields+methods synthesizes behavior; kind \"page\" with title+sections+footer synthesizes a webpage\n",
            "- imports: array of strings\n",
            "- test: array of { name: string, assertions: [string] } — NO code\n",
            "- platform: \"android\" | \"desktop\" | \"web\"\n",
            "- architecture: string\n",
            "- runtime: string\n",
            "- capabilities: [string]\n",
            "- domains: [string]\n",
            "- constraints: [string]\n",
            "- dependencies: [string]\n",
            "- unknown_requirements: [string] — list what you do NOT know\n",
            "- confidence: number 0.0-1.0\n"
        );

        // Free shared-pool models throttle (HTTP 429) routinely. Try the
        // configured model first, then known-good free fallbacks — the
        // translator output is validated identically either way, so a
        // fallback can never weaken the determinism boundary.
        let mut candidates = vec![self.config.model.clone()];
        for fallback in FREE_FALLBACK_MODELS {
            if !candidates.iter().any(|m| m == fallback) {
                candidates.push(fallback.to_string());
            }
        }
        let mut last_err = String::new();
        let mut body = None;
        for (i, model) in candidates.iter().enumerate() {
            if i > 0 {
                // Brief backoff; shared pools recover in seconds.
                tokio::time::sleep(std::time::Duration::from_secs(2 * i as u64)).await;
                log::info!(
                    "Translator fallback {}/{}: {}",
                    i + 1,
                    candidates.len(),
                    model
                );
            }
            // crate::http verifies with bundled roots — no platform trust
            // store, so this cannot hit the Android JNI verifier abort.
            match crate::http::post_json(
                &format!("{}/chat/completions", self.config.base_url),
                self.config.openrouter_key.as_deref(),
                &serde_json::json!({
                    "model": model,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ]
                }),
            )
            .await
            {
                Ok(b) => {
                    body = Some(b);
                    break;
                }
                Err(e) if is_rate_limited(&e) && i + 1 < candidates.len() => {
                    log::warn!("Translator throttled on {}: {}", model, truncate(&e, 160));
                    last_err = e;
                }
                Err(e) => return Err(e),
            }
        }
        let body = body.ok_or_else(|| {
            format!(
                "All translator models throttled. Last: {}. Retry shortly or set a faster model in Settings.",
                truncate(&last_err, 300)
            )
        })?;
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .ok_or("No content in response")?;

        let cleaned = strip_json_fences(content);
        // Models often wrap JSON in prose — extract the first {...} block.
        let candidate = extract_json_object(&cleaned).unwrap_or_else(|| cleaned.clone());
        // 1. Strict path: model followed the schema.
        if let Ok(intent) = serde_json::from_str::<StructuredIntent>(&candidate) {
            return Ok(intent);
        }
        // 2. Repair path: accept any JSON (or prose) and normalize it into a
        //    grounded intent. Only metadata is extracted — any `code` the
        //    model emitted is dropped here and would be rejected downstream
        //    anyway by the planner-only boundary.
        let value: serde_json::Value =
            serde_json::from_str(&candidate).unwrap_or(serde_json::Value::Null);
        Ok(normalize_intent(&value, prompt))
    }
}

/// Build a grounded intent from an arbitrary model output + the original prompt.
///
/// Safety contract:
/// - NEVER carries model-provided `code` into the intent.
/// - Imports/references are restricted to patterns found verbatim in the
///   model output or prompt that match `std::...` / `use ...` shapes or a
///   small keyword table (HashMap, HashSet, Vec, ...).
/// - Anything unrecognized becomes `unknown_requirements`, never a guess.
fn normalize_intent(v: &serde_json::Value, prompt: &str) -> StructuredIntent {
    let mut imports: Vec<String> = Vec::new();
    let mut references: Vec<String> = Vec::new();

    // Pull structured fields when present (imports/references/file/etc.).
    if let Some(arr) = v.get("imports").and_then(|x| x.as_array()) {
        for x in arr.iter().filter_map(|x| x.as_str()) {
            imports.push(x.to_string());
        }
    }
    if let Some(arr) = v.get("references").and_then(|x| x.as_array()) {
        for x in arr.iter().filter_map(|x| x.as_str()) {
            references.push(x.to_string());
        }
    }

    // Scan all strings in the model output for `use X;` / `std::...` shapes.
    let mut texts: Vec<String> = Vec::new();
    collect_strings(v, &mut texts);
    texts.push(prompt.to_string());
    for t in &texts {
        for cap in scan_use_paths(t) {
            if !imports.contains(&cap) {
                imports.push(cap.clone());
            }
            if !references.contains(&cap) {
                references.push(cap);
            }
        }
    }

    // Keyword table for the boring path (verified std symbols only).
    let lower = texts.join(" ").to_lowercase();
    let table = [
        ("hashmap", "std::collections::HashMap"),
        ("hashset", "std::collections::HashSet"),
        ("btreemap", "std::collections::BTreeMap"),
    ];
    for (kw, path) in table {
        if lower.contains(kw) && !imports.contains(&path.to_string()) {
            imports.push(path.to_string());
            references.push(path.to_string());
        }
    }
    imports.sort();
    imports.dedup();
    references.sort();
    references.dedup();

    // Webpage requests: free models almost never emit page structure, so
    // detect from the prompt. Content slots carry the user's own words
    // verbatim — the engine escapes and verifies them, never invents copy.
    let page = detect_page_request(&lower, prompt, v);

    // Target file: explicit field wins, else scan for src/... in output+prompt.
    let mut file: Option<String> = v
        .get("file")
        .and_then(|x| x.as_str())
        .map(|s| s.trim_start_matches("./").to_string());
    if file.is_none() {
        file = texts.iter().find_map(|t| scan_src_file(t));
    }
    if file.is_none() && (!imports.is_empty() || lower.contains("src/lib")) {
        file = Some("src/lib.rs".to_string());
    }
    if file.is_none() && page.is_some() {
        file = Some("index.html".to_string());
    }

    let language = v
        .get("language")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            if file.as_deref().is_some_and(|f| f.ends_with(".rs")) || lower.contains("rust") {
                Some("rust".to_string())
            } else if file.as_deref().is_some_and(|f| f.ends_with(".kt")) {
                Some("kotlin".to_string())
            } else if file
                .as_deref()
                .is_some_and(|f| f.ends_with(".html") || f.ends_with(".htm"))
                || page.is_some()
            {
                Some("html".to_string())
            } else {
                None
            }
        });

    let goal = v
        .get("goal")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| prompt.to_string());

    let define = match &page {
        Some(p) => vec![Some(crate::engine::IntentDefinition {
            name: "homepage".to_string(),
            kind: "page".to_string(),
            references: Vec::new(),
            signature: None,
            cases: Vec::new(),
            fields: Vec::new(),
            methods: Vec::new(),
            title: Some(p.0.clone()),
            sections: p
                .1
                .iter()
                .map(|(h, b)| crate::engine::SectionDef {
                    heading: h.clone(),
                    body: b.clone(),
                })
                .collect(),
            footer: Some(p.2.clone()),
        })],
        None => Vec::new(),
    };

    StructuredIntent {
        platform: v
            .get("platform")
            .and_then(|x| x.as_str())
            .unwrap_or("desktop")
            .to_string(),
        architecture: v
            .get("architecture")
            .and_then(|x| x.as_str())
            .unwrap_or("native")
            .to_string(),
        runtime: Default::default(),
        capabilities: Default::default(),
        domains: Default::default(),
        constraints: Default::default(),
        dependencies: Default::default(),
        unknown_requirements: if imports.is_empty() && page.is_none() {
            vec![format!("unresolved request: {}", truncate(prompt, 160))]
        } else {
            Vec::new()
        },
        goal,
        file,
        language,
        imports,
        confidence: 0.5,
        actions: Vec::new(),
        define,
        test: Vec::new(),
        references,
    }
}

/// Detected page content: title, (heading, body) sections, footer.
type PageContent = (String, Vec<(String, String)>, String);

/// Detect a webpage request in the prompt.
/// Sections prefer model-provided structure when present; otherwise one
/// Welcome section carries the prompt verbatim (user's words, escaped and
/// verified downstream — never model-invented copy).
fn detect_page_request(lower: &str, prompt: &str, v: &serde_json::Value) -> Option<PageContent> {
    const TRIGGERS: &[&str] = &[
        "html",
        "website",
        "webpage",
        "homepage",
        "landing page",
        "web site",
        "web page",
    ];
    if !TRIGGERS.iter().any(|t| lower.contains(t)) {
        return None;
    }
    // Model-provided sections win when well-formed.
    let mut sections: Vec<(String, String)> = v
        .get("sections")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| {
                    Some((
                        s.get("heading")?.as_str()?.to_string(),
                        s.get("body")?.as_str()?.to_string(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default();
    if sections.is_empty() {
        sections.push((
            "Welcome".to_string(),
            prompt.trim().chars().take(200).collect(),
        ));
    }
    let footer = v
        .get("footer")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    Some((page_title(lower, prompt), sections, footer))
}

/// Title from "for X" / "about X", else the prompt's own lead words.
/// Word-based matching (ASCII case-insensitive): no byte offsets cross
/// case mappings, so Unicode prompts can never misindex.
fn page_title(_lower: &str, prompt: &str) -> String {
    let words: Vec<&str> = prompt.split_whitespace().collect();
    for (i, w) in words.iter().enumerate() {
        if w.eq_ignore_ascii_case("for") || w.eq_ignore_ascii_case("about") {
            let tail: Vec<&str> = words
                .iter()
                .skip(i + 1)
                .filter(|w| !["a", "an", "the"].contains(&w.to_lowercase().as_str()))
                .take(6)
                .cloned()
                .collect();
            if !tail.is_empty() {
                let title = tail
                    .join(" ")
                    .trim_end_matches(['.', ',', '!', '?'])
                    .to_string();
                return capitalize_first(&title);
            }
        }
    }
    // Fall back: strip leading verbs, keep the lead.
    let skip = [
        "build", "create", "make", "write", "generate", "a", "an", "the", "simple", "new",
    ];
    let words: Vec<&str> = prompt
        .split_whitespace()
        .filter(|w| !skip.contains(&w.to_lowercase().as_str()))
        .take(6)
        .collect();
    capitalize_first(&words.join(" "))
}

fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => "Homepage".to_string(),
    }
}

fn collect_strings(v: &serde_json::Value, out: &mut Vec<String>) {
    collect_strings_depth(v, out, 0);
}

/// Depth-capped recursion: hostile model output nested past this is
/// truncated, never a stack overflow (which the OS reports as death,
/// not as a catchable panic).
fn collect_strings_depth(v: &serde_json::Value, out: &mut Vec<String>, depth: u8) {
    if depth > 32 || out.len() > 512 {
        return;
    }
    match v {
        serde_json::Value::String(s) => out.push(s.clone()),
        serde_json::Value::Array(a) => {
            for x in a {
                collect_strings_depth(x, out, depth + 1);
            }
        }
        serde_json::Value::Object(m) => {
            for (k, x) in m {
                // Skip anything that looks like emitted code.
                if k.eq_ignore_ascii_case("code") {
                    continue;
                }
                collect_strings_depth(x, out, depth + 1);
            }
        }
        _ => {}
    }
}

/// Scan for `use a::b` paths and bare `std::...` tokens.
/// Byte-safe by construction: all matching is done on bytes (which never
/// panic), and string slices are built only over ASCII runs, whose every
/// index is a char boundary. Multibyte model output can never abort this.
pub(crate) fn scan_use_paths(t: &str) -> Vec<String> {
    fn push_token(bytes: &[u8], start: usize, end: usize, out: &mut Vec<String>) {
        if end <= start {
            return;
        }
        let p = String::from_utf8_lossy(&bytes[start..end])
            .trim_end_matches(':')
            .to_string();
        if p.contains("::") && !out.contains(&p) {
            out.push(p);
        }
    }
    let mut out = Vec::new();
    let bytes = t.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Match `use <path>` where path looks like a::b::C.
        if bytes[i..].starts_with(b"use ") || bytes[i..].starts_with(b"use\t") {
            let mut j = i + 4;
            while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                j += 1;
            }
            let start = j;
            while j < bytes.len()
                && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_' || bytes[j] == b':')
            {
                j += 1;
            }
            push_token(bytes, start, j, &mut out);
            i = j;
        } else if bytes[i..].starts_with(b"std::") {
            let mut j = i;
            while j < bytes.len()
                && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_' || bytes[j] == b':')
            {
                j += 1;
            }
            push_token(bytes, i, j, &mut out);
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

fn scan_src_file(t: &str) -> Option<String> {
    // Char-boundary safe: `from` only ever holds boundary offsets, and
    // every slice goes through boundary-checked arithmetic below.
    let mut from = 0;
    while from <= t.len() {
        let rest = t.get(from..)?;
        let pos = rest.find("src/")?;
        let start = from + pos;
        let mut end = start;
        for (idx, c) in t[start..].char_indices() {
            if c.is_whitespace() || matches!(c, '"' | '\'' | '`' | ')' | ',' | ';') {
                break;
            }
            end = start + idx + c.len_utf8();
        }
        let p = t
            .get(start..end)
            .unwrap_or("")
            .trim_end_matches(['.', ',', ';', ':', '"', '\''])
            .to_string();
        if p.ends_with(".rs") || p.ends_with(".kt") || p.ends_with(".java") {
            return Some(p);
        }
        // "src/" is 4 ASCII bytes, so start + 4 is always a boundary.
        from = end.max(start + 4);
    }
    None
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        let mut m = n;
        while !s.is_char_boundary(m) {
            m -= 1;
        }
        format!("{}…", &s[..m])
    }
}

/// Extract the first balanced {...} JSON object from a response.
/// Returns None if no plausible object is found.
fn extract_json_object(s: &str) -> Option<String> {
    let start = s.find('{')?;
    let mut depth = 0usize;
    let mut in_str = false;
    let mut esc = false;
    for (i, c) in s[start..].char_indices() {
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[start..start + i + 1].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// Strip markdown code fences (```json ... ```) around the LLM's JSON reply.
/// LLMs routinely wrap JSON in fences; serde cannot parse those directly.
fn strip_json_fences(content: &str) -> String {
    let trimmed = content.trim();
    let inner = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```JSON"))
        .or_else(|| trimmed.strip_prefix("```"))
        .and_then(|rest| rest.strip_suffix("```"))
        .map(str::trim)
        .unwrap_or(trimmed);
    inner.to_string()
}

/// Check if an API key is configured.
pub fn has_api_key(config: &ApiConfig) -> bool {
    config.openrouter_key.is_some()
}

/// Get the config file path for the current platform.
///
/// `dirs::config_dir()` returns `None` inside an Android app process (no
/// XDG home), which used to surface as "Cannot locate config directory".
/// Fall back through every plausible base dir, ending at the temp dir
/// (always writable), and honor `GROUNDING_CONFIG` as an override.
/// The resolved path is also shown in Settings so it never lies.
pub fn config_path() -> Result<String, String> {
    if let Ok(custom) = std::env::var("GROUNDING_CONFIG")
        && !custom.trim().is_empty()
    {
        return Ok(custom);
    }
    dirs::config_dir()
        .or_else(dirs::data_dir)
        .or_else(dirs::cache_dir)
        .or_else(dirs::home_dir)
        .map(|p| {
            p.join("grounding-coder")
                .join("config.json")
                .to_string_lossy()
                .to_string()
        })
        .or_else(|| {
            Some(
                std::env::temp_dir()
                    .join("grounding-coder")
                    .join("config.json")
                    .to_string_lossy()
                    .to_string(),
            )
        })
        .ok_or_else(|| "Cannot determine config directory".to_string())
}

/// Load config from default path.
pub fn load_config_default() -> ApiConfig {
    load_config(&config_path().unwrap_or_default())
}

/// Save config to file (e.g., for settings panel). Parent dirs are
/// created — a missing app dir must never fail a save.
pub fn save_config(config: &ApiConfig, path: &str) -> Result<(), String> {
    let json =
        serde_json::to_string_pretty(config).map_err(|e| format!("Serialize error: {}", e))?;
    let p = std::path::Path::new(path);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Mkdir error: {}", e))?;
    }
    std::fs::write(path, json).map_err(|e| format!("Write error: {}", e))?;
    Ok(())
}

/// Load config from file.
pub fn load_config(path: &str) -> ApiConfig {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<ApiConfig>(&s).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Multibyte model output must never abort string scanning.
    /// This class of bug killed the app on-device with no message.
    #[test]
    fn scanners_survive_multibyte() {
        let t =
            "Add HashMap — 日本語テスト 🎉 use std::collections::HashMap; see src/lib.rs — done —";
        let paths = scan_use_paths(t);
        assert!(paths.contains(&"std::collections::HashMap".to_string()));
        assert_eq!(scan_src_file(t).as_deref(), Some("src/lib.rs"));
        let _ = truncate(&"é".repeat(200), 120);
        let _ = truncate("emoji 🎉🎉🎉 boundary", 8);
        let _ = extract_json_object(t);
    }

    #[test]
    fn normalize_unicode_prompt_without_panic() {
        let v = serde_json::json!({"goal": "do things — fast 🚀"});
        let intent = normalize_intent(&v, "Add HashMap — now 🚀 src/lib.rs");
        assert!(
            intent
                .imports
                .contains(&"std::collections::HashMap".to_string())
        );
    }

    /// The reported on-device failure: a free model returns prose with no
    /// structure for a dentist homepage. The translator must still produce
    /// a page definition the engine can prove — never a stub dead-end.
    #[test]
    fn dentist_prompt_without_structure_still_routes_to_page() {
        let v = serde_json::json!({"status": "needs_more_info", "message": "which framework?"});
        let intent = normalize_intent(&v, "Build a simple HTML website for a dentist");
        assert_eq!(intent.file.as_deref(), Some("index.html"));
        assert_eq!(intent.language.as_deref(), Some("html"));
        assert_eq!(intent.define.len(), 1);
        let def = intent.define[0].as_ref().expect("page define");
        assert_eq!(def.kind, "page");
        assert!(def.title.as_deref().unwrap_or("").contains("Dentist"));
        assert!(!def.sections.is_empty());
    }

    #[test]
    fn page_title_prefers_for_object() {
        assert_eq!(
            page_title(
                "build a homepage for bright smile dental",
                "Build a homepage for Bright Smile Dental"
            ),
            "Bright Smile Dental"
        );
    }
}
