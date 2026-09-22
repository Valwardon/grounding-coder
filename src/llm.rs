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

/// API key storage for the LLM translator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub openrouter_key: Option<String>,
    pub github_key: Option<String>,
    pub model: String,
    pub base_url: String,
}

impl Default for ApiConfig {
    fn default() -> Self {
        ApiConfig {
            openrouter_key: None,
            github_key: None,
            model: "google/gemini-2.0-flash-001".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
        }
    }
}

/// The structured intent JSON — output of the LLM translator only.
/// The engine takes over from here: all code is generated deterministically.
pub use crate::engine::StructuredIntent;

/// OpenRouter client — sends NL prompt, receives structured intent.
/// UNVERIFIED: the LLM's output is never trusted. It's validated as JSON
/// and then decomposed into deterministic sub-tasks.
pub struct LlmClient {
    config: Arc<ApiConfig>,
    client: reqwest::Client,
}

impl LlmClient {
    pub fn new(config: ApiConfig) -> Self {
        LlmClient {
            config: Arc::new(config),
            client: reqwest::Client::new(),
        }
    }

    /// Send a natural language prompt and receive a structured intent.
    /// NEVER writes code — only translates NL → intent JSON.
    pub async fn translate(&self, prompt: &str) -> Result<StructuredIntent, String> {
        let system = concat!(
            "You translate natural language into a structured JSON intent. ",
            "You are an unverified layer ONLY. Do NOT write code. ",
            "Output ONLY a JSON object (no markdown fences). ",
            "Required fields and types:\n",
            "- goal: string (what the human wants)\n",
            "- file: string (target file path relative to project root)\n",
            "- language: \"kotlin\" | \"rust\" | etc.\n",
            "- actions: array of { action: string, params: object }\n",
            "- references: array of strings — every symbol fully qualified (e.g. android.widget.Button, solenoid::Part)\n",
            "- define: null or { name: string, kind: string, code: string, references: [string] }\n",
            "- imports: array of strings\n",
            "- test: null or { name: string, assertions: [string], code: string }\n",
            "- platform: \"android\" | \"desktop\" | \"web\"\n",
            "- architecture: \"native\" | \"cross-platform\" | \"hybrid\"\n",
            "- runtime: null or { min_sdk?: number, target_sdk?: number, package?: string, api_constraints: [string] }\n",
            "- capabilities: array of strings (network, filesystem, background execution, ...)\n",
            "- domains: array of strings\n",
            "- constraints: array of strings\n",
            "- dependencies: array of strings\n",
            "- unknown_requirements: array of strings\n",
            "- confidence: number (0.0-1.0)\n"
        );

        let resp = self
            .client
            .post(format!("{}/chat/completions", self.config.base_url))
            .header(
                "Authorization",
                format!(
                    "Bearer {}",
                    self.config.openrouter_key.as_deref().unwrap_or("")
                ),
            )
            .json(&serde_json::json!({
                "model": &self.config.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]
            }))
            .send()
            .await
            .map_err(|e| format!("HTTP error: {}", e))?;

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("Parse error: {}", e))?;
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .ok_or("No content in response")?;

        let cleaned = strip_json_fences(content);
        serde_json::from_str::<StructuredIntent>(&cleaned)
            .map_err(|e| format!("Intent parse error: {}", e))
    }
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
pub fn config_path() -> Result<String, String> {
    dirs::config_dir()
        .map(|p| {
            p.join("grounding-coder")
                .join("config.json")
                .to_string_lossy()
                .to_string()
        })
        .ok_or_else(|| "Cannot determine config directory".to_string())
}

/// Load config from default path.
pub fn load_config_default() -> ApiConfig {
    load_config(&config_path().unwrap_or_default())
}

/// Save config to file (e.g., for settings panel).
pub fn save_config(config: &ApiConfig, path: &str) -> Result<(), String> {
    let json =
        serde_json::to_string_pretty(config).map_err(|e| format!("Serialize error: {}", e))?;
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
