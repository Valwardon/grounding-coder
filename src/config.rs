//! App settings — everything configurable that is NOT a model.
//!
//! The translator is gone: prose reaches the engine through the
//! deterministic understander (`engine::understand`), never through an
//! external model. What remains here is device-local configuration:
//! where the project lives, how hard the repair loop may try, and the
//! GitHub token for the publish actor. Extra fields carry
//! `serde(default)` so older config files keep loading.

use serde::{Deserialize, Serialize};

fn default_project_path() -> String {
    ".".to_string()
}

fn default_max_retries() -> u32 {
    5
}

/// Device-local settings. No keys for any model — there is no model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Project directory the engine works in (device-local path).
    #[serde(default = "default_project_path")]
    pub project_path: String,
    /// Max correction attempts per task (RetryBudget).
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// GitHub token for the publish actor (stays on device).
    #[serde(default)]
    pub github_key: Option<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        AppConfig {
            project_path: default_project_path(),
            max_retries: default_max_retries(),
            github_key: None,
        }
    }
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
pub fn load_config_default() -> AppConfig {
    load_config(&config_path().unwrap_or_default())
}

/// Save config to file (e.g., for settings panel). Parent dirs are
/// created — a missing app dir must never fail a save.
pub fn save_config(config: &AppConfig, path: &str) -> Result<(), String> {
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
pub fn load_config(path: &str) -> AppConfig {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<AppConfig>(&s).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_path_resolves() {
        assert!(config_path().is_ok());
    }

    #[test]
    fn save_load_roundtrip_without_model_fields() {
        let dir = std::env::temp_dir().join(format!(
            "gc-cfg-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("config.json").to_string_lossy().to_string();
        let cfg = AppConfig {
            project_path: "/tmp/p".to_string(),
            max_retries: 7,
            github_key: Some("ghp_x".to_string()),
        };
        save_config(&cfg, &path).expect("save");
        let back = load_config(&path);
        assert_eq!(back.project_path, "/tmp/p");
        assert_eq!(back.max_retries, 7);
        // Old files carrying model keys still load (serde default).
        std::fs::write(
            &path,
            r#"{"project_path": "/tmp/q", "openrouter_key": "sk-x", "model": "m"}"#,
        )
        .unwrap();
        let old = load_config(&path);
        assert_eq!(old.project_path, "/tmp/q");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
