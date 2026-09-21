// Android project workspace and manifest management
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectManifest {
    pub project: ProjectInfo,
    pub runtime: RuntimeInfo,
    pub capabilities: Vec<String>,
    pub domains: Vec<String>,
    pub constraints: Vec<String>,
    pub dependencies: Vec<String>,
    pub source_hash: String,
    pub created_at: u64,
    pub last_verified: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectInfo {
    pub platform: String,
    pub architecture: String,
    pub languages: Vec<String>,
    pub name: String,
    pub identifier: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeInfo {
    pub min_sdk: u32,
    pub target_sdk: u32,
    pub package: String,
    pub version_code: u32,
    pub version_name: String,
}

pub fn load_project_manifest(project_path: &str) -> Option<ProjectManifest> {
    let manifest_path = format!("{}/project_manifest.json", project_path);
    match std::fs::read_to_string(&manifest_path) {
        Ok(content) => {
            match serde_json::from_str(&content) {
                Ok(manifest) => Some(manifest),
                Err(e) => {
                    log::warn!("Failed to parse manifest: {}", e);
                    None
                }
            }
        }
        Err(_) => None,
    }
}

pub fn create_project_manifest(
    project_path: &str,
    platform: &str,
    architecture: &str,
    runtime: &crate::android::runtime::AndroidRuntime,
    capabilities: &[String],
    domains: &[String],
    constraints: &[String],
    dependencies: &[String],
) -> Result<(), String> {
    let manifest = ProjectManifest {
        project: ProjectInfo {
            platform: platform.to_string(),
            architecture: architecture.to_string(),
            languages: vec!["rust".to_string(), "kotlin".to_string()],
            name: "grounding-coder".to_string(),
            identifier: runtime.package.clone(),
        },
        runtime: RuntimeInfo {
            min_sdk: runtime.min_sdk,
            target_sdk: runtime.target_sdk,
            package: runtime.package.clone(),
            version_code: 1,
            version_name: "0.1.0".to_string(),
        },
        capabilities: capabilities.to_vec(),
        domains: domains.to_vec(),
        constraints: constraints.to_vec(),
        dependencies: dependencies.to_vec(),
        source_hash: "initial".to_string(),
        created_at: chrono::Utc::now().timestamp() as u64,
        last_verified: chrono::Utc::now().timestamp() as u64,
    };

    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| format!("Failed to serialize manifest: {}", e))?;

    let manifest_path = format!("{}/project_manifest.json", project_path);
    std::fs::write(&manifest_path, manifest_json)
        .map_err(|e| format!("Failed to write manifest: {}", e))?;

    log::info!("Created project manifest at {}", manifest_path);
    Ok(())
}

pub fn validate_project_manifest(manifest: &ProjectManifest) -> bool {
    if manifest.project.platform != "android" && manifest.project.platform != "desktop" {
        log::warn!("Unsupported platform: {}", manifest.project.platform);
        return false;
    }

    if manifest.project.architecture != "native" && manifest.project.architecture != "cross-platform" {
        log::warn!("Unsupported architecture: {}", manifest.project.architecture);
        return false;
    }

    if manifest.project.languages.is_empty() {
        log::warn!("No languages specified");
        return false;
    }

    true
}

pub fn has_valid_project(project_path: &str) -> bool {
    let manifest_path = format!("{}/project_manifest.json", project_path);
    if !std::path::Path::new(&manifest_path).exists() {
        return false;
    }

    match load_project_manifest(project_path) {
        Some(manifest) => validate_project_manifest(&manifest),
        None => false,
    }
}

pub fn initialize_android_project(project_path: &str) -> Result<(), String> {
    let runtime = crate::android::runtime::initialize_android_runtime();
    let capabilities = crate::android::capabilities::get_android_capabilities()
        .iter()
        .filter(|c| c.available)
        .map(|c| c.name.clone())
        .collect::<Vec<_>>();
    let domains: Vec<String> = vec![];
    let constraints = vec![
        "no_private_keys_in_logs".to_string(),
        "encrypted_secrets".to_string(),
        "offline_safe".to_string(),
    ];
    let dependencies = vec![
        "serde".to_string(),
        "serde_json".to_string(),
        "tokio".to_string(),
    ];

    create_project_manifest(
        project_path,
        "android",
        "native",
        &runtime,
        &capabilities,
        &domains,
        &constraints,
        &dependencies,
    )
}
