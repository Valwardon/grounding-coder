// Android-specific runtime and capability management
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndroidRuntime {
    pub min_sdk: u32,
    pub target_sdk: u32,
    pub package: String,
    pub activities: Vec<String>,
    pub permissions: Vec<String>,
    pub build_gradients: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndroidCapability {
    pub name: String,
    pub api_level: u32,
    pub description: String,
    pub available: bool,
}

pub fn initialize_android_runtime() -> AndroidRuntime {
    AndroidRuntime {
        min_sdk: 24,
        target_sdk: 34,
        package: "com.grounding.coder".to_string(),
        activities: vec!["MainActivity".to_string()],
        permissions: vec![
            "INTERNET".to_string(),
            "ACCESS_NETWORK_STATE".to_string(),
            "VIBRATE".to_string(),
            "CAMERA".to_string(),
            "RECORD_AUDIO".to_string(),
        ],
        build_gradients: [
            ("compile".to_string(), "21".to_string()),
            ("minify".to_string(), "true".to_string()),
        ].iter().cloned().collect(),
    }
}

pub fn get_android_capabilities() -> Vec<AndroidCapability> {
    vec![
        AndroidCapability {
            name: "network".to_string(),
            api_level: 1,
            description: "Network connectivity and HTTP requests".to_string(),
            available: true,
        },
        AndroidCapability {
            name: "filesystem".to_string(),
            api_level: 1,
            description: "Read/write files on device storage".to_string(),
            available: true,
        },
        AndroidCapability {
            name: "background_execution".to_string(),
            api_level: 1,
            description: "Run tasks in background".to_string(),
            available: true,
        },
        AndroidCapability {
            name: "notifications".to_string(),
            api_level: 1,
            description: "Send system notifications".to_string(),
            available: true,
        },
        AndroidCapability {
            name: "location".to_string(),
            api_level: 1,
            description: "Access device location".to_string(),
            available: true,
        },
        AndroidCapability {
            name: "camera".to_string(),
            api_level: 1,
            description: "Access camera hardware".to_string(),
            available: true,
        },
        AndroidCapability {
            name: "audio".to_string(),
            api_level: 1,
            description: "Record and play audio".to_string(),
            available: true,
        },
        AndroidCapability {
            name: "vibration".to_string(),
            api_level: 1,
            description: "Control device vibration".to_string(),
            available: true,
        },
    ]
}

pub fn validate_android_capability(capability: &str, runtime: &AndroidRuntime) -> bool {
    get_android_capabilities()
        .iter()
        .any(|c| c.name == capability && c.available)
}
