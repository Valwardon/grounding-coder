// Android capability registry for project manifest
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectCapability {
    pub name: String,
    pub description: String,
    pub required: bool,
    pub platform_specific: bool,
    pub api_level: u32,
    pub default_implementation: String,
    pub dependencies: Vec<String>,
}

pub fn initialize_capability_registry() -> HashMap<String, ProjectCapability> {
    let mut registry = HashMap::new();

    // Core capabilities
    registry.insert(
        "network".to_string(),
        ProjectCapability {
            name: "network".to_string(),
            description: "Network connectivity and HTTP requests".to_string(),
            required: true,
            platform_specific: false,
            api_level: 1,
            default_implementation: "crate::http (hyper + webpki roots)".to_string(),
            dependencies: vec![
                "hyper".to_string(),
                "hyper-util".to_string(),
                "hyper-rustls".to_string(),
                "tokio".to_string(),
            ],
        },
    );

    registry.insert(
        "filesystem".to_string(),
        ProjectCapability {
            name: "filesystem".to_string(),
            description: "Read/write files on device storage".to_string(),
            required: true,
            platform_specific: false,
            api_level: 1,
            default_implementation: "std::fs".to_string(),
            dependencies: vec![],
        },
    );

    registry.insert(
        "background_execution".to_string(),
        ProjectCapability {
            name: "background_execution".to_string(),
            description: "Run tasks in background".to_string(),
            required: true,
            platform_specific: true,
            api_level: 24,
            default_implementation: "android.app.Service".to_string(),
            dependencies: vec![],
        },
    );

    registry.insert(
        "notifications".to_string(),
        ProjectCapability {
            name: "notifications".to_string(),
            description: "Send system notifications".to_string(),
            required: false,
            platform_specific: true,
            api_level: 1,
            default_implementation: "android.app.NotificationManager".to_string(),
            dependencies: vec![],
        },
    );

    registry.insert(
        "location".to_string(),
        ProjectCapability {
            name: "location".to_string(),
            description: "Access device location".to_string(),
            required: false,
            platform_specific: true,
            api_level: 1,
            default_implementation: "android.location.LocationManager".to_string(),
            dependencies: vec![],
        },
    );

    registry.insert(
        "camera".to_string(),
        ProjectCapability {
            name: "camera".to_string(),
            description: "Access camera hardware".to_string(),
            required: false,
            platform_specific: true,
            api_level: 1,
            default_implementation: "android.hardware.Camera".to_string(),
            dependencies: vec![],
        },
    );

    // Domain-specific capabilities
    registry.insert(
        "solana".to_string(),
        ProjectCapability {
            name: "solana".to_string(),
            description: "Solana blockchain interaction".to_string(),
            required: false,
            platform_specific: false,
            api_level: 1,
            default_implementation: "solana_client::rpc_client::RpcClient".to_string(),
            dependencies: vec!["solana-sdk".to_string(), "solana-client".to_string()],
        },
    );

    registry.insert(
        "websocket".to_string(),
        ProjectCapability {
            name: "websocket".to_string(),
            description: "WebSocket communication".to_string(),
            required: false,
            platform_specific: false,
            api_level: 1,
            default_implementation: "async_tungstenite::connect_async".to_string(),
            dependencies: vec!["tokio-tungstenite".to_string(), "futures".to_string()],
        },
    );

    registry
}

pub fn validate_project_capability(
    capability: &str,
    registry: &HashMap<String, ProjectCapability>,
) -> bool {
    registry.contains_key(capability)
}
