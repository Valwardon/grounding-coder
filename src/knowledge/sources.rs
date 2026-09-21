// Source management for CastleStore
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceEntry {
    pub url: String,
    pub content_hash: String,
    pub version: String,
    pub retrieved_at: u64,
    pub compressed_content: String,
    pub source_type: SourceType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SourceType {
    DocsRs,
    CratesIo,
    AndroidSDK,
    KotlinLang,
    GitHub,
    OfficialDocs,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceCache {
    pub entries: HashMap<String, SourceEntry>,
    pub max_entries: usize,
    pub max_content_size: usize,
}

impl SourceCache {
    pub fn new() -> Self {
        SourceCache {
            entries: HashMap::new(),
            max_entries: 1000,
            max_content_size: 10 * 1024 * 1024, // 10MB
        }
    }

    pub fn add_source(&mut self, url: String, content_hash: String, version: String, content: String) {
        let entry = SourceEntry {
            url: url.clone(),
            content_hash,
            version,
            retrieved_at: chrono::Utc::now().timestamp() as u64,
            compressed_content: content,
            source_type: SourceType::DocsRs, // Simplified - would be detected from URL
        };

        self.entries.insert(url, entry);

        // Clean up old entries if needed
        if self.entries.len() > self.max_entries {
            let keys: Vec<String> = self.entries.keys().take(self.entries.len() - self.max_entries + 1).cloned().collect();
            for key in keys {
                self.entries.remove(&key);
            }
        }
    }

    pub fn get_source(&self, url: &str) -> Option<&SourceEntry> {
        self.entries.get(url)
    }

    pub fn has_source(&self, url: &str) -> bool {
        self.entries.contains_key(url)
    }

    pub fn get_compressed_content(&self, url: &str) -> Option<String> {
        self.entries.get(url).map(|entry| entry.compressed_content.clone())
    }
}

pub fn compress_content(content: &str) -> String {
    // Simple compression: remove extra whitespace and newlines
    // In a real implementation, would use proper compression algorithms
    let cleaned = content.replace(&['\n', '\r', '\t'][..], " ")
        .replace("  ", " ")
        .replace("   ", " ")
        .replace("    ", " ")
        .replace("     ", " ")
        .trim()
        .to_string();

    // Limit size
    if cleaned.len() > 10_000 {
        cleaned[..10_000].to_string()
    } else {
        cleaned
    }
}
