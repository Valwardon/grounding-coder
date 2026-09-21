// Knowledge adapters for different sources
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeResult {
    pub symbols: Vec<CodeSymbolInfo>,
    pub facts: Vec<VerifiedFact>,
    pub patterns: Vec<CodePattern>,
    pub source_urls: Vec<String>,
    pub compiler_verified: bool,
    pub retrieval_time: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSymbolInfo {
    pub qname: String,
    pub language: String,
    pub kind: String,
    pub module: String,
    pub signature: String,
    pub source_urls: Vec<String>,
    pub compiler_verified: bool,
    pub api_level: Option<u32>,
    pub patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedFact {
    pub content: String,
    pub confidence: f64,
    pub source_urls: Vec<String>,
    pub compiler_verified: bool,
    pub last_used: u64,
    pub utility: f64,
    pub usage_count: u32,
    pub dependency_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodePattern {
    pub name: String,
    pub description: String,
    pub language: String,
    pub pattern_type: PatternType,
    pub confidence: f64,
    pub usage_count: u32,
    pub source_urls: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PatternType {
    AndroidIntegration,
    SolanaRpc,
    ErrorHandling,
    BackgroundTask,
    Networking,
    Testing,
    TradingStrategy,
    DataAccess,
    Security,
    Performance,
}

/// Main knowledge oracle trait
pub trait KnowledgeAdapter {
    /// Get the name of this oracle
    fn name(&self) -> &str;

    /// Get the priority of this oracle (higher = tried first)
    fn priority(&self) -> u32;

    /// Get the language this oracle supports
    fn language(&self) -> &str;

    /// Research a symbol from this oracle
    async fn research(&self, symbol: &str) -> Option<KnowledgeResult>;

    /// Check if this oracle can handle the symbol
    fn can_handle(&self, symbol: &str) -> bool;
}

/// Generic knowledge oracle implementation
pub struct KnowledgeOracle {
    adapters: Vec<Box<dyn KnowledgeAdapter>>,
}

impl KnowledgeOracle {
    pub fn new() -> Self {
        KnowledgeOracle {
            adapters: Vec::new(),
        }
    }

    pub fn add_adapter<T: KnowledgeAdapter + 'static>(&mut self, adapter: T) {
        self.adapters.push(Box::new(adapter));
    }

    /// Research a symbol from all available adapters
    pub async fn research(&mut self, symbol: &str, language: &str) -> Option<KnowledgeResult> {
        for adapter in &self.adapters {
            if adapter.language() == language || adapter.language() == "multi" {
                if let Some(result) = adapter.research(symbol).await {
                    return Some(result);
                }
            }
        }
        None
    }

    /// Get all available adapters
    pub fn get_adapters(&self) -> Vec<String> {
        self.adapters.iter().map(|a| a.name().to_string()).collect()
    }
}
