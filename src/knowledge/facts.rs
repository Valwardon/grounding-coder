// Fact management for CastleStore
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedFact {
    pub qname: String,
    pub language: String,
    pub kind: String,
    pub module: String,
    pub signature: String,
    pub description: String,
    pub examples: Vec<String>,
    pub source_urls: Vec<String>,
    pub compiler_verified: bool,
    pub source_url: String,
    pub confidence: f64,
    pub last_used: u64,
    pub utility: f64,
    pub usage_count: u32,
    pub dependency_count: u32,
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

pub fn extract_facts_from_researched(researched: &[crate::engine::research::ResearchedDef]) -> Vec<VerifiedFact> {
    let mut facts = Vec::new();

    for def in researched {
        let mut fact = VerifiedFact {
            qname: def.qname.clone(),
            language: def.language.clone(),
            kind: def.kind.clone(),
            module: def.module.clone(),
            signature: def.signature.clone(),
            description: def.description.clone(),
            examples: def.examples.clone(),
            source_urls: vec![def.source_url.clone()],
            compiler_verified: def.compiler_verified,
            source_url: def.source_url.clone(),
            confidence: if def.compiler_verified { 1.0 } else { 0.5 },
            last_used: chrono::Utc::now().timestamp() as u64,
            utility: 1.0,
            usage_count: 0,
            dependency_count: 0,
        };

        // Extract key facts based on content
        if !def.signature.is_empty() {
            fact.description = format!("API: {} in {}", def.signature, def.module);
        }

        if def.kind == "function" || def.kind == "trait" {
            fact.examples.push(format!("// Example usage: Not available"));
        }

        facts.push(fact);
    }

    facts
}

pub fn extract_symbols_from_researched(researched: &[crate::engine::research::ResearchedDef]) -> Vec<CodeSymbolInfo> {
    let mut symbols = Vec::new();

    for def in researched {
        let symbol = CodeSymbolInfo {
            qname: def.qname.clone(),
            language: def.language.clone(),
            kind: def.kind.clone(),
            module: def.module.clone(),
            signature: def.signature.clone(),
            source_urls: vec![def.source_url.clone()],
            compiler_verified: def.compiler_verified,
            api_level: None, // Would be extracted from source if available
            patterns: Vec::new(),
        };

        symbols.push(symbol);
    }

    symbols
}

pub fn extract_patterns_from_researched(researched: &[crate::engine::research::ResearchedDef]) -> Vec<CodePattern> {
    let mut patterns = Vec::new();

    for def in researched {
        let pattern_type = match def.kind.as_str() {
            "function" => PatternType::DataAccess,
            "trait" => PatternType::DataAccess,
            "struct" => PatternType::AndroidIntegration,
            _ => PatternType::DataAccess,
        };

        let pattern = CodePattern {
            name: format!("{}", def.qname),
            description: def.description.clone(),
            language: def.language.clone(),
            pattern_type,
            confidence: if def.compiler_verified { 1.0 } else { 0.5 },
            usage_count: 0,
            source_urls: vec![def.source_url.clone()],
        };

        patterns.push(pattern);
    }

    patterns
}
