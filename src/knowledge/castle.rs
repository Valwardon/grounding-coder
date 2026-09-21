// CastleStore — engineering knowledge base with compression and retrieval
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CastleStore {
    pub facts: HashMap<String, Vec<VerifiedFact>>, // concept -> facts
    pub symbols: HashMap<String, CodeSymbolInfo>,    // qualified name -> symbol info
    pub patterns: HashMap<String, CodePattern>,     // pattern name -> pattern info
    pub recipes: Vec<String>,                       // learned fix recipes
    pub source_hashes: HashMap<String, String>,     // URL -> content hash
    pub verification_results: HashMap<String, VerificationResult>, // symbol -> verification
    pub projects: Vec<ProjectMetadata>,            // project info
    pub tasks: Vec<TaskMetadata>,                  // task history
    pub max_size: usize,
    pub pruning_threshold: usize,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub symbol: String,
    pub verified: bool,
    pub compiler_output: String,
    pub verification_time: u64,
    pub success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMetadata {
    pub path: String,
    pub name: String,
    pub languages: Vec<String>,
    pub capabilities: Vec<String>,
    pub domains: Vec<String>,
    pub created_at: u64,
    pub last_used: u64,
    pub tasks_completed: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetadata {
    pub id: u64,
    pub description: String,
    pub completed_at: u64,
    pub success: bool,
    pub knowledge_added: Vec<String>, // fact keys added
    pub patterns_used: Vec<String>,  // pattern names used
}

impl CastleStore {
    pub fn new() -> Self {
        CastleStore {
            facts: HashMap::new(),
            symbols: HashMap::new(),
            patterns: HashMap::new(),
            recipes: Vec::new(),
            source_hashes: HashMap::new(),
            verification_results: HashMap::new(),
            projects: Vec::new(),
            tasks: Vec::new(),
            max_size: 10_000, // Max facts
            pruning_threshold: 5_000, // Start pruning when exceeding
        }
    }

    pub fn add_fact(&mut self, fact: VerifiedFact) {
        let concept = fact.content.split('\n').next().unwrap_or("unknown").to_string();
        self.facts
            .entry(concept)
            .or_default()
            .push(fact);

        self.prune_facts();
    }

    pub fn add_symbol(&mut self, symbol: CodeSymbolInfo) {
        self.symbols.insert(symbol.qname.clone(), symbol);
    }

    pub fn add_pattern(&mut self, pattern: CodePattern) {
        self.patterns.insert(pattern.name.clone(), pattern);
    }

    pub fn add_recipe(&mut self, recipe: String) {
        self.recipes.push(recipe);
        if self.recipes.len() > 1000 {
            self.recipes.drain(0..100);
        }
    }

    pub fn search_facts(&self, query: &str) -> Vec<VerifiedFact> {
        let mut results = Vec::new();
        for facts in self.facts.values() {
            for fact in facts {
                if fact.content.contains(query) || fact.source_urls.iter().any(|url| url.contains(query)) {
                    results.push(fact.clone());
                }
            }
        }
        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    pub fn search_patterns(&self, language: &str, domain: &str) -> Vec<CodePattern> {
        let mut results = Vec::new();
        for pattern in self.patterns.values() {
            if pattern.language == language && pattern.usage_count > 0 {
                results.push(pattern.clone());
            }
        }
        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    pub fn get_relevant_context(&self, query: &str, max_facts: usize, max_patterns: usize) -> String {
        let mut context = String::new();

        // Add relevant facts
        let facts = self.search_facts(query);
        for fact in facts.iter().take(max_facts) {
            context.push_str(&format!(
                "Fact: {}\nConfidence: {}\nSources: {}\n\n",
                fact.content,
                fact.confidence,
                fact.source_urls.join(", ")
            ));
        }

        // Add relevant patterns
        context.push_str("\n--- Patterns ---\n");
        for pattern in self.search_patterns("rust", "trading")
            .iter()
            .take(max_patterns) {
            context.push_str(&format!(
                "Pattern: {}\nType: {:?} \nConfidence: {}\n\n",
                pattern.description,
                pattern.pattern_type,
                pattern.confidence
            ));
        }

        context
    }

    fn prune_facts(&mut self) {
        if self.facts.len() <= self.pruning_threshold {
            return;
        }

        // Sort by utility * confidence * last_used
        let mut all_facts: Vec<(String, usize, VerifiedFact)> = Vec::new();
        for (concept, facts) in &self.facts {
            for (idx, fact) in facts.iter().enumerate() {
                let score = fact.utility * fact.confidence * (fact.last_used as f64);
                all_facts.push((concept.clone(), idx, fact.clone()));
            }
        }

        all_facts.sort_by(|a, b| {
            let score_a = a.2.utility * a.2.confidence * (a.2.last_used as f64);
            let score_b = b.2.utility * b.2.confidence * (b.2.last_used as f64);
            b.2.compiler_verified.cmp(&a.2.compiler_verified).then_with(|| score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal))
        });

        // Remove lowest-scoring facts
        let remove_count = self.facts.len() - self.max_size / 2;
        for (concept, idx, _) in all_facts.iter().take(remove_count) {
            if let Some(facts) = self.facts.get_mut(*concept) {
                facts.remove(*idx);
                if facts.is_empty() {
                    self.facts.remove(*concept);
                }
            }
        }
    }
}

/// Load CastleStore from SQLite database
pub fn load_castle_store(db_path: &str) -> CastleStore {
    let mut store = CastleStore::new();

    if let Ok(conn) = rusqlite::Connection::open(db_path) {
        // Load facts
        let mut stmt = conn.prepare("SELECT concept, content, confidence, source_urls, compiler_verified, last_used, utility, usage_count, dependency_count FROM facts").unwrap();
        let facts_iter = stmt.query_map([], |row| {
            let concept: String = row.get(0)?;
            let content: String = row.get(1)?;
            let confidence: f64 = row.get(2)?;
            let source_urls: String = row.get(3)?;
            let compiler_verified: bool = row.get(4)?;
            let last_used: u64 = row.get(5)?;
            let utility: f64 = row.get(6)?;
            let usage_count: u32 = row.get(7)?;
            let dependency_count: u32 = row.get(8)?;

            let source_urls_vec: Vec<String> = if source_urls.is_empty() {
                Vec::new()
            } else {
                source_urls.split(';').map(|s| s.to_string()).collect()
            };

            Ok((concept, VerifiedFact {
                content,
                confidence,
                source_urls: source_urls_vec,
                compiler_verified,
                last_used,
                utility,
                usage_count,
                dependency_count,
            }))
        }).unwrap();

        for row_result in facts_iter {
            if let Ok((concept, fact)) = row_result {
                store.add_fact(fact);
            }
        }

        // Load symbols
        let mut stmt = conn.prepare("SELECT qname, language, kind, module, signature, source_urls, compiler_verified, api_level FROM symbols").unwrap();
        let symbols_iter = stmt.query_map([], |row| {
            let qname: String = row.get(0)?;
            let language: String = row.get(1)?;
            let kind: String = row.get(2)?;
            let module: String = row.get(3)?;
            let signature: String = row.get(4)?;
            let source_urls: String = row.get(5)?;
            let compiler_verified: bool = row.get(6)?;
            let api_level: i64 = row.get(7)?;

            let source_urls_vec: Vec<String> = if source_urls.is_empty() {
                Vec::new()
            } else {
                source_urls.split(';').map(|s| s.to_string()).collect()
            };

            Ok(CodeSymbolInfo {
                qname,
                language,
                kind,
                module,
                signature,
                source_urls: source_urls_vec,
                compiler_verified,
                api_level: if api_level > 0 { Some(api_level as u32) } else { None },
                patterns: Vec::new(),
            })
        }).unwrap();

        for row_result in symbols_iter {
            if let Ok(symbol) = row_result {
                store.add_symbol(symbol);
            }
        }
    }

    store
}

/// Save CastleStore to SQLite database
pub fn save_castle_store(store: &CastleStore, db_path: &str) -> Result<(), String> {
    let conn = rusqlite::Connection::open(db_path)
        .map_err(|e| format!("Failed to open database: {}", e))?;

    // Create tables
    conn.execute(
        "CREATE TABLE IF NOT EXISTS facts (
            concept TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL NOT NULL,
            source_urls TEXT NOT NULL,
            compiler_verified BOOLEAN NOT NULL,
            last_used INTEGER NOT NULL,
            utility REAL NOT NULL,
            usage_count INTEGER NOT NULL,
            dependency_count INTEGER NOT NULL
        )",
        [],
    ).map_err(|e| format!("Failed to create facts table: {}", e))?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS symbols (
            qname TEXT PRIMARY KEY NOT NULL,
            language TEXT NOT NULL,
            kind TEXT NOT NULL,
            module TEXT NOT NULL,
            signature TEXT NOT NULL,
            source_urls TEXT NOT NULL,
            compiler_verified BOOLEAN NOT NULL,
            api_level INTEGER
        )",
        [],
    ).map_err(|e| format!("Failed to create symbols table: {}", e))?;

    // Clear and repopulate facts
    conn.execute("DELETE FROM facts", []).unwrap();
    for (concept, facts) in &store.facts {
        for fact in facts {
            let source_urls = fact.source_urls.join(";");
            conn.execute(
                "INSERT INTO facts (concept, content, confidence, source_urls, compiler_verified, last_used, utility, usage_count, dependency_count) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![concept, fact.content, fact.confidence, source_urls, fact.compiler_verified, fact.last_used, fact.utility, fact.usage_count, fact.dependency_count],
            ).map_err(|e| format!("Failed to insert fact: {}", e))?;
        }
    }

    // Clear and repopulate symbols
    conn.execute("DELETE FROM symbols", []).unwrap();
    for symbol in store.symbols.values() {
        let source_urls = symbol.source_urls.join(";");
        conn.execute(
            "INSERT INTO symbols (qname, language, kind, module, signature, source_urls, compiler_verified, api_level) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![symbol.qname, symbol.language, symbol.kind, symbol.module, symbol.signature, source_urls, symbol.compiler_verified, symbol.api_level.unwrap_or(-1)],
        ).map_err(|e| format!("Failed to insert symbol: {}", e))?;
    }

    log::info!("Saved CastleStore to {} ({} facts, {} symbols)", db_path, store.facts.values().map(|v| v.len()).sum::<usize>(), store.symbols.len());
    Ok(())
}
