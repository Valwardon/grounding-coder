//! The catalog: researched definitions the bot banked for later.
//!
//! Receive → catalog → utilize. When repair hits an unknown identifier,
//! the ResearchOracle fetches a compiler-verified definition from a
//! canonical source; the catalog persists it per-project
//! (`.grounding/catalog.jsonl`, capped) and indexes it into the symbol
//! table, so Step 0 of future runs already knows it. Utilization goes
//! through the normal `AddImport` path — the compiler judges whether
//! the import resolves anything. Nothing here authors code; the file
//! only remembers verified facts.

use super::research::ResearchedDef;

const MAX_ENTRIES: usize = 500;

fn path_for(project_dir: &std::path::Path) -> std::path::PathBuf {
    project_dir.join(".grounding").join("catalog.jsonl")
}

/// Load cataloged definitions (missing/corrupt file = cold start).
pub fn load(project_dir: &std::path::Path) -> Vec<ResearchedDef> {
    let Ok(content) = std::fs::read_to_string(path_for(project_dir)) else {
        return Vec::new();
    };
    content
        .lines()
        .filter_map(|l| serde_json::from_str::<ResearchedDef>(l).ok())
        .collect()
}

/// Append definitions, deduplicated by (qname, source_url), capped.
pub fn save(project_dir: &std::path::Path, defs: &[ResearchedDef]) {
    if defs.is_empty() {
        return;
    }
    let dir = project_dir.join(".grounding");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let mut existing = load(project_dir);
    for d in defs {
        if !existing
            .iter()
            .any(|e| e.qname == d.qname && e.source_url == d.source_url)
        {
            existing.push(d.clone());
        }
    }
    if existing.len() > MAX_ENTRIES {
        let drop = existing.len() - MAX_ENTRIES;
        existing.drain(..drop);
    }
    let mut out = String::new();
    for d in &existing {
        if let Ok(line) = serde_json::to_string(d) {
            out.push_str(&line);
            out.push('\n');
        }
    }
    let _ = std::fs::write(path_for(project_dir), out);
}

/// Extract an unknown identifier from a diagnostic, if it names one.
/// Handles unresolved imports/paths (E0432/E0433/E0412/E0437), missing
/// methods (E0599 `no method named`), and unknown-symbol kinds. Only
/// the first backticked ident is taken; anything else refuses.
pub fn unknown_ident(code: &str, message: &str, kind: &super::error::ErrorKind) -> Option<String> {
    use super::error::ErrorKind as K;
    let relevant = matches!(
        kind,
        K::UnresolvedSymbol | K::MissingImport | K::MissingFieldMethod
    ) || matches!(
        code,
        "E0432" | "E0433" | "E0412" | "E0437" | "E0599" | "E0603"
    );
    if !relevant {
        return None;
    }
    let start = message.find('`')? + 1;
    let name: String = message[start..].chars().take_while(|c| *c != '`').collect();
    // Methods (`no method named`) and paths (`a::b::c`) reduce to their
    // last segment for research; the full shape stays in the message.
    let short = name.rsplit("::").next().unwrap_or(&name).to_string();
    if short.is_empty() {
        return None;
    }
    let ok = short
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '<' || c == '>' || c == ',');
    if !ok || short.trim().is_empty() {
        return None;
    }
    Some(short)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::error::ErrorKind;

    #[test]
    fn extracts_first_backticked_ident() {
        let m = unknown_ident(
            "E0433",
            "failed to resolve: use of undeclared crate or module `serde_json`",
            &ErrorKind::UnresolvedSymbol,
        );
        assert_eq!(m, Some("serde_json".to_string()));
    }

    #[test]
    fn reduces_paths_to_last_segment() {
        let m = unknown_ident(
            "E0599",
            "no method named `read` found for struct `std::io::Stdin`",
            &ErrorKind::MissingFieldMethod,
        );
        assert_eq!(m, Some("read".to_string()));
    }

    #[test]
    fn refuses_irrelevant_codes() {
        assert_eq!(
            unknown_ident("E0596", "cannot borrow `x` as mutable", &ErrorKind::Other),
            None
        );
        assert_eq!(
            unknown_ident("E0308", "mismatched `u32` types", &ErrorKind::Other),
            None
        );
    }

    #[test]
    fn catalog_roundtrip_dedups() {
        let dir = std::env::temp_dir().join(format!(
            "gc-cat-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let def = ResearchedDef {
            qname: "std::collections::HashMap".to_string(),
            language: "rust".to_string(),
            kind: "struct".to_string(),
            module: "std::collections".to_string(),
            signature: "pub struct HashMap<K, V>".to_string(),
            description: "a map".to_string(),
            examples: Vec::new(),
            source_url: "https://docs.rs/x".to_string(),
            compiler_verified: true,
        };
        save(&dir, &[def.clone(), def]);
        let back = load(&dir);
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].qname, "std::collections::HashMap");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
