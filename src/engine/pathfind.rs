//! Pathfinding: from point A (an unknown identifier in a diagnostic)
//! to point B (an importable path for it) without guessing.
//!
//! Order is cost-first and fully deterministic:
//! 1. The arena — symbols the bot already indexed from the project.
//! 2. Dependency sources — `pub` items in the project's locked deps,
//!    with module paths reconstructed from file layout.
//! 3. (Elsewhere, not here) the network oracle for crates/symbols
//!    outside the reachable tree.
//!
//! Every hit carries file:line evidence. Caps bound every walk; the
//! compiler still judges every derived import next round.

use super::arena::{CodeArena, CodeSymbol, SymbolKind};

/// One located definition plus how to import it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathHit {
    /// Import path, e.g. `std::io::Read` or `mycrate::net::Client`.
    pub import_path: String,
    /// Evidence: `file:line` where the definition was found.
    pub evidence: String,
}

const MAX_HITS: usize = 8;
const MAX_SCANNED_FILES: usize = 1500;

/// Arena lookup: qnames ending in `::name` (or equal), importable
/// kinds only, with file evidence. Private/local items never qualify.
pub fn find_in_arena(arena: &CodeArena, name: &str) -> Vec<PathHit> {
    let mut out = Vec::new();
    for id in arena.lookup_simple(name) {
        let Some(node) = arena.get(id) else {
            continue;
        };
        let sym: CodeSymbol = node.read().clone();
        if !matches!(
            sym.kind,
            SymbolKind::Struct
                | SymbolKind::Enum
                | SymbolKind::Function
                | SymbolKind::Trait
                | SymbolKind::Module
                | SymbolKind::Const
        ) {
            continue;
        }
        if !(sym.qname == name || sym.qname.ends_with(&format!("::{}", name))) {
            continue;
        }
        out.push(PathHit {
            import_path: sym.qname.clone(),
            evidence: format!("arena {}:{}", sym.file_path, sym.line),
        });
        if out.len() >= MAX_HITS {
            break;
        }
    }
    out.sort_by(|a, b| a.import_path.cmp(&b.import_path));
    out
}

/// Dependency sources: for each locked dependency, scan its sources
/// for `pub` items named `name` and reconstruct the import path from
/// file layout (`src/foo/bar.rs` → `crate_name::foo::bar::Name`).
/// Method names additionally match the enclosing `pub trait` (the
/// classic missing-import fix). All bounded.
pub fn find_in_deps(project_dir: &std::path::Path, name: &str) -> Vec<PathHit> {
    let mut out = Vec::new();
    let mut scanned = 0usize;
    for (crate_name, crate_dir) in dep_crate_dirs(project_dir) {
        if out.len() >= MAX_HITS {
            break;
        }
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        collect_rs(&crate_dir, &mut files, &mut scanned);
        files.sort();
        for file in files {
            if out.len() >= MAX_HITS {
                break;
            }
            let Ok(content) = std::fs::read_to_string(&file) else {
                continue;
            };
            let rel = file
                .strip_prefix(&crate_dir)
                .unwrap_or(&file)
                .to_string_lossy()
                .to_string();
            // Item definitions: `pub (fn|struct|trait|enum|mod|const|use) NAME`.
            for (i, line) in content.lines().enumerate() {
                let t = line.trim_start();
                if !t.starts_with("pub ") && !t.starts_with("pub(") {
                    continue;
                }
                if let Some(item) = defined_item(t)
                    && item.0 == name
                {
                    let path = module_path(&crate_name, &rel, &item.0);
                    out.push(PathHit {
                        import_path: path,
                        evidence: format!("{}:{}", rel, i + 1),
                    });
                    if out.len() >= MAX_HITS {
                        break;
                    }
                }
            }
            // Trait containers for method names: smallest enclosing
            // `pub trait T` above a `fn name(` line (cap 120 lines up).
            if out.len() < MAX_HITS {
                let lines: Vec<&str> = content.lines().collect();
                for (i, line) in lines.iter().enumerate() {
                    if !line.contains(&format!("fn {}(", name)) {
                        continue;
                    }
                    // Nearest `pub trait` at or above (cap 120 lines
                    // up): single-line `trait T { fn m(); }` shapes keep
                    // both on one line. Nested items can misattribute;
                    // the compiler judges the import next round, so a
                    // wrong guess only costs one.
                    let mut found_trait: Option<String> = None;
                    let mut j = i + 1;
                    while j > 0 && i + 1 - j <= 120 {
                        j -= 1;
                        let t = lines[j].trim_start();
                        if !t.starts_with("pub trait ") {
                            continue;
                        }
                        let tname: String = t["pub trait ".len()..]
                            .chars()
                            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                            .collect();
                        if !tname.is_empty() {
                            found_trait = Some(tname);
                        }
                        break;
                    }
                    if let Some(tname) = found_trait {
                        let path = module_path(&crate_name, &rel, &tname);
                        out.push(PathHit {
                            import_path: path,
                            evidence: format!("{}:{}", rel, j + 1),
                        });
                        if out.len() >= MAX_HITS {
                            break;
                        }
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| a.import_path.cmp(&b.import_path));
    out.dedup_by(|a, b| a.import_path == b.import_path);
    out
}

/// `(name, kind-word)` from a `pub …` line, e.g.
/// `pub fn foo(` → `("foo", "fn")`. Re-exports (`pub use a::b::Foo;`)
/// resolve to the final segment.
fn defined_item(line: &str) -> Option<(String, &'static str)> {
    let t = line.trim_start();
    let rest = t
        .strip_prefix("pub(crate) ")
        .or_else(|| t.strip_prefix("pub(super) "))
        .or_else(|| t.strip_prefix("pub "))?;
    // pub(...) restricted visibility still counts as importable here;
    // the compiler judges actual reachability next round.
    let rest = if rest.starts_with('(') {
        let end = rest.find(')')?;
        rest[end + 1..].trim_start()
    } else {
        rest
    };
    for (kw, kind) in [
        ("fn ", "fn"),
        ("struct ", "struct"),
        ("trait ", "trait"),
        ("enum ", "enum"),
        ("mod ", "mod"),
        ("const ", "const"),
        ("static ", "static"),
        ("type ", "type"),
        ("use ", "use"),
    ] {
        if let Some(after) = rest.strip_prefix(kw) {
            // For `use a::b::Name;` take the final segment.
            // Relative re-exports (`self::`, `super::`, `crate::`)
            // are not cross-crate import candidates.
            if kind == "use"
                && (after.starts_with("self::")
                    || after.starts_with("super::")
                    || after.starts_with("crate::"))
            {
                return None;
            }
            let name: String = if kind == "use" {
                after
                    .trim_end_matches([';', ' ', '\t'])
                    .rsplit("::")
                    .next()
                    .unwrap_or("")
                    .trim_end_matches('}')
                    .trim()
                    .to_string()
            } else {
                after
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect()
            };
            if name.is_empty() || name == "self" || name == "super" || name == "crate" {
                return None;
            }
            return Some((name, kind));
        }
    }
    None
}

/// Module path from crate layout: `src/a/b.rs` → `krate::a::b::Name`,
/// `src/a/mod.rs` → `krate::a::Name`, `src/lib.rs` → `krate::Name`.
fn module_path(crate_name: &str, rel: &str, name: &str) -> String {
    let krate = crate_name.replace('-', "_");
    let mut segs: Vec<String> = rel
        .trim_start_matches("src/")
        .trim_end_matches(".rs")
        .split('/')
        .map(|s| s.to_string())
        .collect();
    if segs.last().map(|s| s.as_str()) == Some("mod") {
        segs.pop();
    }
    if segs == ["lib"] || segs.is_empty() {
        segs.clear();
    }
    // Re-exported names (`pub use`) live where they are written.
    if segs.is_empty() {
        format!("{}::{}", krate, name)
    } else {
        format!("{}::{}::{}", krate, segs.join("::"), name)
    }
}

/// Locked dependency crate dirs: `[[package]]` names from Cargo.lock
/// mapped into the cargo registry (plus path deps from the manifest).
/// Best-effort throughout — unresolvable entries are skipped, never fatal.
fn dep_crate_dirs(project_dir: &std::path::Path) -> Vec<(String, std::path::PathBuf)> {
    let mut out = Vec::new();
    let lock = project_dir.join("Cargo.lock");
    let Ok(content) = std::fs::read_to_string(&lock) else {
        return out;
    };
    // (name, registry version): registry entries carry a `source`
    // line; path/workspace deps don't (their `version` lines are
    // ignored — only the manifest path resolves them).
    let mut names: Vec<(String, Option<String>)> = Vec::new();
    let mut cur_name: Option<String> = None;
    let mut cur_version: Option<String> = None;
    let mut cur_source = false;
    for line in content.lines() {
        let t = line.trim();
        if t == "[[package]]" {
            // Previous package (if any) is complete: its lines all
            // precede this header.
            if let Some(name) = cur_name.take() {
                let v = if cur_source { cur_version.take() } else { None };
                names.push((name, v));
            }
            cur_version = None;
            cur_source = false;
            continue;
        }
        if let Some(v) = t.strip_prefix("name = ") {
            cur_name = Some(v.trim_matches('"').to_string());
        } else if let Some(v) = t.strip_prefix("version = ") {
            cur_version = Some(v.trim_matches('"').to_string());
        } else if t.starts_with("source = ") {
            cur_source = true;
        }
    }
    if let Some(name) = cur_name.take() {
        let v = if cur_source { cur_version.take() } else { None };
        names.push((name, v));
    }
    // Registry root: CARGO_HOME or ~/.cargo.
    let cargo_home = std::env::var("CARGO_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            dirs_home()
                .map(|h| h.join(".cargo"))
                .unwrap_or_else(|| std::path::PathBuf::from("/root/.cargo"))
        });
    let index_hashes: Vec<String> = std::fs::read_dir(cargo_home.join("registry/src"))
        .ok()
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| e.path().is_dir())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect()
        })
        .unwrap_or_default();
    for (name, version) in names {
        // Registry deps have versions; path/workspace deps don't.
        if let Some(v) = version {
            for h in &index_hashes {
                let dir = cargo_home
                    .join("registry/src")
                    .join(h)
                    .join(format!("{}-{}", name, v));
                if dir.is_dir() {
                    out.push((name.clone(), dir));
                    break;
                }
            }
        } else if let Some(dir) = manifest_path_dep(project_dir, &name) {
            out.push((name, dir));
        }
        if out.len() >= 40 {
            break;
        }
    }
    out
}

/// Resolve `name = { path = "…" }` from the root manifest (best effort).
fn manifest_path_dep(project_dir: &std::path::Path, name: &str) -> Option<std::path::PathBuf> {
    let content = std::fs::read_to_string(project_dir.join("Cargo.toml")).ok()?;
    for line in content.lines() {
        let t = line.trim();
        // Match `name = { path = "dir" ...` with exact name.
        // Lines without `=` (headers, blanks) are skipped, never fatal.
        let Some((lhs, rhs)) = t.split_once('=') else {
            continue;
        };
        if lhs.trim() != name || !rhs.contains("path") {
            continue;
        }
        let start = rhs.find("path")?;
        let after = &rhs[start + 4..];
        let eq = after.find('=')?;
        // First quoted string after `path =` (values may trail more keys).
        let rest = after[eq + 1..].trim_start();
        let quote = rest.chars().next()?;
        if quote != '"' && quote != '\'' {
            return None;
        }
        let end = rest[1..].find(quote)?;
        let dir = project_dir.join(&rest[1..1 + end]);
        if dir.is_dir() {
            return Some(dir);
        }
    }
    None
}

fn dirs_home() -> Option<std::path::PathBuf> {
    std::env::var("HOME").ok().map(std::path::PathBuf::from)
}

/// Bounded recursive collection of `.rs` files.
fn collect_rs(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>, scanned: &mut usize) {
    if *scanned > MAX_SCANNED_FILES || out.len() > MAX_SCANNED_FILES {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        if *scanned > MAX_SCANNED_FILES {
            return;
        }
        let path = entry.path();
        if path.is_dir() {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if name.starts_with('.') || name == "target" || name == "tests" || name == "benches" {
                continue;
            }
            collect_rs(&path, out, scanned);
        } else if path.extension().is_some_and(|e| e == "rs") {
            *scanned += 1;
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defined_item_shapes() {
        assert_eq!(defined_item("pub fn foo("), Some(("foo".to_string(), "fn")));
        assert_eq!(
            defined_item("pub(crate) struct Bar {"),
            Some(("Bar".to_string(), "struct"))
        );
        assert_eq!(
            defined_item("pub use a::b::Baz;"),
            Some(("Baz".to_string(), "use"))
        );
        assert_eq!(defined_item("fn private("), None);
        assert_eq!(defined_item("pub use self::x;"), None);
    }

    #[test]
    fn module_path_layouts() {
        assert_eq!(
            module_path("my-crate", "src/lib.rs", "Foo"),
            "my_crate::Foo".to_string()
        );
        assert_eq!(
            module_path("my-crate", "src/net/client.rs", "Client"),
            "my_crate::net::client::Client".to_string()
        );
        assert_eq!(
            module_path("my-crate", "src/net/mod.rs", "Client"),
            "my_crate::net::Client".to_string()
        );
    }

    static TREE_CTR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

    fn fake_dep_tree() -> (std::path::PathBuf, std::path::PathBuf) {
        // Atomic counter: parallel tests must never share a scratch dir
        // (wall-clock nanos collide within one tick and readers race
        // the other test's cleanup).
        let id = TREE_CTR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let base = std::env::temp_dir().join(format!("gc-path-{}-{}", std::process::id(), id));
        let proj = base.join("proj");
        let dep = base.join("fakelib");
        std::fs::create_dir_all(dep.join("src")).unwrap();
        std::fs::write(dep.join("Cargo.toml"), "[package]\nname = \"fakelib\"\n").unwrap();
        std::fs::write(dep.join("src/lib.rs"), "pub mod net;\n").unwrap();
        std::fs::write(
            dep.join("src/net.rs"),
            "pub trait Fetch { fn fetch(&self); }\npub struct Client;\n",
        )
        .unwrap();
        std::fs::create_dir_all(proj.join("src")).unwrap();
        std::fs::write(
            proj.join("Cargo.toml"),
            "[package]\nname = \"p\"\n[dependencies]\nfakelib = { path = \"../fakelib\" }\n",
        )
        .unwrap();
        std::fs::write(
            proj.join("Cargo.lock"),
            "[[package]]\nname = \"fakelib\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        (base, proj)
    }

    #[test]
    fn finds_items_and_trait_holders_in_path_dep() {
        let (base, proj) = fake_dep_tree();
        let hits = find_in_deps(&proj, "Client");
        assert!(
            hits.iter().any(|h| h.import_path == "fakelib::net::Client"),
            "got: {:?}",
            hits
        );
        // Method name resolves to the enclosing pub trait.
        let traits = find_in_deps(&proj, "fetch");
        assert!(
            traits
                .iter()
                .any(|h| h.import_path == "fakelib::net::Fetch"),
            "got: {:?}",
            traits
        );
        // Unknown names find nothing (never an invented path).
        assert!(find_in_deps(&proj, "Nope").is_empty());
        let _ = std::fs::remove_dir_all(&base);
    }
}
