//! The palace: a read-only, deterministic index over everything the
//! loop has committed. Where the stores each speak their own schema
//! (`decisions.jsonl`, `lessons.jsonl`, `encounters.jsonl`,
//! `clarifications.jsonl`, `data/visual/*.json`, the knowledge graph),
//! the palace speaks one: **wing → hall → room**, and every leaf points
//! back at its source line.
//!
//! It invents nothing. A leaf is only ever made from a record already
//! on disk, and [`Palace::unresolved`] re-reads each source to prove the
//! leaf still resolves to it — memory you cannot view is not trusted.
//! Same files in, same index out, byte for byte.

use std::path::Path;

/// Where a leaf came from: the file and the 1-based line that holds it.
/// For a whole-file JSON record (the measured plates) the line is 1 and
/// the resolver reads the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source {
    pub file: String,
    pub line: usize,
}

/// One committed record, addressed by the palace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Leaf {
    /// Which store it lives in: decisions, lessons, encounters,
    /// clarifications, visual, knowledge.
    pub wing: String,
    /// The record's kind within its store (a status, a lesson kind, a
    /// knowledge state, "measured").
    pub hall: String,
    /// The record's own key: a topic, a lesson id, an aspect, a
    /// concept, a plate title.
    pub room: String,
    pub source: Source,
}

/// The whole index, in a fixed order (store order, then file order).
#[derive(Debug, Clone, Default)]
pub struct Palace {
    pub leaves: Vec<Leaf>,
}

impl Palace {
    /// Build the index from a project root: `<project>/data/*.jsonl`,
    /// `<project>/data/visual/*.json`, `<project>/.grounding/knowledge.jsonl`.
    /// Missing files contribute nothing (an empty store is honest).
    pub fn build(project: &Path) -> Palace {
        let data = project.join("data");
        let mut leaves = Vec::new();

        // decisions: date, topic, decision, basis, evidence, alternatives, status
        index_jsonl(
            &mut leaves,
            &data.join("decisions.jsonl"),
            "decisions",
            "status",
            |v| text(v, "topic").unwrap_or_else(|| "?".to_string()),
        );

        // lessons: id, name, statement, where, evidence, kind, applies_to
        index_jsonl(
            &mut leaves,
            &data.join("lessons.jsonl"),
            "lessons",
            "kind",
            |v| text(v, "id").unwrap_or_else(|| "?".to_string()),
        );

        // encounters: kind, aspect, stimulus, answer, action, result, date
        index_jsonl(
            &mut leaves,
            &data.join("encounters.jsonl"),
            "encounters",
            "kind",
            |v| text(v, "aspect").unwrap_or_else(|| "?".to_string()),
        );

        // clarifications: kind, plate/title, aspect(s), question, answer, resolved
        index_jsonl(
            &mut leaves,
            &data.join("clarifications.jsonl"),
            "clarifications",
            "kind",
            |v| {
                first_text(v, "aspect")
                    .or_else(|| first_text(v, "aspects"))
                    .or_else(|| text(v, "title"))
                    .or_else(|| text(v, "kind"))
                    .unwrap_or_else(|| "?".to_string())
            },
        );

        index_visual(&mut leaves, &data.join("visual"));

        // knowledge graph: concept, state, dependencies, provenance, dormant
        index_jsonl(
            &mut leaves,
            &project.join(".grounding/knowledge.jsonl"),
            "knowledge",
            "state",
            |v| text(v, "concept").unwrap_or_else(|| "?".to_string()),
        );

        Palace { leaves }
    }

    /// Every leaf matching `needle` (case-insensitive) in wing, hall, or
    /// room, in index order.
    pub fn query(&self, needle: &str) -> Vec<&Leaf> {
        let n = needle.trim().to_lowercase();
        if n.is_empty() {
            return self.leaves.iter().collect();
        }
        self.leaves
            .iter()
            .filter(|l| {
                l.wing.to_lowercase().contains(&n)
                    || l.hall.to_lowercase().contains(&n)
                    || l.room.to_lowercase().contains(&n)
            })
            .collect()
    }

    /// Leaves whose source no longer exists or no longer contains the
    /// room verbatim — memory that cannot be viewed. Empty is the only
    /// trustworthy answer.
    pub fn unresolved(&self) -> Vec<&Leaf> {
        self.leaves.iter().filter(|l| !resolves(l)).collect()
    }

    /// Leaf counts per wing, in first-seen (store) order.
    pub fn wing_counts(&self) -> Vec<(String, usize)> {
        let mut out: Vec<(String, usize)> = Vec::new();
        for l in &self.leaves {
            match out.iter_mut().find(|(w, _)| *w == l.wing) {
                Some((_, c)) => *c += 1,
                None => out.push((l.wing.clone(), 1)),
            }
        }
        out
    }

    /// A readable map: wing → hall → room [source].
    pub fn render(&self) -> String {
        let mut out = format!(
            "palace: {} leaves over {} wings\n",
            self.leaves.len(),
            self.wing_counts().len()
        );
        let mut wing = String::new();
        let mut hall = String::new();
        for l in &self.leaves {
            if l.wing != wing {
                out.push_str(&format!("{}\n", l.wing));
                wing = l.wing.clone();
                hall.clear();
            }
            if l.hall != hall {
                out.push_str(&format!("  {}\n", l.hall));
                hall = l.hall.clone();
            }
            out.push_str(&format!(
                "    {}  [{}:{}]\n",
                l.room, l.source.file, l.source.line
            ));
        }
        out
    }
}

/// Does a leaf still resolve to its source? The source line (or the
/// whole file for a JSON record) must exist and contain the room —
/// compared against the parsed JSON, so a room with quotes matches the
/// escaped form the file actually stores.
fn resolves(leaf: &Leaf) -> bool {
    let Ok(text) = std::fs::read_to_string(&leaf.source.file) else {
        return false;
    };
    let record = if leaf.source.file.ends_with(".json") {
        text.as_str()
    } else {
        match text.lines().nth(leaf.source.line.saturating_sub(1)) {
            Some(line) => line,
            None => return false,
        }
    };
    record_contains(record, &leaf.room)
}

/// The room appears in the record, raw or as a parsed JSON string
/// (escapes unrolled).
fn record_contains(record: &str, room: &str) -> bool {
    record.contains(room)
        || serde_json::from_str::<serde_json::Value>(record)
            .map(|v| json_string_contains(&v, room))
            .unwrap_or(false)
}

fn json_string_contains(v: &serde_json::Value, needle: &str) -> bool {
    match v {
        serde_json::Value::String(s) => s.contains(needle),
        serde_json::Value::Array(a) => a.iter().any(|x| json_string_contains(x, needle)),
        serde_json::Value::Object(o) => o.values().any(|x| json_string_contains(x, needle)),
        _ => false,
    }
}

fn text(v: &serde_json::Value, key: &str) -> Option<String> {
    v.get(key).and_then(|x| x.as_str()).map(str::to_string)
}

/// A string field, or the first element when the field is an array
/// (the journal stores `aspect` on most records and `aspects` on one).
fn first_text(v: &serde_json::Value, key: &str) -> Option<String> {
    match v.get(key) {
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(serde_json::Value::Array(a)) => a.first().and_then(|x| x.as_str()).map(str::to_string),
        _ => None,
    }
}

/// Append a leaf per non-blank JSONL record, `room` taken by `room_of`.
/// The stored source is the real path the leaf was read from, so the
/// resolver can always read it back.
fn index_jsonl(
    leaves: &mut Vec<Leaf>,
    path: &Path,
    wing: &str,
    hall_key: &str,
    room_of: impl Fn(&serde_json::Value) -> String,
) {
    let Ok(body) = std::fs::read_to_string(path) else {
        return;
    };
    let file = path.to_string_lossy().to_string();
    for (i, line) in body.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        leaves.push(Leaf {
            wing: wing.to_string(),
            hall: text(&v, hall_key).unwrap_or_else(|| "?".to_string()),
            room: room_of(&v),
            source: Source {
                file: file.clone(),
                line: i + 1,
            },
        });
    }
}

/// Measured plates: one whole-file JSON record per plate, indexed by
/// `plate.title` (falling back to the file stem). Files sorted for
/// determinism.
fn index_visual(leaves: &mut Vec<Leaf>, dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<std::path::PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    for path in files {
        let Ok(body) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) else {
            continue;
        };
        let room = v
            .get("plate")
            .and_then(|p| p.get("title"))
            .and_then(|t| t.as_str())
            .map(str::to_string)
            .or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "?".to_string());
        leaves.push(Leaf {
            wing: "visual".to_string(),
            hall: "measured".to_string(),
            room,
            source: Source {
                file: path.to_string_lossy().to_string(),
                line: 1,
            },
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "gc-palace-{}-{}",
            tag,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        std::fs::create_dir_all(dir.join("data/visual")).expect("dirs");
        dir
    }

    fn write(path: &Path, body: &str) {
        let mut f = std::fs::File::create(path).expect("create");
        f.write_all(body.as_bytes()).expect("write");
    }

    fn seeded() -> std::path::PathBuf {
        let p = scratch("seeded");
        write(
            &p.join("data/decisions.jsonl"),
            "{\"date\":\"2026-10-09\",\"topic\":\"fold-reaches-plan\",\"decision\":\"d\",\"basis\":\"b\",\"evidence\":\"e\",\"alternatives\":\"a\",\"status\":\"active\"}\n",
        );
        write(
            &p.join("data/lessons.jsonl"),
            "{\"id\":\"L-x\",\"kind\":\"discrimination\",\"name\":\"n\",\"statement\":\"s\",\"where\":\"w\",\"evidence\":\"e\",\"applies_to\":\"a\"}\n",
        );
        write(
            &p.join("data/encounters.jsonl"),
            "{\"kind\":\"image\",\"aspect\":\"subject-region\",\"stimulus\":\"x\",\"answer\":null,\"action\":\"committed\",\"result\":\"r\",\"date\":\"2026-10-09\"}\n",
        );
        write(
            &p.join("data/clarifications.jsonl"),
            "{\"kind\":\"measure\",\"plate\":3,\"title\":\"Woman\",\"aspect\":\"subject-region\",\"question\":\"q\",\"answer\":null,\"resolved\":false,\"date\":\"2026-10-09\"}\n",
        );
        write(
            &p.join("data/visual/plate-0000.json"),
            "{\"plate\":{\"title\":\"Frans Hals Woman\"},\"subject\":{}}\n",
        );
        std::fs::create_dir_all(p.join(".grounding")).expect("kg dir");
        write(
            &p.join(".grounding/knowledge.jsonl"),
            "{\"concept\":\"flux\",\"state\":\"Verified\",\"dependencies\":[],\"provenance\":{\"source\":\"s\",\"experiment\":null,\"verification\":\"v\",\"rejection\":null,\"tier\":\"Sourced\"},\"visits\":0,\"failures\":0}\n",
        );
        p
    }

    #[test]
    fn every_store_becomes_leaves_with_the_right_shape() {
        let p = seeded();
        let palace = Palace::build(&p);
        assert_eq!(palace.leaves.len(), 6, "{:?}", palace.leaves);
        assert!(palace.unresolved().is_empty(), "{:?}", palace.unresolved());

        let dec = &palace.leaves[0];
        assert_eq!(dec.wing, "decisions");
        assert_eq!(dec.hall, "active");
        assert_eq!(dec.room, "fold-reaches-plan");
        assert_eq!(dec.source.line, 1);

        assert!(
            palace
                .query("subject-region")
                .iter()
                .any(|l| l.wing == "encounters")
        );
        assert!(
            palace
                .query("subject-region")
                .iter()
                .any(|l| l.wing == "clarifications")
        );
        assert!(palace.query("flux").iter().any(|l| l.hall == "Verified"));
        assert_eq!(palace.query("Frans Hals").len(), 1);
    }

    #[test]
    fn empty_wings_contribute_nothing() {
        let p = scratch("empty");
        let palace = Palace::build(&p);
        assert!(palace.leaves.is_empty());
        assert!(palace.unresolved().is_empty());
        assert_eq!(palace.query("").len(), 0);
    }

    #[test]
    fn build_is_deterministic() {
        let p = seeded();
        let a = Palace::build(&p);
        let b = Palace::build(&p);
        assert_eq!(a.render(), b.render());
        assert_eq!(a.leaves, b.leaves);
    }

    #[test]
    fn a_leaf_whose_source_changed_before_the_room_does_not_resolve() {
        let p = seeded();
        let mut palace = Palace::build(&p);
        // Forge a leaf that no line supports: memory that cannot be viewed.
        palace.leaves.push(Leaf {
            wing: "decisions".to_string(),
            hall: "active".to_string(),
            room: "invented-never-committed".to_string(),
            source: Source {
                file: p.join("data/decisions.jsonl").to_string_lossy().to_string(),
                line: 1,
            },
        });
        let orphans = palace.unresolved();
        assert_eq!(orphans.len(), 1);
        assert_eq!(orphans[0].room, "invented-never-committed");
    }
}
