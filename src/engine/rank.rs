//! Learning-to-rank over the bot's own repair history.
//!
//! The model NEVER authors code, NEVER overrides evidence, and NEVER
//! blocks a repair. It answers exactly one question: for an error group
//! (diagnostic code + file shape), did past rounds produce applicable
//! edits? Groups predicted plannable go first; ties and cold starts
//! fall back to appearance order. Every training row comes from
//! compiler-judged rounds — the model learns which verified moves to
//! try first, nothing more.
//!
//! Storage is per-project (`.grounding/ranklog.jsonl`), so clean-room
//! holds: no project ever learns from another's bytes. Rows are capped;
//! training a depth-4 tree on hundreds of rows takes milliseconds.

use std::collections::HashMap;

const MAX_ROWS: usize = 5000;
const MIN_ROWS: usize = 10;

/// One judged round for one error.
#[derive(Debug, Clone)]
pub struct RankRow {
    pub code: String,
    pub ext: String,
    pub planned: bool,
}

/// One judged OUTCOME: a recipe was applied for an error, and the next
/// verify either still shows it (fixed=false) or not (fixed=true).
/// This is what "smarter" trains on — what worked, not what was busy.
#[derive(Debug, Clone)]
pub struct OutcomeRow {
    pub code: String,
    pub ext: String,
    pub recipe: String,
    pub fixed: bool,
}

/// Recipe vocabulary index. Unknown/future names share one bucket so
/// new recipes start neutral instead of invisible.
fn recipe_idx(name: &str) -> f64 {
    match name {
        "rsx-let" => 0.0,
        "resource-read" => 1.0,
        "await-spawn" => 2.0,
        "derive-clone" => 3.0,
        "mut-binding" => 4.0,
        "fnmut-param" => 5.0,
        "fnmut-call" => 6.0,
        "into-string" => 7.0,
        _ => 8.0,
    }
}

/// Map a diagnostic code to a feature bucket. Unknown codes share one.
fn code_idx(code: &str) -> f64 {
    match code {
        "E0596" => 0.0,
        "E0283" => 1.0,
        "E0618" => 2.0,
        "E0728" => 3.0,
        "E0277" => 4.0,
        "error" => 5.0,
        _ => 6.0,
    }
}

/// Map a file extension to a feature bucket.
fn ext_idx(ext: &str) -> f64 {
    match ext.to_lowercase().as_str() {
        "rs" => 0.0,
        "kt" | "java" => 1.0,
        "c" | "h" => 2.0,
        "py" => 3.0,
        _ => 4.0,
    }
}

/// Extension of a file path ("" when absent).
fn file_ext(file: &str) -> String {
    file.rsplit('.').next().unwrap_or("").to_string()
}

fn features(code: &str, ext: &str) -> [f64; 2] {
    [code_idx(code), ext_idx(ext)]
}

/// Load judged rows for a project (missing file = cold start).
pub fn load_rows(project_dir: &std::path::Path) -> Vec<RankRow> {
    let path = project_dir.join(".grounding").join("ranklog.jsonl");
    let Ok(content) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    content
        .lines()
        .filter_map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).ok()?;
            Some(RankRow {
                code: v.get("code")?.as_str()?.to_string(),
                ext: v.get("ext")?.as_str()?.to_string(),
                planned: v.get("planned")?.as_bool()?,
            })
        })
        .collect()
}

/// Load outcome rows (missing file = cold start).
pub fn load_outcomes(project_dir: &std::path::Path) -> Vec<OutcomeRow> {
    let path = project_dir.join(".grounding").join("outcomes.jsonl");
    let Ok(content) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    content
        .lines()
        .filter_map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).ok()?;
            Some(OutcomeRow {
                code: v.get("code")?.as_str()?.to_string(),
                ext: v.get("ext")?.as_str()?.to_string(),
                recipe: v.get("recipe")?.as_str()?.to_string(),
                fixed: v.get("fixed")?.as_bool()?,
            })
        })
        .collect()
}

/// Append outcome rows (capped like rank rows).
pub fn save_outcomes(project_dir: &std::path::Path, rows: &[OutcomeRow]) {
    if rows.is_empty() {
        return;
    }
    let dir = project_dir.join(".grounding");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = dir.join("outcomes.jsonl");
    let mut existing = load_outcomes(project_dir);
    for r in rows {
        existing.push(OutcomeRow {
            code: r.code.clone(),
            ext: r.ext.clone(),
            recipe: r.recipe.clone(),
            fixed: r.fixed,
        });
    }
    if existing.len() > MAX_ROWS {
        let drop = existing.len() - MAX_ROWS;
        existing.drain(..drop);
    }
    let mut out = String::new();
    for r in &existing {
        out.push_str(
            &serde_json::json!({
                "code": r.code,
                "ext": r.ext,
                "recipe": r.recipe,
                "fixed": r.fixed,
            })
            .to_string(),
        );
        out.push('\n');
    }
    let _ = std::fs::write(&path, out);
}

/// Choose among candidate recipe names for one error: highest learned
/// P(fix | code, ext, recipe), ties and cold starts take index 0
/// (the deterministic priority order). Pure, testable.
pub fn choose_recipe(outcomes: &[OutcomeRow], code: &str, ext: &str, candidates: &[&str]) -> usize {
    if candidates.is_empty() || outcomes.len() < MIN_ROWS {
        return 0;
    }
    use linfa::prelude::*;
    let n = outcomes.len();
    let mut feats = Vec::with_capacity(n * 3);
    let mut labels = Vec::with_capacity(n);
    for r in outcomes {
        let f = [code_idx(&r.code), ext_idx(&r.ext), recipe_idx(&r.recipe)];
        feats.extend_from_slice(&f);
        labels.push(usize::from(r.fixed));
    }
    let xs = match ndarray::Array2::from_shape_vec((n, 3), feats) {
        Ok(a) => a,
        Err(_) => return 0,
    };
    let ds = Dataset::new(xs, ndarray::Array1::from_vec(labels));
    let model = match linfa_trees::DecisionTree::params()
        .max_depth(Some(5))
        .fit(&ds)
    {
        Ok(m) => m,
        Err(_) => return 0,
    };
    let mut best = 0;
    let mut best_score = -1.0f64;
    // Strictly greater wins; ties keep the earlier (priority-order)
    // candidate, so learning reorders only on evidence.
    for (i, name) in candidates.iter().enumerate() {
        let f = [code_idx(code), ext_idx(ext), recipe_idx(name)];
        let x = match ndarray::Array2::from_shape_vec((1, 3), vec![f[0], f[1], f[2]]) {
            Ok(a) => a,
            Err(_) => continue,
        };
        let s = model.predict(&x)[0] as f64;
        if s > best_score {
            best_score = s;
            best = i;
        }
    }
    best
}

/// Append judged rows (capped; oldest dropped first).
pub fn save_rows(project_dir: &std::path::Path, rows: &[RankRow]) {
    if rows.is_empty() {
        return;
    }
    let dir = project_dir.join(".grounding");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = dir.join("ranklog.jsonl");
    let mut existing = load_rows(project_dir);
    for r in rows {
        existing.push(RankRow {
            code: r.code.clone(),
            ext: r.ext.clone(),
            planned: r.planned,
        });
    }
    if existing.len() > MAX_ROWS {
        let drop = existing.len() - MAX_ROWS;
        existing.drain(..drop);
    }
    let mut out = String::new();
    for r in &existing {
        out.push_str(
            &serde_json::json!({"code": r.code, "ext": r.ext, "planned": r.planned}).to_string(),
        );
        out.push('\n');
    }
    let _ = std::fs::write(&path, out);
}

/// Train a depth-capped decision tree. `None` on cold starts, tiny
/// data, or any training failure — the caller falls back to appearance
/// order, never blocks.
fn train(rows: &[RankRow]) -> Option<linfa_trees::DecisionTree<f64, usize>> {
    if rows.len() < MIN_ROWS {
        return None;
    }
    use linfa::prelude::*;
    let n = rows.len();
    let mut feats = Vec::with_capacity(n * 2);
    let mut labels = Vec::with_capacity(n);
    for r in rows {
        let f = features(&r.code, &r.ext);
        feats.push(f[0]);
        feats.push(f[1]);
        labels.push(usize::from(r.planned));
    }
    let xs = ndarray::Array2::from_shape_vec((n, 2), feats).ok()?;
    let ys = ndarray::Array1::from_vec(labels);
    let ds = Dataset::new(xs, ys);
    linfa_trees::DecisionTree::params()
        .max_depth(Some(4))
        .fit(&ds)
        .ok()
}

/// Predict P(planned) class for one group: 1.0/0.0 from the tree, or
/// 0.5 when there is no model (cold start ties everything).
pub fn score(model: &Option<linfa_trees::DecisionTree<f64, usize>>, code: &str, file: &str) -> f64 {
    match model {
        None => 0.5,
        Some(m) => {
            use linfa::prelude::*;
            let f = features(code, &file_ext(file));
            let x = ndarray::Array2::from_shape_vec((1, 2), vec![f[0], f[1]]);
            match x {
                Ok(x) => m.predict(&x)[0] as f64,
                Err(_) => 0.5,
            }
        }
    }
}

/// Order group codes by predicted plannability (stable: ties keep
/// appearance order). Pure function of rows + codes — testable.
pub fn order_codes(
    rows: &[RankRow],
    codes: Vec<&str>,
    first_file: &HashMap<String, String>,
) -> Vec<String> {
    let model = train(rows);
    let mut scored: Vec<(usize, f64, String)> = codes
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let file = first_file.get(*c).map(|s| s.as_str()).unwrap_or("");
            (i, score(&model, c, file), c.to_string())
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(_, _, c)| c).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(code: &str, ext: &str, planned: bool) -> RankRow {
        RankRow {
            code: code.to_string(),
            ext: ext.to_string(),
            planned,
        }
    }

    #[test]
    fn cold_start_ties_keep_appearance_order() {
        let rows: Vec<RankRow> = Vec::new();
        let mut files = HashMap::new();
        files.insert("E0596".to_string(), "a.rs".to_string());
        files.insert("error".to_string(), "b.rs".to_string());
        let out = order_codes(&rows, vec!["E0596", "error"], &files);
        assert_eq!(out, vec!["E0596".to_string(), "error".to_string()]);
    }

    #[test]
    fn learned_plannable_codes_sort_first() {
        // E0596 on .rs always planned; bare parse "error" never did.
        let mut rows = Vec::new();
        for _ in 0..8 {
            rows.push(row("E0596", "rs", true));
            rows.push(row("error", "rs", false));
        }
        let mut files = HashMap::new();
        files.insert("E0596".to_string(), "a.rs".to_string());
        files.insert("error".to_string(), "b.rs".to_string());
        let out = order_codes(&rows, vec!["error", "E0596"], &files);
        assert_eq!(out[0], "E0596".to_string());
    }

    fn orow(code: &str, ext: &str, recipe: &str, fixed: bool) -> OutcomeRow {
        OutcomeRow {
            code: code.to_string(),
            ext: ext.to_string(),
            recipe: recipe.to_string(),
            fixed,
        }
    }

    #[test]
    fn choose_prefers_learned_winner() {
        // into-string fixed E0283/rs 8/8; rsx-let never did.
        let mut outcomes = Vec::new();
        for _ in 0..8 {
            outcomes.push(orow("E0283", "rs", "into-string", true));
            outcomes.push(orow("E0283", "rs", "rsx-let", false));
        }
        let pick = choose_recipe(&outcomes, "E0283", "rs", &["rsx-let", "into-string"]);
        assert_eq!(pick, 1);
    }

    #[test]
    fn choose_cold_and_ties_take_priority_order() {
        assert_eq!(
            choose_recipe(&[], "E0596", "rs", &["mut-binding", "fnmut-param"]),
            0
        );
        // All-fixed history: every candidate scores 1, earliest wins.
        let outcomes: Vec<OutcomeRow> = (0..12)
            .map(|i| {
                orow(
                    "E0596",
                    "rs",
                    if i % 2 == 0 {
                        "mut-binding"
                    } else {
                        "fnmut-param"
                    },
                    true,
                )
            })
            .collect();
        assert_eq!(
            choose_recipe(&outcomes, "E0596", "rs", &["mut-binding", "fnmut-param"]),
            0
        );
    }

    #[test]
    fn roundtrip_jsonl() {
        let dir = std::env::temp_dir().join(format!(
            "gc-rank-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        save_rows(&dir, &[row("E0596", "rs", true)]);
        let back = load_rows(&dir);
        assert_eq!(back.len(), 1);
        assert!(back[0].planned);
        save_outcomes(&dir, &[orow("E0596", "rs", "mut-binding", true)]);
        let oback = load_outcomes(&dir);
        assert_eq!(oback.len(), 1);
        assert_eq!(oback[0].recipe, "mut-binding");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
