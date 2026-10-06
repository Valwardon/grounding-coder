//! Taxonomy: researched is-a chains, never shipped knowledge.
//!
//! Doctrine: the program ships ZERO taxonomy — no animal lists, no
//! body plans, no per-creature anything (a shipped tree of every
//! living thing would be the memorization trap at install size).
//! Given a noun ("elephant"), the system researches its parent taxa
//! through Wikidata's structured claims (instance-of / subclass-of /
//! parent-taxon — machine relations, never prose guessing), banks
//! each edge with its source, and walks the researched chain. An
//! unresearched noun is a gap with a research order, never a guess.
//!
//! Body plans attach to researched nodes later (staged gap); this
//! module proves the chain machinery first: resolve, bank, replay
//! deterministically from the bank alone.

use std::collections::HashMap;

/// One researched taxon edge: child → parent with its source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaxonEdge {
    pub child: String,
    pub child_qid: String,
    pub parent: String,
    pub parent_qid: String,
    pub relation: String,
    pub source: String,
}

/// The researched graph: edges banked per request/project, replayed
/// without network. Starts empty every time — knowledge arrives
/// through research acts, never through code.
#[derive(Debug, Clone, Default)]
pub struct TaxonGraph {
    edges: HashMap<String, TaxonEdge>,
}

impl TaxonGraph {
    pub fn new() -> Self {
        TaxonGraph {
            edges: HashMap::new(),
        }
    }

    pub fn bank(&mut self, edge: TaxonEdge) {
        self.edges.insert(edge.child.clone(), edge);
    }

    pub fn parent_of(&self, name: &str) -> Option<&TaxonEdge> {
        self.edges.get(&name.to_lowercase())
    }

    /// Walk researched parents from a noun: ["elephant", "mammal",
    /// ...] as far as banked edges reach. Unknown nouns stop the
    /// walk immediately with the research order for the missing
    /// link. Deterministic over the bank.
    pub fn chain(&self, name: &str) -> (Vec<String>, Option<String>) {
        let mut chain = vec![name.to_lowercase()];
        let mut missing = None;
        for _ in 0..16 {
            let last = chain.last().cloned().unwrap_or_default();
            match self.edges.get(&last) {
                Some(e) => chain.push(e.parent.clone()),
                None => {
                    missing = Some(format!("resolve taxonomy for {:?}", last));
                    break;
                }
            }
        }
        (chain, missing)
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

/// Parse a Wikidata entity-search response: first result's QID and
/// English label, if any. Pure over canned or live JSON.
pub fn parse_entity_search(body: &serde_json::Value) -> Option<(String, String)> {
    let hit = body.get("search")?.as_array()?.first()?;
    let qid = hit.get("id")?.as_str()?.to_string();
    let label = hit
        .get("label")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if qid.is_empty() {
        return None;
    }
    Some((qid, label))
}

/// Parent-taxon claim properties, strongest first: parent taxon
/// (P171, the biological parent), then subclass-of (P279), then
/// instance-of (P31, weakest — "this individual is an X").
pub const PARENT_PROPS: &[(&str, &str)] = &[
    ("P171", "parent-taxon"),
    ("P279", "subclass-of"),
    ("P31", "instance-of"),
];

/// Parse an entity's claims into the first parent edge found
/// (P171 → P279 → P31). Returns (parent QID, relation). Pure.
pub fn parse_parent_claim(entity: &serde_json::Value) -> Option<(String, String)> {
    let claims = entity.get("claims")?;
    for (prop, relation) in PARENT_PROPS {
        let arr = claims.get(*prop)?.as_array()?;
        for claim in arr {
            let qid = claim
                .get("mainsnak")?
                .get("datavalue")?
                .get("value")?
                .get("id")?
                .as_str()?;
            if !qid.is_empty() {
                return Some((qid.to_string(), relation.to_string()));
            }
        }
    }
    None
}

/// Parse an entity's English label. Pure.
pub fn parse_label(entity: &serde_json::Value, qid: &str) -> Option<String> {
    entity
        .get("labels")?
        .get("en")?
        .get("value")?
        .as_str()
        .map(|s| s.to_string())
        .or_else(|| Some(qid.to_string()))
}

/// Research one parent link for a noun: search entities, take the
/// first hit (up to 5) whose claims yield a parent edge, preferring
/// parent-taxon (biological) over subclass over instance. Search
/// top-hits lie routinely ("cat" → Catalan the language, CAT scans,
/// churches) — claims arbitrate, never the rank. When nothing
/// parents, one disambiguated retry ("<noun> animal", logged as
/// such) precedes the refusal: popularity ranking buries animals
/// under acronyms, and the retry is a question, not an answer.
/// Every hop is a cited Wikidata read; failures name the hop. No
/// knowledge enters except through here.
/// Shared-infrastructure courtesy: 2s between hops (Wikidata answers
/// 429 to bursts, measured live).
pub async fn research_parent(name: &str) -> Result<TaxonEdge, String> {
    // Search formulations in order: the bare noun, then
    // disambiguations. Popularity ranking buries animals under
    // acronyms and languages ("cat" → Catalan, CAT scans), so each
    // miss tries a more specific question. Every attempt is logged
    // in the final refusal — the trail shows the asking, never an
    // asserted answer.
    let queries = vec![
        name.to_string(),
        format!("{} animal", name),
        format!("house {}", name),
    ];
    let mut trails = Vec::new();
    for query in &queries {
        match research_parent_in(query, name).await {
            Ok(edge) => return Ok(edge),
            Err(e) => trails.push(format!("{:?}: {}", query, e)),
        }
    }
    Err(format!(
        "no parented entity for {:?} ({})",
        name,
        trails.join("; ")
    ))
}

async fn research_parent_in(query: &str, name: &str) -> Result<TaxonEdge, String> {
    let search_url = format!(
        "https://www.wikidata.org/w/api.php?action=wbsearchentities&search={}&language=en&format=json&limit=5",
        percent_encode(query)
    );
    let search: serde_json::Value = crate::http::get_json(&search_url, None, None)
        .await
        .map_err(|e| format!("taxonomy search failed for {:?}: {}", name, e))?;
    let hits = search
        .get("search")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if hits.is_empty() {
        return Err(format!("no Wikidata entity for {:?}", name));
    }
    let mut tried = Vec::new();
    let mut best: Option<(usize, String, String, String, String)> = None;
    for hit in hits.iter().take(5) {
        let Some(qid) = hit.get("id").and_then(|v| v.as_str()) else {
            continue;
        };
        let label = hit
            .get("label")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let label = if label.is_empty() {
            name.to_string()
        } else {
            label
        };
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let ent_url = format!(
            "https://www.wikidata.org/w/api.php?action=wbgetentities&ids={}&props=claims&format=json",
            qid
        );
        let ent: serde_json::Value = match crate::http::get_json(&ent_url, None, None).await {
            Ok(v) => v,
            Err(e) => {
                tried.push(format!("{}: claims fetch failed ({})", qid, e));
                continue;
            }
        };
        let Some(entity) = ent.get("entities").and_then(|e| e.get(qid)) else {
            tried.push(format!("{}: no entity body", qid));
            continue;
        };
        let Some((parent_qid, relation)) = parse_parent_claim(entity) else {
            tried.push(format!("{} ({}): no parent claim", label, qid));
            continue;
        };
        let rank = PARENT_PROPS
            .iter()
            .position(|(_, r)| *r == relation)
            .unwrap_or(99);
        let better = best
            .as_ref()
            .map(|(br, _, _, _, _)| rank < *br)
            .unwrap_or(true);
        if better {
            best = Some((rank, qid.to_string(), label, parent_qid, relation));
            if rank == 0 {
                break;
            }
        }
    }
    let (_, qid, _label, parent_qid, relation) = best.ok_or_else(|| {
        format!(
            "no parented entity for {:?} (tried: {})",
            name,
            tried.join("; ")
        )
    })?;
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    let lab_url = format!(
        "https://www.wikidata.org/w/api.php?action=wbgetentities&ids={}&props=labels&languages=en&format=json",
        parent_qid
    );
    let lab: serde_json::Value = crate::http::get_json(&lab_url, None, None)
        .await
        .map_err(|e| format!("taxonomy label failed for {}: {}", parent_qid, e))?;
    let parent_entity = lab
        .get("entities")
        .and_then(|e| e.get(&parent_qid))
        .ok_or_else(|| format!("no label body for {}", parent_qid))?;
    let parent = parse_label(parent_entity, &parent_qid).unwrap_or(parent_qid.clone());
    Ok(TaxonEdge {
        child: name.to_lowercase(),
        child_qid: qid,
        parent: parent.to_lowercase(),
        parent_qid,
        relation,
        source: "wikidata-claims".to_string(),
    })
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) {
            out.push(b as char);
        } else if b == b' ' {
            out.push_str("%20");
        } else {
            out.push_str(&format!("%{:02X}", b));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canned_search() -> serde_json::Value {
        serde_json::from_str(
            r#"{"search": [{"id": "Q7378", "label": "elephant", "description": "large mammal"}]}"#,
        )
        .unwrap()
    }

    fn canned_claims() -> serde_json::Value {
        serde_json::from_str(
            r#"{"entities": {"Q7378": {"claims": {
                "P171": [{"mainsnak": {"datavalue": {"value": {"id": "Q125854"}}}}],
                "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q12345"}}}}]
            }}}}"#,
        )
        .unwrap()
    }

    #[test]
    fn entity_search_parses_first_hit() {
        let (qid, label) = parse_entity_search(&canned_search()).expect("parses");
        assert_eq!((qid.as_str(), label.as_str()), ("Q7378", "elephant"));
        assert!(parse_entity_search(&serde_json::json!({})).is_none());
        assert!(parse_entity_search(&serde_json::json!({"search": []})).is_none());
    }

    #[test]
    fn parent_claim_prefers_parent_taxon() {
        // P171 beats P31 even though both present.
        let canned = canned_claims();
        let (qid, rel) = parse_parent_claim(&canned["entities"]["Q7378"]).expect("parses");
        assert_eq!(qid, "Q125854");
        assert_eq!(rel, "parent-taxon");
    }

    #[test]
    fn graph_walks_banked_edges_and_names_gaps() {
        let mut g = TaxonGraph::new();
        // Empty graph: the walk stops immediately with the order.
        let (chain, missing) = g.chain("elephant");
        assert_eq!(chain, vec!["elephant".to_string()]);
        assert!(missing.unwrap().contains("elephant"));
        // Banked edges walk without network.
        g.bank(TaxonEdge {
            child: "elephant".to_string(),
            child_qid: "Q7378".to_string(),
            parent: "mammal".to_string(),
            parent_qid: "Q7377".to_string(),
            relation: "parent-taxon".to_string(),
            source: "test".to_string(),
        });
        let (chain, missing) = g.chain("elephant");
        assert_eq!(chain, vec!["elephant".to_string(), "mammal".to_string()]);
        assert!(missing.unwrap().contains("mammal"));
        assert_eq!(g.edge_count(), 1);
    }
}
