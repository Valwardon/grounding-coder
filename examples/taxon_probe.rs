//! Taxonomy probe (dev tool): research parent chains live.
//! `cargo run --example taxon_probe -- <out.json> elephant cat dog`
//!
//! Each noun climbs Wikidata parent-taxon claims (P171 → P279 →
//! P31), banking every edge with its source. Nothing is known
//! upfront — the graph starts empty every run and fills only
//! through cited reads. Failures name the hop. The receipt file is
//! written BY THE PROGRAM (never transcribed): chains, gaps, method.
use grounding_coder::engine::taxon::{TaxonGraph, research_parent};

fn main() {
    let mut raw = std::env::args().skip(1);
    let out = raw.next().expect("usage: taxon_probe <out.json> <noun>...");
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    rt.block_on(async move {
        let mut graph = TaxonGraph::new();
        // Resume: bank edges from a previous receipt so runs
        // continue climbing instead of re-researching. Previously
        // researched nouns keep their chains unless re-requested —
        // the file only ever grows, never drops researched nouns.
        // The file is program output either way — never transcribed.
        let mut chains = serde_json::Map::new();
        if let Ok(prev) = std::fs::read_to_string(&out)
            && let Ok(val) = serde_json::from_str::<serde_json::Value>(&prev)
            && let Some(prior) = val.get("chains").and_then(|c| c.as_object())
        {
            for (k, v) in prior {
                chains.insert(k.clone(), v.clone());
            }
            for (_, ch) in prior {
                if let Some(edges) = ch.get("edges").and_then(|e| e.as_array()) {
                    for e in edges {
                        if let (Some(child), Some(cqid), Some(parent), Some(pqid), Some(rel), Some(src)) = (
                            e.get(0).and_then(|v| v.as_str()),
                            e.get(1).and_then(|v| v.as_str()),
                            e.get(2).and_then(|v| v.as_str()),
                            e.get(3).and_then(|v| v.as_str()),
                            e.get(4).and_then(|v| v.as_str()),
                            e.get(5).and_then(|v| v.as_str()),
                        ) {
                            graph.bank(grounding_coder::engine::taxon::TaxonEdge {
                                child: child.to_string(),
                                child_qid: cqid.to_string(),
                                parent: parent.to_string(),
                                parent_qid: pqid.to_string(),
                                relation: rel.to_string(),
                                source: src.to_string(),
                            });
                        }
                    }
                }
            }
            println!("taxon: resumed with {} banked edges", graph.edge_count());
        }
        for noun in raw {
            let (chain, missing) = graph.chain(&noun);
            if chain.len() >= 7 {
                println!("taxon: {} already climbed ({:?}) — skipping", noun, chain);
                // Re-emit the banked edges for this noun so the
                // receipt stays complete without re-research.
                let mut edges = Vec::new();
                for w in chain.windows(2) {
                    if let Some(e) = graph.parent_of(&w[0]) {
                        edges.push(serde_json::json!([
                            e.child, e.child_qid, e.parent, e.parent_qid, e.relation, e.source
                        ]));
                    }
                }
                chains.insert(
                    noun.clone(),
                    serde_json::json!({
                        "edges": edges,
                        "chain": chain,
                        "gap": missing,
                        "relation": "parent-taxon-first",
                        "source": "wikidata-claims",
                    }),
                );
                continue;
            }
            let mut current = chain.last().cloned().unwrap_or_else(|| noun.clone());
            let mut edges = Vec::new();
            // Re-list banked prefix edges for receipt completeness.
            for w in chain.windows(2) {
                if let Some(e) = graph.parent_of(&w[0]) {
                    edges.push(serde_json::json!([
                        e.child, e.child_qid, e.parent, e.parent_qid, e.relation, e.source
                    ]));
                }
            }
            for _ in chain.len()..7 {
                match research_parent(&current).await {
                    Ok(edge) => {
                        println!(
                            "taxon: {} ({}) -[{}]-> {} ({}) [{}]",
                            edge.child,
                            edge.child_qid,
                            edge.relation,
                            edge.parent,
                            edge.parent_qid,
                            edge.source
                        );
                        edges.push(serde_json::json!([
                            edge.child,
                            edge.child_qid,
                            edge.parent,
                            edge.parent_qid,
                            edge.relation,
                            edge.source
                        ]));
                        let next = edge.parent.clone();
                        graph.bank(edge);
                        current = next;
                    }
                    Err(e) => {
                        println!("taxon: STOP: {}", e);
                        edges.push(serde_json::json!({"stop": e}));
                        break;
                    }
                }
            }
            let (chain, missing) = graph.chain(&noun);
            println!("taxon: chain {:?} (edges {})", chain, graph.edge_count());
            if let Some(m) = &missing {
                println!("taxon: gap: {}", m);
            }
            chains.insert(
                noun.clone(),
                serde_json::json!({
                    "edges": edges,
                    "chain": chain,
                    "gap": missing,
                    "relation": "parent-taxon-first",
                    "source": "wikidata-claims",
                }),
            );
        }
        let receipt = serde_json::json!({
            "chains": chains,
            "method": "Wikidata parent-taxon claims (P171 first, then P279, then P31); claims arbitrate over search rank; 2s politeness between hops",
            "programmed_knowledge": "none: graph starts empty every run, fills only through cited reads",
            "produced_by": "taxon_probe (program output, never transcribed)",
        });
        std::fs::write(&out, serde_json::to_string_pretty(&receipt).unwrap()).expect("write receipt");
        println!("taxon: wrote {}", out);
    });
}
