//! Taxonomy probe (dev tool): research parent chains live.
//! `cargo run --example taxon_probe -- elephant cat dog`
//!
//! Each noun climbs Wikidata parent-taxon claims (P171 → P279 →
//! P31), banking every edge with its source. Nothing is known
//! upfront — the graph starts empty every run and fills only
//! through cited reads. Failures name the hop.
use grounding_coder::engine::taxon::{TaxonGraph, research_parent};

#[tokio::main]
async fn main() {
    let mut graph = TaxonGraph::new();
    for noun in std::env::args().skip(1) {
        let mut current = noun.clone();
        for _ in 0..6 {
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
                    let next = edge.parent.clone();
                    graph.bank(edge);
                    current = next;
                }
                Err(e) => {
                    println!("taxon: STOP: {}", e);
                    break;
                }
            }
        }
        let (chain, missing) = graph.chain(&noun);
        println!("taxon: chain {:?} (edges {})", chain, graph.edge_count());
        if let Some(m) = missing {
            println!("taxon: gap: {}", m);
        }
    }
}
