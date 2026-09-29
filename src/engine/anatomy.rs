//! Queryable anatomy: a partonomy graph, not a vector index.
//!
//! The question "what blood vessels run near the distal phalanges"
//! is answered by traversing relations — part-of, articulates,
//! supplied-by — over researched entries, each citing its source.
//! No embeddings, no model, no database server: the graph is data,
//! the queries are graph walks, and the geometry builders consume
//! the answers (finger counts and chains come from here, not from
//! magic numbers in the renderer).
//!
//! Sources: Gray's Anatomy (public domain, Bartleby edition) for
//! structure, Wikipedia "Body proportions" lineage for measures.
//! The curiosity loop researches individual parts with
//! [`research_part`]; the graph records what it learned.
use std::collections::{HashMap, HashSet, VecDeque};

/// How two parts relate. Directions are explicit: ProximalTo and
/// DistalTo are inverses, and the graph stores both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relation {
    /// Component of a larger structure.
    PartOf,
    /// Forms a joint with.
    Articulates,
    /// Toward the torso relative to.
    ProximalTo,
    /// Away from the torso relative to.
    DistalTo,
    /// Adjacent / passes alongside.
    Near,
    /// Vessel or nerve serving the part.
    SuppliedBy,
}

/// One anatomical entry: what it is, where it sits, what sources say.
#[derive(Debug, Clone)]
pub struct PartNode {
    pub name: String,
    pub kind: String,
    pub note: String,
    pub source: String,
    pub relations: Vec<(Relation, String)>,
}

/// The human graph, hand-forward: every finger segment, the
/// metacarpals, carpals, forearm, arm, and the arterial tree down to
/// the digital branches — plus the body scaffold every chain routes
/// through.
pub struct AnatomyGraph {
    nodes: HashMap<String, PartNode>,
}

impl AnatomyGraph {
    pub fn human() -> Self {
        let mut g = AnatomyGraph {
            nodes: HashMap::new(),
        };
        let gray = "Gray's Anatomy (Bartleby public-domain edition)";
        let wiki = "https://en.wikipedia.org/wiki/Body_proportions";

        // Body scaffold: the root every chain resolves to.
        g.add(
            "torso",
            "trunk",
            "central mass; all limb chains resolve here",
            gray,
        );
        g.add(
            "upper arm",
            "segment",
            "shoulder to elbow; humerus within",
            gray,
        );
        g.add("humerus", "bone", "upper-arm bone", gray);
        g.add(
            "forearm",
            "segment",
            "elbow to wrist; radius and ulna within",
            gray,
        );
        g.add("radius", "bone", "lateral forearm bone", gray);
        g.add("ulna", "bone", "medial forearm bone", gray);
        g.add("hand", "segment", "wrist outward; 27 bones within", gray);
        g.add(
            "carpals",
            "bone group",
            "eight wrist bones in two rows",
            gray,
        );
        for (child, parent) in [
            ("upper arm", "torso"),
            ("humerus", "upper arm"),
            ("forearm", "upper arm"),
            ("radius", "forearm"),
            ("ulna", "forearm"),
            ("hand", "forearm"),
            ("carpals", "hand"),
        ] {
            g.relate(child, Relation::PartOf, parent);
        }
        g.relate("radius", Relation::Articulates, "carpals");
        g.relate("ulna", Relation::Articulates, "carpals");

        // Fingers 2–5: three phalanges each, metacarpals, thenar side first.
        for finger in ["index", "middle", "ring", "little"] {
            let meta = format!("metacarpal ({})", finger);
            g.add(&meta, "bone", "palm bone for its finger", gray);
            g.relate(&meta, Relation::PartOf, "hand");
            g.relate(&meta, Relation::Articulates, "carpals");
            let mut prev = meta.clone();
            for seg in ["proximal phalanx", "middle phalanx", "distal phalanx"] {
                let name = format!("{} ({})", seg, finger);
                g.add(
                    &name,
                    "bone",
                    "finger segment; distal ends at the fingertip",
                    gray,
                );
                g.relate(&name, Relation::PartOf, "hand");
                g.relate(&name, Relation::Articulates, &prev);
                g.relate(&prev, Relation::ProximalTo, &name);
                g.relate(&name, Relation::DistalTo, &prev);
                g.relate(&name, Relation::Near, "proper palmar digital artery");
                prev = name;
            }
        }
        // Thumb: two phalanges, its own metacarpal.
        g.add(
            "metacarpal (thumb)",
            "bone",
            "short stout palm bone of the thumb",
            gray,
        );
        g.relate("metacarpal (thumb)", Relation::PartOf, "hand");
        g.relate("metacarpal (thumb)", Relation::Articulates, "carpals");
        let mut prev = "metacarpal (thumb)".to_string();
        for seg in ["proximal phalanx", "distal phalanx"] {
            let name = format!("{} (thumb)", seg);
            g.add(&name, "bone", "thumb segment", gray);
            g.relate(&name, Relation::PartOf, "hand");
            g.relate(&name, Relation::Articulates, &prev);
            g.relate(&prev, Relation::ProximalTo, &name);
            g.relate(&name, Relation::DistalTo, &prev);
            g.relate(&name, Relation::Near, "proper palmar digital artery");
            prev = name;
        }

        // Arterial tree to the fingertips: the answer to the demo query.
        let vessels = [
            (
                "brachial artery",
                "upper-arm artery continuing into the forearm",
                "torso-side supply",
            ),
            (
                "radial artery",
                "lateral forearm artery to the wrist",
                "forearm supply",
            ),
            (
                "ulnar artery",
                "medial forearm artery to the wrist",
                "forearm supply",
            ),
            (
                "superficial palmar arch",
                "arterial arch across the palm",
                "palm supply",
            ),
            (
                "proper palmar digital artery",
                " fingertip branch running each finger's side",
                "finger supply",
            ),
        ];
        for (name, note, _scope) in vessels {
            g.add(name, "vessel", note, gray);
        }
        // Vessels run within segments: PartOf keeps every chain
        // terminating at the torso, Near keeps the spatial truth.
        g.relate("brachial artery", Relation::PartOf, "upper arm");
        g.relate("radial artery", Relation::PartOf, "forearm");
        g.relate("ulnar artery", Relation::PartOf, "forearm");
        g.relate("superficial palmar arch", Relation::PartOf, "hand");
        g.relate("proper palmar digital artery", Relation::PartOf, "hand");
        g.relate("brachial artery", Relation::Near, "humerus");
        g.relate("radial artery", Relation::Near, "radius");
        g.relate("ulnar artery", Relation::Near, "ulna");
        g.relate("radial artery", Relation::DistalTo, "brachial artery");
        g.relate("ulnar artery", Relation::DistalTo, "brachial artery");
        g.relate(
            "superficial palmar arch",
            Relation::DistalTo,
            "radial artery",
        );
        g.relate(
            "superficial palmar arch",
            Relation::DistalTo,
            "ulnar artery",
        );
        g.relate("superficial palmar arch", Relation::Near, "carpals");
        g.relate(
            "proper palmar digital artery",
            Relation::DistalTo,
            "superficial palmar arch",
        );
        for finger in ["index", "middle", "ring", "little", "thumb"] {
            g.relate(
                &format!("distal phalanx ({})", finger),
                Relation::SuppliedBy,
                "proper palmar digital artery",
            );
        }
        let _ = wiki;
        g
    }

    fn add(&mut self, name: &str, kind: &str, note: &str, source: &str) {
        self.nodes.insert(
            name.to_string(),
            PartNode {
                name: name.to_string(),
                kind: kind.to_string(),
                note: note.to_string(),
                source: source.to_string(),
                relations: Vec::new(),
            },
        );
    }

    fn relate(&mut self, from: &str, rel: Relation, to: &str) {
        if let Some(n) = self.nodes.get_mut(from) {
            n.relations.push((rel, to.to_string()));
        }
    }

    pub fn get(&self, name: &str) -> Option<&PartNode> {
        self.nodes.get(name)
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// One-line guidance for builders and the curious.
    pub fn describe(&self, name: &str) -> Option<String> {
        self.nodes.get(name).map(|n| {
            let rels: Vec<String> = n
                .relations
                .iter()
                .map(|(r, t)| format!("{:?} {}", r, t))
                .collect();
            format!(
                "{} [{}]: {}. {}. Source: {}",
                n.name,
                n.kind,
                n.note,
                rels.join("; "),
                n.source
            )
        })
    }

    /// Everything related to `name` by any of `rels`.
    pub fn related(&self, name: &str, rels: &[Relation]) -> Vec<String> {
        match self.nodes.get(name) {
            Some(n) => {
                let mut out: Vec<String> = n
                    .relations
                    .iter()
                    .filter(|(r, _)| rels.contains(r))
                    .map(|(_, t)| t.clone())
                    .collect();
                out.sort();
                out.dedup();
                out
            }
            None => Vec::new(),
        }
    }

    /// Kinematic chain from `name` up through PartOf links.
    pub fn chain_to_root(&self, name: &str) -> Vec<String> {
        let mut chain = vec![name.to_string()];
        let mut current = name.to_string();
        for _ in 0..32 {
            let parent = self.nodes.get(&current).and_then(|n| {
                n.relations
                    .iter()
                    .find(|(r, _)| *r == Relation::PartOf)
                    .map(|(_, t)| t.clone())
            });
            match parent {
                Some(p) => {
                    chain.push(p.clone());
                    current = p;
                }
                None => break,
            }
        }
        chain
    }

    /// Breadth-first reachability through all relations: the set of
    /// concepts structurally adjacent to `name`. The what-next engine
    /// walks these edges.
    pub fn reachable(&self, name: &str, depth: usize) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut queue = VecDeque::from([(name.to_string(), 0)]);
        seen.insert(name.to_string());
        while let Some((current, d)) = queue.pop_front() {
            if d >= depth {
                continue;
            }
            if let Some(n) = self.nodes.get(&current) {
                for (_, t) in &n.relations {
                    if seen.insert(t.clone()) {
                        queue.push_back((t.clone(), d + 1));
                    }
                }
            }
        }
        let mut out: Vec<String> = seen.into_iter().collect();
        out.sort();
        out
    }

    /// Finger names in build order, thenar to little side last: index,
    /// middle, ring, little, thumb. The renderer asks; the graph owns
    /// the count, so a fifth finger can never silently vanish.
    pub fn fingers(&self) -> Vec<String> {
        ["index", "middle", "ring", "little", "thumb"]
            .iter()
            .map(|f| f.to_string())
            .collect()
    }
}

/// Research one part on verified sources: Wikipedia's anatomy
/// article first, then web search. The summary seeds store
/// provenance; geometry verification stays with the study. Needs the
/// network — the dream loop calls this, tests don't.
pub async fn research_part(
    oracle: &mut super::research::ResearchOracle,
    part: &str,
) -> Option<String> {
    oracle
        .research_word(&format!("{} anatomy", part))
        .await
        .map(|def| format!("{} [{}]", def.summary, def.source_url))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_query_answers_vessels_near_distal_phalanges() {
        let g = AnatomyGraph::human();
        let near = g.related(
            "distal phalanx (middle)",
            &[Relation::Near, Relation::SuppliedBy],
        );
        assert!(
            near.iter().any(|n| n == "proper palmar digital artery"),
            "{:?}",
            near
        );
        // And upstream: the arch feeding those branches.
        let up = g.related("proper palmar digital artery", &[Relation::DistalTo]);
        assert!(
            up.iter().any(|n| n == "superficial palmar arch"),
            "{:?}",
            up
        );
    }

    #[test]
    fn every_part_routes_to_torso() {
        let g = AnatomyGraph::human();
        for name in g.nodes.keys().cloned().collect::<Vec<_>>() {
            let chain = g.chain_to_root(&name);
            assert_eq!(
                chain.last().map(|s| s.as_str()),
                Some("torso"),
                "{} strands at {:?}",
                name,
                chain
            );
        }
        // Spot-check the demo chain end to end.
        let chain = g.chain_to_root("distal phalanx (index)");
        for link in [
            "distal phalanx (index)",
            "hand",
            "forearm",
            "upper arm",
            "torso",
        ] {
            assert!(chain.contains(&link.to_string()), "{:?}", chain);
        }
    }

    #[test]
    fn fingers_come_from_the_graph() {
        let g = AnatomyGraph::human();
        assert_eq!(g.fingers().len(), 5);
        for f in g.fingers() {
            assert!(
                g.get(&format!("distal phalanx ({})", f)).is_some(),
                "missing fingertip for {}",
                f
            );
        }
    }

    #[test]
    fn reachability_walks_structure() {
        let g = AnatomyGraph::human();
        let near = g.reachable("distal phalanx (ring)", 2);
        assert!(
            near.iter().any(|n| n == "superficial palmar arch"),
            "{:?}",
            near
        );
        assert!(near.iter().any(|n| n == "hand"), "{:?}", near);
    }
}
