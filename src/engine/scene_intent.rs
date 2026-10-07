//! Intent-to-research IR: what the prose asks the world for.
//!
//! The understander routes prose to engine frames (build/create/fix…).
//! This module answers the narrower question for pictures: WHO does
//! WHAT with WHICH things — as a compositional, inspectable structure
//! with confidence and explicitly unresolved relationships. No model,
//! no guessing: a controlled vocabulary, deterministic phrase rules,
//! and an ontology that says what each concept needs.
//!
//! ```text
//! prose → SceneSpec → ResearchPlan → CapabilityVerdict
//! ```
//!
//! Ambiguity is preserved, never invented away: "waving" with no
//! object stays ambiguous (hand or flag) until evidence resolves it.
//! Capabilities are matched honestly: a requirement the engine cannot
//! build (articulated joints, cloth) reports the missing capability
//! instead of silently substituting a block figure.

use serde::{Deserialize, Serialize};

/// Lowercase alphanumeric tokens, order kept.
fn tokens(prose: &str) -> Vec<String> {
    prose
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

const STOPWORDS: &[&str] = &[
    "a", "an", "the", "of", "on", "in", "with", "and", "or", "to", "for", "is", "are", "it", "at",
    "by", "from", "that", "this",
];

/// Controlled vocabulary: surface form → canonical concept. Spelling
/// variants, morphological variants, and synonyms map here; adding a
/// row is a product decision, never learning.
const LEXICON: &[(&str, &str)] = &[
    ("man", "man"),
    ("gentleman", "man"),
    ("gentlemen", "man"),
    ("male", "man"),
    ("males", "man"),
    ("woman", "woman"),
    ("women", "woman"),
    ("lady", "woman"),
    ("ladies", "woman"),
    ("female", "woman"),
    ("females", "woman"),
    ("girl", "woman"),
    ("girls", "woman"),
    ("boy", "man"),
    ("boys", "man"),
    ("human", "human"),
    ("humans", "human"),
    ("person", "human"),
    ("people", "human"),
    ("kid", "human"),
    ("kids", "human"),
    ("child", "human"),
    ("children", "human"),
    ("cat", "cat"),
    ("cats", "cat"),
    ("dog", "dog"),
    ("dogs", "dog"),
    ("elephant", "elephant"),
    ("elephants", "elephant"),
    ("horse", "horse"),
    ("horses", "horse"),
    ("bird", "bird"),
    ("birds", "bird"),
    ("fish", "fish"),
    ("lion", "lion"),
    ("lions", "lion"),
    ("tiger", "tiger"),
    ("tigers", "tiger"),
    ("bear", "bear"),
    ("bears", "bear"),
    ("car", "car"),
    ("cars", "car"),
    ("tree", "tree"),
    ("trees", "tree"),
    ("house", "house"),
    ("houses", "house"),
    ("cake", "cake"),
    ("jetski", "jetski"),
    ("lake", "lake"),
    ("lakes", "lake"),
    ("ocean", "ocean"),
    ("forest", "forest"),
    ("forests", "forest"),
    ("desert", "desert"),
    ("deserts", "desert"),
    ("sky", "sky"),
    ("beach", "beach"),
    ("beaches", "beach"),
    ("salute", "salute"),
    ("salutes", "salute"),
    ("saluting", "salute"),
    ("saluted", "salute"),
    ("wave", "wave"),
    ("waves", "wave"),
    ("waved", "wave"),
    ("waving", "wave"),
    ("sit", "sit"),
    ("sits", "sit"),
    ("sitting", "sit"),
    ("sat", "sit"),
    ("cross", "cross"),
    ("crosses", "cross"),
    ("crossed", "cross"),
    ("crossing", "cross"),
    ("stand", "stand"),
    ("stands", "stand"),
    ("standing", "stand"),
    ("stood", "stand"),
    ("give", "give"),
    ("gives", "give"),
    ("giving", "give"),
    ("gave", "give"),
    ("wear", "wear"),
    ("wears", "wear"),
    ("wearing", "wear"),
    ("wore", "wear"),
    ("worn", "wear"),
    ("hat", "hat"),
    ("hats", "hat"),
    ("flag", "flag"),
    ("flags", "flag"),
    ("mountain", "mountain"),
    ("mountains", "mountain"),
    ("river", "river"),
    ("rivers", "river"),
    ("american_flag", "american_flag"),
    ("peace_sign", "peace_sign"),
];

/// Multi-word concepts, longest first: the token scan must see
/// "american flag" before the single words "american" / "flag".
const PHRASES: &[(&[&str], &str)] = &[
    (&["united", "states", "flag"], "american_flag"),
    (&["american", "flag"], "american_flag"),
    (&["us", "flag"], "american_flag"),
    (&["peace", "sign"], "peace_sign"),
];

/// What kind of thing a concept is — this decides what references
/// the research planner hunts and what the capability matcher checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConceptKind {
    Human,
    Fauna,
    Flora,
    Thing,
    Action,
    Wearable,
    Fabric,
    Place,
}

/// Ontology: concept → (kind, what it needs). The "needs" are
/// research targets, not promises — the capability matcher later
/// says which needs the engine can actually satisfy.
const ONTOLOGY: &[(&str, ConceptKind, &[&str])] = &[
    ("man", ConceptKind::Human, &["proportions", "photograph"]),
    ("woman", ConceptKind::Human, &["proportions", "photograph"]),
    ("human", ConceptKind::Human, &["proportions", "photograph"]),
    ("cat", ConceptKind::Fauna, &["anatomy", "photograph"]),
    ("dog", ConceptKind::Fauna, &["anatomy", "photograph"]),
    ("elephant", ConceptKind::Fauna, &["anatomy", "photograph"]),
    ("horse", ConceptKind::Fauna, &["anatomy", "photograph"]),
    ("bird", ConceptKind::Fauna, &["anatomy", "photograph"]),
    ("fish", ConceptKind::Fauna, &["anatomy", "photograph"]),
    ("lion", ConceptKind::Fauna, &["anatomy", "photograph"]),
    ("tiger", ConceptKind::Fauna, &["anatomy", "photograph"]),
    ("bear", ConceptKind::Fauna, &["anatomy", "photograph"]),
    ("tree", ConceptKind::Flora, &["silhouette", "photograph"]),
    ("car", ConceptKind::Thing, &["silhouette", "photograph"]),
    ("house", ConceptKind::Thing, &["silhouette", "photograph"]),
    ("cake", ConceptKind::Thing, &["silhouette", "photograph"]),
    ("jetski", ConceptKind::Thing, &["silhouette", "photograph"]),
    ("lake", ConceptKind::Place, &["photograph"]),
    ("ocean", ConceptKind::Place, &["photograph"]),
    ("forest", ConceptKind::Place, &["photograph"]),
    ("desert", ConceptKind::Place, &["photograph"]),
    ("sky", ConceptKind::Place, &["photograph"]),
    ("beach", ConceptKind::Place, &["photograph"]),
    ("salute", ConceptKind::Action, &["pose-reference", "joints"]),
    ("sit", ConceptKind::Action, &["pose-reference", "seat"]),
    ("cross", ConceptKind::Action, &["motion-reference", "place"]),
    (
        "wave",
        ConceptKind::Action,
        &["pose-reference", "object-or-hand"],
    ),
    ("stand", ConceptKind::Action, &["pose-reference", "place"]),
    ("peace_sign", ConceptKind::Action, &["hand-reference"]),
    ("hat", ConceptKind::Wearable, &["silhouette", "asset"]),
    ("flag", ConceptKind::Fabric, &["design", "cloth"]),
    ("american_flag", ConceptKind::Fabric, &["design", "cloth"]),
    ("mountain", ConceptKind::Place, &["photograph"]),
    ("river", ConceptKind::Place, &["photograph"]),
];

fn kind_of(concept: &str) -> Option<ConceptKind> {
    ONTOLOGY
        .iter()
        .find(|(c, _, _)| *c == concept)
        .map(|(_, k, _)| *k)
}

/// Who the picture is about.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subject {
    /// Canonical type: "man" | "woman" | "human".
    #[serde(rename = "type")]
    pub stype: String,
    /// Modifiers that refined the type ("human woman" → ["woman"]).
    pub attributes: Vec<String>,
}

/// Something done, by someone, to something.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneAction {
    /// Canonical action: "salute" | "wave" | "stand" | "peace_sign".
    #[serde(rename = "type")]
    pub atype: String,
    pub actor: String,
    pub target: Option<String>,
    pub object: Option<String>,
    /// True when the prose underdetermines the reading ("waving"
    /// with no object: hand or flag). Preserved, never defaulted.
    pub ambiguous: bool,
}

/// A thing in the picture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneObject {
    pub id: String,
    /// Canonical type: "hat" | "flag" | "american_flag" | "mountain".
    #[serde(rename = "type")]
    pub otype: String,
    pub attributes: Vec<String>,
    pub state: Vec<String>,
    pub worn_by: Option<String>,
    /// Construction material: ANY material word bound by the edge
    /// grammar lives here (straw, glass, steel — e.g. pasta), never
    /// as an unrelated object in the scene.
    pub material: Option<String>,
}

/// The structured scene specification: everything the parser could
/// justify, plus everything it could not.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneSpec {
    pub subjects: Vec<Subject>,
    pub actions: Vec<SceneAction>,
    pub objects: Vec<SceneObject>,
    /// Relationships the prose leaves open ("hat wearer: unnamed").
    /// Evidence resolves these later; the parser never invents them.
    pub unresolved: Vec<String>,
    /// Mapped content tokens over content tokens. 1.0 means every
    /// content word landed in the ontology (light verbs forgiven,
    /// material-edge markers counted when the edge fires).
    pub confidence: f64,
}

/// Fold phrase hits into the token stream: multi-word concepts
/// become single tokens ("american flag" → "american_flag") so the
/// rest of the parser sees one concept, not two words.
fn fold_phrases(words: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let mut hit: Option<(&[&str], &str)> = None;
        for (pat, concept) in PHRASES {
            if words[i..].len() >= pat.len() && pat.iter().zip(&words[i..]).all(|(p, w)| p == w) {
                hit = Some((*pat, *concept));
                break;
            }
        }
        match hit {
            Some((pat, concept)) => {
                out.push(concept.to_string());
                i += pat.len();
            }
            None => {
                out.push(words[i].clone());
                i += 1;
            }
        }
    }
    out
}

/// Canonical concept for a surface word, if the lexicon knows it.
fn concept_of(word: &str) -> Option<&'static str> {
    LEXICON.iter().find(|(s, _)| *s == word).map(|(_, c)| *c)
}

/// Parse prose into a scene specification. Pure: same input, same
/// output, no I/O, no model.
pub fn parse_scene(prose: &str) -> SceneSpec {
    let folded = fold_phrases(&tokens(prose));
    let concepts: Vec<&str> = folded.iter().filter_map(|w| concept_of(w)).collect();

    // Subject: first human, fauna, thing, or flora concept; sibling
    // subject words refine it ("human woman" → human + [woman]).
    // Non-human subjects (cat, car, tree, …) are first-class
    // subjects for research — the parser proposes, research
    // disposes. Wearables/places stay objects, never subjects.
    let subject_at = concepts.iter().position(|c| {
        kind_of(c).is_some_and(|k| {
            k == ConceptKind::Human
                || k == ConceptKind::Fauna
                || k == ConceptKind::Thing
                || k == ConceptKind::Flora
        })
    });
    let mut subjects = Vec::new();
    if let Some(si) = subject_at {
        // Sibling subject words refine the primary ("human woman")
        // when adjacent; separated by an action verb they are
        // COMPANION subjects ("cat sitting in human's lap" — the
        // human is researched too, never folded away as a modifier).
        let action_at: Vec<usize> = concepts
            .iter()
            .enumerate()
            .filter(|(_, c)| kind_of(c).is_some_and(|k| k == ConceptKind::Action))
            .map(|(i, _)| i)
            .collect();
        let separated = |a: usize, b: usize| {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            action_at.iter().any(|v| *v > lo && *v < hi)
        };
        let mut attributes = Vec::new();
        let mut companions: Vec<String> = Vec::new();
        for (i, c) in concepts.iter().enumerate() {
            if i != si
                && kind_of(c).is_some_and(|k| k == ConceptKind::Human || k == ConceptKind::Fauna)
                && *c != concepts[si]
                && !companions.iter().any(|s| s == c)
            {
                if separated(i, si) {
                    companions.push(c.to_string());
                } else {
                    attributes.push(c.to_string());
                }
            }
        }
        subjects.push(Subject {
            stype: concepts[si].to_string(),
            attributes,
        });
        for comp in companions {
            subjects.push(Subject {
                stype: comp,
                attributes: Vec::new(),
            });
        }
    }
    // Material edge (Wikidata `material-used` analogue): "X made of
    // Y" binds ANY Y to X's material slot via the understander's
    // deterministic grammar — no material is ever a scene object.
    // Needed before role assignment below (edge words belong
    // elsewhere, never as subjects or objects).
    let material_edge = super::understand::extract_material(prose);
    // Open vocabulary role assignment: unknown content words take
    // roles by POSITION relative to the first action verb — before
    // it they are subject candidates ("cat sitting…", "elephant
    // crossing…"), after it object candidates ("…in human's lap").
    // With no verb at all, unknowns are object candidates only (never
    // phantom subjects for verbless prose like "do the thing").
    // Doctrine: the parser proposes, research disposes — an
    // unclassified noun is a work order for the knowledge graph,
    // never a silent drop and never a programmed classification.
    let edge_markers = ["made", "out", "from", "built", "constructed"];
    let edge_words: Vec<&str> = material_edge
        .as_ref()
        .map(|(b, m)| vec![b.as_str(), m.as_str()])
        .unwrap_or_default();
    let unknown_content = |w: &String| {
        !STOPWORDS.contains(&w.as_str())
            && concept_of(w).is_none()
            && !edge_markers.contains(&w.as_str())
            && !edge_words.contains(&w.as_str())
            && w.len() > 2
    };
    let verb_at = folded.iter().position(|w| {
        concept_of(w).is_some_and(|c| kind_of(c).is_some_and(|k| k == ConceptKind::Action))
    });
    let mut generic_subjects: Vec<String> = Vec::new();
    let mut generic_objects: Vec<String> = Vec::new();
    for (i, w) in folded.iter().enumerate() {
        if !unknown_content(w) || generic_subjects.contains(w) || generic_objects.contains(w) {
            continue;
        }
        match verb_at {
            Some(v) if i < v => generic_subjects.push(w.clone()),
            Some(_) => generic_objects.push(w.clone()),
            None => generic_objects.push(w.clone()),
        }
    }
    for word in &generic_subjects {
        subjects.push(Subject {
            stype: word.clone(),
            attributes: vec!["unclassified".to_string()],
        });
    }
    // Actor: nearest subject before the verb (the doer precedes the
    // deed: "cat sitting" → cat), else the first subject.
    let actor = match verb_at {
        Some(v) => {
            let folded_subjects: Vec<&String> = folded[..v]
                .iter()
                .filter(|w| {
                    subjects
                        .iter()
                        .any(|s| &s.stype == *w || s.attributes.iter().any(|a| a == *w))
                })
                .collect();
            folded_subjects
                .last()
                .map(|w| w.to_string())
                .or_else(|| subjects.first().map(|s| s.stype.clone()))
                .unwrap_or_else(|| "subject".to_string())
        }
        None => subjects
            .first()
            .map(|s| s.stype.clone())
            .unwrap_or_else(|| "subject".to_string()),
    };

    // Objects: wearable / fabric / place concepts in prose order,
    // plus thing/flora concepts NOT already taken as the subject
    // (a "car crossing" is a subject; a "man beside a car" keeps
    // the car as an object). Fauna concepts are subjects, never
    // objects.
    let subject_concept: Option<String> = subject_at.map(|si| concepts[si].to_string());
    let mut objects: Vec<SceneObject> = Vec::new();
    let mut counter = 0u32;
    let mut subject_skipped = false;
    for c in &concepts {
        match kind_of(c) {
            Some(ConceptKind::Wearable) | Some(ConceptKind::Fabric) | Some(ConceptKind::Place) => {
                counter += 1;
                let id = if *c == "american_flag" {
                    "american_flag".to_string()
                } else {
                    format!("{}_{}", c, counter)
                };
                let mut obj = SceneObject {
                    id,
                    otype: c.to_string(),
                    attributes: Vec::new(),
                    state: Vec::new(),
                    worn_by: None,
                    material: None,
                };
                if *c == "american_flag" {
                    obj.attributes.push("american".to_string());
                }
                objects.push(obj);
            }
            Some(ConceptKind::Thing) | Some(ConceptKind::Flora) => {
                // Taken as the subject already? Then it is not also
                // an object. Otherwise it is a research-candidate
                // object ("a man beside a car").
                if !subject_skipped && subject_concept.as_deref() == Some(*c) {
                    subject_skipped = true;
                    continue;
                }
                counter += 1;
                objects.push(SceneObject {
                    id: format!("{}_{}", c, counter),
                    otype: c.to_string(),
                    attributes: Vec::new(),
                    state: Vec::new(),
                    worn_by: None,
                    material: None,
                });
            }
            _ => {}
        }
    }

    // Adjective materials: an unknown content word immediately
    // before a wearable/fabric object names what it is made of
    // ("straw hat", "silk flag") — grammar, not a word list: ANY
    // substance binds here, and the material never becomes a scene
    // object of its own. (The "made of" edge below handles the
    // explicit form; this handles the attributive form.)
    let mut adjective_materials: Vec<(String, String)> = Vec::new();
    {
        let articles = ["a", "an", "the"];
        let is_wearable_fabric = |w: &String| {
            concept_of(w).is_some_and(|c| {
                kind_of(c).is_some_and(|k| k == ConceptKind::Wearable || k == ConceptKind::Fabric)
            })
        };
        for (i, w) in folded.iter().enumerate() {
            if !is_wearable_fabric(w) || i == 0 {
                continue;
            }
            // Walk back past articles to the candidate substance.
            let mut j = i - 1;
            while j > 0 && articles.contains(&folded[j].as_str()) {
                j -= 1;
            }
            let cand = &folded[j];
            if unknown_content(cand) && generic_objects.contains(cand) {
                adjective_materials.push((cand.clone(), w.clone()));
            }
        }
    }
    generic_objects.retain(|w| !adjective_materials.iter().any(|(m, _)| m == w));
    // Generic objects (open vocabulary): post-verb unknown nouns
    // become research-candidate objects ("…in human's lap" → lap).
    // Same doctrine as generic subjects: proposed, never classified.
    for word in &generic_objects {
        counter += 1;
        objects.push(SceneObject {
            id: format!("{}_{}", word, counter),
            otype: word.clone(),
            attributes: vec!["unclassified".to_string()],
            state: Vec::new(),
            worn_by: None,
            material: None,
        });
    }
    // Bind adjective materials onto their objects (research queries
    // attach downstream through the standard material path).
    for (mat, obj_word) in &adjective_materials {
        if let Some(obj) = objects.iter_mut().find(|o| {
            o.otype == *obj_word
                && o.material.is_none()
                && !o.attributes.contains(&"unclassified".to_string())
        }) {
            obj.material = Some(mat.clone());
        }
    }

    // Bind the material edge onto the matching object — or onto a
    // generic object named by the base word itself. ANY base works
    // (tower, bowl, statue): no object noun is special-cased, and no
    // material ever becomes a scene object.
    let mut unresolved = Vec::new();
    if let Some((base, material)) = &material_edge {
        if let Some(obj) = objects
            .iter_mut()
            .find(|o| o.otype == *base || o.id.starts_with(base.as_str()))
        {
            obj.material = Some(material.clone());
        } else if !base.is_empty() {
            objects.push(SceneObject {
                id: format!("{}_{}", base, objects.len() + 1),
                otype: base.clone(),
                attributes: Vec::new(),
                state: Vec::new(),
                worn_by: None,
                material: Some(material.clone()),
            });
        }
    }

    // Worn-by: explicit wear verbs only ("wearing a hat"). A wearable
    // with no wearer stays open — the parser does not invent wearing.
    let wears = concepts.contains(&"wear");
    for obj in objects.iter_mut() {
        if kind_of(&obj.otype).is_some_and(|k| k == ConceptKind::Wearable) && wears {
            obj.worn_by = Some(actor.clone());
        }
    }
    for obj in &objects {
        if kind_of(&obj.otype).is_some_and(|k| k == ConceptKind::Wearable) && obj.worn_by.is_none()
        {
            unresolved.push(format!("{} wearer: unnamed — needs evidence", obj.otype));
        }
    }

    // Actions, in prose order.
    let mut actions = Vec::new();
    let has = |c: &str| concepts.contains(&c);
    if has("salute") {
        actions.push(SceneAction {
            atype: "salute".to_string(),
            actor: actor.clone(),
            // Ontology default, inspectable: a salute goes to the head.
            target: Some("head".to_string()),
            object: None,
            ambiguous: false,
        });
    }
    if has("wave") {
        // "wave" with a named fabric object waves THAT; bare "waving"
        // stays ambiguous (hand or flag) — first fabric wins,
        // deterministic.
        let waved_obj = objects
            .iter()
            .find(|o| kind_of(&o.otype).is_some_and(|k| k == ConceptKind::Fabric))
            .map(|o| o.id.clone());
        let ambiguous = waved_obj.is_none();
        if ambiguous {
            unresolved.push("wave: hand or flag — no object named".to_string());
        } else if let Some(id) = &waved_obj
            && let Some(obj) = objects.iter_mut().find(|o| &o.id == id)
        {
            obj.state.push("waving".to_string());
        }
        actions.push(SceneAction {
            atype: "wave".to_string(),
            actor: actor.clone(),
            target: None,
            object: waved_obj,
            ambiguous,
        });
    }
    if has("stand") {
        // "standing on X" stands on the named place, if any.
        let place = objects
            .iter()
            .find(|o| kind_of(&o.otype).is_some_and(|k| k == ConceptKind::Place))
            .map(|o| o.otype.clone());
        if place.is_none() {
            unresolved.push("stand: no place named — needs evidence".to_string());
        }
        actions.push(SceneAction {
            atype: "stand".to_string(),
            actor: actor.clone(),
            target: place,
            object: None,
            ambiguous: false,
        });
    }
    if has("sit") {
        // "sitting in/on X" sits in the named object or place, if any.
        let seat = objects.first().map(|o| o.otype.clone());
        if seat.is_none() {
            unresolved.push("sit: no seat named — needs evidence".to_string());
        }
        actions.push(SceneAction {
            atype: "sit".to_string(),
            actor: actor.clone(),
            target: seat,
            object: None,
            ambiguous: false,
        });
    }
    if has("peace_sign") {
        actions.push(SceneAction {
            atype: "peace_sign".to_string(),
            actor: actor.clone(),
            target: None,
            object: None,
            ambiguous: false,
        });
    }
    if has("cross") {
        // "crossing X" crosses the named place, if any.
        let place = objects
            .iter()
            .find(|o| kind_of(&o.otype).is_some_and(|k| k == ConceptKind::Place))
            .map(|o| o.otype.clone());
        if place.is_none() {
            unresolved.push("cross: no place named — needs evidence".to_string());
        }
        actions.push(SceneAction {
            atype: "cross".to_string(),
            actor: actor.clone(),
            target: place,
            object: None,
            ambiguous: false,
        });
    }

    // Confidence: ontology-mapped content tokens over content tokens.
    // The light verb "give" ("giving a sign") is forgiven; words bound
    // by the material edge count as consumed — ANY material, since the
    // edge (not a word list) recognized them.
    let edge_markers = ["made", "out", "from", "built", "constructed"];
    let edge_fired = material_edge.is_some();
    let edge_material = material_edge
        .as_ref()
        .map(|(_, m)| m.clone())
        .unwrap_or_default();
    let edge_base = material_edge
        .as_ref()
        .map(|(b, _)| b.clone())
        .unwrap_or_default();
    let content: Vec<&String> = folded
        .iter()
        .filter(|w| !STOPWORDS.contains(&w.as_str()))
        .collect();
    let total = content
        .iter()
        .filter(|w| concept_of(w).is_none_or(|c| c != "give"))
        .count()
        .max(1);
    let mapped = content
        .iter()
        .filter(|w| match concept_of(w) {
            Some("give") | None => {
                edge_fired
                    && (edge_markers.contains(&w.as_str())
                        || w.as_str() == edge_material
                        || w.as_str() == edge_base)
            }
            // A wear verb that actually bound a wearer is consumed.
            Some("wear") => wears,
            Some(c) => kind_of(c).is_some(),
        })
        .count();
    let confidence = (mapped as f64 / total as f64).clamp(0.0, 1.0);

    SceneSpec {
        subjects,
        actions,
        objects,
        unresolved,
        confidence,
    }
}

/// One research requirement: what to hunt, the exact queries, and
/// which engine capability must consume the findings.
#[derive(Debug, Clone)]
pub struct ResearchQuery {
    pub requirement: String,
    pub queries: Vec<String>,
    pub capability: &'static str,
}

/// Research planner: scene requirements → targeted searches (the
/// table, as code). Finding a salute photo and being ABLE to salute
/// are different things — the capability tag keeps them apart.
pub fn plan_research(spec: &SceneSpec) -> Vec<ResearchQuery> {
    let mut out = Vec::new();
    for s in &spec.subjects {
        let label = if s.attributes.is_empty() {
            s.stype.clone()
        } else {
            format!("{} {}", s.attributes.join(" "), s.stype)
        };
        // Humans research through the human path; fauna through
        // comparative anatomy (skeleton, proportions, photographs of
        // the animal itself); flora/things through silhouette and
        // reference photographs. Unclassified leftovers research
        // through anatomy for the knowledge graph to resolve, never
        // for the parser to guess.
        let is_human = matches!(s.stype.as_str(), "man" | "woman" | "human");
        let is_fauna = matches!(
            s.stype.as_str(),
            "cat" | "dog" | "elephant" | "horse" | "bird" | "fish" | "lion" | "tiger" | "bear"
        );
        let is_flora_thing = matches!(
            s.stype.as_str(),
            "tree" | "car" | "house" | "cake" | "jetski"
        );
        if is_human {
            out.push(ResearchQuery {
                requirement: format!("subject: {}", label),
                queries: vec![
                    format!("human {} proportions reference", label),
                    format!("human {} photograph", label),
                    format!("human {} skeleton reference", label),
                    format!("human {} skin hair reference", label),
                ],
                capability: "photographic-subject",
            });
        } else if is_fauna {
            out.push(ResearchQuery {
                requirement: format!("subject: {}", s.stype),
                queries: vec![
                    format!("{} anatomy reference", s.stype),
                    format!("{} skeleton proportions reference", s.stype),
                    format!("{} photograph", s.stype),
                ],
                capability: "photographic-subject",
            });
        } else if is_flora_thing {
            out.push(ResearchQuery {
                requirement: format!("subject: {}", s.stype),
                queries: vec![
                    format!("{} silhouette reference", s.stype),
                    format!("{} photograph", s.stype),
                ],
                capability: "photo-asset",
            });
        } else {
            out.push(ResearchQuery {
                requirement: format!("subject: {}", s.stype),
                queries: vec![
                    format!("{} anatomy reference", s.stype),
                    format!("{} body proportions reference", s.stype),
                    format!("{} photograph", s.stype),
                ],
                capability: "photographic-subject",
            });
        }
    }
    for a in &spec.actions {
        match a.atype.as_str() {
            "salute" => out.push(ResearchQuery {
                requirement: "action: salute".to_string(),
                queries: vec![
                    "saluting hand-to-forehead pose reference".to_string(),
                    "shoulder rotation elbow angle reference".to_string(),
                ],
                capability: "articulated-pose",
            }),
            "wave" => {
                if a.object.is_some() {
                    out.push(ResearchQuery {
                        requirement: "action: wave flag".to_string(),
                        queries: vec![
                            "united states flag design reference".to_string(),
                            "waving cloth reference".to_string(),
                        ],
                        capability: "photo-cloth",
                    });
                } else {
                    out.push(ResearchQuery {
                        requirement: "action: wave (ambiguous)".to_string(),
                        queries: vec!["waving hand pose reference".to_string()],
                        capability: "articulated-pose",
                    });
                }
            }
            "stand" => out.push(ResearchQuery {
                requirement: format!("action: stand {}", a.target.as_deref().unwrap_or("nowhere")),
                queries: vec!["standing full-body pose reference".to_string()],
                capability: "articulated-pose",
            }),
            // Sitting is researched posture, not articulated joints:
            // measured seated donors drive fresh construction — no
            // rig, no montage, honestly stated.
            "sit" => out.push(ResearchQuery {
                requirement: format!("action: sit {}", a.target.as_deref().unwrap_or("nowhere")),
                queries: vec![
                    "sitting pose reference".to_string(),
                    "seated figure photograph".to_string(),
                ],
                capability: "fresh-construction",
            }),
            "peace_sign" => out.push(ResearchQuery {
                requirement: "action: peace_sign".to_string(),
                queries: vec!["peace sign hand pose reference".to_string()],
                capability: "articulated-pose",
            }),
            // Crossing is researched motion, not articulated joints:
            // measured crossing donors drive fresh construction — no
            // rig, no montage, honestly stated.
            "cross" => out.push(ResearchQuery {
                requirement: format!("action: cross {}", a.target.as_deref().unwrap_or("nowhere")),
                queries: vec![
                    format!(
                        "crossing {} photograph",
                        a.target.as_deref().unwrap_or("place")
                    ),
                    "crossing motion reference".to_string(),
                ],
                capability: "fresh-construction",
            }),
            _ => {}
        }
    }
    for o in &spec.objects {
        match o.otype.as_str() {
            "hat" => {
                out.push(ResearchQuery {
                    requirement: format!("object: {}", o.id),
                    queries: vec!["hat silhouette reference".to_string()],
                    capability: "photo-asset",
                });
            }
            "flag" | "american_flag" => {
                if !spec
                    .actions
                    .iter()
                    .any(|a| a.object.as_deref() == Some(&o.id))
                {
                    out.push(ResearchQuery {
                        requirement: format!("object: {}", o.id),
                        queries: vec!["united states flag design reference".to_string()],
                        capability: "photo-asset",
                    });
                }
            }
            "mountain" => out.push(ResearchQuery {
                requirement: format!("place: {}", o.id),
                queries: vec!["mountain landscape photograph".to_string()],
                capability: "photographic-place",
            }),
            "river" => out.push(ResearchQuery {
                requirement: format!("place: {}", o.id),
                queries: vec!["river landscape photograph".to_string()],
                capability: "photographic-place",
            }),
            // Generic object (any base noun): silhouette + reference.
            other => out.push(ResearchQuery {
                requirement: format!("object: {}", o.id),
                queries: vec![
                    format!("{} silhouette reference", other),
                    format!("{} reference", other),
                ],
                capability: "photo-asset",
            }),
        }
        // Material queries attach to whatever object carries them —
        // generic over both object and material.
        if let Some(m) = &o.material {
            out.push(ResearchQuery {
                requirement: format!("material: {} on {}", m, o.id),
                queries: vec![
                    format!("{} geometry texture reference", m),
                    format!("{} material reference", m),
                ],
                capability: "photo-asset",
            });
        }
    }
    out
}

/// Engine capability registry: what the renderer can actually do.
/// `true` = research-measurement + fresh-construction paths that
/// exist (plates teach numbers, the raytracer makes pixels).
/// `false` = honestly missing — the matcher names the gap instead
/// of the engine substituting a block figure.
const CAPABILITIES: &[(&str, bool, &str)] = &[
    (
        "photographic-subject",
        true,
        "plates as research: measured tone/proportion donors, fresh render",
    ),
    (
        "photographic-place",
        true,
        "plates as research: measured palette donors, rendered backdrop",
    ),
    (
        "fresh-construction",
        true,
        "all pixels rendered from researched measurements, never copied",
    ),
    (
        "photo-asset",
        true,
        "researched structural measurements with provenance",
    ),
    (
        "articulated-pose",
        false,
        "no joint rig — needs photographic pose match (not implemented)",
    ),
    (
        "photo-cloth",
        false,
        "no cloth solver — needs photographic cloth match (not implemented)",
    ),
    (
        "procedural-human-geometry",
        false,
        "deleted — capsule people read as Minecraft",
    ),
];

/// What the engine can satisfy, requirement by requirement.
#[derive(Debug, Clone)]
pub struct CapabilityVerdict {
    pub requirement: String,
    pub supported: bool,
    /// Set exactly when unsupported: the missing capability, named.
    pub missing: Option<String>,
}

/// Capability matcher: every research requirement meets the
/// registry. Unsupported requirements name their missing capability
/// — that name IS the next work order, not a silent substitution.
pub fn match_capabilities(plan: &[ResearchQuery]) -> Vec<CapabilityVerdict> {
    plan.iter()
        .map(|q| {
            let supported = CAPABILITIES
                .iter()
                .find(|(c, _, _)| *c == q.capability)
                .is_some_and(|(_, ok, _)| *ok);
            CapabilityVerdict {
                requirement: q.requirement.clone(),
                missing: if supported {
                    None
                } else {
                    Some(q.capability.to_string())
                },
                supported,
            }
        })
        .collect()
}

/// Capability description for logs and receipts.
pub fn describe_capability(name: &str) -> Option<&'static str> {
    CAPABILITIES
        .iter()
        .find(|(c, _, _)| *c == name)
        .map(|(_, _, d)| *d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phrases_fold_before_words() {
        assert_eq!(
            fold_phrases(&["american".into(), "flag".into(), "day".into()]),
            vec!["american_flag".to_string(), "day".to_string()]
        );
        assert_eq!(
            fold_phrases(&["peace".into(), "sign".into()]),
            vec!["peace_sign".to_string()]
        );
    }
}
