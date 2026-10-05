//! Conditioning contract: the B→C interface of the pipeline.
//!
//! Research extracts truth from temporary photographs; generation
//! consumes conditioning. Raw photographs NEVER cross this boundary
//! — only measurements, each carrying its provenance. The phone
//! researches, extracts, discards the bytes, and keeps this: a
//! compact, serializable, hash-pinned description of what the
//! evidence supports.
//!
//! Determinism contract: same truth inputs build byte-identical
//! conditioning. The hash covers content only (methods, values,
//! donor counts) — wall-clock audit metadata rides alongside but
//! outside the hash, so re-runs reproduce bit-for-bit. Fields the
//! truth cannot support are None with the gap named, never defaults
//! smuggled in as knowledge.

use serde::{Deserialize, Serialize};

/// What the scene contains, in researched nouns.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SceneCond {
    pub subjects: Vec<String>,
    pub action: Option<String>,
    pub places: Vec<String>,
}

/// Pose truth: the action plus what constrains it. Joint-level
/// articulation stays None until a pose model measures it — the
/// generator must compose photographically, never invent joints.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PoseCond {
    pub action: String,
    pub actor_kind: String,
    pub target_place: Option<String>,
    /// Donor examples behind the pose (count, not pixels).
    pub donor_count: usize,
}

/// Framing truth: measured subject fill + camera convention. Shot
/// descriptions ("medium-long", "eye-level") derive from fill
/// fractions through fixed bands — stated below, never felt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompositionCond {
    /// Subject height as a fraction of frame height, when measured.
    pub subject_fill: Option<f64>,
    /// Banded shot description, deterministic from fill.
    pub shot: String,
}

impl CompositionCond {
    pub fn from_fill(fill: Option<f64>) -> Self {
        let shot = match fill {
            Some(f) if f >= 0.7 => "close-up",
            Some(f) if f >= 0.4 => "medium shot",
            Some(f) if f >= 0.15 => "medium-long shot",
            Some(_) => "wide shot",
            None => "unconstrained",
        }
        .to_string();
        CompositionCond {
            subject_fill: fill,
            shot,
        }
    }
}

/// Light truth: measured backdrop palette + quality. Quality words
/// come from measured spread (low spread = diffuse/even, high =
/// directional/contrasty) — vocabulary from arithmetic, not taste.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LightCond {
    pub top: Option<(u8, u8, u8)>,
    pub bottom: Option<(u8, u8, u8)>,
    pub quality: String,
}

impl LightCond {
    pub fn from_palette(top: Option<(u8, u8, u8)>, bottom: Option<(u8, u8, u8)>) -> Self {
        let spread = match (top, bottom) {
            (Some(t), Some(b)) => {
                ((t.0 as i32 - b.0 as i32).abs()
                    + (t.1 as i32 - b.1 as i32).abs()
                    + (t.2 as i32 - b.2 as i32).abs()) as f64
                    / 3.0
            }
            _ => -1.0,
        };
        let quality = if spread < 0.0 {
            "unmeasured"
        } else if spread < 25.0 {
            "diffuse daylight"
        } else {
            "directional light"
        }
        .to_string();
        LightCond {
            top,
            bottom,
            quality,
        }
    }
}

/// Environment truth: setting nouns from the researched scene plus
/// the palette they were measured in.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvironmentCond {
    pub setting: Vec<String>,
    pub palette: Vec<(u8, u8, u8)>,
}

/// Subject appearance truth: measured palette + donor count. A
/// subject with no measured palette contributes its nouns only —
/// the generator must not dress a person it cannot see.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SubjectCond {
    pub kind: String,
    pub palette: Option<(u8, u8, u8)>,
    pub palette_spread: Option<(u8, u8, u8)>,
    pub donor_count: usize,
}

/// One truth reference: which requirement, by what method, over how
/// many donors. The audit trail from conditioning back to evidence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TruthRef {
    pub requirement: String,
    pub method: String,
    pub donors: usize,
}

/// Corpus versions behind the research: content hashes where the
/// source is versioned (Open Images metadata files), query text
/// otherwise. Same corpus + same procedure = same conditioning.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct CorpusVersion {
    pub openimages_bbox_sha: Option<String>,
    pub openimages_images_sha: Option<String>,
    pub queries: Vec<String>,
}

/// The full conditioning: everything generation may use, nothing
/// it may not. Compact, serializable, hash-pinned — this (plus a
/// seed) is the complete input to the single forward pass.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Conditioning {
    pub scene: SceneCond,
    pub pose: Option<PoseCond>,
    pub composition: CompositionCond,
    pub light: LightCond,
    pub environment: EnvironmentCond,
    pub subjects: Vec<SubjectCond>,
    pub refs: Vec<TruthRef>,
    pub corpus: CorpusVersion,
    /// Gaps the truth could not fill: the seeker's work orders,
    /// carried into generation so it knows its own uncertainty.
    pub gaps: Vec<String>,
}

/// Canonical hash of conditioning content (audit timestamp
/// excluded): same truth in, same hash out, on any run.
pub fn conditioning_hash(c: &Conditioning) -> String {
    use sha2::{Digest, Sha256};
    let json = serde_json::to_string(c).unwrap_or_default();
    let mut h = Sha256::new();
    h.update(json.as_bytes());
    format!("{:x}", h.finalize())
}

/// Build subject conditioning from measured face truth. Unmeasured
/// fields become palette gaps — the subject keeps its nouns, never
/// gains invented tones.
pub fn subject_from_face_truth(
    kind: &str,
    truth: &super::truth::FaceTruth,
    donors: usize,
) -> SubjectCond {
    match &truth.skin {
        super::truth::Truth::Measured { value, .. } => SubjectCond {
            kind: kind.to_string(),
            palette: Some((value.median.r, value.median.g, value.median.b)),
            palette_spread: Some(value.spread),
            donor_count: donors,
        },
        super::truth::Truth::Insufficient { .. } => SubjectCond {
            kind: kind.to_string(),
            palette: None,
            palette_spread: None,
            donor_count: donors,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elephant_cond() -> Conditioning {
        Conditioning {
            scene: SceneCond {
                subjects: vec!["elephant".to_string()],
                action: Some("cross".to_string()),
                places: vec!["river".to_string()],
            },
            pose: Some(PoseCond {
                action: "cross".to_string(),
                actor_kind: "elephant".to_string(),
                target_place: Some("river".to_string()),
                donor_count: 9,
            }),
            composition: CompositionCond::from_fill(Some(0.37)),
            light: LightCond::from_palette(Some((143, 124, 99)), Some((120, 110, 95))),
            environment: EnvironmentCond {
                setting: vec!["river".to_string()],
                palette: vec![(143, 124, 99)],
            },
            subjects: vec![SubjectCond {
                kind: "elephant".to_string(),
                palette: Some((143, 124, 99)),
                palette_spread: Some((40, 38, 35)),
                donor_count: 9,
            }],
            refs: vec![TruthRef {
                requirement: "subject: elephant".to_string(),
                method: "anatomy-reference".to_string(),
                donors: 9,
            }],
            corpus: CorpusVersion {
                queries: vec!["elephant river".to_string()],
                ..Default::default()
            },
            gaps: vec!["hair structure unmeasured — needs hair extractor".to_string()],
        }
    }

    #[test]
    fn conditioning_is_deterministic_and_pinned() {
        let a = elephant_cond();
        let b = elephant_cond();
        assert_eq!(conditioning_hash(&a), conditioning_hash(&b));
        assert_eq!(conditioning_hash(&a).len(), 64);
        // Different truth → different hash, never silently equal.
        let mut c = elephant_cond();
        c.subjects[0].palette = Some((1, 2, 3));
        assert_ne!(conditioning_hash(&a), conditioning_hash(&c));
    }

    #[test]
    fn conditioning_roundtrips_for_the_phone() {
        // The phone carries this over the wire or in the index: it
        // must survive serialization exactly.
        let a = elephant_cond();
        let back: Conditioning = serde_json::from_str(&serde_json::to_string(&a).unwrap()).unwrap();
        assert_eq!(a, back);
        assert_eq!(conditioning_hash(&a), conditioning_hash(&back));
    }

    #[test]
    fn bands_derive_from_arithmetic() {
        assert_eq!(CompositionCond::from_fill(Some(0.8)).shot, "close-up");
        assert_eq!(
            CompositionCond::from_fill(Some(0.37)).shot,
            "medium-long shot"
        );
        assert_eq!(CompositionCond::from_fill(None).shot, "unconstrained");
        assert_eq!(
            LightCond::from_palette(Some((143, 124, 99)), Some((120, 110, 95))).quality,
            "diffuse daylight"
        );
        assert_eq!(LightCond::from_palette(None, None).quality, "unmeasured");
    }

    #[test]
    fn gaps_travel_never_defaults() {
        // Unmeasured skin: nouns kept, tones absent.
        let crops = vec![crate::engine::vision::Image::blank(
            16,
            16,
            crate::engine::vision::Rgb::new(60, 80, 120),
        )];
        let truth = crate::engine::truth::face_truth(&crops, &[None]);
        let s = subject_from_face_truth("woman", &truth, 1);
        assert_eq!(s.kind, "woman");
        assert!(s.palette.is_none(), "no tones without measurement");
    }
}
