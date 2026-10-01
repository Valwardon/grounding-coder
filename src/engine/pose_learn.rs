//! Pose learning — Phase 2 of the image-creation system.
//!
//! Examples → normalize → cluster → representative pose → apply.
//! Pose examples carry their source (ROM-table named poses and their
//! validated variants — never invented angles); normalization mirrors
//! left-side poses into right-side canonical form; clustering is
//! greedy deterministic complete-linkage; the representative is the
//! medoid (a real validated member — means can violate ROM, medoids
//! cannot) with confidence from cluster tightness.
//!
//! HONEST LIMIT: examples come from the ROM-validated pose library,
//! not from photographs — photo-derived pose needs fine joints, which
//! `body_measure` reports missing. When researched references arrive
//! they ingest through the same `PoseExample` gate (ROM-validated or
//! refused). Applying a pose means FK on the measurement scaffold
//! (Phase 1), never a rendered human.

use super::skeleton::{BodyProportions, Joint, JointAngles, Pose, V3, forward_kinematics};
use std::collections::HashMap;

/// One pose example: joint angles plus where they came from.
/// Ingestion validates against ROM — an impossible example refuses
/// at the gate, never enters a cluster.
#[derive(Debug, Clone)]
pub struct PoseExample {
    pub name: String,
    pub angles: HashMap<Joint, JointAngles>,
    pub source: String,
}

impl PoseExample {
    /// Build from a named pose. Validates before accepting.
    pub fn from_pose(name: &str, pose: &Pose, source: &str) -> Result<Self, String> {
        super::skeleton::validate_pose_rom(pose)?;
        let mut angles = HashMap::new();
        for j in Joint::all() {
            angles.insert(j, pose.get(j));
        }
        Ok(PoseExample {
            name: name.to_string(),
            angles,
            source: source.to_string(),
        })
    }

    pub fn to_pose(&self) -> Pose {
        let mut p = Pose::rest();
        for (j, a) in &self.angles {
            p.set(*j, *a);
        }
        p
    }
}

/// Mirror a joint left↔right. Center joints map to themselves.
pub fn mirror_joint(j: Joint) -> Joint {
    use Joint::*;
    match j {
        ShoulderL => ShoulderR,
        ShoulderR => ShoulderL,
        ElbowL => ElbowR,
        ElbowR => ElbowL,
        WristL => WristR,
        WristR => WristL,
        HandL => HandR,
        HandR => HandL,
        HipL => HipR,
        HipR => HipL,
        KneeL => KneeR,
        KneeR => KneeL,
        AnkleL => AnkleR,
        AnkleR => AnkleL,
        FootL => FootR,
        FootR => FootL,
        c => c,
    }
}

/// Laterality of a joint: right, left, or center (None).
fn side(j: Joint) -> Option<bool> {
    use Joint::*;
    match j {
        ShoulderR | ElbowR | WristR | HandR | HipR | KneeR | AnkleR | FootR => Some(true),
        ShoulderL | ElbowL | WristL | HandL | HipL | KneeL | AnkleL | FootL => Some(false),
        _ => None,
    }
}

/// Normalize into canonical (right-side) form: the dominant side's
/// numbers move onto the right joints, the other side rests, center
/// joints pass through. A left salute and a right salute normalize
/// to the same pose — that is the point.
///
/// Scope, stated: single-sided poses. Both-sided input (gait) takes
/// the dominant side; comparing gaits needs phase alignment, which
/// is not this function.
pub fn normalize(ex: &PoseExample) -> PoseExample {
    let energy = |right: bool| -> f64 {
        Joint::all()
            .iter()
            .filter(|j| side(**j) == Some(right))
            .map(|j| {
                let a = ex.angles.get(j).copied().unwrap_or(JointAngles::rest());
                a.flex.abs() + a.abd.abs()
            })
            .sum()
    };
    let from_right = energy(true) >= energy(false);
    let mut angles = HashMap::new();
    for j in Joint::all() {
        let a = match side(j) {
            None => ex.angles.get(&j).copied().unwrap_or(JointAngles::rest()),
            Some(true) => {
                let src = if from_right { j } else { mirror_joint(j) };
                ex.angles.get(&src).copied().unwrap_or(JointAngles::rest())
            }
            Some(false) => JointAngles::rest(),
        };
        angles.insert(j, a);
    }
    PoseExample {
        name: format!("{}:normalized", ex.name),
        angles,
        source: ex.source.clone(),
    }
}

/// RMS joint-angle distance in degrees. Deterministic.
pub fn pose_distance(a: &PoseExample, b: &PoseExample) -> f64 {
    let mut sum = 0.0;
    let mut n = 0u32;
    for j in Joint::all() {
        let x = a.angles.get(&j).copied().unwrap_or(JointAngles::rest());
        let y = b.angles.get(&j).copied().unwrap_or(JointAngles::rest());
        sum += (x.flex - y.flex).powi(2) + (x.abd - y.abd).powi(2);
        n += 1;
    }
    (sum / n as f64).sqrt()
}

/// Greedy deterministic complete-linkage clustering: examples in
/// name order join the first cluster whose every member sits within
/// `threshold` degrees, else open a new cluster.
pub fn cluster(mut examples: Vec<PoseExample>, threshold: f64) -> Vec<Vec<PoseExample>> {
    examples.sort_by(|a, b| a.name.cmp(&b.name));
    let mut clusters: Vec<Vec<PoseExample>> = Vec::new();
    for ex in examples {
        let mut placed = false;
        for c in clusters.iter_mut() {
            if c.iter().all(|m| pose_distance(&ex, m) < threshold) {
                c.push(ex.clone());
                placed = true;
                break;
            }
        }
        if !placed {
            clusters.push(vec![ex]);
        }
    }
    clusters
}

/// The derived pose: medoid rotations, per-joint ranges, tightness
/// confidence, member count, and every source. The spec's HumanPose.
#[derive(Debug, Clone)]
pub struct HumanPose {
    pub rotations: HashMap<Joint, JointAngles>,
    pub ranges: HashMap<Joint, (f64, f64)>,
    /// 1/(1+mean medoid distance): 1.0 identical, →0 scattered.
    pub confidence: f64,
    pub n: usize,
    pub sources: Vec<String>,
}

/// Derive the representative pose of a cluster. Empty clusters
/// refuse — there is nothing to represent.
pub fn representative(cluster: &[PoseExample]) -> Result<HumanPose, String> {
    if cluster.is_empty() {
        return Err("empty cluster has no representative — refusing".to_string());
    }
    // Medoid: the member closest to all others. Always a validated
    // example, so the representative is ROM-valid by construction.
    let mut best = 0usize;
    let mut best_total = f64::INFINITY;
    for (i, cand) in cluster.iter().enumerate() {
        let total: f64 = cluster.iter().map(|m| pose_distance(cand, m)).sum();
        if total < best_total {
            best_total = total;
            best = i;
        }
    }
    let medoid = &cluster[best];
    let mean_dist = best_total / cluster.len() as f64;
    let mut rotations = HashMap::new();
    let mut ranges = HashMap::new();
    for j in Joint::all() {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for m in cluster {
            let a = m.angles.get(&j).copied().unwrap_or(JointAngles::rest());
            lo = lo.min(a.flex);
            hi = hi.max(a.flex);
        }
        rotations.insert(
            j,
            medoid
                .angles
                .get(&j)
                .copied()
                .unwrap_or(JointAngles::rest()),
        );
        ranges.insert(j, (lo, hi));
    }
    Ok(HumanPose {
        rotations,
        ranges,
        confidence: 1.0 / (1.0 + mean_dist),
        n: cluster.len(),
        sources: cluster.iter().map(|m| m.source.clone()).collect(),
    })
}

impl HumanPose {
    /// Apply to a proportioned body: FK joint positions.
    pub fn apply(&self, props: &BodyProportions) -> Result<HashMap<Joint, V3>, String> {
        let mut p = Pose::rest();
        for (j, a) in &self.rotations {
            p.set(*j, *a);
        }
        forward_kinematics(props, &p)
    }
}

/// Validated example records per action — the learnable registry.
/// Unknown actions yield nothing (no examples is not a guess).
/// Each record cites its ROM-table source.
pub fn examples_for(action: &str) -> Vec<PoseExample> {
    let mut out = Vec::new();
    let mut add = |name: String, pose: Pose, source: String| {
        if let Ok(ex) = PoseExample::from_pose(&name, &pose, &source) {
            out.push(ex);
        }
    };
    match action {
        "salute" => {
            add(
                "salute-right".into(),
                Pose::salute(true),
                "rom-table:Pose::salute(right)".into(),
            );
            add(
                "salute-left".into(),
                Pose::salute(false),
                "rom-table:Pose::salute(left)".into(),
            );
            // Validated variants: nearby angles inside ROM.
            for (i, (sh, el)) in [(150.0, 128.0), (155.0, 132.0), (145.0, 124.0)]
                .into_iter()
                .enumerate()
            {
                let mut p = Pose::rest();
                p.set(Joint::ShoulderR, JointAngles::flex(sh));
                p.set(Joint::ElbowR, JointAngles::flex(el));
                p.set(
                    Joint::WristR,
                    JointAngles {
                        flex: 10.0,
                        abd: 10.0,
                    },
                );
                add(
                    format!("salute-variant-{}", i),
                    p,
                    "rom-table:validated variant".into(),
                );
            }
        }
        "wave" => {
            add(
                "wave-right".into(),
                Pose::wave(true),
                "rom-table:Pose::wave(right)".into(),
            );
            add(
                "wave-left".into(),
                Pose::wave(false),
                "rom-table:Pose::wave(left)".into(),
            );
        }
        "raise" => {
            add(
                "raise-right".into(),
                Pose::raise_hand(true),
                "rom-table:Pose::raise_hand(right)".into(),
            );
            add(
                "raise-left".into(),
                Pose::raise_hand(false),
                "rom-table:Pose::raise_hand(left)".into(),
            );
        }
        "point" => {
            add(
                "point-right".into(),
                Pose::point(true),
                "rom-table:Pose::point(right)".into(),
            );
            add(
                "point-left".into(),
                Pose::point(false),
                "rom-table:Pose::point(left)".into(),
            );
        }
        "sit" => {
            add(
                "sit".into(),
                Pose::sitting(),
                "rom-table:Pose::sitting".into(),
            );
        }
        "walk" => {
            for (i, ph) in [
                0.0,
                std::f64::consts::FRAC_PI_2,
                std::f64::consts::PI,
                3.0 * std::f64::consts::FRAC_PI_2,
            ]
            .into_iter()
            .enumerate()
            {
                add(
                    format!("walk-phase-{}", i),
                    Pose::walking(ph),
                    "rom-table:Pose::walking(phase)".into(),
                );
            }
        }
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_unites_mirror_sides() {
        let l = PoseExample::from_pose("l", &Pose::salute(false), "t").unwrap();
        let r = PoseExample::from_pose("r", &Pose::salute(true), "t").unwrap();
        let raw = pose_distance(&l, &r);
        assert!(raw > 10.0, "sides must differ raw: {}", raw);
        let d = pose_distance(&normalize(&l), &normalize(&r));
        assert!(d < 1e-9, "normalized sides must match: {}", d);
    }

    #[test]
    fn impossible_examples_refuse_at_ingestion() {
        let mut p = Pose::rest();
        p.set(Joint::ElbowR, JointAngles::flex(200.0));
        assert!(PoseExample::from_pose("bad", &p, "t").is_err());
    }

    #[test]
    fn empty_cluster_has_no_representative() {
        assert!(representative(&[]).is_err());
    }
}
