//! Phase 2 pose suite: learn poses from validated examples.
//!
//! Normalization unites mirror sides, dissimilar actions cluster
//! apart, representatives apply through FK, confidence tracks
//! tightness, and unknown actions yield nothing — never a guess.
use grounding_coder::engine::pose_learn::{
    PoseExample, cluster, examples_for, normalize, pose_distance, representative,
};
use grounding_coder::engine::skeleton::{BodyProportions, Joint, Pose};

fn normed(action: &str) -> Vec<PoseExample> {
    examples_for(action).iter().map(normalize).collect()
}

#[test]
fn registry_covers_spec_actions() {
    for action in ["raise", "wave", "salute", "point", "sit", "walk"] {
        assert!(
            !examples_for(action).is_empty(),
            "no examples for {}",
            action
        );
    }
    assert!(examples_for("teleport").is_empty());
    assert!(examples_for("dance").is_empty());
}

#[test]
fn mirror_sides_normalize_together() {
    for action in ["salute", "wave", "point", "raise"] {
        let ns = normed(action);
        assert!(ns.len() >= 2, "need both sides for {}", action);
        let d = pose_distance(&ns[0], &ns[1]);
        assert!(d < 1e-9, "{} sides differ after normalize: {}", action, d);
    }
}

#[test]
fn salute_variants_cluster_wave_apart() {
    let mut all = normed("salute");
    all.extend(normed("wave"));
    let clusters = cluster(all, 15.0);
    assert!(clusters.len() >= 2, "salute and wave must separate");
    let salute_cluster = clusters
        .iter()
        .find(|c| c.iter().any(|m| m.name.contains("salute-right")))
        .expect("salute cluster");
    assert!(
        salute_cluster.iter().all(|m| m.name.contains("salute")),
        "salute cluster polluted: {:?}",
        salute_cluster.iter().map(|m| &m.name).collect::<Vec<_>>()
    );
    assert!(
        salute_cluster.len() >= 4,
        "variants must join: {}",
        salute_cluster.len()
    );
}

#[test]
fn salute_cluster_unites_at_fifteen_degrees() {
    let clusters = cluster(normed("salute"), 15.0);
    assert_eq!(clusters.len(), 1, "one salute family");
    assert_eq!(clusters[0].len(), 5);
}

#[test]
fn representative_salute_applies_to_head() {
    let clusters = cluster(normed("salute"), 15.0);
    let rep = representative(&clusters[0]).expect("representative");
    assert_eq!(rep.n, 5);
    assert!(rep.confidence > 0.2, "tight cluster: {}", rep.confidence);
    assert!(
        rep.sources.iter().all(|s| s.contains("rom-table")),
        "every source cited"
    );
    // Medoid is a real member: shoulder flexion is a variant value.
    let sh = rep.rotations[&Joint::ShoulderR].flex;
    assert!(
        [145.0, 150.0, 155.0].contains(&sh),
        "medoid must be a member: {}",
        sh
    );
    // Ranges span the variants.
    let (lo, hi) = rep.ranges[&Joint::ShoulderR];
    assert!(lo <= 145.0 && hi >= 155.0, "range {:?}", (lo, hi));
    // Apply: the learned salute reaches the head.
    let pos = rep.apply(&BodyProportions::adult_male()).unwrap();
    let hand = pos[&Joint::HandR];
    let head = pos[&Joint::Head];
    let d = ((hand.x - head.x).powi(2) + (hand.y - head.y).powi(2)).sqrt();
    assert!(d < 0.45, "learned salute must reach: {:.2}m", d);
}

#[test]
fn confidence_tracks_tightness() {
    let tight = cluster(normed("salute"), 15.0);
    let loose_input: Vec<PoseExample> =
        normed("salute").into_iter().chain(normed("wave")).collect();
    let loose = cluster(loose_input, 1000.0);
    assert_eq!(loose.len(), 1);
    let ct = representative(&tight[0]).unwrap().confidence;
    let cl = representative(&loose[0]).unwrap().confidence;
    assert!(ct > cl, "tight {} must beat loose {}", ct, cl);
}

#[test]
fn identical_examples_score_one() {
    let ex = examples_for("sit");
    assert_eq!(ex.len(), 1);
    let pair = vec![ex[0].clone(), ex[0].clone()];
    let rep = representative(&pair).unwrap();
    assert!((rep.confidence - 1.0).abs() < 1e-9);
}

#[test]
fn every_registry_action_applies_clean() {
    // Each learnable action derives a representative that FK-applies
    // without ROM refusal — synthesis works before any photograph.
    for action in ["raise", "wave", "salute", "point", "sit", "walk"] {
        let clusters = cluster(normed(action), 45.0);
        assert!(!clusters.is_empty(), "{}", action);
        for c in &clusters {
            let rep = representative(c).unwrap();
            rep.apply(&BodyProportions::adult_male())
                .unwrap_or_else(|e| panic!("{} representative refuses: {}", action, e));
        }
    }
}

#[test]
fn ingestion_refuses_impossible_angles() {
    use grounding_coder::engine::pose_learn::PoseExample as PE;
    use grounding_coder::engine::skeleton::JointAngles;
    let mut p = Pose::rest();
    p.set(Joint::KneeL, JointAngles::flex(200.0));
    assert!(PE::from_pose("bad-knee", &p, "test").is_err());
}

#[test]
fn empty_cluster_refuses_representative() {
    use grounding_coder::engine::pose_learn::representative as rep;
    assert!(rep(&[]).is_err());
    assert!(cluster(vec![], 15.0).is_empty());
}
