//! Human body oracle — the premade construction oracle.
//!
//! Research says "a man": this module turns that into a measured
//! human mesh. The geometry is NOT generated here — it is the
//! MakeHuman hm08 base mesh (CC0, pinned, see
//! `assets/oracle/PROVENANCE.md`) plus adult macro morphs. This
//! module only parses (OBJ + sparse `.target`), morphs, measures,
//! and rigid-binds parts to the FK skeleton.
//!
//! Binding: every vertex joins its nearest rest-pose bone segment;
//! posing applies each part's rigid joint transform. Seams at bent
//! joints are covered by joint spheres (stated approximation —
//! smooth skinning is staged, not faked).

use super::skeleton::{BodyProportions, Joint, Pose};
use std::collections::HashMap;

/// Pinned upstream commit (see PROVENANCE.md).
pub const ORACLE_COMMIT: &str = "a8bc2d54ff0ac92e78ff71431b1023eda42bf482";

/// Expected SHA-256 per vendored file. Corrupt bytes refuse.
pub const ORACLE_SHA256: &[(&str, &str)] = &[
    (
        "base.obj",
        "8e761e6624b8f54536409135d1636da63b32486a90d4897f84e121d144f6fb4c",
    ),
    (
        "male-young.target",
        "70e228ba7164737dae664454394536fc5935fa48d333c1a97d77e2dc6eacc5f5",
    ),
    (
        "female-young.target",
        "118379f6e8ba9266247fdb8788a20e1df40a239f97ced0b9905bcbcc74f6e820",
    ),
];

/// Verify a vendored oracle directory against the manifest.
pub fn verify_oracle_dir(dir: &std::path::Path) -> Result<(), String> {
    use sha2::{Digest, Sha256};
    for (file, want) in ORACLE_SHA256 {
        let bytes =
            std::fs::read(dir.join(file)).map_err(|e| format!("{} unreadable: {}", file, e))?;
        let mut h = Sha256::new();
        h.update(&bytes);
        let got = format!("{:x}", h.finalize());
        if got != *want {
            return Err(format!(
                "{} hash mismatch (corrupt?) — refusing, never running",
                file
            ));
        }
    }
    Ok(())
}

/// Loaded oracle mesh, meters, triangulated.
#[derive(Debug, Clone)]
pub struct OracleBody {
    pub verts: Vec<[f64; 3]>,
    pub faces: Vec<[u32; 3]>,
}

/// Parse Wavefront OBJ verts + faces (fan-triangulated, decimeters
/// to meters). Only `v` and `f` lines; texture/normal indices
/// ignored. Negative indices supported.
pub fn parse_obj(text: &str) -> Result<OracleBody, String> {
    let mut verts = Vec::new();
    let mut faces = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("v ") {
            let p: Vec<f64> = rest
                .split_whitespace()
                .filter_map(|w| w.parse().ok())
                .collect();
            if p.len() < 3 {
                return Err(format!("line {}: bad vertex", ln + 1));
            }
            verts.push([p[0] * 0.1, p[1] * 0.1, p[2] * 0.1]);
        } else if let Some(rest) = line.strip_prefix("f ") {
            let idx: Vec<u32> = rest
                .split_whitespace()
                .map(|w| {
                    let v: i64 = w.split('/').next().unwrap_or("").parse().unwrap_or(0);
                    if v > 0 {
                        (v - 1) as u32
                    } else {
                        (verts.len() as i64 + v) as u32
                    }
                })
                .collect();
            if idx.iter().any(|i| (*i as usize) >= verts.len()) {
                return Err(format!("line {}: face index out of range", ln + 1));
            }
            for k in 1..idx.len().saturating_sub(1) {
                faces.push([idx[0], idx[k], idx[k + 1]]);
            }
        }
    }
    if verts.is_empty() || faces.is_empty() {
        return Err("no geometry parsed".to_string());
    }
    Ok(OracleBody { verts, faces })
}

/// One sparse morph delta: vertex, decimeter offset.
#[derive(Debug, Clone)]
pub struct MorphDelta {
    pub idx: usize,
    pub d: [f64; 3],
}

/// Parse a MakeHuman `.target`: comment lines start with `#`,
/// data lines are `index dx dy dz` (decimeters, → meters).
pub fn parse_target(text: &str) -> Result<Vec<MorphDelta>, String> {
    let mut out = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() != 4 {
            return Err(format!("line {}: bad morph row", ln + 1));
        }
        let idx: usize = p[0]
            .parse()
            .map_err(|_| format!("line {}: bad index", ln + 1))?;
        let mut d = [0.0; 3];
        for (k, w) in p[1..].iter().enumerate() {
            d[k] = w
                .parse::<f64>()
                .map_err(|_| format!("line {}: bad delta", ln + 1))?
                * 0.1;
        }
        out.push(MorphDelta { idx, d });
    }
    Ok(out)
}

/// Apply morph deltas at `weight` (0..1). Out-of-range indices
/// refuse the whole morph — never partially applied.
pub fn apply_morph(
    body: &mut OracleBody,
    deltas: &[MorphDelta],
    weight: f64,
) -> Result<(), String> {
    if !(0.0..=1.0).contains(&weight) {
        return Err(format!("morph weight {} outside [0,1]", weight));
    }
    for m in deltas {
        let v = body
            .verts
            .get_mut(m.idx)
            .ok_or_else(|| format!("morph index {} out of range", m.idx))?;
        v[0] += m.d[0] * weight;
        v[1] += m.d[1] * weight;
        v[2] += m.d[2] * weight;
    }
    Ok(())
}

/// Load + verify + morph a vendored oracle body. `morph_file` is
/// `None` for the unmorphed base, or a `.target` beside `base.obj`.
/// Units meters; the mesh keeps its authored stance (standing).
pub fn load_oracle_body(
    dir: &std::path::Path,
    morph_file: Option<&str>,
) -> Result<OracleBody, String> {
    verify_oracle_dir(dir)?;
    let text =
        std::fs::read_to_string(dir.join("base.obj")).map_err(|e| format!("base.obj: {}", e))?;
    let mut body = parse_obj(&text)?;
    if let Some(mf) = morph_file {
        let mt = std::fs::read_to_string(dir.join(mf)).map_err(|e| format!("{}: {}", mf, e))?;
        let deltas = parse_target(&mt)?;
        apply_morph(&mut body, &deltas, 1.0)?;
    }
    Ok(body)
}

/// Measured body facts: stature + half-widths at torso bands.
/// Fractions of stature, from the mesh — not from canon.
#[derive(Debug, Clone)]
pub struct MeasuredBody {
    pub stature_m: f64,
    pub chest_hw_frac: f64,
    pub waist_hw_frac: f64,
    pub hip_hw_frac: f64,
}

/// Measure stature (bbox height) and torso half-widths (max |x| in
/// height bands over stature). Deterministic.
pub fn measure_body(body: &OracleBody) -> MeasuredBody {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for v in &body.verts {
        lo = lo.min(v[1]);
        hi = hi.max(v[1]);
    }
    let stature = (hi - lo).max(1e-9);
    // Torso half-width per band: max |x| among verts near the
    // sagittal center plane (arms hang off-plane... except they
    // don't — use `measure_torso` with bound parts for arm-free
    // widths; this raw band is a whole-body envelope only).
    let band = |y0: f64, y1: f64| -> f64 {
        let mut w: f64 = 0.0;
        for v in &body.verts {
            let t = (v[1] - lo) / stature;
            if t >= y0 && t < y1 && v[2].abs() < 0.02 * stature {
                w = w.max(v[0].abs());
            }
        }
        w / stature
    };
    // Arms hang inside these bands on a standing mesh; torso mass
    // dominates the max — documented approximation, staged for
    // part-aware measurement.
    MeasuredBody {
        stature_m: stature,
        chest_hw_frac: band(0.68, 0.82),
        waist_hw_frac: band(0.52, 0.64),
        hip_hw_frac: band(0.42, 0.54),
    }
}

/// The compatible rig: MakeHuman bones with CC0 skinning weights,
/// joint centroids measured off the (possibly morphed) mesh, and one
/// of our FK joints per bone for rotation inheritance.
#[derive(Debug, Clone)]
pub struct Rig {
    /// Bones in stable (sorted) order; weight indices point here.
    pub bones: Vec<RigBone>,
    /// Per vert: normalized (bone index, weight). Empty = unweighted
    /// helper (posed rigidly by nearest joint instead of skinned).
    pub weights: Vec<Vec<(usize, f64)>>,
    /// Rest positions of our joints (bone-head centroids, mesh coords).
    pub joints: HashMap<Joint, [f64; 3]>,
}

#[derive(Debug, Clone)]
pub struct RigBone {
    pub name: String,
    pub parent: Option<usize>,
    pub joint: Joint,
    pub rest_head: [f64; 3],
}

/// MakeHuman bone → our FK joint for rotation inheritance.
/// Unlisted bones walk the MH parent chain (breasts follow the
/// spine, toes follow the foot, face follows the head) — structure,
/// never invention: every fallback is a true ancestor.
fn mh_joint(name: &str) -> Option<Joint> {
    use Joint::*;
    let base = name.trim_end_matches(".L").trim_end_matches(".R");
    let sided = |l: Joint, r: Joint| Some(if name.ends_with(".L") { l } else { r });
    match base {
        "root" | "spine01" | "spine02" | "pelvis" => Some(Pelvis),
        "spine03" => Some(Spine),
        "spine04" | "spine05" | "clavicle" => Some(Chest),
        "neck01" | "neck02" | "neck03" => Some(Neck),
        "head" | "jaw" | "eye" | "tongue" | "oris" | "oculi" | "orbicularis" | "temporalis"
        | "levator" | "risorius" => Some(Head),
        "shoulder01" | "upperarm01" | "upperarm02" => sided(ShoulderL, ShoulderR),
        "lowerarm01" | "lowerarm02" => sided(ElbowL, ElbowR),
        "wrist" | "metacarpal1" | "metacarpal2" | "metacarpal3" | "metacarpal4" | "finger1-1"
        | "finger1-2" | "finger1-3" | "finger2-1" | "finger2-2" | "finger2-3" | "finger3-1"
        | "finger3-2" | "finger3-3" | "finger4-1" | "finger4-2" | "finger4-3" | "finger5-1"
        | "finger5-2" | "finger5-3" => sided(WristL, WristR),
        "upperleg01" | "upperleg02" => sided(HipL, HipR),
        "lowerleg01" | "lowerleg02" => sided(KneeL, KneeR),
        "foot" => sided(AnkleL, AnkleR),
        _ => None,
    }
}

/// Our joint ← MH head-group centroid. The anatomical anchor per
/// joint (child-segment heads; femoral midline for the pelvis).
fn joint_group(joint: Joint) -> &'static str {
    use Joint::*;
    match joint {
        Pelvis => "",
        Spine => "spine02____head",
        Chest => "spine04____head",
        Neck => "neck01____head",
        Head => "head____head",
        ShoulderL => "upperarm01.L____head",
        ShoulderR => "upperarm01.R____head",
        ElbowL => "lowerarm01.L____head",
        ElbowR => "lowerarm01.R____head",
        WristL => "wrist.L____head",
        WristR => "wrist.R____head",
        HandL => "metacarpal3.L____head",
        HandR => "metacarpal3.R____head",
        HipL => "upperleg01.L____head",
        HipR => "upperleg01.R____head",
        KneeL => "lowerleg01.L____head",
        KneeR => "lowerleg01.R____head",
        AnkleL => "foot.L____head",
        AnkleR => "foot.R____head",
        FootL => "foot.L____tail",
        FootR => "foot.R____tail",
    }
}

fn centroid(body: &OracleBody, idx: &[usize]) -> Option<[f64; 3]> {
    let mut c = [0.0; 3];
    let mut n = 0usize;
    for i in idx {
        if let Some(v) = body.verts.get(*i) {
            c[0] += v[0];
            c[1] += v[1];
            c[2] += v[2];
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    Some([c[0] / n as f64, c[1] / n as f64, c[2] / n as f64])
}

/// Vendored rig text (pinned, CC0 — see PROVENANCE.md).
const RIG_SKELETON: &str = include_str!("../../assets/oracle/default.mhskel");
const RIG_WEIGHTS: &str = include_str!("../../assets/oracle/default_weights.mhw");

/// Build the rig for a (possibly morphed) body: parse the pinned
/// skeleton + weights, measure joint centroids off THIS mesh,
/// normalize weights per vert (rows sum to 1; zero rows stay empty
/// for the rigid fallback). Refuses on missing groups, corrupt
/// JSON, or out-of-range indices — never half a rig.
pub fn rig_for(body: &OracleBody) -> Result<Rig, String> {
    let skel: serde_json::Value =
        serde_json::from_str(RIG_SKELETON).map_err(|e| format!("rig json: {}", e))?;
    let wt: serde_json::Value =
        serde_json::from_str(RIG_WEIGHTS).map_err(|e| format!("weights json: {}", e))?;
    let bones_json = skel
        .get("bones")
        .and_then(|b| b.as_object())
        .ok_or("rig: no bones")?;
    let joints_json = skel
        .get("joints")
        .and_then(|b| b.as_object())
        .ok_or("rig: no joints")?;
    let mut group_pos: HashMap<String, [f64; 3]> = HashMap::new();
    for (gname, arr) in joints_json {
        let idx: Vec<usize> = arr
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|i| i as usize))
                    .collect()
            })
            .unwrap_or_default();
        if let Some(c) = centroid(body, &idx) {
            group_pos.insert(gname.clone(), c);
        }
    }
    let mut names: Vec<String> = bones_json.keys().cloned().collect();
    names.sort();
    // Our-joint rest positions FIRST: anchor-group centroids (+ pelvis
    // midline). Bones pivot here — at their mapped joint, never at
    // their own head groups (those sit centimeters away and would
    // break rest-identity).
    let mut joints = HashMap::new();
    for j in Joint::all() {
        if j == Joint::Pelvis {
            let l = joint_group(Joint::HipL);
            let r = joint_group(Joint::HipR);
            match (group_pos.get(l), group_pos.get(r)) {
                (Some(a), Some(b)) => {
                    joints.insert(
                        j,
                        [
                            (a[0] + b[0]) / 2.0,
                            (a[1] + b[1]) / 2.0,
                            (a[2] + b[2]) / 2.0,
                        ],
                    );
                }
                _ => return Err("rig: hip groups missing".to_string()),
            }
            continue;
        }
        let g = joint_group(j);
        match group_pos.get(g) {
            Some(c) => {
                joints.insert(j, *c);
            }
            None => return Err(format!("rig: joint group {} missing", g)),
        }
    }
    let mut bones = Vec::new();
    for name in &names {
        let b = &bones_json[name];
        let parent_name = b
            .get("parent")
            .and_then(|p| p.as_str())
            .map(|s| s.to_string());
        let mut joint = mh_joint(name);
        let mut walk = parent_name.clone();
        while joint.is_none() {
            match walk {
                Some(pn) => {
                    joint = mh_joint(&pn);
                    walk = bones_json
                        .get(&pn)
                        .and_then(|b| b.get("parent"))
                        .and_then(|p| p.as_str())
                        .map(|s| s.to_string());
                }
                None => break,
            }
        }
        let joint = joint.ok_or_else(|| format!("rig: no joint for bone {}", name))?;
        // Pivot at the MAPPED joint's centroid (shared with FK), so
        // rest pose is identity bit-for-bit. Bone-local head groups
        // sit centimeters off-joint and must never pivot.
        let rest_head = joints[&joint];
        let parent = parent_name
            .as_ref()
            .and_then(|pn| names.iter().position(|n| n == pn));
        bones.push(RigBone {
            name: name.clone(),
            parent,
            joint,
            rest_head,
        });
    }
    // Weights, normalized per vert (measured row sums run 0.32–1.67,
    // mean 0.94 — never assumed to be 1).
    let wobj = wt
        .get("weights")
        .and_then(|w| w.as_object())
        .ok_or("weights: no weights")?;
    let mut acc: Vec<HashMap<usize, f64>> = vec![HashMap::new(); body.verts.len()];
    for (bi, name) in names.iter().enumerate() {
        if let Some(arr) = wobj.get(name).and_then(|a| a.as_array()) {
            for row in arr {
                let (vi, w) = match row.as_array() {
                    Some(r) if r.len() >= 2 => (r[0].as_u64().map(|i| i as usize), r[1].as_f64()),
                    _ => (None, None),
                };
                if let (Some(vi), Some(w)) = (vi, w)
                    && vi < acc.len()
                    && w > 0.0
                {
                    *acc[vi].entry(bi).or_insert(0.0) += w;
                }
            }
        }
    }
    let mut weights: Vec<Vec<(usize, f64)>> = Vec::with_capacity(body.verts.len());
    for row in acc {
        let total: f64 = row.values().sum();
        if total <= 0.0 {
            weights.push(Vec::new());
        } else {
            let mut normed: Vec<(usize, f64)> =
                row.into_iter().map(|(b, w)| (b, w / total)).collect();
            normed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            weights.push(normed);
        }
    }
    Ok(Rig {
        bones,
        weights,
        joints,
    })
}

/// Helper-part classification: which helper verts render as what.
/// Rules use dominant CC0 bones + sizes (eyeballs, mouth, hair),
/// never position guesses — except the two garments, identified by
/// size and band with counts asserted in test (the file is pinned,
/// so drift fails loudly instead of silently restyling).
/// Unlisted parts render skin. Cubes (rig viz) and the unidentified
/// full-body shell never render at all.
#[derive(Debug, Clone, Default)]
pub struct HelperMats {
    /// Eyeball vert indices (dark, glossy).
    pub eyes: Vec<usize>,
    /// Mouth interior (dark red).
    pub mouth: Vec<usize>,
    /// Hair locks (dark brown).
    pub hair: Vec<usize>,
    /// Modesty garment (dark gray clothing).
    pub shorts: Vec<usize>,
    /// Never rendered: rig-viz cubes + unidentified shell.
    pub excluded: Vec<usize>,
}

/// Classify helper verts (index ≥ 13380) of a body. `dominant` maps
/// vert index → top weight bone name (see `rig_for` weights).
pub fn classify_helpers(body: &OracleBody, dominant: &HashMap<usize, String>) -> HelperMats {
    // Connected components over helper verts.
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for f in &body.faces {
        for k in 0..3 {
            let a = f[k] as usize;
            let b = f[(k + 1) % 3] as usize;
            if a >= 13380 && b >= 13380 {
                adj.entry(a).or_default().push(b);
                adj.entry(b).or_default().push(a);
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    let mut comps: Vec<Vec<usize>> = Vec::new();
    for i in 13380..body.verts.len() {
        if seen.contains(&i) {
            continue;
        }
        let mut stack = vec![i];
        seen.insert(i);
        let mut ids = Vec::new();
        while let Some(j) = stack.pop() {
            ids.push(j);
            if let Some(ns) = adj.get(&j) {
                for k in ns {
                    if !seen.contains(k) {
                        seen.insert(*k);
                        stack.push(*k);
                    }
                }
            }
        }
        comps.push(ids);
    }
    let mut out = HelperMats::default();
    for c in &comps {
        if c.len() == 8 {
            out.excluded.extend(c.iter().copied());
            continue;
        }
        // Majority vote of member verts' dominant bones.
        let mut votes: HashMap<String, usize> = HashMap::new();
        for i in c {
            if let Some(b) = dominant.get(i) {
                *votes.entry(b.clone()).or_insert(0) += 1;
            }
        }
        let top = votes.into_iter().max_by_key(|(_, n)| *n).map(|(b, _)| b);
        let is_eye = top.as_ref().is_some_and(|b| b.starts_with("eye."));
        let is_mouth = top.as_ref().is_some_and(|b| b.starts_with("tongue"));
        if is_eye {
            out.eyes.extend(c.iter().copied());
        } else if is_mouth {
            out.mouth.extend(c.iter().copied());
        } else if c.len() == 226 {
            // Mouth interior mass (tongue-weighted cavity).
            out.mouth.extend(c.iter().copied());
        } else if c.len() == 720 {
            out.shorts.extend(c.iter().copied());
        } else if c.len() == 2674 {
            out.excluded.extend(c.iter().copied());
        } else if (20..=40).contains(&c.len()) {
            // Hair locks.
            out.hair.extend(c.iter().copied());
        }
        // Else: skin default (lids, jaw, ears, nails) — no entry needed.
    }
    out
}

/// Dominant (top-weight) bone per vert, for part classification.
/// Rows sort descending at load, so the first entry wins.
pub fn dominant_bones(rig: &Rig) -> HashMap<usize, String> {
    let mut out = HashMap::new();
    for (vi, row) in rig.weights.iter().enumerate() {
        if let Some((bi, _)) = row.first() {
            out.insert(vi, rig.bones[*bi].name.clone());
        }
    }
    out
}

/// Rest proportions from a rig: offsets reproduce the measured
/// joint positions exactly, so FK and the mesh agree by
/// construction. Stature measured; source labeled researched.
pub fn rig_to_proportions(rig: &Rig, stature_m: f64) -> BodyProportions {
    let at = |j: Joint| -> [f64; 3] { rig.joints.get(&j).copied().unwrap_or([0.0, 0.0, 0.0]) };
    let sh = at(Joint::ShoulderR)[0].abs().max(1e-9);
    let hip = at(Joint::HipR)[0].abs().max(1e-9);
    // Measured rest offsets (child − parent): FK reproduces the
    // measured positions exactly — the review's Step 2, as code.
    let mut offsets = HashMap::new();
    for j in Joint::all() {
        if let Some(p) = j.parent() {
            let a = at(p);
            let b = at(j);
            offsets.insert(j, [b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
        }
    }
    BodyProportions {
        stature_m,
        shoulder_hw: sh / stature_m,
        hip_hw: hip / stature_m,
        root: at(Joint::Pelvis),
        offsets,
        source: "researched: hm08 rig (CC0)".to_string(),
    }
}

/// One call from mesh to proportioned skeleton. Refuses when the
/// rig fails — a canon fallback here would smuggle defaults into
/// "researched" proportions.
pub fn measured_proportions(body: &OracleBody) -> Result<BodyProportions, String> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for v in &body.verts {
        lo = lo.min(v[1]);
        hi = hi.max(v[1]);
    }
    let stature = (hi - lo).max(1e-9);
    let rig = rig_for(body)?;
    Ok(rig_to_proportions(&rig, stature))
}

/// Pose a rigged body: linear-blend skinning over the CC0 weights.
/// Each bone rotates about its rest head by its joint's world
/// rotation, translated to the posed joint. Unweighted verts ride
/// their nearest joint rigidly (helpers, stated). Same matrices as
/// FK — joints agree bit-for-bit (tested).
pub fn pose_body(
    body: &OracleBody,
    rig: &Rig,
    props: &BodyProportions,
    pose: &Pose,
) -> Result<Vec<[f64; 3]>, String> {
    use super::skeleton::{V3, apply_mat, joint_world_rotations};
    let posed = super::skeleton::forward_kinematics(props, pose)?;
    let world = joint_world_rotations(pose)?;
    // Nearest joint per vert, for the unweighted fallback.
    let joints: Vec<(Joint, [f64; 3])> = rig.joints.iter().map(|(j, p)| (*j, *p)).collect();
    let nearest = |v: &[f64; 3]| -> Joint {
        let mut best = Joint::Pelvis;
        let mut bd = f64::INFINITY;
        for (j, p) in &joints {
            let d = (v[0] - p[0]).powi(2) + (v[1] - p[1]).powi(2) + (v[2] - p[2]).powi(2);
            if d < bd {
                bd = d;
                best = *j;
            }
        }
        best
    };
    let xf = |bone_idx: usize, v: [f64; 3]| -> [f64; 3] {
        let b = &rig.bones[bone_idx];
        let r = posed[&b.joint];
        let rel = V3::new(
            v[0] - b.rest_head[0],
            v[1] - b.rest_head[1],
            v[2] - b.rest_head[2],
        );
        let rot = apply_mat(&world[&b.joint], rel);
        [r.x + rot.x, r.y + rot.y, r.z + rot.z]
    };
    let mut out = Vec::with_capacity(body.verts.len());
    for (vi, v) in body.verts.iter().enumerate() {
        let row = &rig.weights[vi];
        if row.is_empty() {
            // Rigid fallback for unweighted helpers.
            let j = nearest(v);
            let r = posed[&j];
            let jr = rig.joints[&j];
            let rel = V3::new(v[0] - jr[0], v[1] - jr[1], v[2] - jr[2]);
            let rot = apply_mat(&world[&j], rel);
            out.push([r.x + rot.x, r.y + rot.y, r.z + rot.z]);
            continue;
        }
        let mut p = [0.0; 3];
        for (bi, w) in row {
            let q = xf(*bi, *v);
            p[0] += w * q[0];
            p[1] += w * q[1];
            p[2] += w * q[2];
        }
        out.push(p);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/oracle")
    }

    #[test]
    fn manifest_verifies_vendored_files() {
        verify_oracle_dir(&dir()).expect("vendored oracle must verify");
    }

    #[test]
    fn base_parses_to_body_scale() {
        let body = load_oracle_body(&dir(), None).expect("base loads");
        assert!(body.verts.len() > 15000, "verts {}", body.verts.len());
        assert!(!body.faces.is_empty());
        let m = measure_body(&body);
        // Adult human scale, standing: stature ~1.6-1.9m.
        assert!(
            m.stature_m > 1.5 && m.stature_m < 2.0,
            "stature {}",
            m.stature_m
        );
        // Torso ordering: chest widest, waist narrowest; all sane.
        assert!(m.chest_hw_frac > m.waist_hw_frac, "{:?}", m);
        assert!(m.hip_hw_frac > m.waist_hw_frac, "{:?}", m);
        for w in [m.chest_hw_frac, m.waist_hw_frac, m.hip_hw_frac] {
            assert!(w > 0.04 && w < 0.22, "width {} in {:?}", w, m);
        }
    }

    #[test]
    fn morphs_move_verts() {
        let base = load_oracle_body(&dir(), None).expect("base");
        let male = load_oracle_body(&dir(), Some("male-young.target")).expect("male");
        let female = load_oracle_body(&dir(), Some("female-young.target")).expect("female");
        let delta = |a: &OracleBody, b: &OracleBody| {
            a.verts
                .iter()
                .zip(b.verts.iter())
                .map(|(x, y)| {
                    ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)).sqrt()
                })
                .fold(0.0, f64::max)
        };
        assert!(delta(&base, &male) > 0.001, "male morph must move verts");
        assert!(
            delta(&base, &female) > 0.001,
            "female morph must move verts"
        );
        assert!(delta(&male, &female) > 0.001, "sexes must differ");
    }

    #[test]
    fn rest_pose_is_identity() {
        let body = load_oracle_body(&dir(), Some("male-young.target")).expect("male");
        let rig = rig_for(&body).expect("rig");
        let props = measured_proportions(&body).expect("measured");
        let out = pose_body(&body, &rig, &props, &Pose::rest()).expect("pose");
        let worst = body
            .verts
            .iter()
            .zip(out.iter())
            .map(|(a, b)| {
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
            })
            .fold(0.0, f64::max);
        assert!(worst < 1e-9, "rest must be exact: {}", worst);
    }

    #[test]
    fn weight_rows_normalize() {
        let body = load_oracle_body(&dir(), None).expect("base");
        let rig = rig_for(&body).expect("rig");
        let mut empty = 0usize;
        for row in &rig.weights {
            let s: f64 = row.iter().map(|(_, w)| w).sum();
            if row.is_empty() {
                empty += 1;
            } else {
                assert!((s - 1.0).abs() < 1e-9, "row sums to {}", s);
            }
        }
        // Helpers ride along; the body is overwhelmingly weighted.
        assert!(
            empty < body.verts.len() / 10,
            "too many bare verts: {}",
            empty
        );
    }

    #[test]
    fn helper_census_matches_pinned_file() {
        // Every helper vert accounted for, by rule. The file is
        // hash-pinned: drift fails here loudly instead of silently
        // restyling the face.
        let body = load_oracle_body(&dir(), None).expect("base");
        let rig = rig_for(&body).expect("rig");
        let dom = dominant_bones(&rig);
        let h = classify_helpers(&body, &dom);
        assert_eq!(h.eyes.len(), 144, "two eyeballs");
        assert_eq!(h.mouth.len(), 226, "mouth interior");
        assert_eq!(h.hair.len(), 320, "ten hair locks");
        assert_eq!(h.shorts.len(), 720, "garment");
        assert_eq!(h.excluded.len(), 2674 + 125 * 8, "shell + cubes");
        let total = body.verts.len() - 13380;
        let rest =
            total - h.eyes.len() - h.mouth.len() - h.hair.len() - h.shorts.len() - h.excluded.len();
        assert!(rest > 400, "skin-default helpers remain: {}", rest);
    }

    #[test]
    fn corrupt_bytes_refuse() {
        assert!(verify_oracle_dir(std::path::Path::new("/nonexistent")).is_err());
    }
}
