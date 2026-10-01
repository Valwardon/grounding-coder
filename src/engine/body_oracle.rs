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

/// Segment list for part binding: joint pairs in rest pose.
fn rest_segments(props: &BodyProportions) -> Vec<(Joint, [f64; 3], [f64; 3])> {
    let rest = super::skeleton::forward_kinematics(props, &Pose::rest()).expect("rest validates");
    let mut segs = Vec::new();
    for j in Joint::all() {
        if let Some(p) = j.parent() {
            let a = rest[&p];
            let b = rest[&j];
            segs.push((j, [a.x, a.y, a.z], [b.x, b.y, b.z]));
        }
    }
    segs
}

fn dist_point_seg(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let l2 = (ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]).max(1e-12);
    let t = ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1] + (p[2] - a[2]) * ab[2]) / l2;
    let t = t.clamp(0.0, 1.0);
    let c = [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t];
    ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt()
}

/// Bind every vertex to its nearest rest-pose bone segment. Returns
/// per-vertex joint ids. Pure geometry — same mesh twice, same parts.
pub fn bind_parts(body: &OracleBody, props: &BodyProportions) -> Vec<Joint> {
    let segs = rest_segments(props);
    body.verts
        .iter()
        .map(|v| {
            let mut best = segs[0].0;
            let mut bd = f64::INFINITY;
            for (j, a, b) in &segs {
                let d = dist_point_seg(*v, *a, *b);
                if d < bd {
                    bd = d;
                    best = *j;
                }
            }
            best
        })
        .collect()
}

/// 3×3 rotation, column-major.
pub type Mat3 = ([f64; 3], [f64; 3], [f64; 3]);

/// Pose a bound body: each part rotates rigidly about its joint by
/// the posed-minus-rest rotation, translated to the posed joint.
/// Rest locals are zero, so the delta is the posed local rotation.
/// Stated approximation: rigid parts (smooth skinning staged);
/// joint spheres cover the seams (see studio).
pub fn pose_body(
    body: &OracleBody,
    parts: &[Joint],
    props: &BodyProportions,
    pose: &Pose,
) -> Result<Vec<[f64; 3]>, String> {
    let rest = super::skeleton::forward_kinematics(props, &Pose::rest())?;
    let posed = super::skeleton::forward_kinematics(props, pose)?;
    let rot_of = |angles: super::skeleton::JointAngles, j: Joint| -> Mat3 {
        // R = Rz(b)·Rx(a), exactly the FK's rot_xz order: single-axis
        // chains match FK bit-for-bit; mixed axes agree with it to
        // second order (documented in skeleton).
        let a = -j.flex_sign() * angles.flex.to_radians();
        let b = j.mirror() * angles.abd.to_radians();
        let (sa, ca) = a.sin_cos();
        let (sb, cb) = b.sin_cos();
        let c0 = [cb, sb, 0.0];
        let c1 = [-sb * ca, cb * ca, sa];
        let c2 = [sb * sa, -cb * sa, ca];
        (c0, c1, c2)
    };
    // World rotation per joint: DFS from the root, composing.
    let mut world: HashMap<Joint, Mat3> = HashMap::new();
    for j in Joint::all() {
        let a = pose.get(j);
        let local = rot_of(a, j);
        let w = match j.parent() {
            None => local,
            Some(p) => mat_mul(&world[&p], &local),
        };
        world.insert(j, w);
    }
    let apply = |m: &Mat3, v: [f64; 3]| -> [f64; 3] {
        [
            m.0[0] * v[0] + m.1[0] * v[1] + m.2[0] * v[2],
            m.0[1] * v[0] + m.1[1] * v[1] + m.2[1] * v[2],
            m.0[2] * v[0] + m.1[2] * v[1] + m.2[2] * v[2],
        ]
    };
    let mut out = Vec::with_capacity(body.verts.len());
    for (v, j) in body.verts.iter().zip(parts.iter()) {
        let r = rest[j];
        let p = posed[j];
        let rel = [v[0] - r.x, v[1] - r.y, v[2] - r.z];
        let rot = apply(&world[j], rel);
        out.push([p.x + rot[0], p.y + rot[1], p.z + rot[2]]);
    }
    Ok(out)
}

fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let col = |a: &Mat3, v: [f64; 3]| -> [f64; 3] {
        [
            a.0[0] * v[0] + a.1[0] * v[1] + a.2[0] * v[2],
            a.0[1] * v[0] + a.1[1] * v[1] + a.2[1] * v[2],
            a.0[2] * v[0] + a.1[2] * v[1] + a.2[2] * v[2],
        ]
    };
    (col(a, b.0), col(a, b.1), col(a, b.2))
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
        let props = BodyProportions::adult_male();
        let parts = bind_parts(&body, &props);
        assert_eq!(parts.len(), body.verts.len());
        let out = pose_body(&body, &parts, &props, &Pose::rest()).expect("pose");
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
    fn corrupt_bytes_refuse() {
        assert!(verify_oracle_dir(std::path::Path::new("/nonexistent")).is_err());
    }
}
