//! Studio composition — Phase 7 of the image-creation system.
//!
//! `create_image(prompt)`: scene requirements → learned pose +
//! canon proportions → tapered-body figure with parametric head →
//! bound clothing/objects → studio light, sweep, and finish.
//!
//! HONESTY CONTRACT, enforced by the receipt: every requirement
//! reports researched-or-defaulted. Poses are learned
//! representatives (ROM library); proportions are canon; skin,
//! straw, and cloth are stated defaults — NO researched photo
//! models exist yet, so nothing here claims to be one. The output
//! is a constructed studio maquette, never a photograph.
//! `create_image_strict` refuses until researched models land,
//! listing exactly what is missing.

use super::mesh::Mesh;
use super::scene::{Camera, Light, Material, Scene, Shape, Vec3};
use super::skeleton::{BodyProportions, Joint, Pose, V3};
use super::vision::{Image, Rgb};
use std::collections::HashMap;

/// Per-requirement provenance in the finished image.
#[derive(Debug, Clone)]
pub struct ReqStatus {
    pub requirement: String,
    pub researched: bool,
    pub note: String,
}

/// A finished creation: pixels plus what they are made of.
#[derive(Debug, Clone)]
pub struct Creation {
    pub image: Image,
    pub receipt: Vec<ReqStatus>,
    pub prompt: String,
    /// Per-material pixel counts from the render (sky included):
    /// the composition audit trail.
    pub pixel_counts: Vec<(String, u64)>,
}

/// Why creation refused.
#[derive(Debug, Clone)]
pub enum CreationError {
    /// No human subject parsed — nothing to pose or dress.
    UnsupportedPrompt(String),
    /// Human present, but no poseable action parsed.
    MissingPose(String),
    /// Strict mode: researched models missing, named one by one.
    MissingEvidence(Vec<String>),
}

impl std::fmt::Display for CreationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CreationError::UnsupportedPrompt(s) => write!(f, "no human subject: {}", s),
            CreationError::MissingPose(s) => write!(f, "no poseable action: {}", s),
            CreationError::MissingEvidence(ids) => {
                write!(f, "missing researched models: {}", ids.join(", "))
            }
        }
    }
}

/// Default skin tone. Procedural default — no researched skin
/// examples exist yet; the receipt says exactly this.
pub const DEFAULT_SKIN: (u8, u8, u8) = (200, 150, 115);
/// Default straw tone. Same status as skin: a stated placeholder.
pub const DEFAULT_STRAW: (u8, u8, u8) = (205, 175, 105);
/// Default unmarked cloth. Gray because no flag model is researched;
/// stripes would invent provenance.
pub const DEFAULT_CLOTH: (u8, u8, u8) = (210, 210, 215);

fn skin_mat() -> Material {
    Material::skin(
        "skin",
        Rgb::new(DEFAULT_SKIN.0, DEFAULT_SKIN.1, DEFAULT_SKIN.2),
    )
}

/// Learned representative pose for an action: largest normalized
/// cluster wins (deterministic). Unknown actions yield nothing.
fn learned_pose(action: &str) -> Option<Pose> {
    let examples: Vec<_> = super::pose_learn::examples_for(action)
        .into_iter()
        .map(|e| super::pose_learn::normalize(&e))
        .collect();
    if examples.is_empty() {
        return None;
    }
    let clusters = super::pose_learn::cluster(examples, 45.0);
    let biggest = clusters.into_iter().max_by_key(|c| c.len())?;
    let rep = super::pose_learn::representative(&biggest).ok()?;
    let mut p = Pose::rest();
    for (j, a) in &rep.rotations {
        p.set(*j, *a);
    }
    Some(p)
}

/// Override one side's arm with an explicit pose (salute right while
/// the body waves left, etc.).
fn set_arm(pose: &mut Pose, right: bool, src: &Pose) {
    let (sh, el, wr) = if right {
        (Joint::ShoulderR, Joint::ElbowR, Joint::WristR)
    } else {
        (Joint::ShoulderL, Joint::ElbowL, Joint::WristL)
    };
    pose.set(sh, src.get(sh));
    pose.set(el, src.get(el));
    pose.set(wr, src.get(wr));
}

/// Vendored oracle bytes: base mesh + adult macro morphs (CC0,
/// pinned — see assets/oracle/PROVENANCE.md). Baked into the binary
/// so the app never fetches bodies at runtime.
const ORACLE_BASE_OBJ: &[u8] = include_bytes!("../../assets/oracle/base.obj");

fn oracle_text(name: &str) -> &'static str {
    match name {
        "male" => include_str!("../../assets/oracle/male-young.target"),
        _ => include_str!("../../assets/oracle/female-young.target"),
    }
}

use std::sync::OnceLock;

fn oracle_base() -> &'static super::body_oracle::OracleBody {
    static BASE: OnceLock<super::body_oracle::OracleBody> = OnceLock::new();
    BASE.get_or_init(|| {
        let text = std::str::from_utf8(ORACLE_BASE_OBJ).expect("vendored obj is utf8");
        super::body_oracle::parse_obj(text).expect("vendored base parses")
    })
}

fn oracle_deltas(sex: &str) -> Vec<super::body_oracle::MorphDelta> {
    super::body_oracle::parse_target(oracle_text(sex)).expect("vendored morph parses")
}

/// Morphed oracle body for a sex. The research product everything
/// below measures from — not canon fractions.
fn morphed_body(female: bool) -> super::body_oracle::OracleBody {
    let mut body = oracle_base().clone();
    let deltas = oracle_deltas(if female { "female" } else { "male" });
    super::body_oracle::apply_morph(&mut body, &deltas, 1.0).expect("vendored morph fits");
    body
}

/// Build the posed figure from the premade oracle: morphed base
/// mesh, linear-blend skinning, helper parts in their own materials
/// (eyes dark, mouth dark red, hair brown, shorts charcoal),
/// eyebrows, eye catchlights, rim light from the scene. Cubes and
/// the unidentified shell never render. The `female` flag selects
/// the macro morph; ground contact holds the lowest vert at y=0.
fn build_figure(
    props: &BodyProportions,
    pose: &Pose,
    female: bool,
    skin: &Material,
    shapes: &mut Vec<Shape>,
) {
    let s = props.stature_m;
    let mut body = morphed_body(female);
    // Scale is a no-op here (measured stature already), kept so a
    // future target stature needs no new code path.
    let m = super::body_oracle::measure_body(&body);
    let k = s / m.stature_m;
    for v in &mut body.verts {
        v[0] *= k;
        v[1] *= k;
        v[2] *= k;
    }
    // Rig measured off the scaled mesh, then linear-blend skinning.
    // Helper parts split into their own materials below.
    let rig = super::body_oracle::rig_for(&body).expect("vendored rig fits vendored mesh");
    let dom = super::body_oracle::dominant_bones(&rig);
    let help = super::body_oracle::classify_helpers(&body, &dom);
    let posed =
        super::body_oracle::pose_body(&body, &rig, props, pose).expect("learned poses validate");
    let min_y = posed.iter().map(|v| v[1]).fold(f64::INFINITY, f64::min);
    let mut verts: Vec<[f64; 3]> = posed.iter().map(|v| [v[0], v[1] - min_y, v[2]]).collect();
    // Eyeballs sit proud of their sockets (real corneas protrude).
    for i in &help.eyes {
        verts[*i][2] += 0.002;
    }
    // Eyelids part around the posed eye centers: rule-selected
    // verts split up/down with distance falloff, like smile_deltas.
    {
        let mut centers = Vec::new();
        for side in [-1.0f64, 1.0] {
            let mut c = [0.0; 3];
            let mut n = 0usize;
            for i in &help.eyes {
                let v = verts[*i];
                if (v[0] < 0.0) == (side < 0.0) {
                    c[0] += v[0];
                    c[1] += v[1];
                    c[2] += v[2];
                    n += 1;
                }
            }
            if n > 0 {
                centers.push([c[0] / n as f64, c[1] / n as f64, c[2] / n as f64]);
            }
        }
        if !centers.is_empty() {
            let tmp = Mesh {
                verts: verts.clone(),
                faces: body.faces.clone(),
                normals: Vec::new(),
            };
            for (i, d) in tmp.eyelid_deltas(&centers, &help.eyes, 0.025) {
                let v = &mut verts[i as usize];
                v[0] += d[0];
                v[1] += d[1];
                v[2] += d[2];
            }
        }
    }
    // Lips: blush the skin toward red around the posed mouth center.
    // The center comes from posed mouth verts — measured, not placed.
    let mouth_c = {
        let n = help.mouth.len().max(1) as f64;
        let mut c = [0.0; 3];
        for i in &help.mouth {
            let v = verts[*i];
            c[0] += v[0];
            c[1] += v[1];
            c[2] += v[2];
        }
        [c[0] / n, c[1] / n, c[2] / n]
    };
    let mut skin_here = skin.clone();
    if !help.mouth.is_empty() {
        skin_here.blush = Some((mouth_c, Rgb::new(150, 82, 68), 0.028 * s));
    }
    // Nose tip: front-most skin vert above the mouth; nostrils sit
    // just under and beside it. Geometry-measured per render.
    let nose_tip = verts
        .iter()
        .filter(|v| v[1] > mouth_c[1] + 0.015 && v[1] < mouth_c[1] + 0.075 && v[0].abs() < 0.03)
        .max_by(|a, b| a[2].partial_cmp(&b[2]).unwrap_or(std::cmp::Ordering::Equal))
        .copied();
    let nostril_mat = Material::named("nostril", Rgb::new(28, 16, 13));
    let mats = [
        skin_here,
        Material::named("eye", Rgb::new(22, 13, 9)),
        Material::named("mouth", Rgb::new(88, 28, 22)),
        Material::named("hair", Rgb::new(48, 30, 17)),
        Material::named("shorts", Rgb::new(38, 38, 44)),
    ];
    // Face classes by vert membership (helpers never share verts
    // with the body — disconnected by construction, asserted).
    let in_set = |f: &[u32; 3], set: &[usize]| f.iter().any(|i| set.contains(&(*i as usize)));
    let all_excluded = |f: &[u32; 3]| f.iter().all(|i| help.excluded.contains(&(*i as usize)));
    let class_of = |f: &[u32; 3]| -> u8 {
        if all_excluded(f) {
            255
        } else if in_set(f, &help.eyes) {
            1
        } else if in_set(f, &help.mouth) {
            2
        } else if in_set(f, &help.hair) {
            3
        } else if in_set(f, &help.shorts) {
            4
        } else {
            0
        }
    };
    for class in 0u8..=4u8 {
        let mut remap: HashMap<usize, u32> = HashMap::new();
        let mut vout: Vec<[f64; 3]> = Vec::new();
        let mut fout: Vec<[u32; 3]> = Vec::new();
        for f in &body.faces {
            if class_of(f) != class {
                continue;
            }
            let mut tri = [0u32; 3];
            for (k, i) in f.iter().enumerate() {
                let iu = *i as usize;
                let nu = *remap.entry(iu).or_insert_with(|| {
                    vout.push(verts[iu]);
                    (vout.len() - 1) as u32
                });
                tri[k] = nu;
            }
            fout.push(tri);
        }
        if fout.is_empty() {
            continue;
        }
        let mut mesh = Mesh {
            verts: vout,
            faces: fout,
            normals: Vec::new(),
        };
        mesh.compute_normals();
        shapes.push(Shape::Mesh {
            mesh,
            mat: mats[class as usize].clone(),
        });
    }
    // Eyebrows: dark tubes above the eye centers (posed space).
    let mut eye_l = Vec::new();
    let mut eye_r = Vec::new();
    for i in &help.eyes {
        let v = verts[*i];
        if v[0] < 0.0 {
            eye_l.push(v);
        } else {
            eye_r.push(v);
        }
    }
    let brow_mat = Material::named("brow", Rgb::new(48, 30, 17));
    for eye in [&eye_l, &eye_r] {
        if eye.is_empty() {
            continue;
        }
        let n = eye.len() as f64;
        let c = [
            eye.iter().map(|v| v[0]).sum::<f64>() / n,
            eye.iter().map(|v| v[1]).sum::<f64>() / n,
            eye.iter().map(|v| v[2]).sum::<f64>() / n,
        ];
        let dir = if c[0] < 0.0 { -1.0 } else { 1.0 };
        let a = [c[0] - dir * 0.008 * s, c[1] + 0.022 * s, c[2] + 0.004 * s];
        let b = [c[0] + dir * 0.020 * s, c[1] + 0.026 * s, c[2] + 0.004 * s];
        shapes.push(Shape::Mesh {
            mesh: Mesh::tube(a, b, 0.0045 * s, 0.0035 * s, 6),
            mat: brow_mat.clone(),
        });
        // Catchlight: millimeter white glint toward the camera side.
        let glint = Mesh::sphere(6, 4, 0.0022 * s).translated(
            c[0] + dir * 0.004 * s,
            c[1] + 0.006 * s,
            c[2] + 0.012 * s,
        );
        shapes.push(Shape::Mesh {
            mesh: glint,
            mat: Material::named("glint", Rgb::new(245, 245, 248)),
        });
    }
    // Nostrils: millimeter dark dots flanking under the nose tip.
    // Placement measured from geometry; symmetry asserted by test.
    if let Some(tip) = nose_tip {
        for side in [-1.0, 1.0] {
            let dot = Mesh::sphere(6, 4, 0.003 * s).translated(
                tip[0] + side * 0.008,
                tip[1] - 0.006,
                tip[2] - 0.003,
            );
            shapes.push(Shape::Mesh {
                mesh: dot,
                mat: nostril_mat.clone(),
            });
        }
    }
}

/// Measured bounds of figure meshes: center + largest extent.
/// Drives auto-framing so renders compose correctly for any pose.
fn figure_bounds(shapes: &[Shape]) -> ([f64; 3], f64) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    let mut any = false;
    for s in shapes {
        if let Shape::Mesh { mesh, .. } = s {
            for v in &mesh.verts {
                any = true;
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k]);
                    hi[k] = hi[k].max(v[k]);
                }
            }
        }
    }
    if !any {
        return ([0.0, 0.9, 0.0], 1.8);
    }
    let size = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(hi[2] - lo[2]);
    (
        [
            (lo[0] + hi[0]) / 2.0,
            (lo[1] + hi[1]) / 2.0,
            (lo[2] + hi[2]) / 2.0,
        ],
        size,
    )
}

/// Straw hat at an assembly anchor: hemisphere crown + brim disc.
/// Default tan; no researched straw model exists (receipt states it).
fn build_hat(anchor: V3, s: f64, shapes: &mut Vec<Shape>) {
    let straw = Material::named(
        "straw",
        Rgb::new(DEFAULT_STRAW.0, DEFAULT_STRAW.1, DEFAULT_STRAW.2),
    );
    let crown = Mesh::hemisphere(16, 8, 0.075 * s, 1.1).translated(anchor.x, anchor.y, anchor.z);
    shapes.push(Shape::Mesh {
        mesh: crown,
        mat: straw.clone(),
    });
    let mut pts = Vec::new();
    for i in 0..24 {
        let th = i as f64 * std::f64::consts::TAU / 24.0;
        pts.push([0.115 * s * th.cos(), 0.115 * s * th.sin()]);
    }
    let brim = Mesh::fan_polygon(&pts).translated(anchor.x, anchor.y, anchor.z);
    shapes.push(Shape::Mesh {
        mesh: brim,
        mat: straw,
    });
}

/// Unmarked waving cloth + pole at a hand anchor. Gray on purpose:
/// stripes would invent a researched flag model that does not exist.
fn build_cloth(anchor: V3, s: f64, shapes: &mut Vec<Shape>) {
    let cloth = Material::named(
        "cloth",
        Rgb::new(DEFAULT_CLOTH.0, DEFAULT_CLOTH.1, DEFAULT_CLOTH.2),
    );
    let w = 0.34 * s;
    let h = 0.24 * s;
    let mut panel = Mesh::plane_grid(w, h, 24, 12);
    let params = super::deform::WaveParams {
        amplitude: 0.025 * s,
        wavelength: 0.24 * s,
        phase: 0.0,
        pole_x: -w / 2.0,
        width: w,
    };
    super::deform::apply_wave(&mut panel, params);
    // Left edge rides the pole: pole base at the hand.
    let pole_top = [anchor.x, anchor.y + 0.34 * s, anchor.z];
    let pole = Mesh::tube(
        [anchor.x, anchor.y - 0.05 * s, anchor.z],
        pole_top,
        0.008 * s,
        0.008 * s,
        8,
    );
    let gray = Material::named("pole", Rgb::new(60, 60, 64));
    shapes.push(Shape::Mesh {
        mesh: pole,
        mat: gray,
    });
    let panel = panel.translated(
        anchor.x + w / 2.0 + 0.008 * s,
        anchor.y + 0.17 * s,
        anchor.z,
    );
    shapes.push(Shape::Mesh {
        mesh: panel,
        mat: cloth,
    });
}

/// Full pipeline: prompt → requirements → learned pose → figure →
/// bound objects → studio render. See `create_image_with` for the
/// researched-skin variant; this one constructs with stated defaults.
pub fn create_image(prompt: &str) -> Result<Creation, CreationError> {
    create_image_with(prompt, None)
}

/// Full pipeline with an optional researched skin palette
/// (color + provenance note). Receipted as researched when present.
pub fn create_image_with(
    prompt: &str,
    skin: Option<(Rgb, String)>,
) -> Result<Creation, CreationError> {
    let spec = crate::engine::scene_intent::parse_scene(prompt);
    let human = spec
        .subjects
        .first()
        .ok_or_else(|| CreationError::UnsupportedPrompt("no human subject parsed".to_string()))?;
    let female = human
        .attributes
        .iter()
        .any(|a| a == "woman" || a == "lady" || a == "female")
        || human.stype == "woman";
    // Measured rig, not canon: proportions come from the morphed
    // oracle mesh (Step 4 — research affects construction, with the
    // source on the receipt, not smuggled into defaults).
    let mbody = morphed_body(female);
    let props = super::body_oracle::measured_proportions(&mbody)
        .map_err(|e| CreationError::MissingEvidence(vec![e]))?;

    // Pose: first poseable action drives the body; salute claims the
    // right arm while a waved object takes the left (assembly rule).
    let atypes: Vec<&str> = spec.actions.iter().map(|a| a.atype.as_str()).collect();
    if atypes.is_empty() {
        return Err(CreationError::MissingPose(
            "no poseable action parsed".to_string(),
        ));
    }
    let base = atypes[0];
    let mut pose = learned_pose(base).unwrap_or_else(Pose::rest);
    if base != "salute"
        && atypes.contains(&"salute")
        && let Some(sal) = learned_pose("salute")
    {
        set_arm(&mut pose, true, &sal);
    }
    if base == "salute"
        && atypes.contains(&"wave")
        && let Some(wv) = learned_pose("wave")
    {
        set_arm(&mut pose, false, &wv);
    }
    let pos =
        crate::engine::skeleton::forward_kinematics(&props, &pose).expect("learned poses validate");
    let asm = crate::engine::assembly::assemble(&spec, &pos, &props);

    let skin_mat_built = match &skin {
        Some((c, _)) => Material::skin("skin", *c),
        None => skin_mat(),
    };
    let mut shapes: Vec<Shape> = Vec::new();
    build_figure(&props, &pose, female, &skin_mat_built, &mut shapes);
    for a in &asm.attachments {
        if a.object_id.contains("hat") {
            build_hat(a.anchor, props.stature_m, &mut shapes);
        } else if a.object_id.contains("flag") {
            build_cloth(a.anchor, props.stature_m, &mut shapes);
        }
    }

    // Studio: auto-framed on the measured figure bounds (fills
    // ~90% of frame height by construction — no hand-tuned camera
    // to drift). Key from front-above-left so the face reads,
    // fill + rim for form, seamless sweep + floor.
    let (c, size) = figure_bounds(&shapes);
    let half = (size / 2.0).max(0.2);
    let dist = half / (20.0f64.to_radians().tan()) / 0.9;
    let dir = Vec3::new(0.15, 0.12, 1.0);
    let dl = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
    let scene = Scene {
        camera: Camera {
            pos: Vec3::new(
                c[0] + dir.x / dl * dist,
                c[1] + dir.y / dl * dist,
                c[2] + dir.z / dl * dist,
            ),
            look_at: Vec3::new(c[0], c[1], c[2]),
            fov_deg: 40.0,
            width: 640,
            height: 480,
        },
        lights: vec![
            Light::key(Vec3::new(-0.35, 0.55, 0.75)),
            Light::fill(Vec3::new(0.6, 0.25, 0.7), 0.30),
            // Rim from behind-top: edge definition against the sweep.
            Light::fill(Vec3::new(0.3, 0.5, -0.8), 0.35),
        ],
        ambient: 0.38,
        sky_top: Rgb::new(232, 232, 238),
        sky_bottom: Rgb::new(248, 248, 250),
        shapes: {
            let mut all = vec![Shape::Plane {
                y: 0.0,
                mat: Material::named("sweep", Rgb::new(225, 225, 228)),
            }];
            all.extend(shapes);
            all
        },
    };
    let (mut img, pixel_counts) = super::scene::render(&scene);
    img.grade(1.06, 3.0);
    img.vignette(0.22);
    img.grain(0xF16E, 3);

    let plan = super::scene_intent::plan_research(&spec);
    let mut store = super::evidence::EvidenceStore::new();
    store.require_from_spec(&spec, &plan);
    let mut receipt: Vec<ReqStatus> = store
        .requirements
        .iter()
        .map(|r| {
            let palette = skin
                .as_ref()
                .filter(|_| r.category == super::evidence::Category::Subject);
            // Subjects ride researched proportions (measured rig);
            // only the skin palette can still be defaulted.
            let (researched, note) = match (&palette, r.category) {
                (Some((_, src)), super::evidence::Category::Subject) => (
                    true,
                    format!("proportions researched (hm08 rig); skin {}", src),
                ),
                _ if r.category == super::evidence::Category::Subject => (
                    true,
                    "proportions researched (hm08 rig); skin default (no photo tones)".to_string(),
                ),
                _ => (
                    false,
                    "no researched model yet — canon/default construction".to_string(),
                ),
            };
            ReqStatus {
                requirement: r.id.clone(),
                researched,
                note,
            }
        })
        .collect();
    for u in &asm.unresolved {
        receipt.push(ReqStatus {
            requirement: format!("open: {}", u),
            researched: false,
            note: "unresolved at assembly — not rendered".to_string(),
        });
    }
    Ok(Creation {
        image: img,
        receipt,
        prompt: prompt.to_string(),
        pixel_counts,
    })
}

/// Strict mode: refuses until researched models exist, listing every
/// missing requirement. Today that is all of them — the refusal IS
/// the §12 path, tested, not a dead letter.
pub fn create_image_strict(prompt: &str) -> Result<Creation, CreationError> {
    let spec = crate::engine::scene_intent::parse_scene(prompt);
    if spec.subjects.is_empty() {
        return Err(CreationError::UnsupportedPrompt(
            "no human subject parsed".to_string(),
        ));
    }
    let plan = super::scene_intent::plan_research(&spec);
    let mut store = super::evidence::EvidenceStore::new();
    store.require_from_spec(&spec, &plan);
    let missing: Vec<String> = store.requirements.iter().map(|r| r.id.clone()).collect();
    if missing.is_empty() {
        return Err(CreationError::MissingPose(
            "no poseable action parsed".to_string(),
        ));
    }
    Err(CreationError::MissingEvidence(missing))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tubes_point_outward() {
        // Outward winding: ring-0 normals point away from the axis.
        let m = Mesh::tube([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.1, 0.1, 12);
        let n = m.normals[0];
        let radial = [m.verts[0][0], 0.0, m.verts[0][2]];
        let dot = n[0] * radial[0] + n[2] * radial[2];
        assert!(dot > 0.0, "inward normals: {:?}", n);
    }
}
