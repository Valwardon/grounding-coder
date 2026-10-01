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

use super::mesh::{HeadShape, Mesh};
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

fn arr(p: V3) -> [f64; 3] {
    [p.x, p.y, p.z]
}

/// Tapered limb segment between posed joints, extended 12% past
/// both ends so open tube ends bury inside neighboring segments.
fn bone(a: [f64; 3], b: [f64; 3], r0: f64, r1: f64) -> Mesh {
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let a2 = [a[0] - d[0] * 0.12, a[1] - d[1] * 0.12, a[2] - d[2] * 0.12];
    let b2 = [b[0] + d[0] * 0.12, b[1] + d[1] * 0.12, b[2] + d[2] * 0.12];
    Mesh::tube(a2, b2, r0, r1, 10)
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

/// Build the posed figure meshes: tapered tubes on FK bones,
/// parametric head with a mild smile, all skin maquette finish.
fn build_figure(props: &BodyProportions, pos: &HashMap<Joint, V3>, shapes: &mut Vec<Shape>) {
    let s = props.stature_m;
    let skin = skin_mat();
    let seg = |a: Joint, b: Joint, r0: f64, r1: f64| -> Mesh {
        bone(arr(pos[&a]), arr(pos[&b]), r0 * s, r1 * s)
    };
    let mut push = |m: Mesh| {
        shapes.push(Shape::Mesh {
            mesh: m,
            mat: skin.clone(),
        })
    };
    // Torso: pelvis→chest barrel, shoulder + hip bars.
    push(seg(Joint::Pelvis, Joint::Chest, 0.095, 0.105));
    push(seg(Joint::Chest, Joint::Neck, 0.060, 0.048));
    push(seg(Joint::ShoulderL, Joint::ShoulderR, 0.048, 0.048));
    push(seg(Joint::HipL, Joint::HipR, 0.062, 0.062));
    // Arms (hands extended past the wrist joint).
    for (sh, el, wr, ha) in [
        (Joint::ShoulderL, Joint::ElbowL, Joint::WristL, Joint::HandL),
        (Joint::ShoulderR, Joint::ElbowR, Joint::WristR, Joint::HandR),
    ] {
        push(seg(sh, el, 0.042, 0.034));
        push(seg(el, wr, 0.034, 0.026));
        let w = pos[&wr];
        let h = pos[&ha];
        let d = V3::new(h.x - w.x, h.y - w.y, h.z - w.z);
        let l = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt().max(1e-9);
        let tip = V3::new(
            h.x + d.x / l * 0.06 * s,
            h.y + d.y / l * 0.06 * s,
            h.z + d.z / l * 0.06 * s,
        );
        push(bone(arr(w), arr(tip), 0.026 * s, 0.018 * s));
    }
    // Legs (toes extended past the foot joint).
    for (hip, knee, ank, foot) in [
        (Joint::HipL, Joint::KneeL, Joint::AnkleL, Joint::FootL),
        (Joint::HipR, Joint::KneeR, Joint::AnkleR, Joint::FootR),
    ] {
        push(seg(hip, knee, 0.075, 0.056));
        push(seg(knee, ank, 0.056, 0.040));
        let a = pos[&ank];
        let f = pos[&foot];
        let d = V3::new(f.x - a.x, f.y - a.y, f.z - a.z);
        let l = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt().max(1e-9);
        let tip = V3::new(
            f.x + d.x / l * 0.05 * s,
            f.y + d.y / l * 0.05 * s,
            f.z + d.z / l * 0.05 * s,
        );
        push(bone(arr(a), arr(tip), 0.040 * s, 0.030 * s));
    }
    // Head: parametric, scaled to canon head height, mild smile.
    let head_h = s / 7.5;
    let hp = pos[&Joint::Head];
    let mut head = Mesh::parametric_head(24, 18, &HeadShape::default());
    let deltas = head.smile_deltas();
    head = head.blendshape(&deltas, 0.3);
    let center = [hp.x, hp.y + head_h * 0.42, hp.z];
    head.displace(|v| {
        [
            center[0] + v[0] * head_h,
            center[1] + v[1] * head_h,
            center[2] + v[2] * head_h,
        ]
    });
    push(head);
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
/// bound objects → studio render. On success the receipt marks every
/// requirement researched-or-defaulted; nothing researched exists
/// yet, so every status is honestly `false` with its reason.
pub fn create_image(prompt: &str) -> Result<Creation, CreationError> {
    let spec = super::scene_intent::parse_scene(prompt);
    let human = spec
        .subjects
        .first()
        .ok_or_else(|| CreationError::UnsupportedPrompt("no human subject parsed".to_string()))?;
    let female = human
        .attributes
        .iter()
        .any(|a| a == "woman" || a == "lady" || a == "female")
        || human.stype == "woman";
    let props = if female {
        BodyProportions::adult_female()
    } else {
        BodyProportions::adult_male()
    };

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
    let pos = super::skeleton::forward_kinematics(&props, &pose).expect("learned poses validate");
    let asm = super::assembly::assemble(&spec, &pos, &props);

    let mut shapes: Vec<Shape> = Vec::new();
    build_figure(&props, &pos, &mut shapes);
    for a in &asm.attachments {
        if a.object_id.contains("hat") {
            build_hat(a.anchor, props.stature_m, &mut shapes);
        } else if a.object_id.contains("flag") {
            build_cloth(a.anchor, props.stature_m, &mut shapes);
        }
    }

    // Studio: frame the figure, key + fill, seamless sweep, floor.
    let s = props.stature_m;
    let scene = Scene {
        camera: Camera {
            pos: Vec3::new(0.25 * s, 1.05 * s, 2.9 * s),
            look_at: Vec3::new(0.0, 0.85 * s, 0.0),
            fov_deg: 40.0,
            width: 640,
            height: 480,
        },
        lights: vec![
            Light::key(Vec3::new(-0.45, 0.8, 0.35)),
            Light::fill(Vec3::new(0.6, 0.25, 0.7), 0.30),
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
    let (mut img, _) = super::scene::render(&scene);
    img.grade(1.06, 3.0);
    img.vignette(0.22);
    img.grain(0xF16E, 3);

    let plan = super::scene_intent::plan_research(&spec);
    let mut store = super::evidence::EvidenceStore::new();
    store.require_from_spec(&spec, &plan);
    let mut receipt: Vec<ReqStatus> = store
        .requirements
        .iter()
        .map(|r| ReqStatus {
            requirement: r.id.clone(),
            researched: false,
            note: "no researched model yet — canon/default construction".to_string(),
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
    })
}

/// Strict mode: refuses until researched models exist, listing every
/// missing requirement. Today that is all of them — the refusal IS
/// the §12 path, tested, not a dead letter.
pub fn create_image_strict(prompt: &str) -> Result<Creation, CreationError> {
    let spec = super::scene_intent::parse_scene(prompt);
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
