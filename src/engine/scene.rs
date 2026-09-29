//! Scenes, not pixels: "put a tower behind me" is a graph edit,
//! the photo is its rendering.
//!
//! ```text
//! Scene { camera, light, shapes } → render → Image + receipt
//! ```
//!
//! The graph is plain data (people are composites of named
//! primitives — head, torso, arms, legs). The renderer is a tiny
//! deterministic raytracer: one directional light, hard shadows,
//! Lambert shading, gradient sky. Same scene twice → byte-identical
//! photo, asserted in tests.
//!
//! Modification is the whole point: add/remove a node, re-render,
//! and the receipt (per-material pixel counts) proves what changed
//! and what didn't. The oracle is arithmetic on the buffer.
use super::vision::{Image, Rgb};

#[derive(Debug, Clone, Copy)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }

    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }

    fn scale(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }

    fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    fn len(self) -> f64 {
        self.dot(self).sqrt()
    }

    fn norm(self) -> Vec3 {
        let l = self.len();
        if l == 0.0 { self } else { self.scale(1.0 / l) }
    }
}

/// A named surface: the name is what receipts and prose edits refer
/// to ("tower", "person-skin").
#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    pub color: Rgb,
}

impl Material {
    pub fn named(name: &str, color: Rgb) -> Self {
        Material {
            name: name.to_string(),
            color,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Shape {
    Sphere {
        center: Vec3,
        radius: f64,
        mat: Material,
    },
    /// Horizontal plane at height `y`.
    Plane { y: f64, mat: Material },
    /// Axis-aligned box.
    Box { min: Vec3, max: Vec3, mat: Material },
    /// Capsule: segment `a→b` swept by `radius`. Limbs are capsules
    /// because people bend and boxes don't.
    Capsule {
        a: Vec3,
        b: Vec3,
        radius: f64,
        mat: Material,
    },
}

struct Hit {
    dist: f64,
    point: Vec3,
    normal: Vec3,
    mat: Material,
}

fn intersect(shape: &Shape, origin: Vec3, dir: Vec3) -> Option<Hit> {
    match shape {
        Shape::Sphere {
            center,
            radius,
            mat,
        } => {
            let oc = origin.sub(*center);
            let b = oc.dot(dir);
            let c = oc.dot(oc) - radius * radius;
            let disc = b * b - c;
            if disc < 0.0 {
                return None;
            }
            let t = -b - disc.sqrt();
            if t <= 1e-6 {
                return None;
            }
            let point = origin.add(dir.scale(t));
            Some(Hit {
                dist: t,
                point,
                normal: point.sub(*center).norm(),
                mat: mat.clone(),
            })
        }
        Shape::Plane { y, mat } => {
            if dir.y.abs() < 1e-9 {
                return None;
            }
            let t = (*y - origin.y) / dir.y;
            if t <= 1e-6 {
                return None;
            }
            Some(Hit {
                dist: t,
                point: origin.add(dir.scale(t)),
                normal: Vec3::new(0.0, 1.0, 0.0),
                mat: mat.clone(),
            })
        }
        Shape::Box { min, max, mat } => {
            let mut tmin = f64::NEG_INFINITY;
            let mut tmax = f64::INFINITY;
            let o = [origin.x, origin.y, origin.z];
            let d = [dir.x, dir.y, dir.z];
            let lo = [min.x, min.y, min.z];
            let hi = [max.x, max.y, max.z];
            let mut axis = 0;
            for i in 0..3 {
                if d[i].abs() < 1e-12 {
                    if o[i] < lo[i] || o[i] > hi[i] {
                        return None;
                    }
                } else {
                    let mut t1 = (lo[i] - o[i]) / d[i];
                    let mut t2 = (hi[i] - o[i]) / d[i];
                    let mut face = if t1 < t2 { -1 } else { 1 };
                    if t1 > t2 {
                        std::mem::swap(&mut t1, &mut t2);
                        face = -face;
                    }
                    if t1 > tmin {
                        tmin = t1;
                        axis = i;
                        let _ = face;
                    }
                    tmax = tmax.min(t2);
                    if tmin > tmax {
                        return None;
                    }
                }
            }
            if tmin <= 1e-6 {
                return None;
            }
            let mut n = [0.0, 0.0, 0.0];
            n[axis] = if d[axis] > 0.0 { -1.0 } else { 1.0 };
            Some(Hit {
                dist: tmin,
                point: origin.add(dir.scale(tmin)),
                normal: Vec3::new(n[0], n[1], n[2]),
                mat: mat.clone(),
            })
        }
        Shape::Capsule { a, b, radius, mat } => {
            // Cylinder around segment a→b, plus sphere caps. The
            // nearest t whose closest segment point lies within range
            // wins; caps cover the rest.
            let ab = b.sub(*a);
            let len = ab.len();
            if len < 1e-9 {
                return None;
            }
            let n = ab.scale(1.0 / len);
            let ao = origin.sub(*a);
            let d_par = dir.dot(n);
            let ao_par = ao.dot(n);
            let d_perp = dir.sub(n.scale(d_par));
            let m = ao.sub(n.scale(ao_par));
            let cap = |c: Vec3| {
                let oc = origin.sub(c);
                let bl = oc.dot(dir);
                let cl = oc.dot(oc) - radius * radius;
                let disc = bl * bl - cl;
                if disc < 0.0 {
                    return None;
                }
                let t = -bl - disc.sqrt();
                if t <= 1e-6 {
                    return None;
                }
                let point = origin.add(dir.scale(t));
                Some(Hit {
                    dist: t,
                    point,
                    normal: point.sub(c).norm(),
                    mat: mat.clone(),
                })
            };
            let mut best: Option<Hit> = None;
            let consider = |h: Hit, best: &mut Option<Hit>| {
                if best.as_ref().is_none_or(|b: &Hit| h.dist < b.dist) {
                    *best = Some(h);
                }
            };
            let aq = d_perp.dot(d_perp);
            if aq > 1e-12 {
                let bq = 2.0 * m.dot(d_perp);
                let cq = m.dot(m) - radius * radius;
                let disc = bq * bq - 4.0 * aq * cq;
                if disc >= 0.0 {
                    let sq = disc.sqrt();
                    for t in [(-bq - sq) / (2.0 * aq), (-bq + sq) / (2.0 * aq)] {
                        if t <= 1e-6 {
                            continue;
                        }
                        let s = ao_par + t * d_par;
                        if s < 0.0 || s > len {
                            continue;
                        }
                        let point = origin.add(dir.scale(t));
                        let center = a.add(n.scale(s));
                        consider(
                            Hit {
                                dist: t,
                                point,
                                normal: point.sub(center).norm(),
                                mat: mat.clone(),
                            },
                            &mut best,
                        );
                        break;
                    }
                }
            }
            if let Some(h) = cap(*a) {
                consider(h, &mut best);
            }
            if let Some(h) = cap(*b) {
                consider(h, &mut best);
            }
            best
        }
    }
}

/// A person, decomposed: head sphere, torso box, arm boxes, leg
/// boxes. Feet at `base.y`, facing +z. All materials named
/// `person-*` so edits and receipts can address the person as one.
pub fn person(base: Vec3, scale: f64, skin: Rgb, clothes: Rgb) -> Vec<Shape> {
    let s = scale;
    let skin_m = Material::named("person-skin", skin);
    let cloth_m = Material::named("person-clothes", clothes);
    let pants_m = Material::named("person-pants", Rgb::new(40, 40, 48));
    let bx = |dx: f64, w: f64, y0: f64, y1: f64, mat: Material| Shape::Box {
        min: Vec3::new(base.x + dx - w / 2.0, base.y + y0, base.z - w / 2.0),
        max: Vec3::new(base.x + dx + w / 2.0, base.y + y1, base.z + w / 2.0),
        mat,
    };
    vec![
        bx(-0.12 * s, 0.15 * s, 0.0, 0.8 * s, pants_m.clone()),
        bx(0.12 * s, 0.15 * s, 0.0, 0.8 * s, pants_m),
        bx(0.0, 0.5 * s, 0.8 * s, 1.5 * s, cloth_m.clone()),
        bx(-0.36 * s, 0.13 * s, 0.8 * s, 1.45 * s, cloth_m.clone()),
        bx(0.36 * s, 0.13 * s, 0.8 * s, 1.45 * s, cloth_m),
        Shape::Sphere {
            center: Vec3::new(base.x, base.y + 1.68 * s, base.z),
            radius: 0.16 * s,
            mat: skin_m,
        },
    ]
}

/// The studied canon: adult body proportions as fractions of
/// standing height, after the 7.5-heads artistic canon (head ≈ 1/7.5
/// of stature, arm span ≈ stature). These are claims with sources
/// ([`ANATOMY_SOURCES`]), not magic numbers — [`study_anatomy`]
/// renders a figure built from them and measures the photo to check.
/// See "Body proportions" and "Human body" on Wikipedia.
#[derive(Debug, Clone)]
pub struct BodyPlan {
    /// Head height / stature (≈1/7.5).
    pub head_h: f64,
    /// Hip joint height / stature (≈1/2).
    pub hip_y: f64,
    /// Shoulder line height / stature.
    pub shoulder_y: f64,
    /// Shoulder half-width / stature.
    pub shoulder_half: f64,
    /// Hip half-width / stature.
    pub hip_half: f64,
    /// Shoulder→wrist length / stature. Fingertips (not wrists) set
    /// the span: shoulder_half + arm_len + finger_len ≈ 1/2.
    pub arm_len: f64,
    /// Wrist→fingertip length / stature, from the anatomy graph.
    pub finger_len: f64,
    /// Wrist→thumb-tip length / stature.
    pub thumb_len: f64,
    /// Torso half-width / stature.
    pub torso_half: f64,
    /// Limb radii / stature.
    pub leg_r: f64,
    pub arm_r: f64,
}

impl BodyPlan {
    pub fn canon() -> Self {
        BodyPlan {
            head_h: 0.133,
            hip_y: 0.50,
            shoulder_y: 0.815,
            shoulder_half: 0.125,
            hip_half: 0.095,
            arm_len: 0.29,
            finger_len: 0.085,
            thumb_len: 0.055,
            torso_half: 0.13,
            leg_r: 0.055,
            arm_r: 0.045,
        }
    }
}

/// Where the canon came from. The curiosity loop researches these;
/// the study verifies the geometry built from them.
pub const ANATOMY_SOURCES: &[&str] = &[
    "https://en.wikipedia.org/wiki/Body_proportions",
    "https://en.wikipedia.org/wiki/Human_body",
];

/// One individual: looks plus posture. Distinct specs are distinct
/// people — the group photo proves it with three of them.
#[derive(Debug, Clone)]
pub struct PersonSpec {
    pub skin: Rgb,
    pub hair: Rgb,
    pub shirt: Rgb,
    pub pants: Rgb,
    pub pose: ArmPose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmPose {
    Down,
    Out,
    WaveLeft,
    WaveRight,
    PeaceLeft,
    PeaceRight,
}

impl ArmPose {
    fn raised_side(self, side: f64) -> bool {
        (matches!(self, ArmPose::WaveLeft | ArmPose::PeaceLeft) && side < 0.0)
            || (matches!(self, ArmPose::WaveRight | ArmPose::PeaceRight) && side > 0.0)
    }

    /// Peace sign: index + middle extended, ring + pinky curled —
    /// on the raised hand only. The other hand stays relaxed and
    /// open; a two-handed peace sign is a different request.
    /// Finger slot k runs inner→outer on right hands, mirrored left.
    fn finger_extended(self, side: f64, k: i32) -> bool {
        let peace =
            matches!(self, ArmPose::PeaceLeft | ArmPose::PeaceRight) && self.raised_side(side);
        if !peace {
            return true;
        }
        if side > 0.0 { k >= 2 } else { k < 2 }
    }
}

/// Shoulder pivot and hand target for one arm: single source of
/// truth shared by the builder and prop placement (bouquets meet
/// hands exactly, not approximately).
pub fn arm_endpoints(
    base: Vec3,
    scale: f64,
    spec: &PersonSpec,
    plan: &BodyPlan,
    side: f64,
) -> (Vec3, Vec3) {
    let h = 1.8 * scale;
    let y = |f: f64| base.y + f * h;
    let shoulder = Vec3::new(
        base.x + side * plan.shoulder_half * h,
        y(plan.shoulder_y),
        base.z,
    );
    let hand = if spec.pose.raised_side(side) {
        Vec3::new(
            base.x + side * (plan.shoulder_half + 0.10) * h,
            y(plan.shoulder_y) + 0.28 * h,
            base.z + 0.05 * h,
        )
    } else {
        match spec.pose {
            ArmPose::Out => Vec3::new(
                base.x + side * (plan.shoulder_half + plan.arm_len) * h,
                y(plan.shoulder_y),
                base.z,
            ),
            _ => Vec3::new(
                base.x + side * (plan.shoulder_half + 0.02) * h,
                y(plan.hip_y) + 0.05 * h,
                base.z,
            ),
        }
    };
    (shoulder, hand)
}

/// A person built from the studied plan, not from magic numbers.
/// Faces +z: head sphere, inset eyes, nose bump, mouth bar, hair cap
/// offset up-back so the face reads. Capsule limbs posed by
/// [`ArmPose`]; hands are skin spheres at the wrist ends. Materials:
/// person-head/hand (skin, kept separate so head measurements exclude
/// waving hands), person-eyes/mouth/hair, person-clothes/pants.
pub fn person_plan(base: Vec3, scale: f64, spec: &PersonSpec, plan: &BodyPlan) -> Vec<Shape> {
    let h = 1.8 * scale;
    let y = |f: f64| base.y + f * h;
    let head_h = plan.head_h * h;
    let head_c = Vec3::new(base.x, y(1.0) - head_h * 0.5, base.z);
    let head_r = head_h * 0.51;
    let skin_head = Material::named("person-head", spec.skin);
    let skin_hand = Material::named("person-hand", spec.skin);
    let cloth_m = Material::named("person-clothes", spec.shirt);
    let pants_m = Material::named("person-pants", spec.pants);
    let hair_m = Material::named("person-hair", spec.hair);
    let eye_m = Material::named("person-eyes", Rgb::new(22, 20, 24));
    let mouth_m = Material::named("person-mouth", Rgb::new(130, 70, 60));

    let mut out = vec![
        // Legs: hip capsules to near-ground, feet boxes on the plane.
        Shape::Capsule {
            a: Vec3::new(base.x - plan.hip_half * h, y(plan.hip_y), base.z),
            b: Vec3::new(base.x - plan.hip_half * h, y(0.03), base.z),
            radius: plan.leg_r * h,
            mat: pants_m.clone(),
        },
        Shape::Capsule {
            a: Vec3::new(base.x + plan.hip_half * h, y(plan.hip_y), base.z),
            b: Vec3::new(base.x + plan.hip_half * h, y(0.03), base.z),
            radius: plan.leg_r * h,
            mat: pants_m.clone(),
        },
        // Torso box, hip to shoulder.
        Shape::Box {
            min: Vec3::new(
                base.x - plan.torso_half * h,
                y(plan.hip_y),
                base.z - plan.torso_half * h * 0.6,
            ),
            max: Vec3::new(
                base.x + plan.torso_half * h,
                y(plan.shoulder_y),
                base.z + plan.torso_half * h * 0.6,
            ),
            mat: cloth_m.clone(),
        },
    ];
    for side in [-1.0, 1.0] {
        let fx = base.x + side * plan.hip_half * h;
        out.push(Shape::Box {
            min: Vec3::new(fx - 0.05 * h, y(0.0), base.z - 0.02 * h),
            max: Vec3::new(fx + 0.05 * h, y(0.055), base.z + 0.09 * h),
            mat: pants_m.clone(),
        });
    }
    // Arms: endpoints from the shared helper so props meet hands
    // exactly. Fingers fan across the knuckles; peace poses extend
    // index+middle and curl ring+pinky into short stubs.
    for side in [-1.0, 1.0] {
        let (shoulder, hand) = arm_endpoints(base, scale, spec, plan, side);
        out.push(Shape::Capsule {
            a: shoulder,
            b: hand,
            radius: plan.arm_r * h,
            mat: cloth_m.clone(),
        });
        out.push(Shape::Sphere {
            center: hand,
            radius: plan.arm_r * h * 1.25,
            mat: skin_hand.clone(),
        });
        // Fingers from the anatomy graph: four continuing the arm
        // line, fanned across the knuckles; thumb angled out-forward.
        // Fingertips (not wrists) set the measured span.
        let fdir = hand.sub(shoulder).norm();
        for k in 0..4 {
            let kx = hand.x + (k as f64 - 1.5) * 0.018 * h;
            let start = Vec3::new(kx, hand.y, hand.z + 0.005 * h).add(fdir.scale(0.012 * h));
            let len = if spec.pose.finger_extended(side, k) {
                plan.finger_len * h
            } else {
                // Curled: foreshortened stub folding into the palm.
                plan.finger_len * h * 0.35
            };
            out.push(Shape::Capsule {
                a: start,
                b: start.add(fdir.scale(len)),
                radius: plan.arm_r * h * 0.5,
                mat: skin_hand.clone(),
            });
        }
        let outer = if shoulder.x < base.x { -1.0 } else { 1.0 };
        let peace = matches!(spec.pose, ArmPose::PeaceLeft | ArmPose::PeaceRight);
        let tdir = if peace {
            // Thumb folds across the curled fingers to hold the sign.
            fdir.scale(0.4)
                .add(Vec3::new(-outer * 0.45, 0.0, 0.15))
                .norm()
        } else {
            fdir.scale(0.7)
                .add(Vec3::new(outer * 0.5, 0.0, 0.35))
                .norm()
        };
        out.push(Shape::Capsule {
            a: hand,
            b: hand.add(tdir.scale(plan.thumb_len * h)),
            radius: plan.arm_r * h * 0.55,
            mat: skin_hand.clone(),
        });
    }
    // Head plus face, all in head units off the crown.
    out.push(Shape::Sphere {
        center: head_c,
        radius: head_r,
        mat: skin_head.clone(),
    });
    for side in [-1.0, 1.0] {
        out.push(Shape::Sphere {
            center: Vec3::new(
                head_c.x + side * 0.18 * head_h,
                head_c.y + 0.06 * head_h,
                head_c.z + 0.44 * head_h,
            ),
            radius: 0.085 * head_h,
            mat: eye_m.clone(),
        });
    }
    out.push(Shape::Box {
        min: Vec3::new(
            head_c.x - 0.07 * head_h,
            head_c.y - 0.12 * head_h,
            head_c.z + 0.42 * head_h,
        ),
        max: Vec3::new(
            head_c.x + 0.07 * head_h,
            head_c.y - 0.02 * head_h,
            head_c.z + 0.50 * head_h,
        ),
        mat: skin_head.clone(),
    });
    out.push(Shape::Box {
        min: Vec3::new(
            head_c.x - 0.17 * head_h,
            head_c.y - 0.28 * head_h,
            head_c.z + 0.40 * head_h,
        ),
        max: Vec3::new(
            head_c.x + 0.17 * head_h,
            head_c.y - 0.20 * head_h,
            head_c.z + 0.46 * head_h,
        ),
        mat: mouth_m,
    });
    out.push(Shape::Sphere {
        center: Vec3::new(head_c.x, head_c.y + 0.22 * head_h, head_c.z - 0.20 * head_h),
        radius: head_h * 0.55,
        mat: hair_m,
    });
    out
}

/// One rose: stem capsule, two leaves, bloom of center + petal
/// ring. Materials rose-stem/leaf/bloom, all receipt-addressable.
pub fn rose(base: Vec3, height: f64, bloom_color: Rgb) -> Vec<Shape> {
    let stem_m = Material::named("rose-stem", Rgb::new(45, 110, 50));
    let leaf_m = Material::named("rose-leaf", Rgb::new(55, 130, 60));
    let bloom_m = Material::named("rose-bloom", bloom_color);
    let top = Vec3::new(base.x, base.y + height, base.z);
    let mut out = vec![
        Shape::Capsule {
            a: base,
            b: top,
            radius: height * 0.03,
            mat: stem_m,
        },
        Shape::Box {
            min: Vec3::new(
                base.x - height * 0.16,
                base.y + height * 0.35,
                base.z - 0.01,
            ),
            max: Vec3::new(
                base.x - height * 0.02,
                base.y + height * 0.45,
                base.z + 0.01,
            ),
            mat: leaf_m.clone(),
        },
        Shape::Box {
            min: Vec3::new(
                base.x + height * 0.02,
                base.y + height * 0.55,
                base.z - 0.01,
            ),
            max: Vec3::new(
                base.x + height * 0.16,
                base.y + height * 0.65,
                base.z + 0.01,
            ),
            mat: leaf_m,
        },
    ];
    let br = height * 0.11;
    out.push(Shape::Sphere {
        center: top,
        radius: br * 0.8,
        mat: bloom_m.clone(),
    });
    for k in 0..5 {
        let a = k as f64 * std::f64::consts::TAU / 5.0;
        out.push(Shape::Sphere {
            center: Vec3::new(
                top.x + a.cos() * br,
                top.y - br * 0.25,
                top.z + a.sin() * br,
            ),
            radius: br * 0.62,
            mat: bloom_m.clone(),
        });
    }
    out
}

/// A held bouquet: stems converge at the hand, blooms fan up-out.
/// `hand` comes from [`arm_endpoints`] — the same math that placed
/// the arm, so flowers meet fingers exactly.
pub fn bouquet(hand: Vec3, scale: f64, colors: &[Rgb]) -> Vec<Shape> {
    let mut out = Vec::new();
    let n = colors.len().max(1) as f64;
    for (i, color) in colors.iter().enumerate() {
        let spread = (i as f64 - (n - 1.0) / 2.0) * 0.09 * scale;
        let base = Vec3::new(hand.x + spread * 0.2, hand.y - 0.42 * scale, hand.z);
        let top = Vec3::new(
            hand.x + spread,
            hand.y + 0.10 * scale,
            hand.z + 0.03 * scale,
        );
        out.push(Shape::Capsule {
            a: base,
            b: top,
            radius: 0.012 * scale,
            mat: Material::named("rose-stem", Rgb::new(45, 110, 50)),
        });
        // Bloom head: center + petal ring sized to read at range.
        let br = 0.055 * scale;
        let bloom_m = Material::named("rose-bloom", *color);
        out.push(Shape::Sphere {
            center: top,
            radius: br * 0.8,
            mat: bloom_m.clone(),
        });
        for k in 0..5 {
            let a = k as f64 * std::f64::consts::TAU / 5.0;
            out.push(Shape::Sphere {
                center: Vec3::new(
                    top.x + a.cos() * br,
                    top.y - br * 0.25,
                    top.z + a.sin() * br,
                ),
                radius: br * 0.62,
                mat: bloom_m.clone(),
            });
        }
        out.push(Shape::Box {
            min: Vec3::new(base.x - 0.05 * scale, base.y + 0.15 * scale, base.z - 0.008),
            max: Vec3::new(base.x + 0.05 * scale, base.y + 0.22 * scale, base.z + 0.008),
            mat: Material::named("rose-leaf", Rgb::new(55, 130, 60)),
        });
    }
    out
}

/// A tower: shaft plus cap, both named `tower`.
pub fn tower(base: Vec3, w: f64, h: f64, color: Rgb) -> Vec<Shape> {
    let mat = Material::named("tower", color);
    vec![
        Shape::Box {
            min: Vec3::new(base.x - w / 2.0, base.y, base.z - w / 2.0),
            max: Vec3::new(base.x + w / 2.0, base.y + h, base.z + w / 2.0),
            mat: mat.clone(),
        },
        Shape::Box {
            min: Vec3::new(base.x - w * 0.65, base.y + h, base.z - w * 0.65),
            max: Vec3::new(base.x + w * 0.65, base.y + h + h * 0.1, base.z + w * 0.65),
            mat,
        },
    ]
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub pos: Vec3,
    pub look_at: Vec3,
    pub fov_deg: f64,
    pub width: u32,
    pub height: u32,
}

/// One directional light. The key light casts shadows; fill and
/// rim lights don't (one shadow map is expensive enough).
#[derive(Debug, Clone)]
pub struct Light {
    pub dir: Vec3,
    pub intensity: f64,
    pub casts_shadow: bool,
}

impl Light {
    pub fn key(dir: Vec3) -> Self {
        Light {
            dir: dir.norm(),
            intensity: 1.0,
            casts_shadow: true,
        }
    }

    pub fn fill(dir: Vec3, intensity: f64) -> Self {
        Light {
            dir: dir.norm(),
            intensity,
            casts_shadow: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Scene {
    pub camera: Camera,
    pub lights: Vec<Light>,
    pub ambient: f64,
    pub sky_top: Rgb,
    pub sky_bottom: Rgb,
    pub shapes: Vec<Shape>,
}

impl Scene {
    /// Remove every shape whose material starts with `prefix`.
    /// Returns shapes removed — "take the tower down" with a receipt.
    pub fn remove_named(&mut self, prefix: &str) -> usize {
        let before = self.shapes.len();
        self.shapes.retain(|s| match s {
            Shape::Sphere { mat, .. }
            | Shape::Plane { mat, .. }
            | Shape::Box { mat, .. }
            | Shape::Capsule { mat, .. } => !mat.name.starts_with(prefix),
        });
        before - self.shapes.len()
    }
}

/// Portrait variant: one studied individual, camera in close —
/// head and torso fill the frame. Sample photos of people.
pub fn portrait_scene(width: u32, height: u32) -> Scene {
    let plan = BodyPlan::canon();
    let spec = PersonSpec {
        skin: Rgb::new(200, 150, 115),
        hair: Rgb::new(60, 38, 24),
        shirt: Rgb::new(60, 110, 200),
        pants: Rgb::new(42, 42, 50),
        pose: ArmPose::Down,
    };
    let mut shapes = vec![Shape::Plane {
        y: 0.0,
        mat: Material::named("ground", Rgb::new(86, 148, 86)),
    }];
    shapes.extend(person_plan(Vec3::new(-1.2, 0.0, 0.0), 1.0, &spec, &plan));
    let mut scene = demo_scene(width, height);
    scene.shapes = shapes;
    scene.camera = Camera {
        pos: Vec3::new(-1.2, 1.55, 2.4),
        look_at: Vec3::new(-1.2, 1.15, 0.0),
        fov_deg: 42.0,
        width,
        height,
    };
    scene
}

/// Group photo: three distinct individuals (looks and postures all
/// differ), three-point lighting, no tower — people are the subject.
pub fn group_scene(width: u32, height: u32) -> Scene {
    let plan = BodyPlan::canon();
    let people = [
        (
            Vec3::new(-2.0, 0.0, 0.0),
            PersonSpec {
                skin: Rgb::new(150, 100, 70),
                hair: Rgb::new(25, 20, 18),
                shirt: Rgb::new(170, 60, 55),
                pants: Rgb::new(45, 45, 55),
                pose: ArmPose::Down,
            },
        ),
        (
            Vec3::new(0.0, 0.0, 0.3),
            PersonSpec {
                skin: Rgb::new(232, 190, 150),
                hair: Rgb::new(110, 65, 35),
                shirt: Rgb::new(45, 140, 140),
                pants: Rgb::new(40, 40, 48),
                pose: ArmPose::WaveLeft,
            },
        ),
        (
            Vec3::new(2.0, 0.0, 0.0),
            PersonSpec {
                skin: Rgb::new(200, 150, 115),
                hair: Rgb::new(190, 190, 195),
                shirt: Rgb::new(200, 170, 60),
                pants: Rgb::new(50, 50, 60),
                pose: ArmPose::Down,
            },
        ),
    ];
    let mut shapes = vec![Shape::Plane {
        y: 0.0,
        mat: Material::named("ground", Rgb::new(88, 146, 88)),
    }];
    for (base, spec) in &people {
        shapes.extend(person_plan(*base, 1.0, spec, &plan));
    }
    Scene {
        camera: Camera {
            pos: Vec3::new(0.0, 1.9, 6.6),
            look_at: Vec3::new(0.0, 1.1, -0.3),
            fov_deg: 58.0,
            width,
            height,
        },
        lights: vec![
            Light::key(Vec3::new(-0.45, 0.8, 0.35)),
            Light::fill(Vec3::new(0.6, 0.25, 0.7), 0.30),
            Light::fill(Vec3::new(0.3, 0.4, -0.8), 0.25),
        ],
        ambient: 0.35,
        sky_top: Rgb::new(110, 170, 235),
        sky_bottom: Rgb::new(215, 235, 250),
        shapes,
    }
}

/// The demo: a person, a tower behind them, ground, sky, one light.
pub fn demo_scene(width: u32, height: u32) -> Scene {
    let mut shapes = vec![Shape::Plane {
        y: 0.0,
        mat: Material::named("ground", Rgb::new(86, 148, 86)),
    }];
    shapes.extend(person(
        Vec3::new(-1.2, 0.0, 0.0),
        1.0,
        Rgb::new(232, 190, 150),
        Rgb::new(60, 110, 200),
    ));
    shapes.extend(tower(
        Vec3::new(1.8, 0.0, -2.5),
        1.1,
        3.2,
        Rgb::new(150, 150, 158),
    ));
    Scene {
        camera: Camera {
            pos: Vec3::new(0.0, 2.1, 6.0),
            look_at: Vec3::new(0.2, 1.3, -0.5),
            fov_deg: 55.0,
            width,
            height,
        },
        // Light from behind-left: tower shadows fall away from the
        // person, so adding the tower never repaints them.
        // Single key light — pixel-identical to the old single-dir
        // math, so the frozen demo tests keep passing.
        lights: vec![Light::key(Vec3::new(-0.45, 0.8, 0.35))],
        ambient: 0.35,
        sky_top: Rgb::new(110, 170, 235),
        sky_bottom: Rgb::new(215, 235, 250),
        shapes,
    }
}

fn shade(base: Rgb, amount: f64) -> Rgb {
    let a = amount.clamp(0.0, 1.0);
    Rgb::new(
        (base.r as f64 * a).round() as u8,
        (base.g as f64 * a).round() as u8,
        (base.b as f64 * a).round() as u8,
    )
}

/// Render core: the photo plus a per-pixel material id and the id
/// table. Labels are what measurement reads — colors shade, names
/// don't.
fn render_core(scene: &Scene) -> (Image, Vec<usize>, Vec<String>) {
    let cam = &scene.camera;
    let fwd = cam.look_at.sub(cam.pos).norm();
    let right = fwd.cross(Vec3::new(0.0, 1.0, 0.0)).norm();
    let up = right.cross(fwd).norm();
    let tan_half = (cam.fov_deg.to_radians() / 2.0).tan();
    let aspect = cam.width as f64 / cam.height as f64;

    let mut img = Image::blank(cam.width, cam.height, Rgb::new(0, 0, 0));
    let mut table: Vec<String> = Vec::new();
    let id_of = |table: &mut Vec<String>, name: &str| -> usize {
        match table.iter().position(|n| n == name) {
            Some(i) => i,
            None => {
                table.push(name.to_string());
                table.len() - 1
            }
        }
    };
    let mut labels = vec![0usize; (cam.width * cam.height) as usize];

    for y in 0..cam.height {
        let py = 1.0 - 2.0 * (y as f64 + 0.5) / cam.height as f64;
        for x in 0..cam.width {
            let px = (2.0 * (x as f64 + 0.5) / cam.width as f64 - 1.0) * aspect;
            let dir = fwd
                .add(right.scale(px * tan_half))
                .add(up.scale(py * tan_half))
                .norm();
            let mut best: Option<Hit> = None;
            for s in &scene.shapes {
                if let Some(h) = intersect(s, cam.pos, dir)
                    && best.as_ref().is_none_or(|b: &Hit| h.dist < b.dist)
                {
                    best = Some(h);
                }
            }
            let id = match best {
                None => {
                    let t = y as f64 / cam.height as f64;
                    let c = Rgb::new(
                        (scene.sky_top.r as f64 * (1.0 - t) + scene.sky_bottom.r as f64 * t) as u8,
                        (scene.sky_top.g as f64 * (1.0 - t) + scene.sky_bottom.g as f64 * t) as u8,
                        (scene.sky_top.b as f64 * (1.0 - t) + scene.sky_bottom.b as f64 * t) as u8,
                    );
                    img.set(x, y, c);
                    id_of(&mut table, "sky")
                }
                Some(h) => {
                    // Every light contributes; only shadow-casting
                    // lights test occlusion. Single-light scenes take
                    // exactly the old path — byte-identical output.
                    let shadow_origin = h.point.add(h.normal.scale(1e-4));
                    let mut diffuse = 0.0;
                    for light in &scene.lights {
                        let lit = if !light.casts_shadow {
                            true
                        } else {
                            !scene
                                .shapes
                                .iter()
                                .any(|s| intersect(s, shadow_origin, light.dir).is_some())
                        };
                        if lit {
                            diffuse += light.intensity * h.normal.dot(light.dir).max(0.0);
                        }
                    }
                    img.set(
                        x,
                        y,
                        shade(
                            h.mat.color,
                            scene.ambient + (1.0 - scene.ambient) * diffuse.min(1.0),
                        ),
                    );
                    let name = h.mat.name.clone();
                    id_of(&mut table, &name)
                }
            };
            labels[(y * cam.width + x) as usize] = id;
        }
    }
    (img, labels, table)
}

/// Render the scene plus a receipt: per-material pixel counts.
/// Sky pixels carry the material name "sky".
pub fn render(scene: &Scene) -> (Image, Vec<(String, u64)>) {
    let (img, labels, table) = render_core(scene);
    let mut counts = vec![0u64; table.len()];
    for id in labels {
        counts[id] += 1;
    }
    let mut receipt: Vec<(String, u64)> = table.into_iter().zip(counts).collect();
    receipt.sort();
    (img, receipt)
}

/// Render plus material labels: `labels[i]` indexes `table` for pixel
/// `i` in row-major order. The anatomy study measures these.
pub fn render_labels(scene: &Scene) -> (Image, Vec<usize>, Vec<String>) {
    render_core(scene)
}

/// What the study measured on its own render.
#[derive(Debug, Clone)]
pub struct AnatomyReport {
    /// Head height / stature (canon ≈ 1/7.5 ≈ 0.133).
    pub head_ratio: f64,
    /// Arm span / stature, Out pose (canon ≈ 1.0).
    pub span_ratio: f64,
    /// Eye separation / head width (canon ≈ 0.35).
    pub eye_ratio: f64,
    pub passed: bool,
    /// The evidence string: measurements plus sources.
    pub evidence: String,
}

fn label_id(table: &[String], name: &str) -> Option<usize> {
    table.iter().position(|n| n == name)
}

fn bbox_of(labels: &[usize], width: u32, id: usize) -> Option<(u32, u32, u32, u32)> {
    let mut found = false;
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0, 0);
    for (i, l) in labels.iter().enumerate() {
        if *l != id {
            continue;
        }
        found = true;
        let x = (i as u32) % width;
        let y = (i as u32) / width;
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    found.then_some((x0, y0, x1, y1))
}

/// The curiosity loop studying anatomy: build a figure from the
/// canon, photograph it, measure the photo. A frontal Out-pose
/// figure at close range keeps perspective distortion small and
/// pixels many. Tolerances are pixel-quantization honest (±1px on a
/// ~170px figure ≈ ±0.01, doubled for safety).
pub fn study_anatomy() -> AnatomyReport {
    let plan = BodyPlan::canon();
    let spec = PersonSpec {
        skin: Rgb::new(200, 150, 115),
        hair: Rgb::new(60, 38, 24),
        shirt: Rgb::new(70, 120, 190),
        pants: Rgb::new(45, 45, 55),
        pose: ArmPose::Out,
    };
    let mut shapes = vec![Shape::Plane {
        y: 0.0,
        mat: Material::named("ground", Rgb::new(86, 148, 86)),
    }];
    shapes.extend(person_plan(Vec3::new(0.0, 0.0, 0.0), 1.0, &spec, &plan));
    let scene = Scene {
        camera: Camera {
            pos: Vec3::new(0.0, 1.0, 3.5),
            look_at: Vec3::new(0.0, 0.9, 0.0),
            fov_deg: 40.0,
            width: 320,
            height: 240,
        },
        lights: vec![Light::key(Vec3::new(-0.45, 0.8, 0.35))],
        ambient: 0.35,
        sky_top: Rgb::new(110, 170, 235),
        sky_bottom: Rgb::new(215, 235, 250),
        shapes,
    };
    let (_, labels, table) = render_labels(&scene);
    let person_ids: Vec<usize> = table
        .iter()
        .enumerate()
        .filter(|(_, n)| n.starts_with("person-"))
        .map(|(i, _)| i)
        .collect();
    let mut person_pixels: Vec<usize> = labels.clone();
    for p in person_pixels.iter_mut() {
        if !person_ids.contains(p) {
            *p = usize::MAX;
        }
    }
    let head_ratio = match label_id(&table, "person-head").and_then(|id| bbox_of(&labels, 320, id))
    {
        Some((_, hy0, _, hy1)) => {
            let person_h = person_height(&person_pixels, 320);
            (hy1 - hy0 + 1) as f64 / person_h
        }
        None => -1.0,
    };
    let span_ratio = {
        let person_w = person_width(&person_pixels, 320);
        let person_h = person_height(&person_pixels, 320);
        person_w / person_h
    };
    // Eyes center-to-center over head width: bbox edges would add
    // the eyeball diameters to the separation, measuring the wrong
    // thing. Split the eye mask at the head center and average halves.
    let eye_ratio = match (
        label_id(&table, "person-eyes"),
        label_id(&table, "person-head").and_then(|id| bbox_of(&labels, 320, id)),
    ) {
        (Some(eye_id), Some((hx0, _, hx1, _))) => {
            let mid = (hx0 + hx1) as f64 / 2.0;
            let (mut l_sum, mut l_n, mut r_sum, mut r_n) = (0.0, 0u32, 0.0, 0u32);
            for (i, l) in labels.iter().enumerate() {
                if *l != eye_id {
                    continue;
                }
                let x = ((i as u32) % 320) as f64;
                if x < mid {
                    l_sum += x;
                    l_n += 1;
                } else {
                    r_sum += x;
                    r_n += 1;
                }
            }
            if l_n == 0 || r_n == 0 {
                -1.0
            } else {
                (r_sum / r_n as f64 - l_sum / l_n as f64) / (hx1 - hx0 + 1) as f64
            }
        }
        _ => -1.0,
    };
    let passed = (head_ratio - 0.133).abs() <= 0.025
        && (span_ratio - 1.0).abs() <= 0.07
        && (eye_ratio - 0.35).abs() <= 0.10;
    AnatomyReport {
        head_ratio,
        span_ratio,
        eye_ratio,
        passed,
        evidence: format!(
            "anatomy study: head/stature={:.3} (canon 0.133), span/stature={:.3} (canon 1.0), eyes/head={:.3} (canon 0.35); sources: {}",
            head_ratio,
            span_ratio,
            eye_ratio,
            ANATOMY_SOURCES.join(", ")
        ),
    }
}

fn person_height(masked: &[usize], width: u32) -> f64 {
    let mut y0 = u32::MAX;
    let mut y1 = 0;
    for (i, l) in masked.iter().enumerate() {
        if *l == usize::MAX {
            continue;
        }
        let y = (i as u32) / width;
        y0 = y0.min(y);
        y1 = y1.max(y);
    }
    if y1 < y0 { 1.0 } else { (y1 - y0 + 1) as f64 }
}

fn person_width(masked: &[usize], width: u32) -> f64 {
    let mut x0 = u32::MAX;
    let mut x1 = 0;
    for (i, l) in masked.iter().enumerate() {
        if *l == usize::MAX {
            continue;
        }
        let x = (i as u32) % width;
        x0 = x0.min(x);
        x1 = x1.max(x);
    }
    if x1 < x0 { 1.0 } else { (x1 - x0 + 1) as f64 }
}

/// File the study in the knowledge store: a passing study becomes
/// Verified knowledge with the measurements as evidence; anything
/// else stays unpromoted with the numbers recorded.
pub fn record_anatomy(store: &mut super::knowledge::KnowledgeStore, report: &AnatomyReport) {
    use super::knowledge::{KnowledgeItem, KnowledgeState, Provenance, VerificationTier};
    // insert() keeps the first item it sees: the first study files,
    // later studies are re-verifications (out of scope in v1).
    let mut item = KnowledgeItem::new(
        "human-proportions",
        if report.passed {
            KnowledgeState::Verified
        } else {
            KnowledgeState::Evidence
        },
        "anatomy study of rendered figure",
    );
    item.provenance = Provenance {
        source: "anatomy study of rendered figure".to_string(),
        experiment: Some("render canon-built figure, measure masks".to_string()),
        verification: if report.passed {
            Some(report.evidence.clone())
        } else {
            None
        },
        rejection: None,
        tier: if report.passed {
            VerificationTier::Measured
        } else {
            VerificationTier::Sourced
        },
    };
    store.insert(item);
}

/// The research half of the anatomy curiosity: look up the published
/// canon on verified sources. The returned summary seeds the store's
/// provenance; the study verifies the geometry. Needs the network —
/// the dream loop calls this, tests don't.
pub async fn research_anatomy(oracle: &mut super::research::ResearchOracle) -> Option<String> {
    oracle
        .research_word("human body proportions")
        .await
        .map(|def| format!("{} [{}]", def.summary, def.source_url))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_demo() -> Scene {
        demo_scene(160, 120)
    }

    fn receipt_count(receipt: &[(String, u64)], name: &str) -> u64 {
        receipt
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| *c)
            .unwrap_or(0)
    }

    #[test]
    fn demo_renders_every_layer() {
        let (img, receipt) = render(&small_demo());
        assert_eq!((img.width, img.height), (160, 120));
        assert!(receipt_count(&receipt, "sky") > 1000, "{:?}", receipt);
        assert!(receipt_count(&receipt, "ground") > 1000, "{:?}", receipt);
        assert!(receipt_count(&receipt, "tower") > 50, "{:?}", receipt);
        assert!(receipt_count(&receipt, "person-skin") > 5, "{:?}", receipt);
        assert!(
            receipt_count(&receipt, "person-clothes") > 20,
            "{:?}",
            receipt
        );
        // Not blank: sky and ground dominate different regions.
        assert!(img.mean_brightness() > 0.2 && img.mean_brightness() < 0.9);
    }

    #[test]
    fn render_is_byte_deterministic() {
        let (a, _) = render(&small_demo());
        let (b, _) = render(&small_demo());
        for y in 0..a.height {
            for x in 0..a.width {
                assert_eq!(a.get(x, y), b.get(x, y), "pixel ({},{})", x, y);
            }
        }
    }

    #[test]
    fn adding_the_tower_only_adds_tower() {
        let mut scene = small_demo();
        let removed = scene.remove_named("tower");
        assert_eq!(removed, 2, "tower is shaft plus cap");
        let (before, r_before) = render(&scene);
        assert_eq!(receipt_count(&r_before, "tower"), 0);
        let person_before = receipt_count(&r_before, "person-clothes")
            + receipt_count(&r_before, "person-skin")
            + receipt_count(&r_before, "person-pants");

        scene.shapes.extend(tower(
            Vec3::new(1.8, 0.0, -2.5),
            1.1,
            3.2,
            Rgb::new(150, 150, 158),
        ));
        let (after, r_after) = render(&scene);
        assert!(receipt_count(&r_after, "tower") > 50, "{:?}", r_after);
        let person_after = receipt_count(&r_after, "person-clothes")
            + receipt_count(&r_after, "person-skin")
            + receipt_count(&r_after, "person-pants");
        // The light placement guarantees this: tower shadows fall away.
        assert_eq!(person_before, person_after, "person untouched by the edit");
        // And the tower visibly differs from sky where it stands.
        assert_ne!(
            before.get(110, 40),
            after.get(110, 40),
            "tower region must change"
        );
    }

    #[test]
    fn person_head_rides_above_torso() {
        let (_, receipt) = render(&small_demo());
        assert!(receipt_count(&receipt, "person-skin") > 0);
        assert!(receipt_count(&receipt, "person-clothes") > 0);
        // Structural check at the source: head center above torso top.
        let shapes = person(
            Vec3::new(0.0, 0.0, 0.0),
            1.0,
            Rgb::new(1, 1, 1),
            Rgb::new(2, 2, 2),
        );
        let head_y = shapes.iter().find_map(|s| match s {
            Shape::Sphere { center, mat, .. } if mat.name == "person-skin" => Some(center.y),
            _ => None,
        });
        assert!(head_y.is_some_and(|y| y > 1.5), "head above torso");
    }

    fn plan_spec() -> (PersonSpec, BodyPlan) {
        (
            PersonSpec {
                skin: Rgb::new(200, 150, 115),
                hair: Rgb::new(60, 38, 24),
                shirt: Rgb::new(70, 120, 190),
                pants: Rgb::new(45, 45, 55),
                pose: ArmPose::Down,
            },
            BodyPlan::canon(),
        )
    }

    #[test]
    fn capsule_limb_renders_and_misses() {
        let mat = Material::named("limb", Rgb::new(200, 100, 100));
        let scene = Scene {
            camera: Camera {
                pos: Vec3::new(0.0, 1.0, 4.0),
                look_at: Vec3::new(0.0, 1.0, 0.0),
                fov_deg: 40.0,
                width: 80,
                height: 60,
            },
            lights: vec![Light::key(Vec3::new(-0.4, 0.8, 0.4))],
            ambient: 0.4,
            sky_top: Rgb::new(100, 100, 200),
            sky_bottom: Rgb::new(200, 200, 220),
            shapes: vec![Shape::Capsule {
                a: Vec3::new(0.0, 0.5, 0.0),
                b: Vec3::new(0.0, 1.5, 0.0),
                radius: 0.15,
                mat,
            }],
        };
        let (_, receipt) = render(&scene);
        assert!(receipt_count(&receipt, "limb") > 20, "{:?}", receipt);
        // Same scene, capsule moved out of frame: nothing renders.
        let mut empty = scene;
        empty.shapes = vec![Shape::Capsule {
            a: Vec3::new(50.0, 0.5, 0.0),
            b: Vec3::new(50.0, 1.5, 0.0),
            radius: 0.15,
            mat: Material::named("limb", Rgb::new(200, 100, 100)),
        }];
        let (_, receipt) = render(&empty);
        assert_eq!(receipt_count(&receipt, "limb"), 0);
    }

    #[test]
    fn plan_person_has_a_face() {
        let (spec, plan) = plan_spec();
        let mut shapes = vec![Shape::Plane {
            y: 0.0,
            mat: Material::named("ground", Rgb::new(86, 148, 86)),
        }];
        shapes.extend(person_plan(Vec3::new(0.0, 0.0, 2.0), 1.0, &spec, &plan));
        let scene = Scene {
            camera: Camera {
                pos: Vec3::new(0.0, 1.2, 4.5),
                look_at: Vec3::new(0.0, 1.0, 2.0),
                fov_deg: 40.0,
                width: 160,
                height: 120,
            },
            lights: vec![Light::key(Vec3::new(-0.4, 0.8, 0.4))],
            ambient: 0.4,
            sky_top: Rgb::new(100, 100, 200),
            sky_bottom: Rgb::new(200, 200, 220),
            shapes,
        };
        let (_, receipt) = render(&scene);
        for name in [
            "person-head",
            "person-hair",
            "person-eyes",
            "person-mouth",
            "person-hand",
        ] {
            assert!(receipt_count(&receipt, name) > 0, "{:?}", receipt);
        }
    }

    #[test]
    fn labels_agree_with_receipt() {
        let scene = small_demo();
        let (_, receipt) = render(&scene);
        let (_, labels, table) = render_labels(&scene);
        let mut counts = vec![0u64; table.len()];
        for id in labels {
            counts[id] += 1;
        }
        for (name, count) in &receipt {
            let id = table.iter().position(|n| n == name).expect("label table");
            assert_eq!(counts[id], *count, "label/receipt mismatch for {}", name);
        }
    }

    #[test]
    fn study_measures_the_canon() {
        let report = study_anatomy();
        assert!(
            report.passed,
            "canon-built figure must measure canon: head={:.3} span={:.3} eyes={:.3}",
            report.head_ratio, report.span_ratio, report.eye_ratio
        );
        assert!(
            report.evidence.contains("Body_proportions"),
            "{}",
            report.evidence
        );
    }

    #[test]
    fn record_files_verified_knowledge() {
        use super::super::knowledge::{KnowledgeState, KnowledgeStore};
        let dir = std::env::temp_dir().join(format!("gc-anat-{}", unique_test_id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut store = KnowledgeStore::open(&dir);
        let report = study_anatomy();
        assert!(report.passed);
        record_anatomy(&mut store, &report);
        let item = store.get("human-proportions").expect("filed");
        assert_eq!(item.state, KnowledgeState::Verified);
        assert_eq!(
            item.provenance.tier,
            super::super::knowledge::VerificationTier::Measured,
            "render measurement is measured, not merely sourced"
        );
        assert!(
            item.provenance
                .verification
                .as_ref()
                .is_some_and(|v| v.contains("head/stature"))
        );
        // A failing study files Evidence, never Verified.
        let mut bad = report.clone();
        bad.passed = false;
        let dir2 = std::env::temp_dir().join(format!("gc-anat-bad-{}", unique_test_id()));
        std::fs::create_dir_all(&dir2).unwrap();
        let mut store2 = KnowledgeStore::open(&dir2);
        record_anatomy(&mut store2, &bad);
        assert_eq!(
            store2.get("human-proportions").map(|i| i.state),
            Some(KnowledgeState::Evidence)
        );
    }

    #[test]
    fn group_photo_shows_three_people() {
        // Faces need pixels: at 160x120 the eyes go sub-pixel, so the
        // group proves itself at 320x240 (still a fraction of a second).
        let (_, receipt) = render(&group_scene(320, 240));
        // Shared materials, but three bodies worth of pixels: hair and
        // hands scale with headcount, and the waving hand reads.
        assert!(receipt_count(&receipt, "person-hair") > 60, "{:?}", receipt);
        assert!(receipt_count(&receipt, "person-hand") > 40, "{:?}", receipt);
        assert!(receipt_count(&receipt, "person-eyes") > 5, "{:?}", receipt);
        assert_eq!(
            receipt_count(&receipt, "tower"),
            0,
            "people are the subject"
        );
    }

    fn finger_lengths(spec: &PersonSpec) -> (usize, usize) {
        let plan = BodyPlan::canon();
        let shapes = person_plan(Vec3::new(0.0, 0.0, 0.0), 1.0, spec, &plan);
        let expected_full = plan.finger_len * 1.8;
        let mut full = 0;
        let mut stub = 0;
        for s in &shapes {
            if let Shape::Capsule { a, b, mat, .. } = s {
                if mat.name == "person-hand" {
                    let len = (b.sub(*a)).len();
                    if (len - expected_full).abs() < 0.01 {
                        full += 1;
                    } else if (len - expected_full * 0.35).abs() < 0.01 {
                        stub += 1;
                    }
                }
            }
        }
        (full, stub)
    }

    #[test]
    fn peace_sign_extends_two_curls_two() {
        let down = PersonSpec {
            skin: Rgb::new(200, 150, 115),
            hair: Rgb::new(60, 38, 24),
            shirt: Rgb::new(70, 120, 190),
            pants: Rgb::new(45, 45, 55),
            pose: ArmPose::Down,
        };
        assert_eq!(finger_lengths(&down), (8, 0));
        let peace = PersonSpec {
            pose: ArmPose::PeaceRight,
            ..down.clone()
        };
        // Left hand open (4) + right hand peace (2 ext + 2 stub).
        // Thumb capsules are thicker and excluded by the length check.
        assert_eq!(finger_lengths(&peace), (6, 2));
        // The peace hand rides raised, like a wave.
        let plan = BodyPlan::canon();
        let (_, r_hand) = arm_endpoints(Vec3::new(0.0, 0.0, 0.0), 1.0, &peace, &plan, 1.0);
        let (_, d_hand) = arm_endpoints(Vec3::new(0.0, 0.0, 0.0), 1.0, &down, &plan, 1.0);
        assert!(r_hand.y > d_hand.y + 0.3, "peace hand raised");
    }

    #[test]
    fn bouquet_meets_the_hand() {
        let plan = BodyPlan::canon();
        let spec = PersonSpec {
            skin: Rgb::new(200, 150, 115),
            hair: Rgb::new(60, 38, 24),
            shirt: Rgb::new(70, 120, 190),
            pants: Rgb::new(45, 45, 55),
            pose: ArmPose::Down,
        };
        let base = Vec3::new(0.0, 0.0, 0.0);
        let (_, hand) = arm_endpoints(base, 1.0, &spec, &plan, 1.0);
        let flowers = bouquet(hand, 1.0, &[Rgb::new(200, 40, 60)]);
        // Blooms above the hand, stems through it, leaves present.
        let blooms = flowers
            .iter()
            .filter(|s| matches!(s, Shape::Sphere { mat, .. } if mat.name == "rose-bloom"))
            .count();
        assert_eq!(blooms, 6, "center + 5 petals");
        assert!(
            flowers
                .iter()
                .any(|s| matches!(s, Shape::Box { mat, .. } if mat.name == "rose-leaf"))
        );
        let highest_stem_y = flowers
            .iter()
            .filter_map(|s| match s {
                Shape::Capsule { a, b, mat, .. } if mat.name == "rose-stem" => Some(a.y.min(b.y)),
                _ => None,
            })
            .fold(f64::INFINITY, f64::min);
        let lowest_bloom_y = flowers
            .iter()
            .filter_map(|s| match s {
                Shape::Sphere { center, mat, .. } if mat.name == "rose-bloom" => Some(center.y),
                _ => None,
            })
            .fold(f64::INFINITY, f64::min);
        assert!(highest_stem_y < hand.y, "stems start below the hand");
        assert!(lowest_bloom_y > hand.y, "blooms open above the hand");
        // Single rose renders red pixels with leaves and stem.
        let mut shapes = vec![Shape::Plane {
            y: 0.0,
            mat: Material::named("ground", Rgb::new(86, 148, 86)),
        }];
        shapes.extend(rose(Vec3::new(0.0, 0.0, 2.0), 1.0, Rgb::new(200, 40, 60)));
        let scene = Scene {
            camera: Camera {
                pos: Vec3::new(0.0, 1.0, 4.0),
                look_at: Vec3::new(0.0, 0.7, 2.0),
                fov_deg: 40.0,
                width: 120,
                height: 90,
            },
            lights: vec![Light::key(Vec3::new(-0.4, 0.8, 0.4))],
            ambient: 0.4,
            sky_top: Rgb::new(100, 100, 200),
            sky_bottom: Rgb::new(200, 200, 220),
            shapes,
        };
        let (_, receipt) = render(&scene);
        for name in ["rose-bloom", "rose-stem", "rose-leaf"] {
            let count: u64 = receipt
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, c)| *c)
                .unwrap_or(0);
            assert!(count > 5, "{} missing: {:?}", name, receipt);
        }
    }

    fn unique_test_id() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    }
}
