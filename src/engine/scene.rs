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

/// The demo: a tower on open ground under one light. People are
/// researched and composited, never procedurally meshed — the
/// renderer builds worlds, not bodies.
pub fn demo_scene(width: u32, height: u32) -> Scene {
    let mut shapes = vec![Shape::Plane {
        y: 0.0,
        mat: Material::named("ground", Rgb::new(86, 148, 86)),
    }];
    shapes.extend(tower(
        Vec3::new(0.6, 0.0, -2.5),
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

/// REMOVED: the render-measure study built figures from the canon
/// and measured its own renders (self-consistency, not validation).
/// Canon bodies are gone; people come from researched photographs.
/// The measurement machinery (render_labels, masks) stays for
/// measuring real plates — that validation is future work.
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
        // Tower demo: sky, ground, tower. People come from
        // photographs, never from meshes — no person-* materials here.
        let (img, receipt) = render(&small_demo());
        assert_eq!((img.width, img.height), (160, 120));
        assert!(receipt_count(&receipt, "sky") > 1000, "{:?}", receipt);
        assert!(receipt_count(&receipt, "ground") > 1000, "{:?}", receipt);
        assert!(receipt_count(&receipt, "tower") > 50, "{:?}", receipt);
        assert!(
            receipt.iter().all(|(n, _)| !n.starts_with("person-")),
            "no procedural people: {:?}",
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

        scene.shapes.extend(tower(
            Vec3::new(0.6, 0.0, -2.5),
            1.1,
            3.2,
            Rgb::new(150, 150, 158),
        ));
        let (after, r_after) = render(&scene);
        assert!(receipt_count(&r_after, "tower") > 50, "{:?}", r_after);
        // The tower occludes whatever stood behind it — sky, ground,
        // or both — and adds nothing else.
        let bg_before = receipt_count(&r_before, "ground") + receipt_count(&r_before, "sky");
        let bg_after = receipt_count(&r_after, "ground") + receipt_count(&r_after, "sky");
        assert!(bg_after < bg_before, "tower must cover background");
        // And the tower visibly differs from sky where it stands.
        assert_ne!(
            before.get(80, 40),
            after.get(80, 40),
            "tower region must change"
        );
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

    // REMOVED with the canon bodies: study_anatomy measured renders
    // of procedural figures (self-consistency). Real validation —
    // measuring researched plates against the canon — is future work,
    // with the measurement machinery (render_labels) kept for it.

    #[test]
    fn bouquet_meets_the_hand() {
        // Hand position is explicit now (no arm builder to ask).
        let hand = Vec3::new(0.3, 0.9, 0.1);
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
}
