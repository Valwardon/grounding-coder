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

#[derive(Debug, Clone)]
pub struct Scene {
    pub camera: Camera,
    pub light_dir: Vec3,
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
            Shape::Sphere { mat, .. } | Shape::Plane { mat, .. } | Shape::Box { mat, .. } => {
                !mat.name.starts_with(prefix)
            }
        });
        before - self.shapes.len()
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
        light_dir: Vec3::new(-0.45, 0.8, 0.35).norm(),
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

/// Render the scene plus a receipt: per-material pixel counts.
/// Sky pixels carry the material name "sky".
pub fn render(scene: &Scene) -> (Image, Vec<(String, u64)>) {
    let cam = &scene.camera;
    let fwd = cam.look_at.sub(cam.pos).norm();
    let right = fwd.cross(Vec3::new(0.0, 1.0, 0.0)).norm();
    let up = right.cross(fwd).norm();
    let tan_half = (cam.fov_deg.to_radians() / 2.0).tan();
    let aspect = cam.width as f64 / cam.height as f64;

    let mut img = Image::blank(cam.width, cam.height, Rgb::new(0, 0, 0));
    let mut receipt: Vec<(String, u64)> = Vec::new();
    let mut count = |name: &str| match receipt.iter_mut().find(|(n, _)| n == name) {
        Some(e) => e.1 += 1,
        None => receipt.push((name.to_string(), 1)),
    };

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
            match best {
                None => {
                    let t = y as f64 / cam.height as f64;
                    let c = Rgb::new(
                        (scene.sky_top.r as f64 * (1.0 - t) + scene.sky_bottom.r as f64 * t) as u8,
                        (scene.sky_top.g as f64 * (1.0 - t) + scene.sky_bottom.g as f64 * t) as u8,
                        (scene.sky_top.b as f64 * (1.0 - t) + scene.sky_bottom.b as f64 * t) as u8,
                    );
                    img.set(x, y, c);
                    count("sky");
                }
                Some(h) => {
                    // Hard shadow: anything between the point and the
                    // light kills the diffuse term, ambient remains.
                    let to_light = scene.light_dir;
                    let shadow_origin = h.point.add(h.normal.scale(1e-4));
                    let blocked = scene
                        .shapes
                        .iter()
                        .any(|s| intersect(s, shadow_origin, to_light).is_some());
                    let diffuse = if blocked {
                        0.0
                    } else {
                        h.normal.dot(to_light).max(0.0)
                    };
                    img.set(
                        x,
                        y,
                        shade(h.mat.color, scene.ambient + (1.0 - scene.ambient) * diffuse),
                    );
                    let name = h.mat.name.clone();
                    count(&name);
                }
            }
        }
    }
    receipt.sort();
    (img, receipt)
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
}
