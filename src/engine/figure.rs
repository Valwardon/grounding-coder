//! Figures from genomes: the creation test made concrete.
//!
//! "A guy with a big nose waving a Canadian flag while laughing
//! with a noodle hat" decomposes into known operations: a genome
//! with nose morphs maxed, smile at full, a procedural flag (bars +
//! fan-triangulated maple leaf, cloth-waved), a noodle-bowl hat
//! (hemisphere + helical strand capsules), one raised arm. Every
//! pixel below is procedural — no plates, no likenesses, no minors.
//! Adults only, by genome validation.
use super::human::HumanGenome;
use super::mesh::Mesh;
use super::scene::{Camera, Light, Material, Scene, Shape, Vec3};
use super::vision::Rgb;

/// Maple leaf outline, normalized ±1, y up. Simplified 22-point
/// symmetric leaf plus a separate stem box.
fn maple_points() -> Vec<[f64; 2]> {
    vec![
        [0.0, -1.0],
        [0.14, -0.50],
        [0.50, -0.62],
        [0.32, -0.28],
        [0.66, -0.30],
        [0.44, 0.00],
        [0.78, 0.10],
        [0.46, 0.30],
        [0.60, 0.64],
        [0.26, 0.50],
        [0.12, 0.88],
        [0.0, 0.74],
        [-0.12, 0.88],
        [-0.26, 0.50],
        [-0.60, 0.64],
        [-0.46, 0.30],
        [-0.78, 0.10],
        [-0.44, 0.00],
        [-0.66, -0.30],
        [-0.32, -0.28],
        [-0.50, -0.62],
        [-0.14, -0.50],
    ]
}

/// The test figure: laughing genome man with a big nose, noodle-bowl
/// hat, raised right arm waving a rippling Canadian flag. Returns
/// shapes in world space, figure facing +z.
pub fn canadian_waver(genome: &HumanGenome) -> Vec<Shape> {
    let mut out = Vec::new();
    let skin = Material::skin("skin", genome.skin_tone);
    let shirt = Material::named("shirt", Rgb::new(40, 60, 120));
    let wood = Material::named("wood", Rgb::new(120, 85, 50));
    let flag_red = Material::named("flag-red", Rgb::new(200, 30, 40));
    let flag_white = Material::named("flag-white", Rgb::new(235, 235, 235));
    let wheat = Material::named("noodle", Rgb::new(215, 185, 125));
    let bowl_m = Material::named("bowl", Rgb::new(230, 228, 220));

    // Head: genome mesh at 0.30 scale, center y=1.62.
    let head_scale = 0.30;
    let head_c = Vec3::new(0.0, 1.62, 0.0);
    let mut head = genome.head_mesh();
    // Scale about origin then translate: bake world coords.
    for v in head.verts.iter_mut() {
        v[0] = v[0] * head_scale + head_c.x;
        v[1] = v[1] * head_scale + head_c.y;
        v[2] = v[2] * head_scale + head_c.z;
    }
    head.compute_normals();
    out.push(Shape::Mesh {
        mesh: head,
        mat: skin.clone(),
    });

    // Eyeballs: white + pupil, seated in the sockets.
    for side in [-1.0, 1.0] {
        let ec = Vec3::new(side * 0.037, 1.612, 0.123);
        out.push(Shape::Sphere {
            center: ec,
            radius: 0.034,
            mat: Material::named("eye-white", Rgb::new(232, 230, 225)),
        });
        out.push(Shape::Sphere {
            center: Vec3::new(ec.x, ec.y, ec.z + 0.026),
            radius: 0.016,
            mat: Material::named("pupil", Rgb::new(25, 18, 15)),
        });
    }
    // Open laughing mouth: dark inset below the lip ridge.
    out.push(Shape::Sphere {
        center: Vec3::new(0.0, 1.526, 0.100),
        radius: 0.030,
        mat: Material::named("mouth-open", Rgb::new(70, 25, 20)),
    });

    // Neck, shoulders, torso (clothed — no nude render path exists).
    out.push(Shape::Capsule {
        a: Vec3::new(0.0, 1.36, 0.0),
        b: Vec3::new(0.0, 1.50, 0.0),
        radius: 0.058,
        mat: skin.clone(),
    });
    out.push(Shape::Capsule {
        a: Vec3::new(-0.24, 1.22, 0.0),
        b: Vec3::new(0.24, 1.22, 0.0),
        radius: 0.125,
        mat: shirt.clone(),
    });
    out.push(Shape::Box {
        min: Vec3::new(-0.22, 0.72, -0.11),
        max: Vec3::new(0.22, 1.24, 0.11),
        mat: shirt.clone(),
    });
    // Raised right arm in a sleeve, mitt hand.
    let shoulder = Vec3::new(0.24, 1.25, 0.0);
    let hand = Vec3::new(0.52, 1.72, 0.08);
    out.push(Shape::Capsule {
        a: shoulder,
        b: hand,
        radius: 0.068,
        mat: shirt.clone(),
    });
    out.push(Shape::Sphere {
        center: hand,
        radius: 0.058,
        mat: skin.clone(),
    });
    // Left arm down, out of the way.
    out.push(Shape::Capsule {
        a: Vec3::new(-0.24, 1.25, 0.0),
        b: Vec3::new(-0.30, 0.85, 0.02),
        radius: 0.068,
        mat: shirt.clone(),
    });

    // Flag pole from the raised hand.
    let pole_top = Vec3::new(hand.x + 0.03, hand.y + 0.95, hand.z);
    out.push(Shape::Capsule {
        a: hand,
        b: pole_top,
        radius: 0.012,
        mat: wood,
    });
    // Flag: three cloth panels rippling +x, leaf centered.
    // Still-photo wave frozen at t=1.2.
    let wave = |v: [f64; 3]| {
        let t = 1.2;
        let amp = 0.06 * (v[0] / 0.6 + 0.5).clamp(0.0, 1.0) + 0.008;
        [
            v[0],
            v[1],
            v[2] + amp * (v[0] * 9.0 - t * 2.0).sin() + 0.015 * (v[1] * 7.0).sin(),
        ]
    };
    let flag_y = pole_top.y - 0.25;
    let panels = [
        (0.61, flag_red.clone()),
        (0.81, flag_white.clone()),
        (1.01, flag_red.clone()),
    ];
    for (cx, mat) in panels {
        let mut panel = Mesh::plane_grid(0.2, 0.4, 8, 10);
        panel.displace(|v| {
            let w = wave([v[0] + cx - 0.88, v[1], v[2]]);
            [w[0] + cx, w[1] + flag_y, w[2]]
        });
        out.push(Shape::Mesh { mesh: panel, mat });
    }
    // Maple leaf: fan polygon on the white panel, slight offset.
    // Real flags size the leaf ~40% of flag height — small enough
    // to need the white panel behind it, big enough to read.
    let leaf_pts: Vec<[f64; 2]> = maple_points()
        .into_iter()
        .map(|[x, y]| [x * 0.105, y * 0.105])
        .collect();
    let mut leaf = Mesh::fan_polygon(&leaf_pts);
    let lw = wave([0.0, 0.0, 0.0]);
    for v in leaf.verts.iter_mut() {
        v[0] += 0.88;
        v[1] += flag_y;
        v[2] += lw[2] + 0.004;
    }
    leaf.compute_normals();
    out.push(Shape::Mesh {
        mesh: leaf,
        mat: Material::named("leaf", Rgb::new(200, 30, 40)),
    });
    // Leaf stem: thin red box below the leaf.
    out.push(Shape::Box {
        min: Vec3::new(0.874, flag_y - 0.145, lw[2] + 0.001),
        max: Vec3::new(0.886, flag_y - 0.075, lw[2] + 0.009),
        mat: Material::named("leaf", Rgb::new(200, 30, 40)),
    });

    // Noodle-bowl hat: squashed hemisphere on the crown + helical
    // wheat strands spilling out.
    let mut bowl = Mesh::hemisphere(20, 8, 0.17, 0.72);
    for v in bowl.verts.iter_mut() {
        v[0] += 0.0;
        v[1] += 1.80;
        v[2] += -0.01;
    }
    bowl.compute_normals();
    out.push(Shape::Mesh {
        mesh: bowl,
        mat: bowl_m,
    });
    for s in 0..10 {
        let a0 = s as f64 * 2.4;
        let r = 0.05 + 0.075 * ((s as f64 * 0.7).sin().abs());
        let y0 = 1.86 + 0.02 * ((s as f64 * 1.3).sin());
        let mut prev = Vec3::new(a0.cos() * r, y0, -0.01 + a0.sin() * r);
        for k in 1..=6 {
            let a = a0 + k as f64 * 0.55;
            let p = Vec3::new(
                a.cos() * (r + 0.008 * k as f64),
                y0 + 0.016 * k as f64,
                -0.01 + a.sin() * (r + 0.008 * k as f64),
            );
            out.push(Shape::Capsule {
                a: prev,
                b: p,
                radius: 0.008,
                mat: wheat.clone(),
            });
            prev = p;
        }
    }
    out
}

/// The test scene: waver framed head-to-waist against sky.
pub fn waver_scene(width: u32, height: u32, genome: &HumanGenome) -> Scene {
    Scene {
        camera: Camera {
            pos: Vec3::new(0.40, 1.50, 3.0),
            look_at: Vec3::new(0.40, 1.45, 0.0),
            fov_deg: 34.0,
            width,
            height,
        },
        lights: vec![
            Light::key(Vec3::new(-0.45, 0.8, 0.35)),
            Light::fill(Vec3::new(0.6, 0.25, 0.7), 0.30),
        ],
        ambient: 0.38,
        sky_top: Rgb::new(110, 170, 235),
        sky_bottom: Rgb::new(215, 235, 250),
        shapes: canadian_waver(genome),
    }
}

#[cfg(test)]
mod tests {
    use super::super::scene;
    use super::*;

    fn test_genome() -> HumanGenome {
        HumanGenome::with_params(35, 0.5, 0.55, 0.5, 0.95, 0.8, 0.5, 0.6, 0.5, 1.0).unwrap()
    }

    #[test]
    fn waver_builds_all_parts() {
        let shapes = canadian_waver(&test_genome());
        let count = |prefix: &str| {
            shapes
                .iter()
                .filter(|s| match s {
                    Shape::Sphere { mat, .. }
                    | Shape::Plane { mat, .. }
                    | Shape::Box { mat, .. }
                    | Shape::Capsule { mat, .. }
                    | Shape::Mesh { mat, .. } => mat.name.starts_with(prefix),
                })
                .count()
        };
        assert!(count("skin") > 0, "head and hands");
        assert!(count("flag-") >= 3, "bars plus center");
        assert!(count("leaf") >= 1, "maple leaf");
        assert!(count("noodle") >= 50, "strand segments");
        assert!(count("eye-") >= 2, "eyes");
    }

    #[test]
    fn waver_renders_readable_pixels() {
        // The acceptance test, in code: skin, flag red/white, wheat,
        // eyes — one render, measured, no eyeballs required.
        let (img, receipt) = scene::render(&waver_scene(200, 150, &test_genome()));
        let get = |name: &str| {
            receipt
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, c)| *c)
                .unwrap_or(0)
        };
        let skin = get("skin") + get("eye-white");
        assert!(skin > 300, "face reads: {:?}", receipt);
        let red = get("flag-red") + get("leaf") + get("mouth-open");
        assert!(red > 150, "flag reads: {:?}", receipt);
        let white = get("flag-white") + get("eye-white");
        assert!(white > 100, "contrast reads: {:?}", receipt);
        let wheat: u64 = receipt
            .iter()
            .filter(|(n, _)| n == "noodle")
            .map(|(_, c)| *c)
            .sum();
        assert!(wheat > 50, "noodles read: {:?}", receipt);
        assert_eq!((img.width, img.height), (200, 150));
    }
}
