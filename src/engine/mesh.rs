//! Meshes: the geometry substrate of generated humans.
//!
//! A head is a parametric surface, not a sculpted asset: UV-sphere
//! topology deformed by radial feature functions (cranium, jaw,
//! brow, nose, lips, eye sockets, chin), each driven by a genome
//! parameter in 0..1. Same parameters twice → byte-identical verts.
//! Blendshapes (smile and friends) are rule-selected vertex deltas,
//! never hand-picked indices. Normals are smooth-averaged; the
//! renderer culls backfaces and prechecks a bounding sphere.
use std::f64::consts::TAU;

/// Indexed triangle mesh with smooth vertex normals.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub verts: Vec<[f64; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub normals: Vec<[f64; 3]>,
}

/// Head-shape parameters, all 0..1. Defaults 0.5 read as an
/// unremarkable adult head; extremes stay inside the mesh without
/// tearing (displacements are bounded gaussians).
#[derive(Debug, Clone)]
pub struct HeadShape {
    pub cranial_width: f64,
    pub jaw_width: f64,
    pub cheek_projection: f64,
    pub nose_length: f64,
    pub nose_width: f64,
    pub eye_spacing: f64,
    pub lip_fullness: f64,
    pub chin_projection: f64,
}

impl Default for HeadShape {
    fn default() -> Self {
        HeadShape {
            cranial_width: 0.5,
            jaw_width: 0.5,
            cheek_projection: 0.5,
            nose_length: 0.5,
            nose_width: 0.5,
            eye_spacing: 0.5,
            lip_fullness: 0.5,
            chin_projection: 0.5,
        }
    }
}

fn gauss(x: f64, sigma: f64) -> f64 {
    (-x * x / (2.0 * sigma * sigma)).exp()
}

fn ang_dist(u: f64) -> f64 {
    let d = u % TAU;
    d.min(TAU - d)
}

impl Mesh {
    /// Parametric head: y up, +z face, unit-ish height 1.0 centered
    /// near origin. `nu` azimuth steps, `nv` polar steps.
    pub fn parametric_head(nu: u32, nv: u32, p: &HeadShape) -> Self {
        let nu = nu.max(8);
        let nv = nv.max(6);
        let mut verts = Vec::new();
        for j in 0..=nv {
            let v = j as f64 * std::f64::consts::PI / nv as f64;
            let sin_v = v.sin();
            let cos_v = v.cos();
            for i in 0..=nu {
                let u = i as f64 * TAU / nu as f64;
                // Base ellipsoid: wider cranium, slightly long face.
                let cw = 0.78 + 0.12 * p.cranial_width;
                let mut x = 0.5 * sin_v * u.sin() * cw;
                let y = 0.5 * cos_v;
                let mut z = 0.5 * sin_v * u.cos() * 0.92;
                // Jaw taper below the equator: wide jaws taper
                // less (broad chin), narrow jaws taper more.
                if cos_v < 0.0 {
                    let t = -cos_v * (0.95 - 0.45 * p.jaw_width);
                    x *= 1.0 - 0.35 * t;
                    z *= 1.0 - 0.25 * t;
                }
                // Cheeks: forward fullness at mid-face sides.
                let cheek = gauss(ang_dist(u) - 0.9, 0.35)
                    * gauss(v - 1.9, 0.35)
                    * (0.02 + 0.04 * p.cheek_projection);
                z += cheek;
                x += cheek * 0.4 * u.sin().signum();
                // Brow ridge above the eyes.
                z += 0.018 * gauss(ang_dist(u), 0.5) * gauss(v - 1.25, 0.18);
                // Nose: ridge at front-center, sized by params.
                let nw = 0.16 + 0.14 * p.nose_width;
                z +=
                    (0.02 + 0.075 * p.nose_length) * gauss(ang_dist(u), nw) * gauss(v - 1.95, 0.22);
                // Lips: horizontal ridge, lower third.
                z += (0.008 + 0.030 * p.lip_fullness)
                    * gauss(ang_dist(u), 0.30)
                    * gauss(v - 2.25, 0.11);
                // Eye sockets: paired depressions.
                let eye_az = 0.24 + 0.12 * p.eye_spacing;
                let d_eye = (ang_dist(u) - eye_az).abs();
                z -= 0.030 * gauss(d_eye, 0.13) * gauss(v - 1.62, 0.16);
                // Chin: forward nub near the bottom front.
                z += (0.010 + 0.030 * p.chin_projection)
                    * gauss(ang_dist(u), 0.28)
                    * gauss(v - 2.72, 0.16);
                verts.push([x, y, z]);
            }
        }
        let mut faces = Vec::new();
        let row = nu + 1;
        for j in 0..nv {
            for i in 0..nu {
                let a = j * row + i;
                let b = a + 1;
                let c = a + row;
                let d = c + 1;
                faces.push([a, c, b]);
                faces.push([b, c, d]);
            }
        }
        let mut mesh = Mesh {
            verts,
            faces,
            normals: Vec::new(),
        };
        mesh.compute_normals();
        mesh
    }

    /// Smooth normals: face normals averaged at shared verts.
    pub fn compute_normals(&mut self) {
        let mut acc = vec![[0.0; 3]; self.verts.len()];
        for f in &self.faces {
            let a = self.verts[f[0] as usize];
            let b = self.verts[f[1] as usize];
            let c = self.verts[f[2] as usize];
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let n = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            for idx in f {
                acc[*idx as usize][0] += n[0];
                acc[*idx as usize][1] += n[1];
                acc[*idx as usize][2] += n[2];
            }
        }
        self.normals = acc
            .into_iter()
            .map(|n| {
                let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                if l == 0.0 {
                    [0.0, 1.0, 0.0]
                } else {
                    [n[0] / l, n[1] / l, n[2] / l]
                }
            })
            .collect();
    }

    /// Blendshape deltas: rule-selected verts near the mouth corners,
    /// moved up for a smile. The rule references head proportions,
    /// never magic indices — it works on any parametric head.
    pub fn smile_deltas(&self) -> Vec<(u32, [f64; 3])> {
        // Mouth corners sit where the lip ridge ends: recover the
        // lip band (v≈2.25 → y = 0.5·cos(2.25) ≈ −0.31) and take
        // verts at moderate |x| in that band.
        let mut out = Vec::new();
        for (i, v) in self.verts.iter().enumerate() {
            if (v[1] + 0.31).abs() < 0.045 {
                let ax = v[0].abs();
                if (0.06..0.16).contains(&ax) && v[2] > 0.30 {
                    out.push((i as u32, [0.0, 0.035, 0.008]));
                }
            }
        }
        out
    }

    /// Apply named deltas at a weight. Deterministic per (mesh,
    /// deltas, weight); normals recomputed after.
    pub fn blendshape(&self, deltas: &[(u32, [f64; 3])], weight: f64) -> Self {
        let mut verts = self.verts.clone();
        for (i, d) in deltas {
            if let Some(v) = verts.get_mut(*i as usize) {
                v[0] += d[0] * weight;
                v[1] += d[1] * weight;
                v[2] += d[2] * weight;
            }
        }
        let mut mesh = Mesh {
            verts,
            faces: self.faces.clone(),
            normals: Vec::new(),
        };
        mesh.compute_normals();
        mesh
    }

    /// Eyelid-open deltas: verts within `radius` of an eye center
    /// split — upper verts rise, lower verts fall off with distance.
    /// Rule-selected like `smile_deltas` (no sculpted indices); the
    /// caller keeps eyeball verts out via `skip`. Weight scales the
    /// opening; 1.0 parts the lids ~9mm total.
    pub fn eyelid_deltas(
        &self,
        centers: &[[f64; 3]],
        skip: &[usize],
        radius: f64,
    ) -> Vec<(u32, [f64; 3])> {
        let skipped: std::collections::HashSet<usize> = skip.iter().copied().collect();
        let mut out = Vec::new();
        for (i, v) in self.verts.iter().enumerate() {
            if skipped.contains(&i) {
                continue;
            }
            for c in centers {
                let d =
                    ((v[0] - c[0]).powi(2) + (v[1] - c[1]).powi(2) + (v[2] - c[2]).powi(2)).sqrt();
                if d < radius {
                    let side = if v[1] >= c[1] { 1.0 } else { -1.0 };
                    let f = 1.0 - d / radius;
                    out.push((i as u32, [0.0, side * 0.0045 * f, 0.0]));
                    break;
                }
            }
        }
        out
    }

    /// Translate all verts. Returns a new mesh.
    pub fn translated(&self, dx: f64, dy: f64, dz: f64) -> Self {
        let verts = self
            .verts
            .iter()
            .map(|v| [v[0] + dx, v[1] + dy, v[2] + dz])
            .collect();
        Mesh {
            verts,
            faces: self.faces.clone(),
            normals: self.normals.clone(),
        }
    }

    /// Tapered tube from `from` (radius r0) to `to` (radius r1):
    /// limbs, torsos, poles. `n` radial steps, open ends (callers
    /// overlap segments into joints). Plane-grid winding verified
    /// outward analytically; smooth normals via compute_normals.
    pub fn tube(from: [f64; 3], to: [f64; 3], r0: f64, r1: f64, n: u32) -> Self {
        Self::tube_profile(from, to, &|t| r0 + (r1 - r0) * t, 1, n)
    }

    /// Profiled tube: radius follows `r(t)` over `segs` length
    /// segments — muscle bellies, calves, waists. Same winding and
    /// normals as `tube`.
    pub fn tube_profile(
        from: [f64; 3],
        to: [f64; 3],
        r: &dyn Fn(f64) -> f64,
        segs: u32,
        n: u32,
    ) -> Self {
        let n = n.max(6);
        let segs = segs.max(1);
        let ax = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
        let len = (ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2])
            .sqrt()
            .max(1e-9);
        let a = [ax[0] / len, ax[1] / len, ax[2] / len];
        let up = if a[1].abs() < 0.9 {
            [0.0, 1.0, 0.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        let mut u = [
            a[1] * up[2] - a[2] * up[1],
            a[2] * up[0] - a[0] * up[2],
            a[0] * up[1] - a[1] * up[0],
        ];
        let ul = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt().max(1e-9);
        u = [u[0] / ul, u[1] / ul, u[2] / ul];
        let v = [
            a[1] * u[2] - a[2] * u[1],
            a[2] * u[0] - a[0] * u[2],
            a[0] * u[1] - a[1] * u[0],
        ];
        let mut verts = Vec::new();
        for j in 0..=segs {
            let t = j as f64 / segs as f64;
            let rr = r(t).max(1e-6);
            let c = [
                from[0] + ax[0] * t,
                from[1] + ax[1] * t,
                from[2] + ax[2] * t,
            ];
            for i in 0..=n {
                let th = i as f64 * TAU / n as f64;
                let (ct, st) = (th.cos(), th.sin());
                verts.push([
                    c[0] + rr * (u[0] * ct + v[0] * st),
                    c[1] + rr * (u[1] * ct + v[1] * st),
                    c[2] + rr * (u[2] * ct + v[2] * st),
                ]);
            }
        }
        let mut faces = Vec::new();
        let row = n + 1;
        for j in 0..segs {
            for i in 0..n {
                let a0 = j * row + i;
                faces.push([a0, a0 + 1, a0 + row]);
                faces.push([a0 + 1, a0 + row + 1, a0 + row]);
            }
        }
        let mut mesh = Mesh {
            verts,
            faces,
            normals: Vec::new(),
        };
        mesh.compute_normals();
        mesh
    }

    /// Full sphere shell centered origin: joints, hands, feet.
    pub fn sphere(nu: u32, nv: u32, r: f64) -> Self {
        let nu = nu.max(8);
        let nv = nv.max(4);
        let mut verts = Vec::new();
        for j in 0..=nv {
            let v = j as f64 * std::f64::consts::PI / nv as f64;
            for i in 0..=nu {
                let u = i as f64 * TAU / nu as f64;
                verts.push([r * v.sin() * u.sin(), r * v.cos(), r * v.sin() * u.cos()]);
            }
        }
        let mut faces = Vec::new();
        let row = nu + 1;
        for j in 0..nv {
            for i in 0..nu {
                let a = j * row + i;
                faces.push([a, a + row, a + 1]);
                faces.push([a + 1, a + row, a + row + 1]);
            }
        }
        let mut mesh = Mesh {
            verts,
            faces,
            normals: Vec::new(),
        };
        mesh.compute_normals();
        mesh
    }

    /// Flat grid in the XY plane, centered origin: (nu+1)×(nv+1)
    /// verts, outward +z normals. Callers displace then recompute.
    pub fn plane_grid(w: f64, h: f64, nu: u32, nv: u32) -> Self {
        let nu = nu.max(1);
        let nv = nv.max(1);
        let mut verts = Vec::new();
        for j in 0..=nv {
            for i in 0..=nu {
                verts.push([
                    -w / 2.0 + w * i as f64 / nu as f64,
                    -h / 2.0 + h * j as f64 / nv as f64,
                    0.0,
                ]);
            }
        }
        let mut faces = Vec::new();
        let row = nu + 1;
        for j in 0..nv {
            for i in 0..nu {
                let a = j * row + i;
                faces.push([a, a + 1, a + row]);
                faces.push([a + 1, a + row + 1, a + row]);
            }
        }
        let mut mesh = Mesh {
            verts,
            faces,
            normals: Vec::new(),
        };
        mesh.compute_normals();
        mesh
    }

    /// Upper hemisphere shell (pole at +y): bowls, helmets, hats.
    /// `r` radius, `squash` vertical scale.
    pub fn hemisphere(nu: u32, nv: u32, r: f64, squash: f64) -> Self {
        let nu = nu.max(8);
        let nv = nv.max(3);
        let mut verts = Vec::new();
        for j in 0..=nv {
            // Polar from 0 (pole) to ~100° (just past equator).
            let v = j as f64 * 1.75 / nv as f64;
            for i in 0..=nu {
                let u = i as f64 * TAU / nu as f64;
                verts.push([
                    r * v.sin() * u.sin(),
                    r * v.cos() * squash,
                    r * v.sin() * u.cos(),
                ]);
            }
        }
        let mut faces = Vec::new();
        let row = nu + 1;
        for j in 0..nv {
            for i in 0..nu {
                let a = j * row + i;
                faces.push([a, a + row, a + 1]);
                faces.push([a + 1, a + row, a + row + 1]);
            }
        }
        let mut mesh = Mesh {
            verts,
            faces,
            normals: Vec::new(),
        };
        mesh.compute_normals();
        mesh
    }

    /// Fan triangulation of a star-shaped polygon (centroid sees all
    /// verts): maple leaves and other flat emblems. Points in XY.
    pub fn fan_polygon(points: &[[f64; 2]]) -> Self {
        let n = points.len();
        let mut c = [0.0, 0.0];
        for p in points {
            c[0] += p[0];
            c[1] += p[1];
        }
        c[0] /= n.max(1) as f64;
        c[1] /= n.max(1) as f64;
        let mut verts = vec![[c[0], c[1], 0.0]];
        for p in points {
            verts.push([p[0], p[1], 0.0]);
        }
        let mut faces = Vec::new();
        for i in 0..n as u32 {
            let b = 1 + i;
            let c = 1 + (i + 1) % n as u32;
            faces.push([0, b, c]);
        }
        let mut mesh = Mesh {
            verts,
            faces,
            normals: Vec::new(),
        };
        mesh.compute_normals();
        mesh
    }

    /// Displace every vert by `f`, then recompute normals. Cloth
    /// waves, dents, and fitting live here.
    pub fn displace(&mut self, f: impl Fn([f64; 3]) -> [f64; 3]) {
        for v in self.verts.iter_mut() {
            *v = f(*v);
        }
        self.compute_normals();
    }

    /// Bounding sphere (centroid + max radius): the renderer's
    /// precheck before touching triangles.
    pub fn bounding_sphere(&self) -> ([f64; 3], f64) {
        let n = self.verts.len().max(1) as f64;
        let mut c = [0.0; 3];
        for v in &self.verts {
            c[0] += v[0];
            c[1] += v[1];
            c[2] += v[2];
        }
        c[0] /= n;
        c[1] /= n;
        c[2] /= n;
        let mut r: f64 = 0.0;
        for v in &self.verts {
            let d = ((v[0] - c[0]).powi(2) + (v[1] - c[1]).powi(2) + (v[2] - c[2]).powi(2)).sqrt();
            r = r.max(d);
        }
        (c, r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_topology_counts() {
        let m = Mesh::parametric_head(32, 20, &HeadShape::default());
        assert_eq!(m.verts.len(), 33 * 21);
        assert_eq!(m.faces.len(), 32 * 20 * 2);
        assert_eq!(m.normals.len(), m.verts.len());
        // Normals are unit length.
        for n in &m.normals {
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((l - 1.0).abs() < 1e-9, "{}", l);
        }
    }

    #[test]
    fn same_params_same_mesh() {
        let a = Mesh::parametric_head(24, 14, &HeadShape::default());
        let b = Mesh::parametric_head(24, 14, &HeadShape::default());
        assert_eq!(a.verts, b.verts);
        assert_eq!(a.faces, b.faces);
    }

    #[test]
    fn jaw_morph_widens_the_chin_band() {
        let narrow = Mesh::parametric_head(
            32,
            20,
            &HeadShape {
                jaw_width: 0.1,
                ..HeadShape::default()
            },
        );
        let wide = Mesh::parametric_head(
            32,
            20,
            &HeadShape {
                jaw_width: 0.9,
                ..HeadShape::default()
            },
        );
        // Chin band: verts below y=-0.25.
        let width = |m: &Mesh| -> f64 {
            let xs: Vec<f64> = m
                .verts
                .iter()
                .filter(|v| v[1] < -0.25)
                .map(|v| v[0].abs())
                .collect();
            xs.into_iter().fold(0.0f64, f64::max) * 2.0
        };
        let (wn, ww) = (width(&narrow), width(&wide));
        assert!(ww > wn + 0.03, "narrow={} wide={}", wn, ww);
    }

    #[test]
    fn smile_lifts_mouth_corners() {
        let m = Mesh::parametric_head(32, 20, &HeadShape::default());
        let deltas = m.smile_deltas();
        assert!(!deltas.is_empty(), "rule must find mouth corners");
        let smiled = m.blendshape(&deltas, 1.0);
        for (i, _) in &deltas {
            assert!(
                smiled.verts[*i as usize][1] > m.verts[*i as usize][1],
                "corner must rise"
            );
        }
        // Weight zero is the identity.
        let same = m.blendshape(&deltas, 0.0);
        assert_eq!(same.verts, m.verts);
    }

    #[test]
    fn grids_hemispheres_and_fans_build() {
        let g = Mesh::plane_grid(2.0, 1.0, 8, 4);
        assert_eq!(g.verts.len(), 9 * 5);
        assert_eq!(g.faces.len(), 8 * 4 * 2);
        // Flat grid normals face +z.
        assert!(g.normals.iter().all(|n| n[2] > 0.99));
        let h = Mesh::hemisphere(16, 6, 1.0, 0.8);
        assert!(h.verts.iter().all(|v| v[1] > -0.2));
        // Hemisphere normals point outward (away from center axis).
        for (v, n) in h.verts.iter().zip(h.normals.iter()) {
            let radial = (v[0] * n[0] + v[2] * n[2]).signum();
            if v[1].abs() + v[0].abs() + v[2].abs() > 0.3 {
                assert!(radial > 0.0, "inward normal at {:?} {:?}", v, n);
            }
        }
        // Fan of a diamond: 4 triangles, area sanity via bbox.
        let d = Mesh::fan_polygon(&[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]);
        assert_eq!(d.faces.len(), 4);
        // Displacement moves verts and refreshes normals.
        let mut w = Mesh::plane_grid(2.0, 2.0, 4, 4);
        w.displace(|v| [v[0], v[1], v[2] + (v[0] * 3.0).sin() * 0.2]);
        assert!(w.verts.iter().any(|v| v[2].abs() > 0.01));
        assert_eq!(w.normals.len(), w.verts.len());
    }

    #[test]
    fn bounding_sphere_contains_all() {
        let m = Mesh::parametric_head(24, 14, &HeadShape::default());
        let (c, r) = m.bounding_sphere();
        for v in &m.verts {
            let d = ((v[0] - c[0]).powi(2) + (v[1] - c[1]).powi(2) + (v[2] - c[2]).powi(2)).sqrt();
            assert!(d <= r + 1e-9);
        }
    }
}
