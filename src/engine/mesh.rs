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
    fn bounding_sphere_contains_all() {
        let m = Mesh::parametric_head(24, 14, &HeadShape::default());
        let (c, r) = m.bounding_sphere();
        for v in &m.verts {
            let d = ((v[0] - c[0]).powi(2) + (v[1] - c[1]).powi(2) + (v[2] - c[2]).powi(2)).sqrt();
            assert!(d <= r + 1e-9);
        }
    }
}
