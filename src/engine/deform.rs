//! Deformation operators — Phase 6 of the image-creation system.
//!
//! Cloth waves, and other soft bends, as parametric mesh operators:
//! amplitude, wavelength, phase, and a pole-edge taper. The operator
//! is exact math on verts with recomputed normals; the PARAMETERS
//! are procedural defaults, stated as such — deriving them from
//! cloth-photo measurements is staged behind cloth example
//! collection (no such measurements exist yet, and the receipt for
//! any deformed cloth says where its numbers came from).

use super::mesh::Mesh;

/// Cloth wave parameters. All lengths in meters.
#[derive(Debug, Clone, Copy)]
pub struct WaveParams {
    /// Peak displacement in meters.
    pub amplitude: f64,
    /// Spatial period in meters.
    pub wavelength: f64,
    /// Temporal phase in radians (still frames differ honestly).
    pub phase: f64,
    /// Pole-edge x: displacement grows away from it.
    pub pole_x: f64,
    /// Cloth width: taper reaches full amplitude here.
    pub width: f64,
}

impl WaveParams {
    /// Gentle default for demos. Procedural default — NOT measured.
    /// Derivation from waving-cloth examples is staged.
    pub fn gentle_default() -> Self {
        WaveParams {
            amplitude: 0.03,
            wavelength: 0.40,
            phase: 0.0,
            pole_x: 0.0,
            width: 0.60,
        }
    }
}

/// Apply the wave to a cloth mesh in the XY plane: z-displacement
/// grows from the pole edge. Normals recomputed via `displace`.
/// Pure: same mesh and params twice, byte-identical verts.
pub fn apply_wave(mesh: &mut Mesh, p: WaveParams) {
    let lam = p.wavelength.max(1e-6);
    let w = p.width.max(1e-6);
    mesh.displace(|v| {
        let t = ((v[0] - p.pole_x) / w).clamp(0.0, 1.0);
        let dz =
            p.amplitude * ((std::f64::consts::TAU * (v[0] - p.pole_x) / lam) - p.phase).sin() * t;
        [v[0], v[1], v[2] + dz]
    });
}

/// Total mesh area (triangle sum). Approximate — for
/// change-fraction assertions, not for physics.
pub fn mesh_area(mesh: &Mesh) -> f64 {
    let mut area = 0.0;
    for f in &mesh.faces {
        let a = mesh.verts[f[0] as usize];
        let b = mesh.verts[f[1] as usize];
        let c = mesh.verts[f[2] as usize];
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        area += 0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    }
    area
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_amplitude_is_identity() {
        let g = Mesh::plane_grid(1.0, 1.0, 8, 8);
        let mut w = g.clone();
        apply_wave(
            &mut w,
            WaveParams {
                amplitude: 0.0,
                ..WaveParams::gentle_default()
            },
        );
        assert_eq!(w.verts, g.verts);
    }

    #[test]
    fn peak_matches_amplitude() {
        let mut g = Mesh::plane_grid(1.2, 0.6, 24, 6);
        let p = WaveParams {
            amplitude: 0.05,
            wavelength: 1.2,
            phase: 0.0,
            pole_x: -0.6,
            width: 1.2,
        };
        apply_wave(&mut g, p);
        let peak = g.verts.iter().map(|v| v[2].abs()).fold(0.0, f64::max);
        // Tapered peak sits just under amplitude; nonzero regardless.
        assert!(peak > 0.5 * p.amplitude && peak <= p.amplitude + 1e-9);
    }

    #[test]
    fn normals_stay_unit_and_area_near() {
        let mut g = Mesh::plane_grid(1.2, 0.6, 24, 6);
        let a0 = mesh_area(&g);
        apply_wave(&mut g, WaveParams::gentle_default());
        for n in &g.normals {
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((l - 1.0).abs() < 1e-9);
        }
        let r = mesh_area(&g) / a0;
        assert!((r - 1.0).abs() < 0.08, "area ratio {}", r);
    }
}
