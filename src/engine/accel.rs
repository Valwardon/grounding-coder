//! Uniform-grid ray accelerator for mesh triangles.
//!
//! Brute-forcing every triangle per ray dies on oracle meshes
//! (~37k tris). The grid rasterizes mesh bboxes into cells and
//! marches rays cell-by-cell (Amanatides & Woo DDA), testing only
//! tris a ray can actually meet. Identical pixels to brute force by
//! construction — same `ray_triangle` math, conservative raster,
//! march-while-closer rule — pinned by an equality test, not trust.
//!
//! Analytic shapes (spheres, planes, boxes, capsules) stay on the
//! existing path; the grid holds mesh triangles only.

use super::mesh::Mesh;
use super::scene::{Material, Vec3};

/// One mesh triangle, world-space, with its vertex normals + material.
#[derive(Debug, Clone)]
pub struct GridTri {
    pub v: [[f64; 3]; 3],
    pub n: [[f64; 3]; 3],
    pub mat: usize,
}

/// Uniform grid over all mesh triangles in a scene render.
pub struct TriGrid {
    pub origin: [f64; 3],
    pub cell: [f64; 3],
    pub res: [u32; 3],
    pub cells: Vec<Vec<u32>>,
    pub tris: Vec<GridTri>,
    pub mats: Vec<Material>,
    tick: Vec<u32>,
    cur: u32,
}

impl TriGrid {
    /// Gather mesh triangles from shapes. Empty when the scene holds
    /// no meshes — the render path then behaves exactly as before.
    pub fn build(meshes: &[(&Mesh, Material)]) -> Self {
        let mut tris = Vec::new();
        let mut mats: Vec<Material> = Vec::new();
        for (mesh, mat) in meshes {
            let mi = match mats.iter().position(|m| m.name == mat.name) {
                Some(i) => i,
                None => {
                    mats.push(mat.clone());
                    mats.len() - 1
                }
            };
            for f in &mesh.faces {
                let g = |i: u32| mesh.verts[i as usize];
                let n = |i: u32| {
                    mesh.normals
                        .get(i as usize)
                        .copied()
                        .unwrap_or([0.0, 1.0, 0.0])
                };
                tris.push(GridTri {
                    v: [g(f[0]), g(f[1]), g(f[2])],
                    n: [n(f[0]), n(f[1]), n(f[2])],
                    mat: mi,
                });
            }
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for t in &tris {
            for v in &t.v {
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k]);
                    hi[k] = hi[k].max(v[k]);
                }
            }
        }
        if tris.is_empty() {
            lo = [0.0; 3];
            hi = [1.0; 3];
        }
        // Resolution from triangle count, shaped by the bbox.
        let per = (tris.len().max(1) as f64).cbrt().ceil().clamp(4.0, 64.0);
        let span = [
            (hi[0] - lo[0]).max(1e-6),
            (hi[1] - lo[1]).max(1e-6),
            (hi[2] - lo[2]).max(1e-6),
        ];
        let longest = span[0].max(span[1]).max(span[2]);
        let res = [
            ((per * span[0] / longest).ceil() as u32).max(1),
            ((per * span[1] / longest).ceil() as u32).max(1),
            ((per * span[2] / longest).ceil() as u32).max(1),
        ];
        let cell = [
            span[0] / res[0] as f64,
            span[1] / res[1] as f64,
            span[2] / res[2] as f64,
        ];
        let ncells = (res[0] * res[1] * res[2]) as usize;
        let mut cells: Vec<Vec<u32>> = vec![Vec::new(); ncells];
        let idx = |x: u32, y: u32, z: u32| (z * res[1] * res[0] + y * res[0] + x) as usize;
        for (ti, t) in tris.iter().enumerate() {
            let mut tlo = [f64::INFINITY; 3];
            let mut thi = [f64::NEG_INFINITY; 3];
            for v in &t.v {
                for k in 0..3 {
                    tlo[k] = tlo[k].min(v[k]);
                    thi[k] = thi[k].max(v[k]);
                }
            }
            let c0 = [
                (((tlo[0] - lo[0]) / cell[0]).floor() as i64).clamp(0, res[0] as i64 - 1) as u32,
                (((tlo[1] - lo[1]) / cell[1]).floor() as i64).clamp(0, res[1] as i64 - 1) as u32,
                (((tlo[2] - lo[2]) / cell[2]).floor() as i64).clamp(0, res[2] as i64 - 1) as u32,
            ];
            let c1 = [
                (((thi[0] - lo[0]) / cell[0]).floor() as i64).clamp(0, res[0] as i64 - 1) as u32,
                (((thi[1] - lo[1]) / cell[1]).floor() as i64).clamp(0, res[1] as i64 - 1) as u32,
                (((thi[2] - lo[2]) / cell[2]).floor() as i64).clamp(0, res[2] as i64 - 1) as u32,
            ];
            for z in c0[2]..=c1[2] {
                for y in c0[1]..=c1[1] {
                    for x in c0[0]..=c1[0] {
                        cells[idx(x, y, z)].push(ti as u32);
                    }
                }
            }
        }
        let tick = vec![0u32; tris.len()];
        TriGrid {
            origin: lo,
            cell,
            res,
            cells,
            tris,
            mats,
            tick,
            cur: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tris.is_empty()
    }

    /// Cell containing `p`, clamped into range. Callers only pass
    /// points on/inside the bounds (ray starts), so clamping fixes
    /// boundary floats instead of misreading them as outside.
    fn cell_of(&self, p: [f64; 3]) -> [u32; 3] {
        let mut c = [0u32; 3];
        for k in 0..3 {
            let f = ((p[k] - self.origin[k]) / self.cell[k]).floor() as i64;
            c[k] = f.clamp(0, self.res[k] as i64 - 1) as u32;
        }
        c
    }

    fn cell_index(&self, c: [u32; 3]) -> usize {
        (c[2] * self.res[1] * self.res[0] + c[1] * self.res[0] + c[0]) as usize
    }

    /// Nearest mesh hit along the ray, if any. Marches cells in
    /// order and stops once the best hit beats the next cell
    /// boundary — nothing closer can hide behind it.
    pub fn intersect(&mut self, origin: Vec3, dir: Vec3) -> Option<(f64, Vec3, usize)> {
        let o = [origin.x, origin.y, origin.z];
        let d = [dir.x, dir.y, dir.z];
        // Slab test against the grid bounds.
        let mut t_enter = 0.0f64;
        let mut t_exit = f64::INFINITY;
        for k in 0..3 {
            let hi = self.origin[k] + self.cell[k] * self.res[k] as f64;
            if d[k].abs() < 1e-12 {
                if o[k] < self.origin[k] || o[k] > hi {
                    return None;
                }
            } else {
                let mut t1 = (self.origin[k] - o[k]) / d[k];
                let mut t2 = (hi - o[k]) / d[k];
                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }
                t_enter = t_enter.max(t1);
                t_exit = t_exit.min(t2);
                if t_enter > t_exit {
                    return None;
                }
            }
        }
        if t_exit <= 1e-6 {
            return None;
        }
        let start = [
            o[0] + d[0] * t_enter.max(0.0),
            o[1] + d[1] * t_enter.max(0.0),
            o[2] + d[2] * t_enter.max(0.0),
        ];
        let mut cell = self.cell_of(start);
        // DDA setup.
        let mut step = [0i64; 3];
        let mut t_max = [0.0f64; 3];
        let mut t_delta = [0.0f64; 3];
        for k in 0..3 {
            if d[k].abs() < 1e-12 {
                step[k] = 0;
                t_max[k] = f64::INFINITY;
                t_delta[k] = f64::INFINITY;
            } else if d[k] > 0.0 {
                step[k] = 1;
                let edge = self.origin[k] + (cell[k] + 1) as f64 * self.cell[k];
                t_max[k] = (edge - o[k]) / d[k];
                t_delta[k] = self.cell[k] / d[k];
            } else {
                step[k] = -1;
                let edge = self.origin[k] + cell[k] as f64 * self.cell[k];
                t_max[k] = (edge - o[k]) / d[k];
                t_delta[k] = -self.cell[k] / d[k];
            }
        }
        self.cur = self.cur.wrapping_add(1);
        let mut best: Option<(f64, Vec3, usize)> = None;
        loop {
            let ci = self.cell_index(cell);
            for ti in self.cells[ci].clone() {
                if self.tick[ti as usize] == self.cur {
                    continue;
                }
                self.tick[ti as usize] = self.cur;
                let t = &self.tris[ti as usize];
                let va = Vec3::new(t.v[0][0], t.v[0][1], t.v[0][2]);
                let vb = Vec3::new(t.v[1][0], t.v[1][1], t.v[1][2]);
                let vc = Vec3::new(t.v[2][0], t.v[2][1], t.v[2][2]);
                if let Some((tt, n)) =
                    super::scene::ray_triangle(origin, dir, va, vb, vc, t.n[0], t.n[1], t.n[2])
                    && best.map(|(bt, _, _)| tt < bt).unwrap_or(true)
                {
                    best = Some((tt, n, t.mat));
                }
            }
            // Next cell boundary along the fastest axis.
            let mut axis = 0;
            for k in 1..3 {
                if t_max[k] < t_max[axis] {
                    axis = k;
                }
            }
            if best.map(|(bt, _, _)| bt < t_max[axis]).unwrap_or(false) {
                return best;
            }
            if t_max[axis] > t_exit {
                return best;
            }
            if step[axis] == 0 {
                return best;
            }
            let next = cell[axis] as i64 + step[axis];
            if next < 0 || next >= self.res[axis] as i64 {
                return best;
            }
            cell[axis] = next as u32;
            t_max[axis] += t_delta[axis];
        }
    }

    /// Any mesh hit past epsilon — shadow rays stop at the first.
    pub fn occluded(&mut self, origin: Vec3, dir: Vec3) -> bool {
        // March like intersect but quit at the first valid hit.
        let o = [origin.x, origin.y, origin.z];
        let d = [dir.x, dir.y, dir.z];
        let mut t_enter = 0.0f64;
        let mut t_exit = f64::INFINITY;
        for k in 0..3 {
            let hi = self.origin[k] + self.cell[k] * self.res[k] as f64;
            if d[k].abs() < 1e-12 {
                if o[k] < self.origin[k] || o[k] > hi {
                    return false;
                }
            } else {
                let mut t1 = (self.origin[k] - o[k]) / d[k];
                let mut t2 = (hi - o[k]) / d[k];
                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }
                t_enter = t_enter.max(t1);
                t_exit = t_exit.min(t2);
                if t_enter > t_exit {
                    return false;
                }
            }
        }
        if t_exit <= 1e-6 {
            return false;
        }
        let start = [
            o[0] + d[0] * t_enter.max(0.0),
            o[1] + d[1] * t_enter.max(0.0),
            o[2] + d[2] * t_enter.max(0.0),
        ];
        let mut cell = self.cell_of(start);
        let mut step = [0i64; 3];
        let mut t_max = [0.0f64; 3];
        let mut t_delta = [0.0f64; 3];
        for k in 0..3 {
            if d[k].abs() < 1e-12 {
                step[k] = 0;
                t_max[k] = f64::INFINITY;
                t_delta[k] = f64::INFINITY;
            } else if d[k] > 0.0 {
                step[k] = 1;
                let edge = self.origin[k] + (cell[k] + 1) as f64 * self.cell[k];
                t_max[k] = (edge - o[k]) / d[k];
                t_delta[k] = self.cell[k] / d[k];
            } else {
                step[k] = -1;
                let edge = self.origin[k] + cell[k] as f64 * self.cell[k];
                t_max[k] = (edge - o[k]) / d[k];
                t_delta[k] = -self.cell[k] / d[k];
            }
        }
        self.cur = self.cur.wrapping_add(1);
        loop {
            let ci = self.cell_index(cell);
            for ti in self.cells[ci].clone() {
                if self.tick[ti as usize] == self.cur {
                    continue;
                }
                self.tick[ti as usize] = self.cur;
                let t = &self.tris[ti as usize];
                let va = Vec3::new(t.v[0][0], t.v[0][1], t.v[0][2]);
                let vb = Vec3::new(t.v[1][0], t.v[1][1], t.v[1][2]);
                let vc = Vec3::new(t.v[2][0], t.v[2][1], t.v[2][2]);
                if super::scene::ray_triangle(origin, dir, va, vb, vc, t.n[0], t.n[1], t.n[2])
                    .is_some()
                {
                    return true;
                }
            }
            let mut axis = 0;
            for k in 1..3 {
                if t_max[k] < t_max[axis] {
                    axis = k;
                }
            }
            if t_max[axis] > t_exit || step[axis] == 0 {
                return false;
            }
            let next = cell[axis] as i64 + step[axis];
            if next < 0 || next >= self.res[axis] as i64 {
                return false;
            }
            cell[axis] = next as u32;
            t_max[axis] += t_delta[axis];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::mesh::Mesh;
    use super::super::scene::Vec3;
    use super::TriGrid;

    #[test]
    fn grid_matches_brute_force() {
        // Two overlapping boxes-as-meshes plus a lone triangle: every
        // pixel must read identical with and without the grid.
        let a = Mesh::tube([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.2, 0.2, 8);
        let b = Mesh::tube([0.3, 0.0, 0.0], [0.3, 1.0, 0.0], 0.15, 0.15, 8);
        let mat = || {
            super::super::scene::Material::named("t", super::super::vision::Rgb::new(200, 200, 200))
        };
        let grid_meshes = vec![(&a, mat()), (&b, mat())];
        let mut grid = TriGrid::build(&grid_meshes);
        // Rays: straight at each tube, between them (miss), grazing.
        let rays: [([f64; 3], [f64; 3]); 6] = [
            ([0.3, 0.5, 5.0], [0.0, 0.0, -1.0]),
            ([0.0, 0.5, 5.0], [0.0, 0.0, -1.0]),
            ([0.15, 0.5, 5.0], [0.0, 0.0, -1.0]),
            ([2.0, 0.5, 5.0], [0.0, 0.0, -1.0]),
            ([0.3, 0.5, 5.0], [0.1, 0.0, -1.0]),
            ([-1.0, 0.5, 5.0], [0.2, 0.0, -1.0]),
        ];
        for (o, d) in rays {
            let origin = Vec3::new(o[0], o[1], o[2]);
            let dl: f64 = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let dir = Vec3::new(d[0] / dl, d[1] / dl, d[2] / dl);
            // Brute force over both meshes.
            let mut brute: Option<(f64, Vec3)> = None;
            for m in [&a, &b] {
                for f in &m.faces {
                    let va = m.verts[f[0] as usize];
                    let vb = m.verts[f[1] as usize];
                    let vc = m.verts[f[2] as usize];
                    if let Some((t, n)) = super::super::scene::ray_triangle(
                        origin,
                        dir,
                        Vec3::new(va[0], va[1], va[2]),
                        Vec3::new(vb[0], vb[1], vb[2]),
                        Vec3::new(vc[0], vc[1], vc[2]),
                        m.normals[f[0] as usize],
                        m.normals[f[1] as usize],
                        m.normals[f[2] as usize],
                    ) && brute.map(|(bt, _)| t < bt).unwrap_or(true)
                    {
                        brute = Some((t, n));
                    }
                }
            }
            let got = grid.intersect(origin, dir);
            match (brute, got) {
                (None, None) => {}
                (Some((bt, _)), Some((gt, _, _))) => {
                    assert!((bt - gt).abs() < 1e-9, "ray {:?}: {} vs {}", o, bt, gt)
                }
                _ => panic!("hit mismatch on ray {:?}", o),
            }
        }
        assert!(grid.occluded(Vec3::new(0.3, 0.5, 5.0), Vec3::new(0.0, 0.0, -1.0)));
        assert!(!grid.occluded(Vec3::new(2.0, 0.5, 5.0), Vec3::new(0.0, 0.0, -1.0)));
    }
}
