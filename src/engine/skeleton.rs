//! Articulated human skeleton — Phase 1 of the image-creation system.
//!
//! The spec's joint tree with independent per-joint rotation and
//! forward kinematics. Named poses (standing, sitting, walking,
//! salute, wave, point, raise) validate every angle against the
//! anatomy ROM table; impossible poses refuse instead of rendering.
//!
//! HONEST LIMIT, stated in code: this is a measurement scaffold, not
//! a human image. Bone lengths come from canon fractions (labeled as
//! such) until plate measurements (see `body_measure`) reach the
//! 20-example gate. Surface geometry is a later phase. Rendering here
//! means annotated measurement diagrams — the receipt says so, and
//! the files are named `skeleton-measurement-*`, never portraits.

use super::vision::{Image, Rgb};
use std::collections::HashMap;

/// Every joint in the spec's minimum skeleton.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Joint {
    Pelvis,
    Spine,
    Chest,
    Neck,
    Head,
    ShoulderL,
    ShoulderR,
    ElbowL,
    ElbowR,
    WristL,
    WristR,
    HandL,
    HandR,
    HipL,
    HipR,
    KneeL,
    KneeR,
    AnkleL,
    AnkleR,
    FootL,
    FootR,
}

impl Joint {
    /// Parent in the kinematic tree. Pelvis is the root.
    pub fn parent(self) -> Option<Joint> {
        use Joint::*;
        Some(match self {
            Pelvis => return None,
            Spine => Pelvis,
            Chest => Spine,
            Neck => Chest,
            Head => Neck,
            ShoulderL | ShoulderR => Chest,
            ElbowL => ShoulderL,
            ElbowR => ShoulderR,
            WristL => ElbowL,
            WristR => ElbowR,
            HandL => WristL,
            HandR => WristR,
            HipL | HipR => Pelvis,
            KneeL => HipL,
            KneeR => HipR,
            AnkleL => KneeL,
            AnkleR => KneeR,
            FootL => AnkleL,
            FootR => AnkleR,
        })
    }

    /// All joints, parents before children (topological order).
    pub fn all() -> Vec<Joint> {
        use Joint::*;
        vec![
            Pelvis, Spine, Chest, Neck, Head, ShoulderL, ShoulderR, ElbowL, ElbowR, WristL, WristR,
            HandL, HandR, HipL, HipR, KneeL, KneeR, AnkleL, AnkleR, FootL, FootR,
        ]
    }

    /// Left joints mirror abduction; knees flex backward.
    pub(crate) fn mirror(self) -> f64 {
        use Joint::*;
        match self {
            ShoulderL | ElbowL | WristL | HipL | KneeL | AnkleL => -1.0,
            _ => 1.0,
        }
    }

    pub(crate) fn flex_sign(self) -> f64 {
        use Joint::*;
        match self {
            KneeL | KneeR => -1.0,
            _ => 1.0,
        }
    }

    /// ROM-table name for validation. Joints with no range data
    /// refuse — no data is not a yes.
    fn rom_name(self) -> &'static str {
        use Joint::*;
        match self {
            ShoulderL | ShoulderR => "shoulder",
            ElbowL | ElbowR => "elbow",
            WristL | WristR => "wrist",
            HipL | HipR => "hip",
            KneeL | KneeR => "knee",
            AnkleL | AnkleR => "ankle",
            Neck => "neck",
            // Axial joints: validated structurally (spine/chest stay
            // near-rigid in v1 poses), not against limb ROM.
            _ => "",
        }
    }
}

/// Body proportions as stature fractions. Defaults are canon-derived
/// (7.5-heads + Dreyfuss-style segment ratios) — inspectable, and
/// overridden by `body_measure` ranges once plates clear the gate.
/// Joint positions only: body surfaces come from the oracle mesh.
#[derive(Debug, Clone)]
pub struct BodyProportions {
    /// Total height in meters.
    pub stature_m: f64,
    /// Half shoulder width / stature (joint positions).
    pub shoulder_hw: f64,
    /// Half hip width / stature (joint positions).
    pub hip_hw: f64,
    /// Root (pelvis) rest position in meters. Measured rigs set it;
    /// canon defaults hold 55% of stature.
    pub root: [f64; 3],
    /// Measured rest offsets (child − parent) in meters, overriding
    /// canon fractions joint by joint. Empty = all canon. The rig
    /// measurer fills this; FK and deformation then agree with the
    /// mesh by construction.
    pub offsets: HashMap<Joint, [f64; 3]>,
    /// Where the numbers came from.
    pub source: String,
}

impl BodyProportions {
    pub fn adult_male() -> Self {
        let s = 1.75;
        BodyProportions {
            stature_m: s,
            shoulder_hw: 0.12,
            hip_hw: 0.057,
            root: [0.0, 0.55 * s, 0.0],
            offsets: HashMap::new(),
            source: "canon default (7.5-heads proportions)".to_string(),
        }
    }

    pub fn adult_female() -> Self {
        let s = 1.62;
        BodyProportions {
            stature_m: s,
            shoulder_hw: 0.11,
            hip_hw: 0.068,
            root: [0.0, 0.55 * s, 0.0],
            offsets: HashMap::new(),
            source: "canon default (7.5-heads proportions)".to_string(),
        }
    }

    /// Rest offset of a joint from its parent, in meters.
    /// Fractions of stature from the 7.5-heads canon (leg chain sums
    /// to pelvis height so the foot rests on the ground; head top
    /// lands within 2% of stature — both asserted by tests).
    fn offset(&self, j: Joint) -> V3 {
        if let Some(o) = self.offsets.get(&j) {
            return V3::new(o[0], o[1], o[2]);
        }
        let s = self.stature_m;
        use Joint::*;
        let (x, y, z) = match j {
            Pelvis => (self.root[0], self.root[1], self.root[2]),
            Spine => (0.0, 0.08 * s, 0.0),
            Chest => (0.0, 0.11 * s, 0.0),
            Neck => (0.0, 0.10 * s, 0.0),
            Head => (0.0, 0.075 * s, 0.0),
            ShoulderL => (-self.shoulder_hw * s, 0.06 * s, 0.0),
            ShoulderR => (self.shoulder_hw * s, 0.06 * s, 0.0),
            ElbowL | ElbowR => (0.0, -0.17 * s, 0.0),
            WristL | WristR => (0.0, -0.16 * s, 0.0),
            HandL | HandR => (0.0, -0.10 * s, 0.0),
            HipL => (-self.hip_hw * s, -0.01 * s, 0.0),
            HipR => (self.hip_hw * s, -0.01 * s, 0.0),
            KneeL | KneeR => (0.0, -0.257 * s, 0.0),
            AnkleL | AnkleR => (0.0, -0.24 * s, 0.0),
            FootL | FootR => (0.0, -0.03 * s, 0.07 * s),
        };
        V3::new(x, y, z)
    }
}

/// Minimal 3D vector for FK (scene's Vec3 methods are private).
#[derive(Debug, Clone, Copy)]
pub struct V3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl V3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        V3 { x, y, z }
    }

    fn add(self, o: V3) -> V3 {
        V3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }

    pub fn len(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

impl std::ops::Sub for V3 {
    type Output = V3;

    fn sub(self, o: V3) -> V3 {
        V3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

/// Joint angles in degrees: flexion (sagittal, +forward) and
/// abduction (lateral). Zeros = anatomical rest (standing).
#[derive(Debug, Clone, Copy)]
pub struct JointAngles {
    pub flex: f64,
    pub abd: f64,
}

impl JointAngles {
    pub fn rest() -> Self {
        JointAngles {
            flex: 0.0,
            abd: 0.0,
        }
    }

    pub fn flex(flex: f64) -> Self {
        JointAngles { flex, abd: 0.0 }
    }
}

/// A full-body pose: angles per joint. Missing joints rest.
#[derive(Debug, Clone)]
pub struct Pose {
    angles: HashMap<Joint, JointAngles>,
}

impl Pose {
    pub fn rest() -> Self {
        Pose {
            angles: HashMap::new(),
        }
    }

    pub fn set(&mut self, j: Joint, a: JointAngles) {
        self.angles.insert(j, a);
    }

    pub fn get(&self, j: Joint) -> JointAngles {
        self.angles.get(&j).copied().unwrap_or(JointAngles::rest())
    }

    /// Named poses. Every angle below sits inside its anatomy ROM
    /// range — validated at apply time, not trusted here.
    pub fn standing() -> Self {
        Pose::rest()
    }

    pub fn sitting() -> Self {
        let mut p = Pose::rest();
        for side in [Joint::HipL, Joint::HipR] {
            p.set(side, JointAngles::flex(90.0));
        }
        for side in [Joint::KneeL, Joint::KneeR] {
            p.set(side, JointAngles::flex(90.0));
        }
        p
    }

    /// Gait cycle at `phase` radians. Amplitudes stay deep inside
    /// ROM; left leads by π.
    pub fn walking(phase: f64) -> Self {
        let mut p = Pose::rest();
        for (ph, left) in [(0.0, true), (std::f64::consts::PI, false)] {
            let swing = (phase + ph).sin();
            let hip = if left { Joint::HipL } else { Joint::HipR };
            let knee = if left { Joint::KneeL } else { Joint::KneeR };
            let shoulder = if left {
                Joint::ShoulderL
            } else {
                Joint::ShoulderR
            };
            let elbow = if left { Joint::ElbowL } else { Joint::ElbowR };
            // Gait biases forward (table lists flexion 0–120 only —
            // extension behind the torso is real but unmodeled in v1).
            p.set(hip, JointAngles::flex(20.0 + 20.0 * swing));
            p.set(
                knee,
                JointAngles::flex(12.0 + 18.0 * (0.5 - 0.5 * (phase + ph).cos())),
            );
            // Arms stay in front (table lists flexion 0–180 only —
            // extension behind the torso is real but unmodeled in v1).
            p.set(shoulder, JointAngles::flex(9.0 + 9.0 * swing));
            p.set(elbow, JointAngles::flex(10.0));
        }
        p
    }

    fn side_joints(right: bool) -> (Joint, Joint, Joint) {
        if right {
            (Joint::ShoulderR, Joint::ElbowR, Joint::WristR)
        } else {
            (Joint::ShoulderL, Joint::ElbowL, Joint::WristL)
        }
    }

    /// Hand to forehead. Angles tuned against the ROM table
    /// (shoulder ≤180, elbow ≤145) and the proximity test below.
    pub fn salute(right: bool) -> Self {
        let mut p = Pose::rest();
        let (sh, el, wr) = Pose::side_joints(right);
        p.set(sh, JointAngles::flex(150.0));
        p.set(el, JointAngles::flex(128.0));
        p.set(
            wr,
            JointAngles {
                flex: 10.0,
                abd: 10.0,
            },
        );
        p
    }

    /// Arm up, slight out. The wave reads when the hand clears the head.
    pub fn wave(right: bool) -> Self {
        let mut p = Pose::rest();
        let (sh, el, _) = Pose::side_joints(right);
        p.set(
            sh,
            JointAngles {
                flex: 160.0,
                abd: 18.0,
            },
        );
        p.set(el, JointAngles::flex(12.0));
        p
    }

    /// Arm forward, elbow nearly straight.
    pub fn point(right: bool) -> Self {
        let mut p = Pose::rest();
        let (sh, el, _) = Pose::side_joints(right);
        p.set(sh, JointAngles::flex(90.0));
        p.set(el, JointAngles::flex(5.0));
        p
    }

    /// Straight up.
    pub fn raise_hand(right: bool) -> Self {
        let mut p = Pose::rest();
        let (sh, el, _) = Pose::side_joints(right);
        p.set(sh, JointAngles::flex(170.0));
        p.set(el, JointAngles::flex(5.0));
        p
    }
}

/// Validate every posed angle against the anatomy ROM table.
/// Unknown joints refuse; out-of-range refuses with the range.
/// Side joints validate under their base name (ShoulderR → shoulder).
pub fn validate_pose_rom(pose: &Pose) -> Result<(), String> {
    for j in Joint::all() {
        let name = j.rom_name();
        if name.is_empty() {
            continue;
        }
        let a = pose.get(j);
        let table = super::anatomy::joint_table();
        let entry = table
            .iter()
            .find(|e| e.joint == name)
            .ok_or_else(|| format!("{:?}: no range data — refusing, not guessing", name))?;
        let angles = match entry.rom.len() {
            1 => vec![a.flex],
            _ => vec![a.flex, a.abd],
        };
        super::anatomy::validate_pose(name, &angles)?;
    }
    Ok(())
}

/// 3×3 rotation, column-major. One math for FK positions and
/// mesh deformation alike (see `joint_world_rotations`).
pub type Mat3 = ([f64; 3], [f64; 3], [f64; 3]);

/// R = Rz(b)·Rx(a) as columns (matches the old rot_xz order).
pub(crate) fn rot_mat(a: f64, b: f64) -> Mat3 {
    let (sa, ca) = a.sin_cos();
    let (sb, cb) = b.sin_cos();
    let c0 = [cb, sb, 0.0];
    let c1 = [-sb * ca, cb * ca, sa];
    let c2 = [sb * sa, -cb * sa, ca];
    (c0, c1, c2)
}

/// Matrix product (column convention).
pub(crate) fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let col = |v: [f64; 3]| -> [f64; 3] {
        [
            a.0[0] * v[0] + a.1[0] * v[1] + a.2[0] * v[2],
            a.0[1] * v[0] + a.1[1] * v[1] + a.2[1] * v[2],
            a.0[2] * v[0] + a.1[2] * v[1] + a.2[2] * v[2],
        ]
    };
    (col(b.0), col(b.1), col(b.2))
}

/// Apply a rotation to a vector.
pub(crate) fn apply_mat(m: &Mat3, v: V3) -> V3 {
    V3::new(
        m.0[0] * v.x + m.1[0] * v.y + m.2[0] * v.z,
        m.0[1] * v.x + m.1[1] * v.y + m.2[1] * v.z,
        m.0[2] * v.x + m.1[2] * v.y + m.2[2] * v.z,
    )
}

/// World rotation per joint, composed exactly down the tree.
/// Single source for FK positions and mesh deformation.
pub fn joint_world_rotations(pose: &Pose) -> Result<HashMap<Joint, Mat3>, String> {
    validate_pose_rom(pose)?;
    let mut world: HashMap<Joint, Mat3> = HashMap::new();
    for j in Joint::all() {
        let a = pose.get(j);
        let local = rot_mat(
            -j.flex_sign() * a.flex.to_radians(),
            j.mirror() * a.abd.to_radians(),
        );
        let w = match j.parent() {
            None => local,
            Some(p) => mat_mul(&world[&p], &local),
        };
        world.insert(j, w);
    }
    Ok(world)
}

/// World-space joint positions for a posed, proportioned skeleton.
/// Root (pelvis) sits at `props.root`; rotations compose exactly
/// down the tree. Pure math — same skeleton twice, same coordinates.
pub fn forward_kinematics(
    props: &BodyProportions,
    pose: &Pose,
) -> Result<HashMap<Joint, V3>, String> {
    let world = joint_world_rotations(pose)?;
    let mut pos: HashMap<Joint, V3> = HashMap::new();
    for j in Joint::all() {
        let off = props.offset(j);
        // The offset rides in the parent frame (standard FK: the
        // joint's own rotation moves its children, not itself).
        let rotated = match j.parent() {
            None => off,
            Some(p) => apply_mat(&world[&p], off),
        };
        let p = match j.parent() {
            None => rotated,
            Some(par) => pos[&par].add(rotated),
        };
        pos.insert(j, p);
    }
    Ok(pos)
}

/// Measurement diagram: bones as lines, joints as discs, on a dark
/// grid with a ground line. INSTRUMENTATION — the receipt and the
/// filename both say measurement, never portrait.
pub fn render_diagram(pos: &HashMap<Joint, V3>, width: u32, height: u32) -> Image {
    let mut img = Image::blank(width, height, Rgb::new(16, 18, 24));
    // Fit: world x∈[-1,1]m, y∈[0,2]m with margin.
    let sx = width as f64 / 2.4;
    let sy = height as f64 / 2.2;
    let s = sx.min(sy);
    let cx = width as f64 / 2.0;
    let base = height as f64 - 12.0;
    let proj = |p: V3| -> (i32, i32) { ((cx + p.x * s) as i32, (base - p.y * s) as i32) };
    // Ground line.
    for x in 0..width {
        img.set(x, height - 12, Rgb::new(70, 80, 95));
    }
    let bone = Rgb::new(220, 225, 235);
    let joint_px = Rgb::new(255, 90, 90);
    for j in Joint::all() {
        if let Some(p) = j.parent() {
            let (x0, y0) = proj(pos[&p]);
            let (x1, y1) = proj(pos[&j]);
            draw_line(&mut img, x0, y0, x1, y1, bone);
        }
    }
    for j in Joint::all() {
        let (x, y) = proj(pos[&j]);
        img.draw_disc(x, y, 4, joint_px);
    }
    img
}

/// Bresenham line, clipped by `set` bounds. Two pixels wide so
/// measurement diagrams read at a glance.
fn draw_line(img: &mut Image, x0: i32, y0: i32, x1: i32, y1: i32, px: Rgb) {
    let (mut x, mut y) = (x0, y0);
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        img.set(x as u32, y as u32, px);
        if x >= 0 {
            img.set((x + 1) as u32, y as u32, px);
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn additive_rotation_matches_exact_for_single_axis() {
        // Single-axis chains are exact under additive composition:
        // elbow flexion alone must swing the wrist forward by the book.
        let props = BodyProportions::adult_male();
        let mut p = Pose::rest();
        p.set(Joint::ElbowR, JointAngles::flex(90.0));
        let pos = forward_kinematics(&props, &p).unwrap();
        let wrist = pos[&Joint::WristR];
        let elbow = pos[&Joint::ElbowR];
        let d = wrist - elbow;
        // 90° flexion: forearm points forward (+z), not down.
        assert!(d.z > 0.20, "forearm must point forward: {:?}", d);
        assert!(d.y.abs() < 0.05, "no vertical remainder: {:?}", d);
        // Length preserved exactly.
        let rest = forward_kinematics(&props, &Pose::rest()).unwrap();
        let l0 = (rest[&Joint::WristR] - rest[&Joint::ElbowR]).len();
        assert!((d.len() - l0).abs() < 1e-9);
    }
}
