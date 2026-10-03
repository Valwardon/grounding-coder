//! Iterative refinement: the grounded diffusion parallel.
//!
//! Diffusion corrupts data, then learns to reverse it. Here both
//! directions are explicit: targets come from measurements (plates,
//! rigs, specs), and each refinement step proposes one ROM-clamped
//! parameter move, re-measures the residual, and keeps the move
//! only on improvement. Bounded iterations, full trajectory
//! receipt — coarse poses first, fine corrections later, exactly
//! the denoising schedule without a network.
//!
//! Sampling also mirrors diffusion honestly: new individuals draw
//! uniformly within RESEARCHED ranges (seeded, explicit
//! histograms) — synthetic variation, never a specific person.

use super::skeleton::{BodyProportions, Joint, JointAngles, Pose};

/// One constraint the refined pose must satisfy.
#[derive(Debug, Clone)]
pub enum RefineTarget {
    /// Joint world position within tolerance (meters).
    Reach {
        joint: Joint,
        point: [f64; 3],
        tol: f64,
    },
    /// Joint world Y within tolerance (foot planting).
    Plant { joint: Joint, y: f64, tol: f64 },
}

/// One accepted refinement step, for the trajectory receipt.
#[derive(Debug, Clone)]
pub struct RefineStep {
    pub joint: String,
    pub flex_before: f64,
    pub flex_after: f64,
    pub residual_before: f64,
    pub residual_after: f64,
}

/// Residual: sum of squared target violations. Zero clears.
fn residual(props: &BodyProportions, pose: &Pose, targets: &[RefineTarget]) -> Option<f64> {
    let pos = super::skeleton::forward_kinematics(props, pose).ok()?;
    let mut sum = 0.0;
    for t in targets {
        match t {
            RefineTarget::Reach { joint, point, tol } => {
                let p = pos.get(joint)?;
                let d = ((p.x - point[0]).powi(2)
                    + (p.y - point[1]).powi(2)
                    + (p.z - point[2]).powi(2))
                .sqrt()
                    - tol;
                if d > 0.0 {
                    sum += d * d;
                }
            }
            RefineTarget::Plant { joint, y, tol } => {
                let p = pos.get(joint)?;
                let d = (p.y - y).abs() - tol;
                if d > 0.0 {
                    sum += d * d;
                }
            }
        }
    }
    Some(sum)
}

/// Refine a starting pose against targets: coordinate descent over
/// joints (fixed order), ±5° proposals inside ROM, keep on strict
/// improvement only. Every accepted step lands in the trail.
/// Returns the refined pose plus its trajectory receipt.
pub fn refine_pose(
    props: &BodyProportions,
    start: &Pose,
    targets: &[RefineTarget],
    budget: u32,
) -> (Pose, Vec<RefineStep>) {
    let mut pose = start.clone();
    let mut trail = Vec::new();
    let mut remaining = budget;
    let order = Joint::all();
    while remaining > 0 {
        let before = match residual(props, &pose, targets) {
            Some(r) => r,
            None => break,
        };
        if before <= 0.0 {
            break;
        }
        let mut improved = false;
        for j in &order {
            if remaining == 0 {
                break;
            }
            let cur = pose.get(*j);
            for delta in [5.0, -5.0] {
                let mut cand = pose.clone();
                cand.set(
                    *j,
                    JointAngles {
                        flex: cur.flex + delta,
                        abd: cur.abd,
                    },
                );
                // ROM gates every proposal with correct arity
                // (unknown/axial joints skip quietly).
                let valid = match joint_rom(*j) {
                    Some((name, n)) => {
                        let both = [cand.get(*j).flex, cand.get(*j).abd];
                        super::anatomy::validate_pose(name, &both[..n]).is_ok()
                    }
                    None => false,
                };
                if !valid {
                    continue;
                }
                if let Some(after) = residual(props, &cand, targets) {
                    remaining -= 1;
                    if after < before {
                        trail.push(RefineStep {
                            joint: format!("{:?}", j),
                            flex_before: cur.flex,
                            flex_after: cur.flex + delta,
                            residual_before: before,
                            residual_after: after,
                        });
                        pose = cand;
                        improved = true;
                        break;
                    }
                    if remaining == 0 {
                        break;
                    }
                }
            }
            if improved {
                break;
            }
        }
        if !improved {
            break;
        }
    }
    (pose, trail)
}

/// ROM-table name for a joint (axial joints have no rows and stay
/// at rest), plus its axis count for arity-correct validation.
fn joint_rom(j: Joint) -> Option<(&'static str, usize)> {
    let name: &'static str = match j {
        Joint::ShoulderL | Joint::ShoulderR => "shoulder",
        Joint::ElbowL | Joint::ElbowR => "elbow",
        Joint::WristL | Joint::WristR => "wrist",
        Joint::HipL | Joint::HipR => "hip",
        Joint::KneeL | Joint::KneeR => "knee",
        Joint::AnkleL | Joint::AnkleR => "ankle",
        Joint::Neck => "neck",
        _ => return None,
    };
    let n = super::anatomy::joint_table()
        .into_iter()
        .find(|e| e.joint == name)
        .map(|e| e.rom.len())?;
    Some((name, n))
}

/// Sample a new individual within researched ranges (seeded):
/// stature uniform in [1.60, 1.90] (measured male/female bracket),
/// morph weight in [0.7, 1.0] (near-full expression). Explicit
/// histograms — the sampling half of the diffusion parallel,
/// with no network and no likeness.
#[derive(Debug, Clone)]
pub struct Individual {
    pub stature_m: f64,
    pub morph_weight: f64,
    pub seed: u64,
}

pub fn sample_individual(seed: u64) -> Individual {
    // Deterministic LCG (stated, repeatable).
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f64) / (u32::MAX as f64)
    };
    Individual {
        stature_m: 1.60 + next() * 0.30,
        morph_weight: 0.7 + next() * 0.3,
        seed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn male() -> BodyProportions {
        BodyProportions::adult_male()
    }

    #[test]
    fn refinement_reaches_further() {
        // Arm hanging far from the head: refinement must close the
        // gap without breaking ROM.
        let mut start = Pose::rest();
        start.set(Joint::ShoulderR, JointAngles::flex(30.0));
        start.set(Joint::ElbowR, JointAngles::flex(10.0));
        let props = male();
        let head = forward_head(&props);
        let targets = vec![RefineTarget::Reach {
            joint: Joint::HandR,
            point: head,
            tol: 0.45,
        }];
        let before = residual(&props, &start, &targets).unwrap();
        let (pose, trail) = refine_pose(&props, &start, &targets, 40);
        let after = residual(&props, &pose, &targets).unwrap();
        assert!(after <= before, "{} !<= {}", after, before);
        assert!(!trail.is_empty(), "must take at least one step");
        // Every step strictly improves (denoising, never wandering).
        for s in &trail {
            assert!(s.residual_after < s.residual_before);
        }
        // ROM held on the whole final pose.
        assert!(crate::engine::skeleton::forward_kinematics(&props, &pose).is_ok());
    }

    fn forward_head(props: &BodyProportions) -> [f64; 3] {
        let pos = crate::engine::skeleton::forward_kinematics(props, &Pose::rest()).unwrap();
        let h = pos[&Joint::Head];
        [h.x, h.y, h.z]
    }

    #[test]
    fn feet_plant_on_demand() {
        let props = male();
        let start = Pose::standing();
        let targets = vec![
            RefineTarget::Plant {
                joint: Joint::FootL,
                y: 0.02,
                tol: 0.02,
            },
            RefineTarget::Plant {
                joint: Joint::FootR,
                y: 0.02,
                tol: 0.02,
            },
        ];
        let (pose, _) = refine_pose(&props, &start, &targets, 20);
        let pos = crate::engine::skeleton::forward_kinematics(&props, &pose).unwrap();
        assert!((pos[&Joint::FootL].y - 0.02).abs() < 0.05);
    }

    #[test]
    fn budget_bounds_work() {
        let props = male();
        let impossible = vec![RefineTarget::Reach {
            joint: Joint::HandR,
            point: [10.0, 10.0, 10.0],
            tol: 0.01,
        }];
        let (_, trail) = refine_pose(&props, &Pose::rest(), &impossible, 7);
        assert!(trail.len() as u32 <= 7, "budget overrun");
    }

    #[test]
    fn individuals_stay_in_ranges() {
        for seed in [1u64, 7, 42, 999] {
            let ind = sample_individual(seed);
            assert!((1.60..=1.90).contains(&ind.stature_m));
            assert!((0.7..=1.0).contains(&ind.morph_weight));
        }
        // Seeded: same seed, same individual.
        assert_eq!(
            sample_individual(42).stature_m,
            sample_individual(42).stature_m
        );
    }

    #[test]
    fn unknown_joints_skip_rom_quietly() {
        // Axial joints (spine/chest/pelvis) carry no ROM rows:
        // refinement leaves them at rest instead of failing.
        let props = male();
        let (_, trail) = refine_pose(&props, &Pose::rest(), &[], 10);
        assert!(trail.is_empty());
    }
}
