//! Scene assembly — Phase 5 of the image-creation system.
//!
//! Relationships become placements: worn objects anchor to joints,
//! held objects anchor to free hands, resolved against FK joint
//! positions on the measurement scaffold. Every attachment carries
//! its anchor coordinates and the rule that put it there; anything
//! unplaceable stays in `unresolved` — never dropped, never guessed.
//!
//! Stated conventions (inspectable defaults, not intelligence):
//! salutes take the right hand; a waved object goes to the other
//! hand when one salutes, else the right with a note; hats sit
//! above the head joint by half a head height plus a crown gap.
//! HONEST LIMIT: anchors are coordinates for later construction —
//! no meshes are attached here.

use super::skeleton::{BodyProportions, Joint, V3};
use std::collections::HashMap;

/// One placed object: where it anchors and why.
#[derive(Debug, Clone)]
pub struct Attachment {
    pub object_id: String,
    pub joint: Joint,
    /// World-space anchor in meters.
    pub anchor: V3,
    /// Rule that placed it (convention or relationship).
    pub note: String,
}

/// Assembly outcome: placements plus everything left open.
#[derive(Debug, Clone)]
pub struct Assembly {
    pub attachments: Vec<Attachment>,
    pub unresolved: Vec<String>,
}

/// Assemble a parsed scene onto posed joint positions.
/// `positions` come from FK (Phase 1); `saluting`/`waving` name the
/// actions present so hands assign deterministically.
pub fn assemble(
    spec: &super::scene_intent::SceneSpec,
    positions: &HashMap<Joint, V3>,
    props: &BodyProportions,
) -> Assembly {
    let mut out = Assembly {
        attachments: Vec::new(),
        unresolved: Vec::new(),
    };
    let joint_pos = |j: Joint| -> Option<V3> { positions.get(&j).copied() };
    let has_action = |t: &str| spec.actions.iter().any(|a| a.atype == t);
    let saluting = has_action("salute");

    for o in &spec.objects {
        // Hats: worn objects sit above the head joint.
        if o.otype == "hat" {
            match (&o.worn_by, joint_pos(Joint::Head)) {
                (Some(w), Some(head)) => {
                    let crown = props.stature_m / 7.5 / 2.0 + 0.03;
                    out.attachments.push(Attachment {
                        object_id: o.id.clone(),
                        joint: Joint::Head,
                        anchor: V3::new(head.x, head.y + crown, head.z),
                        note: format!("{} worn by {}: crown above head", o.id, w),
                    });
                }
                _ => out
                    .unresolved
                    .push(format!("{}: no wearer or head position — unplaced", o.id)),
            }
            continue;
        }
        // Waved fabric: the hand not saluting holds the pole.
        if o.state.iter().any(|s| s == "waving") {
            let (hand, why) = if saluting {
                (Joint::HandL, "salute takes right; flag to left")
            } else {
                (Joint::HandR, "free hand defaults right")
            };
            match joint_pos(hand) {
                Some(p) => out.attachments.push(Attachment {
                    object_id: o.id.clone(),
                    joint: hand,
                    anchor: p,
                    note: format!("{} held at {:?}: {}", o.id, hand, why),
                }),
                None => out
                    .unresolved
                    .push(format!("{}: no hand position — unplaced", o.id)),
            }
            continue;
        }
        // Anything else: no placement rule exists. Named, not boxed.
        out.unresolved.push(format!(
            "{} ({}): no placement rule — needs evidence",
            o.id, o.otype
        ));
    }

    // Salute check: the saluting hand must actually reach the head.
    if saluting {
        match (joint_pos(Joint::HandR), joint_pos(Joint::Head)) {
            (Some(hand), Some(head)) => {
                let d = ((hand.x - head.x).powi(2) + (hand.y - head.y).powi(2)).sqrt();
                if d > 0.45 {
                    out.unresolved
                        .push(format!("salute hand {:.2}m from head — pose misses", d));
                }
            }
            _ => out
                .unresolved
                .push("salute: hand or head position missing".to_string()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_objects_stay_open() {
        let spec = super::super::scene_intent::parse_scene("A tower out of glass.");
        let props = BodyProportions::adult_male();
        let pos = super::super::skeleton::forward_kinematics(
            &props,
            &super::super::skeleton::Pose::standing(),
        )
        .unwrap();
        let asm = assemble(&spec, &pos, &props);
        assert!(asm.attachments.is_empty());
        assert!(asm.unresolved.iter().any(|u| u.contains("tower")));
    }
}
