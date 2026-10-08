fn main() {
    for p in [
        "A woman standing in a lake.",
        "A woman standing.",
        "An elephant crossing a river.",
        "Write a program that macros a keystroke.",
    ] {
        let spec = grounding_coder::engine::scene_intent::parse_scene(p);
        println!(
            "{p:?}\n   confidence {:.2}\n   subjects {:?}\n   actions {:?}\n   objects {:?}\n   unresolved {:?}",
            spec.confidence,
            spec.subjects.iter().map(|s| &s.stype).collect::<Vec<_>>(),
            spec.actions
                .iter()
                .map(|a| (a.atype.clone(), a.ambiguous))
                .collect::<Vec<_>>(),
            spec.objects
                .iter()
                .map(|o| (&o.id, &o.otype))
                .collect::<Vec<_>>(),
            spec.unresolved
        );
    }
}
