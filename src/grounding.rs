//! GroundingType runtime: what the derive macro describes.
//!
//! Schemas are plain data the deterministic engine reads: which
//! fields exist, their Rust types as strings, doc lines, and morph
//! flags. No reflection, no registry, no surprises.
#[derive(Debug, Clone)]
pub struct GroundingField {
    pub name: &'static str,
    pub ty: &'static str,
    pub doc: &'static str,
    pub is_morph: bool,
}

#[derive(Debug, Clone)]
pub struct GroundingSchema {
    pub type_name: &'static str,
    pub fields: Vec<GroundingField>,
}

impl GroundingSchema {
    /// Morph-target fields only: the deformable parameter space.
    pub fn morphs(&self) -> Vec<&GroundingField> {
        self.fields.iter().filter(|f| f.is_morph).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_filters_morphs() {
        let s = GroundingSchema {
            type_name: "Demo",
            fields: vec![
                GroundingField {
                    name: "a",
                    ty: "f32",
                    doc: "",
                    is_morph: true,
                },
                GroundingField {
                    name: "b",
                    ty: "u8",
                    doc: "",
                    is_morph: false,
                },
            ],
        };
        assert_eq!(s.morphs().len(), 1);
        assert_eq!(s.morphs()[0].name, "a");
    }
}
