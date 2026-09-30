//! Human genomes: synthetic adults from parameters, never likenesses.
//!
//! A genome is validated data, not a mesh asset: age restricted to
//! adults (refused otherwise — no minors, no exceptions), morphs in
//! 0..1, every property carrying its evidence (source + status).
//! The genome drives geometry through [`super::mesh`]; the renderer
//! draws whatever the genome describes. Figures are clothed busts
//! by construction — there is no nude render path.
use super::mesh::{HeadShape, Mesh};
use super::vision::Rgb;

/// Adulthood floor: genomes describe adults only.
pub const ADULT_AGE_MIN: u8 = 18;
pub const ADULT_AGE_MAX: u8 = 100;

/// Where a property's numbers come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropertyEvidence {
    /// Researched (anatomy graph, cited canon).
    Researched(String),
    /// Renderer-chosen default, stated as such.
    Stylistic(String),
}

/// One genome parameter: value plus its evidence story.
#[derive(Debug, Clone)]
pub struct GenomeParam {
    pub value: f64,
    pub evidence: PropertyEvidence,
}

/// A synthetic adult: face morphs, body measures, appearance. All
/// morphs 0..1; age 18+. Invalid genomes refuse at construction —
/// the engine never renders what it cannot validate.
///
/// Inspected via `grounding_schema()` (derive-generated field
/// inventory), never via struct internals.
#[derive(Debug, Clone, grounding_macros::GroundingType)]
pub struct HumanGenome {
    pub age_years: u8,
    pub skin_tone: Rgb,
    pub hair_color: Rgb,
    pub cranial_width: GenomeParam,
    pub jaw_width: GenomeParam,
    pub cheek_projection: GenomeParam,
    pub nose_length: GenomeParam,
    pub nose_width: GenomeParam,
    pub eye_spacing: GenomeParam,
    pub lip_fullness: GenomeParam,
    pub chin_projection: GenomeParam,
    /// Smile weight 0..1 (blendshape).
    pub smile: f64,
}

fn morph(value: f64, evidence: PropertyEvidence) -> Result<GenomeParam, String> {
    if !(0.0..=1.0).contains(&value) {
        return Err(format!("morph out of range [0,1]: {}", value));
    }
    Ok(GenomeParam { value, evidence })
}

impl HumanGenome {
    /// A default adult. Every morph cites the anatomy graph or the
    /// renderer default it actually is — inspectable via
    /// [`HumanGenome::describe`].
    pub fn adult() -> Result<Self, String> {
        Self::with_params(32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_params(
        age: u8,
        cranial_width: f64,
        jaw_width: f64,
        cheek_projection: f64,
        nose_length: f64,
        nose_width: f64,
        eye_spacing: f64,
        lip_fullness: f64,
        chin_projection: f64,
        smile: f64,
    ) -> Result<Self, String> {
        if !(ADULT_AGE_MIN..=ADULT_AGE_MAX).contains(&age) {
            return Err(format!(
                "age {} outside adult range {}-{} — refusing",
                age, ADULT_AGE_MIN, ADULT_AGE_MAX
            ));
        }
        if !(0.0..=1.0).contains(&smile) {
            return Err(format!("smile out of range [0,1]: {}", smile));
        }
        let graph = PropertyEvidence::Researched("anatomy graph + Gray's canon".to_string());
        let style =
            |what: &str| PropertyEvidence::Stylistic(format!("renderer default for {}", what));
        Ok(HumanGenome {
            age_years: age,
            skin_tone: Rgb::new(200, 150, 115),
            hair_color: Rgb::new(60, 38, 24),
            cranial_width: morph(cranial_width, graph.clone())?,
            jaw_width: morph(jaw_width, graph.clone())?,
            cheek_projection: morph(cheek_projection, style("cheeks"))?,
            nose_length: morph(nose_length, graph.clone())?,
            nose_width: morph(nose_width, graph.clone())?,
            eye_spacing: morph(eye_spacing, graph.clone())?,
            lip_fullness: morph(lip_fullness, style("lips"))?,
            chin_projection: morph(chin_projection, style("chin"))?,
            smile,
        })
    }

    fn head_shape(&self) -> HeadShape {
        HeadShape {
            cranial_width: self.cranial_width.value,
            jaw_width: self.jaw_width.value,
            cheek_projection: self.cheek_projection.value,
            nose_length: self.nose_length.value,
            nose_width: self.nose_width.value,
            eye_spacing: self.eye_spacing.value,
            lip_fullness: self.lip_fullness.value,
            chin_projection: self.chin_projection.value,
        }
    }

    /// Build the head mesh at 32×22 resolution with the smile
    /// blendshape applied. Deterministic per genome.
    pub fn head_mesh(&self) -> Mesh {
        let base = Mesh::parametric_head(32, 22, &self.head_shape());
        let deltas = base.smile_deltas();
        base.blendshape(&deltas, self.smile)
    }

    /// Human-readable inspection: every property, value, evidence.
    /// The engine (and editors) read people through this, never
    /// through struct internals.
    pub fn describe(&self) -> Vec<String> {
        let mut out = vec![
            format!(
                "age: {} (adult {}-{})",
                self.age_years, ADULT_AGE_MIN, ADULT_AGE_MAX
            ),
            format!("skin_tone: {:?}", self.skin_tone),
            format!("hair_color: {:?}", self.hair_color),
        ];
        for (name, p) in [
            ("cranial_width", &self.cranial_width),
            ("jaw_width", &self.jaw_width),
            ("cheek_projection", &self.cheek_projection),
            ("nose_length", &self.nose_length),
            ("nose_width", &self.nose_width),
            ("eye_spacing", &self.eye_spacing),
            ("lip_fullness", &self.lip_fullness),
            ("chin_projection", &self.chin_projection),
        ] {
            let ev = match &p.evidence {
                PropertyEvidence::Researched(s) => format!("researched: {}", s),
                PropertyEvidence::Stylistic(s) => format!("stylistic: {}", s),
            };
            out.push(format!("{}: {:.2} [{}]", name, p.value, ev));
        }
        out.push(format!("smile: {:.2}", self.smile));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minors_refuse_at_construction() {
        assert!(HumanGenome::with_params(12, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0).is_err());
        assert!(HumanGenome::with_params(17, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0).is_err());
        assert!(HumanGenome::adult().is_ok());
        assert!(HumanGenome::with_params(25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0).is_ok());
        // Morphs outside 0..1 refuse too.
        assert!(HumanGenome::with_params(25, 1.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0).is_err());
    }

    #[test]
    fn genome_drives_geometry() {
        let narrow =
            HumanGenome::with_params(30, 0.5, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0).unwrap();
        let wide =
            HumanGenome::with_params(30, 0.5, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0).unwrap();
        let (mn, mw) = (narrow.head_mesh(), wide.head_mesh());
        let width = |m: &Mesh| -> f64 {
            m.verts
                .iter()
                .filter(|v| v[1] < -0.25)
                .map(|v| v[0].abs())
                .fold(0.0f64, f64::max)
                * 2.0
        };
        assert!(width(&mw) > width(&mn) + 0.03);
    }

    #[test]
    fn schema_lists_every_field() {
        let schema = HumanGenome::grounding_schema();
        assert_eq!(schema.type_name, "HumanGenome");
        let names: Vec<&str> = schema.fields.iter().map(|f| f.name).collect();
        for want in ["age_years", "skin_tone", "cranial_width", "smile"] {
            assert!(names.contains(&want), "{:?}", names);
        }
        let age = schema
            .fields
            .iter()
            .find(|f| f.name == "age_years")
            .unwrap();
        assert_eq!(age.ty, "u8");
    }

    #[test]
    fn morph_flags_survive_the_macro() {
        use grounding_macros::GroundingType;
        #[derive(GroundingType)]
        struct DemoFace {
            /// Jaw width.
            #[morph]
            jaw: f32,
            /// Label.
            name: String,
        }
        let demo = DemoFace {
            jaw: 0.7,
            name: "test".to_string(),
        };
        assert_eq!(demo.jaw, 0.7);
        assert_eq!(demo.name, "test");
        let schema = DemoFace::grounding_schema();
        assert_eq!(schema.morphs().len(), 1);
        assert_eq!(schema.morphs()[0].name, "jaw");
        assert!(schema.morphs()[0].doc.contains("Jaw"));
    }

    #[test]
    fn every_property_carries_evidence() {
        let g = HumanGenome::adult().unwrap();
        let desc = g.describe();
        // 3 header lines + 8 morphs + smile.
        assert_eq!(desc.len(), 12, "{:?}", desc);
        assert!(desc.iter().all(|l| l.contains('[') || l.contains(':')));
    }
}
