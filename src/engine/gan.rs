//! On-device conditional GAN: the generator this repo deserves.
//!
//! NOT diffusion, NOT an LLM: a tiny feed-forward generator —
//! noise + conditioning in, photo out, ONE forward pass, phone-sized
//! weights, pure-Rust CPU backend. Training data comes ONLY from
//! researched plates (provenance manifest required, license/author
//! gates enforced — web scrapes need not apply). Conditioning is
//! researched words by one generic rule; the model never branches
//! on subject, so a woman, a cat, and a tree all train and run
//! through the same graph.
//!
//! Honesty contract: the generator REFUSES until trained weights
//! exist on disk. Random-init weights never produce photos, and no
//! hosted fallback hides behind this module — callers pick their
//! own fallback and receipt it themselves.

use burn::{
    config::Config,
    module::Module,
    nn,
    optim::{AdamConfig, GradientsParams},
    tensor::{Int, Tensor},
};
use burn_store::{ModuleSnapshot, SafetensorsStore};

/// Device handle.
pub type Device = burn::tensor::Device;

/// Output frame: what one forward pass paints.
pub const GEN_W: u32 = 64;
/// Output frame height.
pub const GEN_H: u32 = 64;
/// Noise vector width: the only randomness, always seeded.
pub const NOISE_DIM: usize = 16;
/// Conditioning vector width (words → embedding mean → project).
pub const COND_DIM: usize = 32;
/// Embedding buckets for hashed words (generic hashing, no vocab).
pub const WORD_BUCKETS: usize = 512;
/// Embedding width per bucket.
pub const WORD_DIM: usize = 16;
/// Max conditioning words (truncated, logged upstream).
pub const MAX_WORDS: usize = 16;
/// Checkpoint file inside a trained weights dir.
pub const WEIGHTS_FILE: &str = "generator.safetensors";
/// Training receipt file inside a trained weights dir.
pub const TRAIN_RECEIPT_FILE: &str = "train_receipt.json";

/// FNV-1a word hash into buckets: every word in every language
/// lands somewhere deterministic — no vocabulary to maintain, no
/// language to special-case.
fn word_bucket(word: &str) -> usize {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in word.to_lowercase().bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    (h % WORD_BUCKETS as u64) as usize
}

/// Conditioning encoder config.
#[derive(Config, Debug)]
pub struct CondEncoderConfig {
    buckets: usize,
    word_dim: usize,
    cond_dim: usize,
}

/// Words → fixed vector: hashed-bucket embedding mean, projected.
/// Same words → same vector (given weights); unknown words hash
/// like any other — nothing is out-of-vocabulary by construction.
#[derive(Module, Debug)]
pub struct CondEncoder {
    embedding: nn::Embedding,
    project: nn::Linear,
}

impl CondEncoderConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> CondEncoder {
        CondEncoder {
            embedding: nn::EmbeddingConfig::new(self.buckets, self.word_dim).init(device),
            project: nn::LinearConfig::new(self.word_dim, self.cond_dim).init(device),
        }
    }
}

impl CondEncoder {
    pub fn forward(&self, words: &[String]) -> Tensor<2> {
        let device = self.embedding.weight.device();
        let idx: Vec<i64> = words
            .iter()
            .take(MAX_WORDS)
            .map(|w| word_bucket(w) as i64)
            .collect();
        let idx = if idx.is_empty() { vec![0] } else { idx };
        let n = idx.len() as f32;
        // [1, n] indices → [1, n, word_dim] embedding rows.
        let input = Tensor::<1, Int>::from_data(idx.as_slice(), &device).unsqueeze_dim(0);
        let emb = self.embedding.forward(input);
        // Mean over words: order-free, count-free.
        let mean = emb.sum_dim(1).div_scalar(n);
        let mean = mean.reshape([1, self.embedding.weight.dims()[1]]);
        self.project.forward(mean)
    }
}

/// Generator config: noise+cond → 64x64 photo, three upscales.
#[derive(Config, Debug)]
pub struct GeneratorConfig {
    noise_dim: usize,
    cond_dim: usize,
}

/// Tiny feed-forward generator: FC → 8x8 → deconv ×3 → tanh.
/// ~300K params. One forward() call is one photo (single pass).
#[derive(Module, Debug)]
pub struct Generator {
    encoder: CondEncoder,
    fc: nn::Linear,
    deconv1: nn::conv::ConvTranspose2d,
    deconv2: nn::conv::ConvTranspose2d,
    deconv3: nn::conv::ConvTranspose2d,
}

impl GeneratorConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> Generator {
        Generator {
            encoder: CondEncoderConfig {
                buckets: WORD_BUCKETS,
                word_dim: WORD_DIM,
                cond_dim: self.cond_dim,
            }
            .init(device),
            fc: nn::LinearConfig::new(self.noise_dim + self.cond_dim, 8 * 8 * 32).init(device),
            deconv1: nn::conv::ConvTranspose2dConfig::new([32, 16], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .with_padding_out([0, 0])
                .init(device),
            deconv2: nn::conv::ConvTranspose2dConfig::new([16, 8], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .with_padding_out([0, 0])
                .init(device),
            deconv3: nn::conv::ConvTranspose2dConfig::new([8, 3], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .with_padding_out([0, 0])
                .init(device),
        }
    }
}

impl Generator {
    /// One pass: words + seeded noise → 64x64 photo tensor in [-1, 1].
    pub fn forward(&self, words: &[String], noise: Tensor<2>) -> Tensor<4> {
        let cond = self.encoder.forward(words);
        let x = Tensor::cat(vec![noise, cond], 1);
        let x = self.fc.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = x.reshape([1, 32, 8, 8]);
        let x = burn::tensor::activation::relu(self.deconv1.forward(x));
        let x = burn::tensor::activation::relu(self.deconv2.forward(x));
        burn::tensor::activation::tanh(self.deconv3.forward(x))
    }
}

/// Discriminator config: photo+cond → real logit.
#[derive(Config, Debug)]
pub struct DiscriminatorConfig {
    cond_dim: usize,
}

/// Tiny conditional discriminator: three strided convs, cond
/// concatenated late, one logit. Sees words, never branches on them.
#[derive(Module, Debug)]
pub struct Discriminator {
    encoder: CondEncoder,
    conv1: nn::conv::Conv2d,
    conv2: nn::conv::Conv2d,
    conv3: nn::conv::Conv2d,
    fc: nn::Linear,
}

impl DiscriminatorConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> Discriminator {
        Discriminator {
            encoder: CondEncoderConfig {
                buckets: WORD_BUCKETS,
                word_dim: WORD_DIM,
                cond_dim: self.cond_dim,
            }
            .init(device),
            conv1: nn::conv::Conv2dConfig::new([3, 16], [4, 4])
                .with_stride([2, 2])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv2: nn::conv::Conv2dConfig::new([16, 32], [4, 4])
                .with_stride([2, 2])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv3: nn::conv::Conv2dConfig::new([32, 64], [4, 4])
                .with_stride([2, 2])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            fc: nn::LinearConfig::new(8 * 8 * 64 + self.cond_dim, 1).init(device),
        }
    }
}

impl Discriminator {
    pub fn forward(&self, img: Tensor<4>, words: &[String]) -> Tensor<2> {
        let cond = self.encoder.forward(words);
        let x = burn::tensor::activation::leaky_relu(self.conv1.forward(img), 0.2);
        let x = burn::tensor::activation::leaky_relu(self.conv2.forward(x), 0.2);
        let x = burn::tensor::activation::leaky_relu(self.conv3.forward(x), 0.2);
        let x = x.reshape([1, 8 * 8 * 64]);
        let x = Tensor::cat(vec![x, cond], 1);
        self.fc.forward(x)
    }
}

/// One researched training example: words from the plate's title,
/// pixels from the plate, provenance for the receipt.
#[derive(Debug, Clone)]
pub struct ResearchExample {
    pub words: Vec<String>,
    pub image: super::vision::Image,
    pub provenance: String,
}

/// Grounded dataset: built ONLY from a bank dir carrying a
/// provenance manifest. Plates missing author or license are
/// refused with reasons — unlicensed pixels never train.
#[derive(Debug, Default)]
pub struct ResearchDataset {
    pub examples: Vec<ResearchExample>,
    pub refused: Vec<String>,
}

impl ResearchDataset {
    pub fn from_bank(bank: &std::path::Path) -> Self {
        let mut out = ResearchDataset::default();
        let manifest = match std::fs::read_to_string(bank.join("provenance.json")) {
            Ok(m) => m,
            Err(e) => {
                out.refused.push(format!("no provenance manifest: {}", e));
                return out;
            }
        };
        let entries: Vec<serde_json::Value> = match serde_json::from_str(&manifest) {
            Ok(v) => v,
            Err(e) => {
                out.refused.push(format!("unreadable manifest: {}", e));
                return out;
            }
        };
        for e in &entries {
            let file = e.get("file").and_then(|v| v.as_str()).unwrap_or("");
            let author = e.get("author").and_then(|v| v.as_str()).unwrap_or("");
            let license = e.get("license").and_then(|v| v.as_str()).unwrap_or("");
            if file.is_empty() || author.is_empty() || license.is_empty() {
                out.refused
                    .push("unlicensed plate refused (file/author/license incomplete)".to_string());
                continue;
            }
            let img = match super::vision::Image::load_bmp(&bank.join(file)) {
                Ok(img) => img,
                Err(e) => {
                    out.refused.push(format!("{} unreadable: {}", file, e));
                    continue;
                }
            };
            let title = e.get("title").and_then(|v| v.as_str()).unwrap_or(file);
            let words: Vec<String> = title
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| !w.is_empty())
                .map(|w| w.to_lowercase())
                .collect();
            out.examples.push(ResearchExample {
                words,
                image: img,
                provenance: format!("{} | {} | {}", author, license, file),
            });
        }
        out
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

/// Engine image → training tensor [1,3,64,64] in [-1,1].
fn image_to_tensor(img: &super::vision::Image, device: &burn::tensor::Device) -> Tensor<4> {
    let small = img.resize_smooth(GEN_W, GEN_H);
    let mut data = Vec::with_capacity((GEN_W * GEN_H * 3) as usize);
    for c in 0..3 {
        for y in 0..GEN_H {
            for x in 0..GEN_W {
                let p = small
                    .get(x, y)
                    .map(|p| [p.r, p.g, p.b][c as usize])
                    .unwrap_or(0);
                data.push(p as f32 / 127.5 - 1.0);
            }
        }
    }
    Tensor::<1>::from_floats(data.as_slice(), device).reshape([
        1,
        3,
        GEN_H as usize,
        GEN_W as usize,
    ])
}

/// Seeded noise [1, NOISE_DIM]: deterministic LCG, std-only — the
/// same seed always draws the same noise, on every backend.
fn seeded_noise(seed: u64, device: &burn::tensor::Device) -> Tensor<2> {
    let mut s = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut data = Vec::with_capacity(NOISE_DIM);
    for _ in 0..NOISE_DIM {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        data.push(((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0);
    }
    Tensor::<1>::from_floats(data.as_slice(), device).reshape([1, NOISE_DIM])
}

/// Sigmoid-cross-entropy against constant labels, mean over batch.
fn bce_with_labels(logits: Tensor<2>, label: f32) -> Tensor<1> {
    let eps = 1e-7;
    let probs = burn::tensor::activation::sigmoid(logits);
    let y = probs.clone().mul_scalar(label)
        + probs
            .clone()
            .mul_scalar(-1.0)
            .add_scalar(1.0)
            .mul_scalar(1.0 - label);
    y.clamp(eps, 1.0 - eps).log().neg().mean()
}

/// Read a scalar loss into a host f32 (NaN if unreadable).
fn first_scalar(t: &Tensor<1>) -> f32 {
    t.clone()
        .into_data()
        .try_to_vec::<f32>()
        .ok()
        .and_then(|v| v.first().copied())
        .unwrap_or(f32::NAN)
}

/// One alternating GAN step over a single example: D learns
/// real-vs-fake, G learns to fool D. Returns the updated models,
/// optimizers, and (d_loss, g_loss). Small, explicit, auditable —
/// the math between gradients and weights is right here.
pub fn train_step(
    genm: Generator,
    disc: Discriminator,
    mut gen_opt: burn::optim::ModuleOptimizer,
    mut disc_opt: burn::optim::ModuleOptimizer,
    ex: &ResearchExample,
    seed: u64,
    device: &burn::tensor::Device,
) -> (
    Generator,
    Discriminator,
    burn::optim::ModuleOptimizer,
    burn::optim::ModuleOptimizer,
    f32,
    f32,
) {
    let lr = 2e-4_f64;

    // D step: real should score 1, detached fake should score 0.
    let real = image_to_tensor(&ex.image, device);
    let noise = seeded_noise(seed, device);
    let fake = genm.forward(&ex.words, noise).detach();
    let d_loss = bce_with_labels(disc.forward(real, &ex.words), 1.0)
        + bce_with_labels(disc.forward(fake, &ex.words), 0.0);
    let d_val = first_scalar(&d_loss);
    let d_grads = GradientsParams::from_grads(d_loss.backward(), &disc);
    let disc = disc_opt.step(lr, disc, d_grads);

    // G step: generated photo should score 1.
    let noise = seeded_noise(seed.wrapping_add(1), device);
    let fake = genm.forward(&ex.words, noise);
    let g_loss = bce_with_labels(disc.forward(fake, &ex.words), 1.0);
    let g_val = first_scalar(&g_loss);
    let g_grads = GradientsParams::from_grads(g_loss.backward(), &genm);
    let genm = gen_opt.step(lr, genm, g_grads);

    (genm, disc, gen_opt, disc_opt, d_val, g_val)
}

/// Fresh models + optimizers on a device.
pub fn new_models(
    device: &burn::tensor::Device,
) -> (
    Generator,
    Discriminator,
    burn::optim::ModuleOptimizer,
    burn::optim::ModuleOptimizer,
) {
    let genm = GeneratorConfig {
        noise_dim: NOISE_DIM,
        cond_dim: COND_DIM,
    }
    .init(device);
    let disc = DiscriminatorConfig { cond_dim: COND_DIM }.init(device);
    let gen_opt = AdamConfig::new().init();
    let disc_opt = AdamConfig::new().init();
    (genm, disc, gen_opt, disc_opt)
}

/// Save the trained generator to `weights_dir/generator.safetensors`.
pub fn save_weights(genm: &Generator, weights_dir: &std::path::Path) -> Result<(), String> {
    std::fs::create_dir_all(weights_dir)
        .map_err(|e| format!("cannot create {}: {}", weights_dir.display(), e))?;
    let path = weights_dir.join(WEIGHTS_FILE);
    let mut store = SafetensorsStore::from_file(path.clone()).overwrite(true);
    genm.save_into(&mut store)
        .map_err(|e| format!("cannot save {}: {}", path.display(), e))
}

/// Train on a research bank: round-robin over the licensed plates for
/// `steps` alternating steps, then write weights + a receipt. Only a
/// bank that produced at least one licensed example may train — an
/// empty bank refuses rather than teaching random pixels.
pub fn train(
    bank: &std::path::Path,
    weights_dir: &std::path::Path,
    steps: u64,
) -> Result<TrainReceipt, String> {
    let ds = ResearchDataset::from_bank(bank);
    if ds.is_empty() {
        return Err(format!(
            "no licensed plates in {} — refusing to train on nothing",
            bank.display()
        ));
    }
    let device = burn::tensor::Device::flex().autodiff();
    let (mut genm, mut disc, mut gen_opt, mut disc_opt) = new_models(&device);
    let mut loss_trail = Vec::new();
    for step in 0..steps {
        let ex = &ds.examples[(step as usize) % ds.len()];
        let (g, d, go, dop, dv, gv) = train_step(genm, disc, gen_opt, disc_opt, ex, step, &device);
        genm = g;
        disc = d;
        gen_opt = go;
        disc_opt = dop;
        loss_trail.push((dv, gv));
    }
    save_weights(&genm, weights_dir)?;
    let receipt = TrainReceipt {
        steps,
        examples: ds.len(),
        refused: ds.refused.clone(),
        provenance: ds.examples.iter().map(|e| e.provenance.clone()).collect(),
        loss_trail,
    };
    let receipt_path = weights_dir.join(TRAIN_RECEIPT_FILE);
    std::fs::write(
        &receipt_path,
        serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("cannot write {}: {}", receipt_path.display(), e))?;
    Ok(receipt)
}

/// Weights path check: trained generator exists and loads.
pub fn weights_ready(weights_dir: &std::path::Path, device: &burn::tensor::Device) -> bool {
    let path = weights_dir.join(WEIGHTS_FILE);
    if !path.exists() {
        return false;
    }
    let mut store = SafetensorsStore::from_file(path);
    let mut r#gen = GeneratorConfig {
        noise_dim: NOISE_DIM,
        cond_dim: COND_DIM,
    }
    .init(device);
    r#gen.load_from(&mut store).is_ok()
}

/// On-device generation: load trained weights, one forward pass
/// from researched words + seed. REFUSES when no trained weights
/// exist — random weights never produce photos, and this function
/// never falls back to anything else.
pub fn generate_on_device(
    weights_dir: &std::path::Path,
    words: &[String],
    seed: u64,
) -> Result<super::vision::Image, String> {
    let device = burn::tensor::Device::flex();
    let path = weights_dir.join(WEIGHTS_FILE);
    if !path.exists() {
        return Err(format!(
            "no trained GAN at {} — refusing (train first, never random-init)",
            path.display()
        ));
    }
    let mut store = SafetensorsStore::from_file(path.clone());
    let mut genm = GeneratorConfig {
        noise_dim: NOISE_DIM,
        cond_dim: COND_DIM,
    }
    .init(&device);
    genm.load_from(&mut store)
        .map_err(|e| format!("unreadable weights {}: {}", path.display(), e))?;
    let noise = seeded_noise(seed, &device);
    let out = genm.forward(words, noise);
    let data: Vec<f32> = out.into_data().try_to_vec::<f32>().unwrap_or_default();
    if data.len() != (3 * GEN_W * GEN_H) as usize {
        return Err(format!(
            "generator emitted {} values — refusing",
            data.len()
        ));
    }
    let mut img = super::vision::Image::blank(GEN_W, GEN_H, super::vision::Rgb::new(0, 0, 0));
    for c in 0..3 {
        for y in 0..GEN_H {
            for x in 0..GEN_W {
                let v = data[(c * GEN_W * GEN_H + y * GEN_W + x) as usize];
                let b = ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                let cur = img.get(x, y).unwrap_or(super::vision::Rgb::new(0, 0, 0));
                let rgb = match c {
                    0 => super::vision::Rgb::new(b, cur.g, cur.b),
                    1 => super::vision::Rgb::new(cur.r, b, cur.b),
                    _ => super::vision::Rgb::new(cur.r, cur.g, b),
                };
                img.set(x, y, rgb);
            }
        }
    }
    Ok(img)
}

/// Training receipt: what taught the weights, exactly.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainReceipt {
    pub steps: u64,
    pub examples: usize,
    pub refused: Vec<String>,
    pub provenance: Vec<String>,
    pub loss_trail: Vec<(f32, f32)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Device {
        burn::tensor::Device::flex().autodiff()
    }

    #[test]
    fn conditioning_is_deterministic_and_shaped() {
        // Same words twice → same vector; shape is [1, COND_DIM].
        // Vocabulary-free: unknown words hash like any other.
        let (r#gen, _, _, _) = new_models(&device());
        let w = vec!["woman".to_string(), "stand".to_string()];
        let a = r#gen.encoder.forward(&w);
        let b = r#gen.encoder.forward(&w);
        let da: Vec<f32> = a.into_data().try_to_vec::<f32>().unwrap();
        let db: Vec<f32> = b.into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(da.len(), COND_DIM);
        assert_eq!(da, db);
        let other = r#gen
            .encoder
            .forward(&["xylophone".to_string(), "quasar".to_string()]);
        let dother: Vec<f32> = other.into_data().try_to_vec::<f32>().unwrap();
        assert_ne!(da, dother, "distinct words must encode distinctly");
    }

    #[test]
    fn generator_paints_64x64_in_one_pass() {
        let (r#gen, _, _, _) = new_models(&device());
        let noise = seeded_noise(7, &device());
        let out = r#gen.forward(&["woman".to_string()], noise);
        assert_eq!(out.dims(), [1, 3, GEN_H as usize, GEN_W as usize]);
    }

    #[test]
    fn discriminator_scores_photo_and_words() {
        let (_, disc, _, _) = new_models(&device());
        let img = Tensor::<4>::zeros([1, 3, GEN_H as usize, GEN_W as usize], &device());
        let logit = disc.forward(img, &["cat".to_string()]);
        assert_eq!(logit.dims(), [1, 1]);
    }

    #[test]
    fn dataset_requires_provenance() {
        // Manifest entries missing author/license refuse; the
        // licensed one banks with title words.
        let dir = std::path::Path::new("/tmp/gan-test-bank");
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();
        let img =
            super::super::vision::Image::blank(32, 32, super::super::vision::Rgb::new(9, 9, 9));
        img.save_bmp(&dir.join("good.bmp")).unwrap();
        img.save_bmp(&dir.join("nolicense.bmp")).unwrap();
        std::fs::write(
            dir.join("provenance.json"),
            serde_json::json!([
                {"file": "good.bmp", "author": "a", "license": "CC-BY", "title": "standing woman"},
                {"file": "nolicense.bmp", "author": "", "license": "", "title": "x"}
            ])
            .to_string(),
        )
        .unwrap();
        let ds = ResearchDataset::from_bank(dir);
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.examples[0].words, vec!["standing", "woman"]);
        assert_eq!(ds.refused.len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn train_step_moves_finite_loss() {
        // Two alternating steps on one synthetic example: losses
        // finite, models still shaped. Smoke only — learning needs
        // a real bank and real steps, receipted when run.
        let device = device();
        let (r#gen, disc, gopt, dopt) = new_models(&device);
        let ex = ResearchExample {
            words: vec!["woman".to_string()],
            image: super::super::vision::Image::blank(
                64,
                64,
                super::super::vision::Rgb::new(120, 110, 100),
            ),
            provenance: "test".to_string(),
        };
        let (r#gen, disc, gopt, dopt, d1, g1) =
            train_step(r#gen, disc, gopt, dopt, &ex, 1, &device);
        let (_, _, _, _, d2, g2) = train_step(r#gen, disc, gopt, dopt, &ex, 2, &device);
        for v in [d1, g1, d2, g2] {
            assert!(v.is_finite(), "loss blew up: {}", v);
        }
    }

    #[test]
    fn generate_refuses_without_weights() {
        // No weights on disk → refusal naming the path, never
        // random-init pixels.
        let err = generate_on_device(
            std::path::Path::new("/tmp/gan-no-weights-here"),
            &["woman".to_string()],
            1,
        )
        .expect_err("must refuse untrained");
        assert!(err.contains("no trained GAN"), "{}", err);
    }

    #[test]
    fn train_writes_weights_and_receipt_that_generate() {
        // End-to-end: a licensed bank trains a few steps, weights
        // reload from disk, and the reloaded generator paints a
        // 64x64 frame — the whole contract, receipted.
        let bank = std::path::Path::new("/tmp/gan-train-bank");
        let weights = std::path::Path::new("/tmp/gan-train-weights");
        let _ = std::fs::remove_dir_all(bank);
        let _ = std::fs::remove_dir_all(weights);
        std::fs::create_dir_all(bank).unwrap();
        let img =
            super::super::vision::Image::blank(48, 48, super::super::vision::Rgb::new(60, 90, 120));
        img.save_bmp(&bank.join("p.bmp")).unwrap();
        std::fs::write(
            bank.join("provenance.json"),
            serde_json::json!([
                {"file": "p.bmp", "author": "a", "license": "CC-BY", "title": "standing woman"}
            ])
            .to_string(),
        )
        .unwrap();
        let receipt = train(bank, weights, 3).expect("training must run");
        assert_eq!(receipt.steps, 3);
        assert_eq!(receipt.examples, 1);
        assert!(weights_ready(weights, &device()), "weights must reload");
        assert!(weights.join(WEIGHTS_FILE).exists());
        assert!(weights.join(TRAIN_RECEIPT_FILE).exists());
        let out = generate_on_device(weights, &["standing".to_string(), "woman".to_string()], 1)
            .expect("reloaded generator must paint");
        assert_eq!((out.width, out.height), (GEN_W, GEN_H));
        let _ = std::fs::remove_dir_all(bank);
        let _ = std::fs::remove_dir_all(weights);
    }

    #[test]
    fn train_refuses_empty_bank() {
        let bank = std::path::Path::new("/tmp/gan-empty-bank");
        let _ = std::fs::remove_dir_all(bank);
        std::fs::create_dir_all(bank).unwrap();
        let err = train(bank, std::path::Path::new("/tmp/gan-empty-w"), 1)
            .expect_err("empty bank must refuse");
        assert!(err.contains("no licensed plates"), "{}", err);
        let _ = std::fs::remove_dir_all(bank);
    }
}
