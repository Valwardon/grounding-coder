//! At-run training: miniature deterministic diffusion.
//!
//! Doctrine: fixed-seed DDIM (eta = 0) over the engine's own images.
//! No stochastic sampler anywhere — same dataset, seed, and schedule
//! twice gives byte-identical adapters and byte-identical pictures.
//! The training set is researched plates (per-request scope); the
//! adapter is hash-pinned into the receipt alongside the seed.
//!
//! Scale honesty, stated upfront: the denoiser here is a 67-parameter
//! MLP over single pixels — it learns tones and simple structure at
//! toy resolutions, not photorealism. It proves the LOOP (dataset →
//! train → pin → seeded sample → receipt), which is what has to be
//! right before any real backbone plugs in behind the same trait.
//! Strength comes later from bigger denoisers; the machinery below
//! does not change.
use super::vision::{Image, Rgb};

/// Diffusion timesteps for toy training/sampling.
pub const TRAIN_STEPS: usize = 10;
/// Hidden width of the pixel denoiser.
pub const HIDDEN: usize = 8;
/// Learning rate for the deterministic trainer.
pub const LEARN_RATE: f64 = 0.05;

/// Linear beta schedule, shared by training and sampling — one
/// schedule everywhere, so adapters and pictures always agree on
/// what "timestep t" means.
pub fn betas(steps: usize) -> Vec<f64> {
    (1..=steps)
        .map(|i| 0.0001 + (0.02 - 0.0001) * (i as f64 - 1.0) / (steps.max(2) - 1) as f64)
        .collect()
}

/// Cumulative product of (1 - beta): signal kept at each step.
pub fn alpha_bars(betas: &[f64]) -> Vec<f64> {
    let mut acc = 1.0;
    betas
        .iter()
        .map(|b| {
            acc *= 1.0 - b;
            acc
        })
        .collect()
}

/// Seeded deterministic RNG (LCG family, like the rest of the
/// engine): uniform output plus Box-Muller gaussians. The seed IS
/// the reproducibility of the whole trajectory — same seed, same
/// noise, same batches, same pictures.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E3779B97F4A7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 33) as f64) / (u32::MAX as f64)
    }

    /// Standard-normal sample from two uniforms (Box-Muller).
    pub fn next_gauss(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Forward diffusion: mix clean signal with gaussian noise at step t
/// (1-based). Pure function of (signal, t, noise) — the trainer and
/// the sampler share it, never reimplement it.
pub fn forward(signal: f64, t: usize, bars: &[f64], noise: f64) -> f64 {
    let a = bars[(t - 1).min(bars.len() - 1)].clamp(1e-6, 1.0);
    a.sqrt() * signal + (1.0 - a).sqrt() * noise
}

/// Tiny pixel denoiser: (r, g, b, t/T) → predicted noise (nr, ng, nb).
/// One hidden layer, tanh, shared across every pixel — a real (if
/// small) learned score function, not a lookup table.
#[derive(Debug, Clone)]
pub struct Denoiser {
    pub w1: [[f64; HIDDEN]; 4],
    pub b1: [f64; HIDDEN],
    pub w2: [[f64; 3]; HIDDEN],
    pub b2: [f64; 3],
}

impl Denoiser {
    /// Deterministic init from seed: uniform ±0.5 everywhere.
    pub fn init(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut w1 = [[0.0; HIDDEN]; 4];
        let mut b1 = [0.0; HIDDEN];
        let mut w2 = [[0.0; 3]; HIDDEN];
        let mut b2 = [0.0; 3];
        for row in w1.iter_mut() {
            for v in row.iter_mut() {
                *v = rng.next_f64() - 0.5;
            }
        }
        for v in b1.iter_mut() {
            *v = rng.next_f64() - 0.5;
        }
        for row in w2.iter_mut() {
            for v in row.iter_mut() {
                *v = rng.next_f64() - 0.5;
            }
        }
        for v in b2.iter_mut() {
            *v = rng.next_f64() - 0.5;
        }
        Denoiser { w1, b1, w2, b2 }
    }

    /// Forward pass: predict noise for one noisy pixel at timestep t.
    pub fn predict(&self, rgb: (f64, f64, f64), t01: f64) -> (f64, f64, f64) {
        let x = [rgb.0, rgb.1, rgb.2, t01];
        let mut h = [0.0; HIDDEN];
        for (j, hj) in h.iter_mut().enumerate() {
            let dot: f64 = x.iter().enumerate().map(|(i, xi)| self.w1[i][j] * xi).sum();
            *hj = (self.b1[j] + dot).tanh();
        }
        let mut o = [0.0; 3];
        for (k, ok) in o.iter_mut().enumerate() {
            let dot: f64 = h.iter().enumerate().map(|(j, hj)| self.w2[j][k] * hj).sum();
            *ok = self.b2[k] + dot;
        }
        (o[0], o[1], o[2])
    }

    /// Flattened weights for hashing and comparison.
    pub fn flat(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(4 * HIDDEN + HIDDEN + HIDDEN * 3 + 3);
        v.extend(self.w1.iter().flat_map(|r| r.iter().copied()));
        v.extend(self.b1.iter().copied());
        v.extend(self.w2.iter().flat_map(|r| r.iter().copied()));
        v.extend(self.b2.iter().copied());
        v
    }
}

/// Hash-pin an adapter: SHA-256 over rounded weights plus the
/// training hyperparameters. Receipts carry this — a different hash
/// is a different model, never silently substituted.
pub fn adapter_hash(den: &Denoiser, epochs: u32, lr: f64, steps: usize) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for w in den.flat() {
        h.update(format!("{:.6}", w).as_bytes());
    }
    h.update(format!("e{}l{:.4}s{}", epochs, lr, steps).as_bytes());
    format!("{:x}", h.finalize())
}

/// One training example: clean plate pixels in [0,1].
fn plate_signal(img: &Image) -> Vec<(f64, f64, f64)> {
    let mut v = Vec::with_capacity((img.width * img.height) as usize);
    for y in 0..img.height {
        for x in 0..img.width {
            if let Some(p) = img.get(x, y) {
                v.push((p.r as f64 / 255.0, p.g as f64 / 255.0, p.b as f64 / 255.0));
            }
        }
    }
    v
}

/// Train a denoiser on plates: each epoch samples every plate, a
/// random timestep, and fresh noise — all from the seeded RNG, so
/// the same inputs train byte-identical adapters. Refuses empty
/// datasets instead of training on nothing.
pub fn train(
    plates: &[Image],
    seed: u64,
    epochs: u32,
    steps: usize,
) -> Result<(Denoiser, TrainingLog), String> {
    if plates.is_empty() {
        return Err("no training plates — refusing to train on nothing".to_string());
    }
    let data: Vec<Vec<(f64, f64, f64)>> = plates.iter().map(plate_signal).collect();
    if data.iter().any(|d| d.is_empty()) {
        return Err("empty plate in dataset — refusing".to_string());
    }
    let bars = alpha_bars(&betas(steps.max(1)));
    let mut den = Denoiser::init(seed);
    let mut rng = Rng::new(seed ^ 0x7EA7);
    let mut first_loss = 0.0;
    let mut loss = 0.0;
    for epoch in 0..epochs {
        loss = 0.0;
        let mut n = 0u64;
        for plate in &data {
            // Deterministic order, seeded noise: the trajectory is a
            // pure function of (dataset, seed).
            let t = 1 + (rng.next_u64() as usize % steps.max(1));
            let t01 = t as f64 / steps.max(1) as f64;
            for &(r, g, b) in plate {
                let (er, eg, eb) = (rng.next_gauss(), rng.next_gauss(), rng.next_gauss());
                let xr = forward(r, t, &bars, er);
                let xg = forward(g, t, &bars, eg);
                let xb = forward(b, t, &bars, eb);
                // Forward activations (tanh hidden).
                let x = [xr, xg, xb, t01];
                let mut h = [0.0; HIDDEN];
                for (j, hj) in h.iter_mut().enumerate() {
                    let dot: f64 = x.iter().enumerate().map(|(i, xi)| den.w1[i][j] * xi).sum();
                    *hj = (den.b1[j] + dot).tanh();
                }
                let mut o = [0.0; 3];
                for (k, ok) in o.iter_mut().enumerate() {
                    let dot: f64 = h.iter().enumerate().map(|(j, hj)| den.w2[j][k] * hj).sum();
                    *ok = den.b2[k] + dot;
                }
                // MSE gradients, backprop by hand (tanh derivative).
                // Deltas are fully computed before any weight moves,
                // so reads never race writes.
                let target = [er, eg, eb];
                let mut do_ = [0.0; 3];
                for (k, dok) in do_.iter_mut().enumerate() {
                    *dok = 2.0 * (o[k] - target[k]) / 3.0;
                    loss += (o[k] - target[k]).powi(2);
                    n += 1;
                }
                let mut dh = [0.0; HIDDEN];
                for (j, dhj) in dh.iter_mut().enumerate() {
                    let back: f64 = do_
                        .iter()
                        .enumerate()
                        .map(|(k, dok)| dok * den.w2[j][k])
                        .sum();
                    *dhj = back * (1.0 - h[j] * h[j]);
                }
                for (k, dok) in do_.iter().enumerate() {
                    den.b2[k] -= LEARN_RATE * dok;
                    for (j, wrow) in den.w2.iter_mut().enumerate() {
                        wrow[k] -= LEARN_RATE * dok * h[j];
                    }
                }
                for (j, dhj) in dh.iter().enumerate() {
                    den.b1[j] -= LEARN_RATE * dhj;
                    for (i, wrow) in den.w1.iter_mut().enumerate() {
                        wrow[j] -= LEARN_RATE * dhj * x[i];
                    }
                }
            }
        }
        loss /= n.max(1) as f64;
        if epoch == 0 {
            first_loss = loss;
        }
    }
    let hash = adapter_hash(&den, epochs, LEARN_RATE, steps);
    Ok((
        den,
        TrainingLog {
            epochs,
            steps,
            first_loss,
            final_loss: loss,
            adapter_hash: hash,
            plates: plates.len(),
        },
    ))
}

/// What one training run did. The adapter hash plus the seed below
/// fully determine every picture sampled from it.
#[derive(Debug, Clone)]
pub struct TrainingLog {
    pub epochs: u32,
    pub steps: usize,
    pub first_loss: f64,
    pub final_loss: f64,
    pub adapter_hash: String,
    pub plates: usize,
}

/// DDIM sampling with eta = 0: fully deterministic given the adapter
/// and seed. No stochasticity anywhere — the only randomness is the
/// initial noise, drawn from the seeded RNG, so the same seed draws
/// the same picture down to the last bit.
pub fn sample(den: &Denoiser, seed: u64, width: u32, height: u32, steps: usize) -> Image {
    let steps = steps.max(1);
    let bars = alpha_bars(&betas(steps));
    let mut rng = Rng::new(seed);
    let mut img: Vec<(f64, f64, f64)> = (0..width * height)
        .map(|_| (rng.next_gauss(), rng.next_gauss(), rng.next_gauss()))
        .collect();
    for t in (1..=steps).rev() {
        let t01 = t as f64 / steps as f64;
        let a_t = bars[t - 1].clamp(1e-6, 1.0);
        let a_prev = if t > 1 {
            bars[t - 2].clamp(1e-6, 1.0)
        } else {
            1.0
        };
        for px in img.iter_mut() {
            let (er, eg, eb) = den.predict(*px, t01);
            let pred = [(px.0, er), (px.1, eg), (px.2, eb)];
            let mut out = [0.0; 3];
            for (k, (x, e)) in pred.iter().enumerate() {
                let x0 = (x - (1.0 - a_t).sqrt() * e) / a_t.sqrt();
                out[k] = a_prev.sqrt() * x0 + (1.0 - a_prev).sqrt() * e;
            }
            *px = (out[0], out[1], out[2]);
        }
    }
    let mut out = Image::blank(width, height, Rgb::new(0, 0, 0));
    for (i, (r, g, b)) in img.iter().enumerate() {
        let q = |v: f64| (v * 255.0).round().clamp(0.0, 255.0) as u8;
        out.set(
            i as u32 % width,
            i as u32 / width,
            Rgb::new(q(*r), q(*g), q(*b)),
        );
    }
    out
}

/// Hash of an image's pixels (dataset pinning for receipts).
pub fn plate_hash(img: &Image) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(img.width.to_le_bytes());
    h.update(img.height.to_le_bytes());
    for y in 0..img.height {
        for x in 0..img.width {
            if let Some(p) = img.get(x, y) {
                h.update([p.r, p.g, p.b]);
            }
        }
    }
    format!("{:x}", h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(r: u8, g: u8, b: u8, w: u32, h: u32) -> Image {
        Image::blank(w, h, Rgb::new(r, g, b))
    }

    #[test]
    fn schedule_is_monotone_signal_loss() {
        // More steps in, less signal kept: the forward process must
        // actually destroy information on schedule.
        let bars = alpha_bars(&betas(TRAIN_STEPS));
        assert_eq!(bars.len(), TRAIN_STEPS);
        for w in bars.windows(2) {
            assert!(w[0] > w[1], "signal must decay: {:?}", bars);
        }
        assert!(bars[0] > 0.9 && bars[TRAIN_STEPS - 1] < 0.95);
    }

    #[test]
    fn oracle_denoiser_roundtrips_exactly() {
        // With perfect noise prediction, DDIM eta=0 inverts the
        // forward process: the sampler math is exact, so any blur in
        // practice is the LEARNED model's error, never the sampler's.
        struct Oracle;
        impl Oracle {
            fn predict(
                &self,
                rgb: (f64, f64, f64),
                t01: f64,
                truth: (f64, f64, f64),
            ) -> (f64, f64, f64) {
                let _ = (rgb, t01);
                truth
            }
        }
        let clean = (0.8, 0.4, 0.2);
        let steps = 6;
        let bars = alpha_bars(&betas(steps));
        let oracle = Oracle;
        // Diffuse to pure noise level, then step back with truth.
        let mut rng = Rng::new(11);
        let t = steps;
        let truth = (rng.next_gauss(), rng.next_gauss(), rng.next_gauss());
        let mut x = (
            forward(clean.0, t, &bars, truth.0),
            forward(clean.1, t, &bars, truth.1),
            forward(clean.2, t, &bars, truth.2),
        );
        for tt in (1..=steps).rev() {
            let t01 = tt as f64 / steps as f64;
            let (e0, e1, e2) = oracle.predict(x, t01, truth);
            let a_t = bars[tt - 1].clamp(1e-6, 1.0);
            let a_prev = if tt > 1 {
                bars[tt - 2].clamp(1e-6, 1.0)
            } else {
                1.0
            };
            let back = |xv: f64, e: f64| {
                let x0 = (xv - (1.0 - a_t).sqrt() * e) / a_t.sqrt();
                a_prev.sqrt() * x0 + (1.0 - a_prev).sqrt() * e
            };
            x = (back(x.0, e0), back(x.1, e1), back(x.2, e2));
        }
        for (got, want) in [(x.0, clean.0), (x.1, clean.1), (x.2, clean.2)] {
            assert!(
                (got - want).abs() < 1e-9,
                "roundtrip must be exact: {:?}",
                x
            );
        }
    }

    #[test]
    fn training_learns_and_pins() {
        // Four constant plates: the denoiser must drive loss down and
        // pin a stable adapter hash.
        let plates = vec![
            solid(200, 150, 115, 8, 8),
            solid(60, 80, 120, 8, 8),
            solid(40, 120, 60, 8, 8),
            solid(210, 210, 215, 8, 8),
        ];
        let (a, log) = train(&plates, 7, 12, 6).expect("trains");
        assert!(
            log.final_loss < log.first_loss,
            "loss must fall: {} -> {}",
            log.first_loss,
            log.final_loss
        );
        assert_eq!(log.adapter_hash.len(), 64);
        // Same inputs retrain byte-identical weights and hash.
        let (b, log2) = train(&plates, 7, 12, 6).expect("retrains");
        assert_eq!(log2.adapter_hash, log.adapter_hash);
        assert_eq!(a.flat(), b.flat());
    }

    #[test]
    fn sampling_is_deterministic_and_novel() {
        // Same seed twice: identical pixels. Different seeds: different
        // pictures. Determinism without mode collapse.
        let plates = vec![solid(200, 150, 115, 8, 8), solid(60, 80, 120, 8, 8)];
        let (den, _) = train(&plates, 3, 8, 6).expect("trains");
        let a = sample(&den, 99, 16, 16, 6);
        let b = sample(&den, 99, 16, 16, 6);
        let c = sample(&den, 100, 16, 16, 6);
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(a.get(x, y), b.get(x, y), "same seed must match");
            }
        }
        let mut diff = 0u32;
        for y in 0..16 {
            for x in 0..16 {
                if a.get(x, y) != c.get(x, y) {
                    diff += 1;
                }
            }
        }
        assert!(diff > 10, "seeds must vary the picture ({} px)", diff);
    }

    #[test]
    fn gates_refuse_honestly() {
        assert!(train(&[], 1, 4, 4).is_err());
        assert!(train(&[Image::blank(0, 0, Rgb::new(0, 0, 0))], 1, 4, 4).is_err());
        assert_ne!(
            plate_hash(&solid(1, 2, 3, 4, 4)),
            plate_hash(&solid(1, 2, 4, 4, 4))
        );
    }
}
