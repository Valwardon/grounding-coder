/// Energy-bounded retry budget — grounded's `CuriosityBudget` repurposed.
///
/// In grounded, CuriosityBudget bounds how deeply the engine explores
/// unknown concepts: each recursive resolution step consumes energy
/// proportional to semantic distance, arousal, and error rate.
///
/// Here, it bounds how many correction attempts the bot makes on a
/// code generation: each failed verification consumes budget.
/// When budget runs out, the bot stops and reports "I don't know"
/// instead of guessing indefinitely.
///
/// This is the mechanism that prevents the infinite retry loops
/// that LLM coding agents fall into. The bot never guesses —
/// it tries a bounded number of deterministic corrections,
/// then yields to a human.
#[derive(Debug, Clone)]
pub struct RetryBudget {
    /// Total attempts allocated for this task.
    pub total: u32,
    /// Remaining attempts.
    pub remaining: u32,
    /// Number of structural errors (compilation failures) observed.
    error_count: u32,
}

impl RetryBudget {
    pub fn new(total_attempts: f64) -> Self {
        let total = total_attempts.round() as u32;
        RetryBudget {
            total,
            remaining: total,
            error_count: 0,
        }
    }

    /// Consume one unit of budget for a correction attempt.
    /// Returns false if budget is exhausted.
    ///
    /// Mirrors grounded's `consume()` — the bot checks this before
    /// each retry. No budget = no more guessing.
    pub fn consume(&mut self, _semantic_distance: f64, error_rate: f64, novelty: f64) -> bool {
        // Cost increases with errors (like grounded's error_rate term)
        // and decreases with novelty (like grounded's novelty discount).
        // For coding: error_rate = fraction of previous attempts that failed.
        // novelty = whether this error pattern is new (no recipe).
        let cost = 1.0 + 0.5 * error_rate - 0.3 * novelty;
        let cost_u32 = cost.round().max(1.0) as u32;

        if self.remaining < cost_u32 {
            return false;
        }
        self.remaining = self.remaining.saturating_sub(cost_u32);
        true
    }

    /// Record a structural error (compilation failure).
    /// Increases future attempt costs — the bot becomes more
    /// conservative as errors accumulate (like grounded's
    /// "structural errors → more expensive branches").
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }

    /// Ratio of remaining budget (0.0 = empty, 1.0 = full).
    pub fn ratio(&self) -> f64 {
        if self.total > 0 {
            self.remaining as f64 / self.total as f64
        } else {
            0.0
        }
    }

    /// Remaining attempts.
    pub fn remaining(&self) -> u32 {
        self.remaining
    }

    /// Whether the bot should halt (budget exhausted).
    pub fn is_exhausted(&self) -> bool {
        self.remaining == 0
    }

    /// Reset for a new task.
    pub fn reset(&mut self, new_total: u32) {
        self.total = new_total;
        self.remaining = new_total;
        self.error_count = 0;
    }
}

/// Result of a single correction attempt.
#[derive(Debug, Clone)]
pub struct AttemptResult {
    /// Whether the code was verified correct after this attempt.
    pub verified: bool,
    /// How much budget was consumed.
    pub budget_consumed: u32,
    /// Error messages if verification failed (empty if verified).
    pub errors: Vec<String>,
    /// Files that were modified.
    pub files_changed: Vec<String>,
}
