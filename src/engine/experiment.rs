//! The first real experiment: does the dream loop measurably reduce
//! future work?
//!
//! A task family (related problems sharing concepts) runs in two
//! arms — a dreaming arm with a shared knowledge store plus idle
//! passes between tasks, and an amnesiac baseline with a fresh store
//! per task. Every leg records the same metrics; the report compares
//! them.
//!
//! ```text
//! Problem A → solve → verify/learn → idle dream → Problem B → ...
//!      │                                          │
//!      └────────────── metrics ───────────────────┘
//! ```
//!
//! Honesty rules for this module:
//! - Solving is real: every task runs through `CodeBot` on a scratch
//!   project. Only the knowledge store carries state between tasks.
//! - The concept graph (used / unknowns / related) is the experiment
//!   design, supplied by the author like stimuli in a psychology
//!   experiment. Everything else is measured.
//! - Durations are recorded, never asserted (timing is noisy).
//! - Research counts are author-reported for live runs, 0 offline.
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::knowledge::{self, CompletedTask, KnowledgeState, KnowledgeStore};
use super::{AgentOutcome, CodeBot};

/// The experiment design for one task: what the task depends on,
/// what it brushes past, and what nearby concepts the author already
/// suspects. These are the stimuli — everything else is measured.
pub struct TaskObservation {
    pub intent_json: String,
    pub concepts_used: Vec<String>,
    pub unknowns_seen: Vec<String>,
    pub related: Vec<String>,
    /// Oracle fetches the run actually performed (0 when offline).
    pub research_requests: u32,
}

/// Everything one task leg teaches us, measured or derived — never
/// estimated.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskMetrics {
    pub goal: String,
    pub solved: bool,
    pub outcome: String,
    pub duration_ms: u64,
    pub budget_consumed: u32,
    pub research_requests: u32,
    /// Unknowns already in the store from earlier tasks: recognized,
    /// not re-learned. The rediscovery metric is built from these.
    pub recognized: Vec<String>,
    /// Used concepts already Verified: directly reusable knowledge.
    pub reused: Vec<String>,
    /// Gaps this task added that no earlier task had surfaced.
    pub new_questions: Vec<String>,
}

/// What one idle pass did.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DreamMetrics {
    pub investigations: u32,
    pub hypotheses_held: u32,
    pub verified: u32,
    pub failures: u32,
}

/// The full family report. `rediscovery_rate` is the headline:
///
/// ```text
/// recognized / (recognized + new)
/// ```
///
/// 1.0 means every gap was already known — the problems "ceased to
/// exist" as learning work. 0.0 means every task started from zero.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct FamilyReport {
    pub arm: String,
    pub tasks: Vec<TaskMetrics>,
    pub dreams: Vec<DreamMetrics>,
    pub generalized: Vec<String>,
    pub rejected: Vec<String>,
}

impl FamilyReport {
    pub fn recognized_total(&self) -> usize {
        self.tasks.iter().map(|t| t.recognized.len()).sum()
    }

    pub fn new_total(&self) -> usize {
        self.tasks.iter().map(|t| t.new_questions.len()).sum()
    }

    pub fn rediscovery_rate(&self) -> f64 {
        let r = self.recognized_total() as f64;
        let n = self.new_total() as f64;
        if r + n == 0.0 { 1.0 } else { r / (r + n) }
    }

    pub fn solved_all(&self) -> bool {
        !self.tasks.is_empty() && self.tasks.iter().all(|t| t.solved)
    }

    pub fn verified_total(&self) -> usize {
        self.dreams.iter().map(|d| d.verified as usize).sum()
    }
}

fn scratch_project(parent: &Path, tag: &str) -> PathBuf {
    let dir = parent.join(format!("task-{}", tag));
    let src = dir.join("src");
    let _ = std::fs::create_dir_all(&src);
    let _ = std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"gc-exp\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    );
    let _ = std::fs::write(src.join("lib.rs"), "pub fn base() {}\n");
    dir
}

/// Run one arm of the family. Each task solves on a fresh scratch
/// project with a fresh bot — solving starts clean, only the store
/// carries state. With `carry_store`, one store spans all tasks plus
/// `dream_budget` idle investigations between tasks; without it, each
/// task gets a blank store (the amnesiac baseline).
pub async fn run_family(
    arm: &str,
    work_dir: &Path,
    tasks: Vec<(String, TaskObservation)>,
    carry_store: bool,
    dream_budget: u32,
    research: bool,
    bot_budget: u32,
) -> FamilyReport {
    let mut report = FamilyReport {
        arm: arm.to_string(),
        ..FamilyReport::default()
    };
    let shared_dir = work_dir.join(format!("store-{}", arm));
    let n = tasks.len();
    for (idx, (goal, obs)) in tasks.into_iter().enumerate() {
        // Amnesiac arm: a blank store per task, so nothing carries.
        let store_dir = if carry_store {
            shared_dir.clone()
        } else {
            work_dir.join(format!("store-{}-{}", arm, idx))
        };
        let mut store = KnowledgeStore::open(&store_dir);
        let m = solve_and_learn(&goal, &obs, &mut store, work_dir, bot_budget).await;
        report.tasks.push(m);
        // Idle dream between tasks — never after the last one, so the
        // final task's metrics describe solving, not dreaming.
        if carry_store && dream_budget > 0 && idx + 1 < n {
            let outcomes = knowledge::dream_loop(&store_dir, dream_budget, research).await;
            let mut d = DreamMetrics {
                investigations: outcomes.len() as u32,
                ..DreamMetrics::default()
            };
            for o in &outcomes {
                match o.to {
                    KnowledgeState::Verified => d.verified += 1,
                    KnowledgeState::Hypothesis if research => d.failures += 1,
                    KnowledgeState::Hypothesis => d.hypotheses_held += 1,
                    _ => {}
                }
            }
            report.dreams.push(d);
        }
    }
    // Report what the store finally holds — the author decides what
    // to generalize; the harness only reads back the verdicts.
    let final_dir = if carry_store {
        shared_dir
    } else {
        work_dir.join(format!("store-{}-{}", arm, n.saturating_sub(1)))
    };
    let store = KnowledgeStore::open(&final_dir);
    for item in store.all() {
        match item.state {
            KnowledgeState::Generalized => report.generalized.push(item.concept.clone()),
            KnowledgeState::Rejected => report.rejected.push(item.concept.clone()),
            _ => {}
        }
    }
    report
}

async fn solve_and_learn(
    goal: &str,
    obs: &TaskObservation,
    store: &mut KnowledgeStore,
    work_dir: &Path,
    bot_budget: u32,
) -> TaskMetrics {
    // Snapshot the store BEFORE gaps: anything the task mentions that
    // is already here is recognized, not discovered.
    let before: HashSet<String> = store.all().iter().map(|i| i.concept.clone()).collect();
    let verified_before: HashSet<String> = store
        .all()
        .iter()
        .filter(|i| i.state == KnowledgeState::Verified)
        .map(|i| i.concept.clone())
        .collect();

    let project = scratch_project(work_dir, &goal_slug(goal));
    let mut bot = CodeBot::new(&project.to_string_lossy(), bot_budget);
    let start = Instant::now();
    let outcome = bot.run_task(&obs.intent_json).await;
    let duration_ms = start.elapsed().as_millis() as u64;
    let budget_consumed = bot_budget.saturating_sub(bot.budget_remaining());

    let (solved, outcome_str) = match &outcome {
        Ok(AgentOutcome::Success(_)) => (true, "SUCCESS".to_string()),
        Ok(o) => (false, format!("{}", o)),
        Err(e) => (false, format!("ERROR: {}", e)),
    };

    let gaps = knowledge::extract_gaps(
        &CompletedTask {
            goal: goal.to_string(),
            concepts_used: obs.concepts_used.clone(),
            unknowns_seen: obs.unknowns_seen.clone(),
            failed_attempts: budget_consumed,
            related: obs.related.clone(),
        },
        store,
    );

    let mut recognized: Vec<String> = obs
        .unknowns_seen
        .iter()
        .chain(obs.related.iter())
        .filter(|c| before.contains(c.as_str()))
        .cloned()
        .collect();
    recognized.sort();
    recognized.dedup();

    let mut reused: Vec<String> = obs
        .concepts_used
        .iter()
        .filter(|c| verified_before.contains(c.as_str()))
        .cloned()
        .collect();
    reused.sort();
    reused.dedup();

    let mut new_questions: Vec<String> = gaps.into_iter().filter(|g| !before.contains(g)).collect();
    new_questions.sort();

    TaskMetrics {
        goal: goal.to_string(),
        solved,
        outcome: outcome_str,
        duration_ms,
        budget_consumed,
        research_requests: obs.research_requests,
        recognized,
        reused,
        new_questions,
    }
}

fn goal_slug(goal: &str) -> String {
    goal.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}
