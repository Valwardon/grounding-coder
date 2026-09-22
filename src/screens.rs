use crate::engine::{AgentOutcome, CodeBot};
use crate::llm::ApiConfig;
use dioxus::prelude::*;

// ---------- Chat ----------

/// Chat tab: natural language in, verified engine result out.
/// Uses the project path + budget from Settings; reports changed files
/// back through `on_done` so the Code tab can highlight them.
#[component]
pub fn Chat(settings: Signal<ApiConfig>, on_done: EventHandler<Vec<String>>) -> Element {
    let mut input = use_signal(String::new);
    let mut history = use_signal(Vec::<(String, bool)>::new);
    let mut working = use_signal(|| false);

    rsx! {
        div { class: "screen",
            div { class: "screen-header",
                h2 { "Chat" }
                span { class: "subtitle", "LLM translates → engine proves → oracle verifies" }
            }
            div { class: "chat-container",
                if history().is_empty() {
                    div { class: "chat-empty",
                        "Ask the bot to write or fix code in {settings().project_path}."
                    }
                }
                for (msg, is_user) in history().iter() {
                    div { class: if *is_user { "chat-bubble user" } else { "chat-bubble bot" },
                        pre { "{msg}" }
                    }
                }
                if working() {
                    div { class: "chat-bubble bot", "Working… (deterministic engine running)" }
                }
            }
            div { class: "chat-input-row",
                input {
                    class: "input",
                    placeholder: "e.g., Add a button that vibrates the phone on click",
                    value: "{input}",
                    oninput: move |e| input.set(e.value().clone()),
                }
                button {
                    class: "btn-primary",
                    disabled: working(),
                    onclick: move |_| {
                        let prompt = input();
                        let cfg = settings();
                        if prompt.trim().is_empty() || working() {
                            return;
                        }
                        working.set(true);
                        history.write().push((prompt.clone(), true));
                        input.set(String::new());
                        spawn(async move {
                            let (text, changed) = run_task_deterministic(&prompt, &cfg).await;
                            history.write().push((text, false));
                            working.set(false);
                            on_done.call(changed);
                        });
                    },
                    "Send"
                }
            }
        }
    }
}

/// The deterministic path: LLM → intent JSON → engine.
/// The LLM is NOT involved in code generation. Returns display text plus
/// the files a successful run changed.
async fn run_task_deterministic(prompt: &str, cfg: &ApiConfig) -> (String, Vec<String>) {
    if cfg.openrouter_key.is_none() {
        return (
            "ERROR: OpenRouter API key not set. Open Settings.".to_string(),
            Vec::new(),
        );
    }

    let llm_client = crate::llm::LlmClient::new(cfg.clone());
    match llm_client.translate(prompt).await {
        Ok(intent) => match serde_json::to_string(&intent) {
            Ok(intent_json) => {
                let mut bot = CodeBot::new(&cfg.project_path, cfg.max_retries);
                match bot.run_task(&intent_json).await {
                    Ok(outcome) => {
                        let changed = match &outcome {
                            AgentOutcome::Success(r) => r.changes.clone(),
                            _ => Vec::new(),
                        };
                        (format!("{}", outcome), changed)
                    }
                    Err(e) => (format!("ENGINE: {}", e), Vec::new()),
                }
            }
            Err(e) => (format!("SERIALIZE: {}", e), Vec::new()),
        },
        Err(e) => (format!("LLM: {}", e), Vec::new()),
    }
}

// ---------- Code ----------

/// Code tab: browse project sources, view files, see what the last run
/// changed. Read-only — all writes go through the engine from Chat.
#[component]
pub fn Code(
    settings: Signal<ApiConfig>,
    last_changes: Signal<Vec<String>>,
    refresh: Signal<u64>,
) -> Element {
    let mut files = use_signal(Vec::<String>::new);
    let mut selected = use_signal(|| None::<String>);
    let mut content = use_signal(String::new);
    let mut error = use_signal(String::new);

    // Re-list whenever Chat reports a completed run.
    use_effect(move || {
        let _tick = refresh();
        let project = settings().project_path.clone();
        spawn(async move {
            match list_sources(&project) {
                Ok(list) => {
                    files.set(list);
                    error.set(String::new());
                }
                Err(e) => error.set(e),
            }
        });
    });

    rsx! {
        div { class: "screen",
            div { class: "screen-header",
                h2 { "Code" }
                span { class: "subtitle", "{settings().project_path} · {files().len()} files" }
            }
            if !error().is_empty() {
                div { class: "status-error", "{error()}" }
            }
            div { class: "code-list",
                for f in files().iter() {
                    button {
                        class: if selected().as_deref() == Some(f.as_str()) { "code-item active" } else { "code-item" },
                        onclick: {
                            let f = f.clone();
                            let project = settings().project_path.clone();
                            move |_| {
                                selected.set(Some(f.clone()));
                                let path = std::path::Path::new(&project).join(&f);
                                match std::fs::read_to_string(&path) {
                                    Ok(text) => {
                                        content.set(truncate_file(&text));
                                        error.set(String::new());
                                    }
                                    Err(e) => {
                                        content.set(String::new());
                                        error.set(format!("Cannot read {}: {}", f, e));
                                    }
                                }
                            }
                        },
                        span { class: "code-name", "{f}" }
                        if last_changes().iter().any(|c| c.ends_with(f.as_str())) {
                            span { class: "badge-changed", "●" }
                        }
                    }
                }
                if files().is_empty() && error().is_empty() {
                    div { class: "hint-text", "No source files found. Check the project path in Settings." }
                }
            }
            if let Some(f) = selected() {
                div { class: "code-view",
                    div { class: "code-view-header", "{f}" }
                    pre { "{content()}" }
                }
            }
        }
    }
}

/// Source files under the project, relative paths, capped for mobile.
fn list_sources(project: &str) -> Result<Vec<String>, String> {
    let root = std::path::Path::new(project);
    if !root.exists() {
        return Err(format!("Project path does not exist: {}", project));
    }
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .max_depth(6)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            !matches!(
                name.as_str(),
                ".git"
                    | "target"
                    | "build"
                    | ".gradle"
                    | ".idea"
                    | "node_modules"
                    | "__pycache__"
                    | ".venv"
                    | "venv"
            )
        })
        .flatten()
    {
        let path = entry.path();
        let is_source = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| crate::engine::lang::KNOWN_EXTENSIONS.contains(&ext));
        if is_source && path.is_file() {
            out.push(
                path.strip_prefix(root)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .to_string(),
            );
        }
        if out.len() >= 300 {
            break;
        }
    }
    out.sort();
    Ok(out)
}

fn truncate_file(text: &str) -> String {
    const MAX: usize = 20_000;
    if text.len() <= MAX {
        return text.to_string();
    }
    format!(
        "{}…\n… [truncated, {} bytes total]",
        &text[..MAX],
        text.len()
    )
}

// ---------- Settings ----------

/// Settings tab: LLM key + model, GitHub token, project path, budget.
/// Saved to the on-device config file; never leaves the device except
/// for the API calls the user explicitly makes (Chat).
#[component]
pub fn Settings(settings: Signal<ApiConfig>) -> Element {
    let mut status = use_signal(String::new);
    let mut engine_info = use_signal(String::new);

    rsx! {
        div { class: "screen",
            div { class: "screen-header",
                h2 { "Settings" }
                span { class: "subtitle", "Keys stay on this device" }
            }
            div { class: "card-block",
                div { class: "group-title", "LLM Translator (unverified layer)" }
                div { class: "field",
                    label { class: "field-label", "OpenRouter API Key" }
                    input {
                        class: "input",
                        r#type: "password",
                        placeholder: "sk-or-…",
                        value: "{settings().openrouter_key.clone().unwrap_or_default()}",
                        oninput: move |e| {
                            let mut s = settings();
                            s.openrouter_key = if e.value().is_empty() { None } else { Some(e.value()) };
                            settings.set(s);
                        },
                    }
                }
                div { class: "field",
                    label { class: "field-label", "Model" }
                    input {
                        class: "input",
                        value: "{settings().model}",
                        oninput: move |e| {
                            let mut s = settings();
                            s.model = e.value();
                            settings.set(s);
                        },
                    }
                }
            }
            div { class: "card-block",
                div { class: "group-title", "GitHub (push / releases)" }
                div { class: "field",
                    label { class: "field-label", "Personal Access Token" }
                    input {
                        class: "input",
                        r#type: "password",
                        placeholder: "ghp_… (needs repo scope)",
                        value: "{settings().github_key.clone().unwrap_or_default()}",
                        oninput: move |e| {
                            let mut s = settings();
                            s.github_key = if e.value().is_empty() { None } else { Some(e.value()) };
                            settings.set(s);
                        },
                    }
                }
                div { class: "hint-text",
                    "Used to upload files and releases. The engine only acts on your Chat instructions."
                }
            }
            div { class: "card-block",
                div { class: "group-title", "Project" }
                div { class: "field",
                    label { class: "field-label", "Project Path" }
                    input {
                        class: "input",
                        value: "{settings().project_path}",
                        oninput: move |e| {
                            let mut s = settings();
                            s.project_path = e.value();
                            settings.set(s);
                        },
                    }
                }
                div { class: "field",
                    label { class: "field-label", "Max Retries" }
                    input {
                        class: "input",
                        r#type: "number",
                        value: "{settings().max_retries}",
                        oninput: move |e| {
                            if let Ok(n) = e.value().parse::<u32>() {
                                let mut s = settings();
                                s.max_retries = n.clamp(1, 50);
                                settings.set(s);
                            }
                        },
                    }
                }
            }
            div { class: "card-block",
                button {
                    class: "btn-primary",
                    onclick: move |_| {
                        match crate::llm::config_path() {
                            Ok(path) => match crate::llm::save_config(&settings(), &path) {
                                Ok(()) => status.set(format!("Saved to {}", path)),
                                Err(e) => status.set(format!("Save failed: {}", e)),
                            },
                            Err(e) => status.set(format!("No writable config dir: {}", e)),
                        }
                    },
                    "Save Settings"
                }
                span { " {status()}" }
            }
            div { class: "card-block",
                button {
                    class: "btn-secondary",
                    onclick: move |_| {
                        let project = settings().project_path.clone();
                        spawn(async move {
                            let bot = CodeBot::new(&project, 5);
                            let syms = bot.symbols().len();
                            let recipes = bot.recipes().len();
                            engine_info.set(if syms <= 1 {
                                format!(
                                    "Engine OK — {} symbol in {}. Point Project Path at a source tree to index code.",
                                    syms, project
                                )
                            } else {
                                format!(
                                    "Engine OK — {} symbols indexed in {}, {} recipes learned.",
                                    syms, project, recipes
                                )
                            });
                        });
                    },
                    "Check Engine"
                }
                span { " {engine_info()}" }
            }
            div { class: "hint-text",
                "⚠ The LLM only translates language to intent JSON. All code is generated and verified deterministically by the engine."
            }
        }
    }
}
