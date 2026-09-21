use dioxus::prelude::*;
use crate::engine::{CodeBot, TaskResult};

#[component]
fn Field(label: String, value: String, oninput: EventHandler<String>) -> Element {
    rsx! {
        div { class: "field",
            label { class: "field-label", "{label}" }
            input {
                class: "input",
                value: "{value}",
                oninput: move |e| oninput(e.value()),
            }
        }
    }
}

#[component]
fn FieldObscure(label: String, value: String, oninput: EventHandler<String>) -> Element {
    rsx! {
        div { class: "field",
            label { class: "field-label", "{label}" }
            input {
                class: "input",
                r#type: "password",
                value: "{value}",
                oninput: move |e| oninput(e.value()),
            }
        }
    }
}

#[component]
fn FieldMultiline(label: String, value: String, oninput: EventHandler<String>) -> Element {
    rsx {
        div { class: "field",
            label { class: "field-label", "{label}" }
            textarea {
                class: "input",
                value: "{value}",
                oninput: move |e| oninput(e.value()),
                rows: "6",
            }
        }
    }
}

#[component]
fn Group(title: String) -> Element {
    rsx! {
        div { class: "group-title", "{title}" }
    }
}

#[component]
pub fn Chat() -> Element {
    let mut input = use_signal(|| String::new());
    let mut history = use_signal(|| Vec::<(String, bool)>::new());
    let mut status = use_signal(|| "Ready.".to_string());
    let mut working = use_signal(|| false);

    let mut settings = use_signal(|| crate::llm::load_config_default());

    let on_send = move |_| {
        let prompt = input();
        if prompt.is_empty() || working() {
            return;
        }
        working.set(true);
        history.modify(|h| h.push((prompt.clone(), true)));
        let p = prompt;
        let cfg = settings();
        spawn(async move {
            let msg = run_task_deterministic(&p, &cfg).await;
            history.modify(|h| h.push((msg, false)));
            working.set(false);
        });
    };

    let messages = history().iter().rev().take(50).rev();

    rsx! {
        div { class: "screen",
            div { class: "screen-header",
                h2 { "Chat" }
                span { class: "subtitle", status() }
            }
            div { class: "chat-container",
                for (msg, is_user) in messages {
                    div { class: if is_user { "chat-bubble user" } else { "chat-bubble bot" },
                        pre { white_space: "pre-wrap", "{msg}" }
                    }
                }
                if working() {
                    div { class: "chat-bubble bot", "Working... (deterministic engine)" }
                }
            }
            div { class: "chat-input-row",
                input {
                    class: "input",
                    placeholder: "Ask the bot to write code (e.g., 'Add a button that vibrates the phone')",
                    value: "{input}",
                    oninput: move |e| input.set(e.value()),
                }
                button { class: "btn-primary", onclick: on_send, "Send" }
            }
        }
    }
}

#[component]
pub fn Settings() -> Element {
    let mut settings = use_signal(crate::llm::load_config_default);
    let mut status = use_signal(|| String::new());

    rsx! {
        div { class: "screen",
            div { class: "screen-header",
                h2 { "Settings" }
                span { class: "subtitle", "API & Project Configuration" }
            }
            div { class: "card-block",
                Group { title: String::from("LLM Translator (Unverified Layer)") }
                FieldObscure {
                    label: "OpenRouter API Key".to_string(),
                    value: settings().openrouter_key.clone().unwrap_or_default(),
                    oninput: move |v| {
                        let mut s = settings();
                        s.openrouter_key = if v.is_empty() { None } else { Some(v) };
                        settings.set(s);
                    },
                }
                Field {
                    label: "Model".to_string(),
                    value: settings().model.clone(),
                    oninput: move |v| {
                        let mut s = settings();
                        s.model = v;
                        settings.set(s);
                    },
                }
                Field {
                    label: "Base URL".to_string(),
                    value: settings().base_url.clone(),
                    oninput: move |v| {
                        let mut s = settings();
                        s.base_url = v;
                        settings.set(s);
                    },
                }
                Field {
                    label: "GitHub Token (optional)".to_string(),
                    value: settings().github_key.clone().unwrap_or_default(),
                    oninput: move |v| {
                        let mut s = settings();
                        s.github_key = if v.is_empty() { None } else { Some(v) };
                        settings.set(s);
                    },
                }
            }

            div { class: "card-block",
                Group { title: String::from("Engine Project") }
                Field {
                    label: "Project Path".to_string(),
                    value: settings().model.clone(), // reuse model field for now
                    oninput: move |v| {
                        let mut s = settings();
                        s.model = v; // placeholder
                        settings.set(s);
                    },
                }
                div { class: "hint-text",
                    "The deterministic engine operates on this directory. \
                    It will scan for .rs/.kt/.java files and build a code symbol graph."
                }
            }

            div { class: "card-block",
                button { class: "btn-primary", onclick: move |_| {
                    if let Ok(path) = crate::llm::config_path() {
                        let _ = crate::llm::save_config(&settings(), &path);
                        status.set("Settings saved.".to_string());
                    } else {
                        status.set("Failed to save settings.".to_string());
                    }
                }, "Save Settings" }
                span { " {status()}" }
            }
        }
    }
}

#[component]
pub fn Symbols() -> Element {
    let mut symbols = use_signal(|| Vec::<(String, crate::engine::CodeSymbol)>::new());
    let mut status = use_signal(|| String::new());

    use_effect(move || {
        status.set("Scanning...".to_string());
        let project = ".";
        spawn(async move {
            let bot = CodeBot::new(project, 5);
            let syms = bot.symbols();
            symbols.set(syms);
            status.set(format!("Indexed {} symbols.", symbols().len()));
        });
        || {}
    });

    let rows: Element = if symbols().is_empty() {
        rsx! { div { "No symbols found. Open a project directory." } }
    } else {
        let rows = symbols().iter().take(100).map(|(label, sym)| {
            rsx! {
                div { class: "symbol-row",
                    span { class: "symbol-name", "{label}" }
                    span { class: "symbol-meta", "[{sym.kind}] @ {sym.location()}" }
                }
            }
        });
        rsx! { {rows} }
    };

    rsx! {
        div { class: "screen",
            div { class: "screen-header",
                h2 { "Code Symbols" }
                span { class: "subtitle", status() }
            }
            div { class: "symbol-table", rows }
        }
    }
}

#[component]
pub fn Recipes() -> Element {
    let mut recipes = use_signal(|| Vec::<String>::new());
    let mut status = use_signal(|| String::new());

    use_effect(move || {
        status.set("Loading...".to_string());
        let project = ".";
        spawn(async move {
            let bot = CodeBot::new(project, 5);
            let rs = bot.recipes();
            recipes.set(rs);
            status.set(format!("{} recipes in log.", recipes().len()));
        });
        || {}
    });

    rsx! {
        div { class: "screen",
            div { class: "screen-header",
                h2 { "Error → Fix Recipes" }
                span { class: "subtitle", status() }
            }
            div { class: "recipe-list",
                for r in recipes() {
                    div { class: "recipe", "{r}" }
                }
            }
        }
    }
}

async fn run_task_deterministic(prompt: &str, cfg: &crate::llm::ApiConfig) -> String {
    if cfg.openrouter_key.is_none() {
        return "ERROR: OpenRouter API key not set. Go to Settings.".to_string();
    }

    let llm_client = crate::llm::LlmClient::new(cfg.clone());
    match llm_client.translate(prompt).await {
        Ok(intent) => {
            let intent_json = serde_json::to_string(&intent)
                .map_err(|e| format!("SERIALIZE ERROR: {}", e))?;
            let mut bot = CodeBot::new(&cfg.model, 5);
            match bot.run_task(&intent_json).await {
                Ok(result) => format!(
                    "SUCCESS\nMessage: {}\nFiles changed: {}\nErrors fixed: {}",
                    result.message, result.changes.len(), result.errors_fixed
                ),
                Err(e) => format!("ENGINE ERROR: {}", e),
            }
        }
        Err(e) => format!("LLM TRANSLATION ERROR: {}", e),
    }
}
