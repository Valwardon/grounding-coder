use dioxus::prelude::*;
use crate::engine::{CodeBot, CodeSymbol};

#[component]
pub fn Chat() -> Element {
    let mut input = use_signal(|| String::new());
    let mut history = use_signal(|| Vec::<(String, bool)>::new());
    let mut working = use_signal(|| false);
    let mut settings = use_signal(crate::llm::load_config_default);

    rsx! {
        div { class: "screen",
            div { class: "screen-header",
                h2 { "Chat" }
                span { class: "subtitle", "LLM is unverified NL→intent only; all code is engine-verified" }
            }
            div { class: "chat-container",
                if history().is_empty() {
                    div { class: "chat-empty",
                        "Ask the bot to write code for your Android project."
                    }
                }
                for (msg, is_user) in history().iter() {
                    div { class: if *is_user { "chat-bubble user" } else { "chat-bubble bot" },
                        pre { "{msg}" }
                    }
                }
                if working() {
                    div { class: "chat-bubble bot", "Working... (deterministic engine running)" }
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
                    onclick: move |_| {
                        let prompt = input();
                        let cfg = settings();
                        if prompt.is_empty() || working() {
                            return;
                        }
                        working.set(true);
                        history.write().push((prompt.clone(), true));
                        input.set(String::new());
                        spawn(async move {
                            let result = run_task_deterministic(&prompt, &cfg).await;
                            history.write().push((result, false));
                            working.set(false);
                        });
                    },
                    "Send"
                }
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
                div { class: "group-title", "LLM Translator (Unverified Layer)" }
                div { class: "field",
                    label { class: "field-label", "OpenRouter API Key" }
                    input {
                        class: "input",
                        r#type: "password",
                        placeholder: "sk-...",
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
                div { class: "field",
                    label { class: "field-label", "Project Path" }
                    input {
                        class: "input",
                        value: ".",
                        oninput: move |_| {},
                    }
                }
                div { class: "field",
                    label { class: "field-label", "Max Retries" }
                    input {
                        class: "input",
                        value: "5",
                        oninput: move |_| {},
                    }
                }
            }
            div { class: "card-block",
                button {
                    class: "btn-primary",
                    onclick: move |_| {
                        if let Ok(path) = crate::llm::config_path() {
                            let _ = crate::llm::save_config(&settings(), &path);
                            status.set("Settings saved.".to_string());
                        }
                    },
                    "Save Settings"
                }
                span { " {status()}" }
            }
            div { class: "hint-text",
                "⚠ The LLM is an unverified layer. It only translates natural language to structured intent JSON. All code is generated and verified deterministically by the engine."
            }
        }
    }
}

#[component]
pub fn Symbols() -> Element {
    let mut symbols = use_signal(|| Vec::<(String, CodeSymbol)>::new());
    let mut status = use_signal(|| String::new());

    use_effect(move || {
        spawn(async move {
            let bot = CodeBot::new(".", 5);
            let syms = bot.symbols();
            symbols.set(syms);
            status.set(format!("Indexed {} symbols.", symbols().len()));
        });
    });

    rsx! {
        div { class: "screen",
            div { class: "screen-header",
                h2 { "Code Symbols" }
                span { class: "subtitle", "{status()}" }
            }
            div { class: "symbol-table",
                for (label, sym) in symbols().iter().take(100) {
                    div { class: "symbol-row",
                        span { class: "symbol-name", "{label} [{sym.kind}]" }
                        span { class: "symbol-meta", "@ {sym.location()}" }
                    }
                }
            }
        }
    }
}

#[component]
pub fn Recipes() -> Element {
    let mut recipes = use_signal(|| Vec::<String>::new());
    let mut status = use_signal(|| String::new());

    use_effect(move || {
        spawn(async move {
            let bot = CodeBot::new(".", 5);
            let rs = bot.recipes();
            recipes.set(rs);
            status.set(format!("{} recipes in log.", recipes().len()));
        });
    });

    rsx! {
        div { class: "screen",
            div { class: "screen-header",
                h2 { "Error → Fix Recipes" }
                span { class: "subtitle", "{status()}" }
            }
            div { class: "recipe-list",
                for r in recipes() {
                    div { class: "recipe", "{r}" }
                }
                if recipes().is_empty() {
                    div { class: "hint-text", "No recipes learned yet. Recipes are added when the bot successfully fixes a compile error." }
                }
            }
        }
    }
}

/// The deterministic path: LLM → intent JSON → engine.
/// The LLM is NOT involved in code generation.
async fn run_task_deterministic(prompt: &str, cfg: &crate::llm::ApiConfig) -> String {
    if cfg.openrouter_key.is_none() {
        return "ERROR: OpenRouter API key not set. Go to Settings.".to_string();
    }

    let llm_client = crate::llm::LlmClient::new(cfg.clone());
    match llm_client.translate(prompt).await {
        Ok(intent) => {
            match serde_json::to_string(&intent) {
                Ok(intent_json) => {
                    let mut bot = CodeBot::new(".", 5);
                    match bot.run_task(&intent_json).await {
                        Ok(result) => format!(
                            "SUCCESS: {}\nFiles: {} | Errors fixed: {} | Budget: {} | Recipes: {}",
                            result.message, result.changes.len(), result.errors_fixed,
                            result.budget_used, result.recipes_learned
                        ),
                        Err(e) => format!("ENGINE: {}", e),
                    }
                }
                Err(e) => format!("SERIALIZE: {}", e),
            }
        }
        Err(e) => format!("LLM: {}", e),
    }
}
