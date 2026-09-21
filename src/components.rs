use dioxus::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Chat,
    Settings,
    Symbols,
    Recipes,
}

const NAV_ITEMS: &[(&str, &str, Screen)] = &[
    ("chat", "Chat", Screen::Chat),
    ("tune", "Settings", Screen::Settings),
    ("code", "Symbols", Screen::Symbols),
    ("list", "Recipes", Screen::Recipes),
];

#[component]
pub fn App() -> Element {
    let mut tab = use_signal(|| Screen::Chat);
    let css = asset!("/assets/main.css");

    rsx! {
        document::Link { rel: "stylesheet", href: css }
        div { class: "app",
            div { class: "content",
                match tab() {
                    Screen::Chat => rsx! { crate::screens::Chat {} },
                    Screen::Settings => rsx! { crate::screens::Settings {} },
                    Screen::Symbols => rsx! { crate::screens::Symbols {} },
                    Screen::Recipes => rsx! { crate::screens::Recipes {} },
                }
            }
            nav { class: "bottom-nav",
                for (i, (icon, label, screen)) in NAV_ITEMS.iter().enumerate() {
                    button {
                        class: "nav-btn",
                        class: if tab() == *screen { "nav-btn active" },
                        onclick: move |_| tab.set(*screen),
                        span { class: "nav-icon", "{icon}" }
                        span { class: "nav-label", "{label}" }
                    }
                }
            }
        }
    }
}
