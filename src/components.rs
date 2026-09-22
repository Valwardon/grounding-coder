use dioxus::prelude::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Screen {
    Chat,
    Settings,
    Symbols,
    Recipes,
}

const NAV_ITEMS: &[(&str, &str, Screen)] = &[
    ("chat", "Chat", Screen::Chat),
    ("tune", "Settings", Screen::Settings),
    ("search", "Symbols", Screen::Symbols),
    ("book", "Recipes", Screen::Recipes),
];

#[component]
pub fn App() -> Element {
    let mut tab = use_signal(|| Screen::Chat);

    rsx! {
        document::Stylesheet { href: asset!("/assets/main.css") }
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
                for (icon, label, screen) in NAV_ITEMS.iter() {
                    button {
                        class: if tab() == *screen { "nav-btn active" } else { "nav-btn" },
                        onclick: move |_| tab.set(*screen),
                        span { class: "nav-icon", "{icon}" }
                        span { class: "nav-label", "{label}" }
                    }
                }
            }
        }
    }
}
