use dioxus::prelude::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Screen {
    Chat,
    Code,
    Settings,
}

const NAV_ITEMS: &[(&str, &str, Screen)] = &[
    ("💬", "Chat", Screen::Chat),
    ("📄", "Code", Screen::Code),
    ("⚙️", "Settings", Screen::Settings),
];

#[component]
pub fn App() -> Element {
    let mut tab = use_signal(|| Screen::Chat);
    // Shared app state: settings persist to disk; last_changes + refresh
    // let the Chat tab tell the Code tab what just landed on disk.
    let settings = use_signal(crate::llm::load_config_default);
    let mut last_changes = use_signal(Vec::<String>::new);
    let mut refresh = use_signal(|| 0u64);

    // Theme is inlined at compile time: the Gradle build does not run the
    // manganis asset pipeline, so a linked stylesheet 404s on-device and
    // the app paints unstyled on white. This guarantees first paint.
    rsx! {
        style { {include_str!("../assets/main.css")} }
        div { class: "app",
            div { class: "content",
                match tab() {
                    Screen::Chat => rsx! {
                        crate::screens::Chat {
                            settings,
                            on_done: move |files: Vec<String>| {
                                last_changes.set(files);
                                refresh += 1;
                            },
                        }
                    },
                    Screen::Code => rsx! {
                        crate::screens::Code {
                            settings,
                            last_changes,
                            refresh,
                        }
                    },
                    Screen::Settings => rsx! { crate::screens::Settings { settings } },
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
