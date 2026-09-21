pub mod engine;
pub mod llm;

#[cfg(feature = "ui")]
pub mod components;

#[cfg(feature = "ui")]
pub mod screens;

pub use engine::CodeBot;
pub use engine::TaskResult;

/// Android mobile entry point.
/// When compiled as a cdylib with the "ui" feature, dioxus provides
/// the android_main glue via the tao/wry android binding.
/// We expose the main() function that dioxus's mobile module looks up.
#[cfg(all(feature = "ui", target_os = "android"))]
include!("android_main.rs");
