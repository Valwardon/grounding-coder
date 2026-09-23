pub mod android;
pub mod engine;
pub mod http;
pub mod knowledge;
pub mod llm;
pub mod oracle;

#[cfg(feature = "ui")]
pub mod components;

#[cfg(feature = "ui")]
pub mod screens;

pub use engine::{AgentOutcome, BlockReason, CodeBot, TaskResult};

/// Android entry point for the Dioxus mobile runtime.
///
/// `dioxus-desktop`'s `start_app()` trampoline locates this symbol with
/// `dlsym(RTLD_DEFAULT, "main")` and **aborts the process when it is
/// missing**. It must live in the cdylib itself: the `android_main` and
/// desktop `main` bin targets are never linked into the `.so`, so without
/// this the app shows a white screen and then dies. Same binary, same
/// process — the UI and the engine stay together with no IPC.
#[cfg(all(target_os = "android", feature = "ui"))]
#[unsafe(no_mangle)]
pub extern "C" fn main() {
    dioxus::launch(crate::components::App);
}
