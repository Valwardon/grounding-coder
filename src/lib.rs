pub mod android;
pub mod engine;
pub mod github;
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

/// App-private files dir, handed over from Kotlin at startup.
///
/// Android gives Rust processes an unreadable CWD, so every engine file
/// op fails without this. `MainActivity.onCreate` calls
/// `nativeInitFilesDir(filesDir.absolutePath)` exactly once; before that
/// (and on non-Android hosts) this is `None` and paths pass through.
#[cfg(target_os = "android")]
static APP_FILES_DIR: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

/// JNI entry called from `MainActivity.onCreate`. Trivial by design: parse
/// one string, store it. Nothing here can fail the boot.
#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
pub extern "C" fn Java_dev_dioxus_main_MainActivity_nativeInitFilesDir(
    mut env: jni::JNIEnv,
    _class: jni::objects::JClass,
    dir: jni::objects::JString,
) {
    if let Ok(s) = env.get_string(&dir) {
        let path: String = s.into();
        let _ = APP_FILES_DIR.set(std::path::PathBuf::from(path));
    }
}

/// Resolve the engine's working dir: bare `"."`/empty means "the app's
/// private dir" when the bridge has handed it over, else passthrough.
pub fn resolve_project_dir(configured: &str) -> String {
    let t = configured.trim();
    if t == "." || t.is_empty() {
        #[cfg(target_os = "android")]
        if let Some(d) = APP_FILES_DIR.get() {
            return d.to_string_lossy().to_string();
        }
        return ".".to_string();
    }
    configured.to_string()
}
