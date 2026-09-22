// Android/iOS mobile entry point (bin target "android_main", requires "ui").
//
// Dioxus's mobile runtime bootstrap for the `grounding-coder` app on Android.
// This binary is defined in Cargo.toml as `android_main` and is only compiled
// when the `ui` feature is enabled, so a plain `cargo build` is unaffected.

// Dioxus's mobile runtime bootstrap for the `grounding-coder` app on Android.
// This binary is defined in Cargo.toml as `android_main` and is only compiled
// when the `ui` feature is enabled, so a plain `cargo build` is unaffected.

fn main() {
    dioxus::launch(grounding_coder::components::App);
}
