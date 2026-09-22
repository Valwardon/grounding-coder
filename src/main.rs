// Desktop app entry point.
//
// The `ui` feature pulls in Dioxus and the mobile/desktop UI. Without it this
// binary still builds, but only prints a hint — so a plain `cargo build`
// (default features) never fails on the UI code.

#[cfg(feature = "ui")]
fn main() {
    dioxus::launch(grounding_coder::components::App);
}

#[cfg(not(feature = "ui"))]
fn main() {
    println!(
        "UI disabled. Build with --features ui to run the desktop app, or run the CLI: cargo run --bin gc -- --help"
    );
}
