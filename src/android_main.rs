/// This file is included by lib.rs when compiling for Android with the "ui" feature.
/// It provides the `main` function that dioxus's mobile module looks up via dlsym
/// to bootstrap the Dioxus application on Android.

use dioxus::prelude::*;

fn main() {
    dioxus::launch(crate::components::App);
}
