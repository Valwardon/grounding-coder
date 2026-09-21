#![cfg(target_os = "android")]

use dioxus::prelude::*;

dioxus::mobile::bootstrapper(grounding_coder::components::App);
