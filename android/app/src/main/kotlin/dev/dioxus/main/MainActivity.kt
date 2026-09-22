package dev.dioxus.main

typealias BuildConfig = com.grounding.coder.BuildConfig

/// App entry point. Matches the Dioxus CLI template exactly: all lifecycle,
/// native boot, WebView creation, and content serving live in [WryActivity]
/// and the Rust cdylib (whose `main` symbol the `start_app` trampoline
/// resolves via `dlsym`). Any override here risks fighting the framework —
/// previous loadUrl/setWebView overrides caused the white screen.
class MainActivity : WryActivity()
