package dev.dioxus.main

import android.webkit.WebView

typealias BuildConfig = com.grounding.coder.BuildConfig

/// App entry point. All lifecycle and native bootstrapping lives in
/// [WryActivity]: `onCreate` calls the Rust `create()` entry, which builds
/// the WebView and routes back through [setWebView] → [onWebViewCreate].
/// This class only picks the first URL. Do NOT override [setWebView] (final)
/// or touch `mWebView` (private) — both fail compilation by design.
class MainActivity : WryActivity() {

    override fun onWebViewCreate(webView: WebView) {
        // WebView is created, load the main app URL
        webView.loadUrl(getInitialUrl())
    }

    private fun getInitialUrl(): String {
        // Load from local assets via WebViewAssetLoader
        return "https://appassets.androidplatform.net/index.html"
    }
}
