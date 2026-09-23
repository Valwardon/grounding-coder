package dev.dioxus.main

import android.os.Bundle
import android.webkit.WebView

typealias BuildConfig = com.grounding.coder.BuildConfig

/// App entry point. Matches the Dioxus CLI template, plus two deliberate
/// additions, both guarded to never break boot:
/// 1. files-dir bridge — the Rust process starts with an unreadable CWD,
///    so the engine needs the app-private dir handed over via JNI.
/// 2. JS bridge registration — `window.Grounding.startWork/stopWork`
///    lets the Rust UI hold a foreground service while engine runs land.
class MainActivity : WryActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            nativeInitFilesDir(filesDir.absolutePath)
        } catch (_: Throwable) {
        }
    }

    override fun onWebViewCreate(webView: WebView) {
        try {
            webView.addJavascriptInterface(GroundingBridge(this), "Grounding")
        } catch (_: Throwable) {
        }
    }

    private external fun nativeInitFilesDir(dir: String)
}
