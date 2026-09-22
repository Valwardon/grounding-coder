package dev.dioxus.main

import android.os.Bundle
import android.webkit.WebView
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

typealias BuildConfig = com.grounding.coder.BuildConfig

class MainActivity : WryActivity() {

    override fun setWebView(webView: RustWebView) {
        mWebView = webView
        onWebViewCreate(webView)
    }

    override fun onWebViewCreate(webView: WebView) {
        // WebView is created, load the main app URL
        val url = getInitialUrl()
        webView.loadUrl(url)
    }

    override fun onWebViewLoadUrl(url: String) {
        mWebView.loadUrl(url)
    }

    override fun onWebViewLoadUrl(url: String, additionalHttpHeaders: java.util.Map<String, String>) {
        mWebView.loadUrl(url, additionalHttpHeaders)
    }

    private fun getInitialUrl(): String {
        // Load from local assets via WebViewAssetLoader
        return "https://appassets.androidplatform.net/index.html"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // The native library is loaded in WryActivity.companion.init
        // but we need to ensure the Rust entry point is called
        CoroutineScope(Dispatchers.IO).launch {
            // Initialize the Rust engine
            initializeRustEngine()
        }
    }

    private external fun initializeRustEngine()

    companion object {
        init {
            System.loadLibrary("grounding_coder")
        }
    }
}
