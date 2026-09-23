package dev.dioxus.main

import android.os.Bundle

typealias BuildConfig = com.grounding.coder.BuildConfig

/// App entry point. Matches the Dioxus CLI template, plus one deliberate
/// addition: handing the app-private files dir to Rust. The Rust process
/// starts with an unreadable CWD, so without this bridge every engine
/// file op fails with "No such file or directory".
class MainActivity : WryActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Bridge the writable dir before anything needs the filesystem.
        // failures here must never break boot, so swallow everything.
        try {
            nativeInitFilesDir(filesDir.absolutePath)
        } catch (_: Throwable) {
        }
    }

    private external fun nativeInitFilesDir(dir: String)
}
