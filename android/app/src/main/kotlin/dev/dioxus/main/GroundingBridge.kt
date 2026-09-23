package dev.dioxus.main

import android.app.Activity
import android.content.Intent
import android.webkit.JavascriptInterface
import androidx.core.content.ContextCompat
import java.lang.ref.WeakReference

/// JS bridge the Rust UI drives around each engine run:
/// `Grounding.startWork()` when Chat sends, `Grounding.stopWork()` when the
/// outcome lands. Best-effort by contract — every call is guarded so a
/// missing service can never break the app.
class GroundingBridge(activity: Activity) {
    private val activityRef = WeakReference(activity)

    @JavascriptInterface
    fun startWork() {
        val activity = activityRef.get() ?: return
        try {
            ContextCompat.startForegroundService(
                activity,
                Intent(activity, GroundingService::class.java).setAction("WORK"),
            )
        } catch (_: Throwable) {
        }
    }

    @JavascriptInterface
    fun stopWork() {
        val activity = activityRef.get() ?: return
        try {
            activity.stopService(Intent(activity, GroundingService::class.java))
        } catch (_: Throwable) {
        }
    }
}
