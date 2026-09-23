package dev.dioxus.main

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat

/// Foreground service holding the process alive while the engine proves
/// code. Android kills background apps under pressure with no warning —
/// the persistent notification buys survival through minimize and doze.
/// Honest scope: this keeps the SAME process (and its in-progress run)
/// alive. A swipe-away kill still ends the run; that needs out-of-process
/// work, which is a later milestone.
class GroundingService : Service() {

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == "STOP") {
            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }

        val manager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        manager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "Grounding Coder work",
                NotificationManager.IMPORTANCE_LOW,
            ),
        )
        val note = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Grounding Coder working")
            .setContentText("Proving code — minimizing is safe.")
            .setSmallIcon(android.R.drawable.stat_sys_upload)
            .setOngoing(true)
            .build()
        ServiceCompat.startForeground(
            this,
            NOTIFICATION_ID,
            note,
            android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
        )
        return START_STICKY
    }

    companion object {
        const val CHANNEL_ID = "grounding-work"
        const val NOTIFICATION_ID = 1
    }
}
