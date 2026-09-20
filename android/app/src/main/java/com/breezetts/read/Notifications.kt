package com.breezetts.read

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat

/**
 * The button that is always there.
 *
 * An ongoing notification, not a foreground service: nothing needs to be
 * running for a button to exist, and a service that did nothing but hold a
 * notification would be a battery entry the person has to forgive. It survives
 * everything except a reboot, which the receiver below covers.
 *
 * Its action goes to [ClipboardActivity] rather than straight to a read, for
 * the clipboard-focus reason explained there.
 */
object Trigger {

    private const val CHANNEL = "breeze-trigger"
    private const val ID = 4201

    /** Put the button up, or take it down if it has been turned off. */
    fun apply(context: Context) {
        if (Settings(context).showNotification) show(context) else hide(context)
    }

    fun show(context: Context) {
        val app = context.applicationContext
        val manager = NotificationManagerCompat.from(app)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL,
                app.getString(R.string.notification_channel_trigger),
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = "The button that reads whatever you have copied"
                setShowBadge(false)
            }
            manager.createNotificationChannel(channel)
        }

        val read = PendingIntent.getActivity(
            app, 0, ClipboardActivity.intent(app),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val open = PendingIntent.getActivity(
            app, 1, Intent(app, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )

        val settings = Settings(app)
        val notification: Notification = NotificationCompat.Builder(app, CHANNEL)
            .setSmallIcon(R.drawable.ic_read)
            .setContentTitle("Read on ${settings.name}")
            .setContentText(
                if (settings.paired) "Copy something, then tap Read"
                else "Tap to pair with your Mac"
            )
            .setOngoing(true)
            .setSilent(true)
            .setShowWhen(false)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setContentIntent(open)
            .addAction(R.drawable.ic_read, "Read the clipboard", read)
            .build()

        try {
            manager.notify(ID, notification)
        } catch (denied: SecurityException) {
            // Notifications are off for this app. The selection-toolbar action
            // and the share sheet both still work, so this is not fatal.
        }
    }

    fun hide(context: Context) {
        NotificationManagerCompat.from(context.applicationContext).cancel(ID)
    }
}

/** Put the button back after a restart. */
class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context?, intent: Intent?) {
        if (context == null) return
        if (intent?.action != Intent.ACTION_BOOT_COMPLETED) return
        if (Settings(context).paired) Trigger.apply(context)
    }
}
