package com.breezetts.read

import android.content.Intent
import androidx.media3.common.ForwardingPlayer
import androidx.media3.common.Player
import androidx.media3.datasource.okhttp.OkHttpDataSource
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.DefaultMediaSourceFactory
import androidx.media3.session.MediaSession
import androidx.media3.session.MediaSessionService
import android.os.Handler
import android.os.Looper

/**
 * The player, and the notification that comes with it for free.
 *
 * Each paragraph of the read is one item in the playlist, which is the whole
 * trick: the previous and next buttons on the lock screen, on a headset, and in
 * the car are already wired to "previous item" and "next item", so they become
 * paragraph skips without this app handling a single button press.
 *
 * Fetching goes through the app's own pinned HTTP client rather than
 * ExoPlayer's default, so audio is held to the same standard as everything
 * else: it can only come from the Mac this phone was paired with.
 */
class PlayerService : MediaSessionService() {

    private var session: MediaSession? = null

    override fun onCreate() {
        super.onCreate()
        val settings = Settings(this)
        val server = Server(this, settings)

        val sources = DefaultMediaSourceFactory(this).setDataSourceFactory(
            OkHttpDataSource.Factory(server.httpClient())
                .setDefaultRequestProperties(
                    mapOf("Authorization" to "Bearer ${settings.token}")
                )
        )

        val player = ExoPlayer.Builder(this)
            .setMediaSourceFactory(sources)
            .setHandleAudioBecomingNoisy(true)
            .build()

        session = MediaSession.Builder(this, DebouncedSkips(player))
            .setId("breeze-read")
            .build()
    }

    override fun onGetSession(controllerInfo: MediaSession.ControllerInfo): MediaSession? =
        session

    override fun onTaskRemoved(rootIntent: Intent?) {
        // Swiping the app away should not silence a read that is still going;
        // the notification is the app as far as this is concerned.
        val player = session?.player
        if (player == null || !player.playWhenReady || player.mediaItemCount == 0) {
            stopSelf()
        }
    }

    override fun onDestroy() {
        session?.run {
            player.release()
            release()
        }
        session = null
        super.onDestroy()
    }
}

/**
 * Skips that wait to see whether another one is coming.
 *
 * Holding the next button should scroll through the document, not synthesize
 * every paragraph on the way past -- the same rule the hotkeys on the Mac
 * follow, for the same reason. Each press moves the target and pushes a
 * deadline out; only when the presses stop does the player actually go there.
 *
 * Two seconds matches the Mac's own debounce, so a read behaves the same way
 * whichever end you are pressing the buttons on.
 */
private class DebouncedSkips(player: Player) : ForwardingPlayer(player) {

    private val handler = Handler(Looper.getMainLooper())
    private var target: Int? = null

    private val land = Runnable {
        val destination = target ?: return@Runnable
        target = null
        val last = wrappedPlayer.mediaItemCount - 1
        if (last >= 0) {
            wrappedPlayer.seekTo(destination.coerceIn(0, last), 0L)
            wrappedPlayer.play()
        }
    }

    override fun seekToNextMediaItem() = aim(+1)

    override fun seekToNext() = aim(+1)

    override fun seekToPreviousMediaItem() = aim(-1)

    override fun seekToPrevious() = aim(-1)

    private fun aim(delta: Int) {
        val from = target ?: wrappedPlayer.currentMediaItemIndex
        target = (from + delta).coerceAtLeast(0)
        handler.removeCallbacks(land)
        handler.postDelayed(land, DEBOUNCE_MS)
    }

    private companion object {
        const val DEBOUNCE_MS = 2_000L
    }
}
