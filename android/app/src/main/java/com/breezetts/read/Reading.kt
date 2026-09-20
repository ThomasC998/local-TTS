package com.breezetts.read

import android.content.ComponentName
import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.media3.common.MediaItem
import androidx.media3.common.MediaMetadata
import androidx.media3.common.Player
import androidx.media3.session.MediaController
import androidx.media3.session.SessionToken
import com.google.common.util.concurrent.ListenableFuture
import org.json.JSONObject
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * One read, from a tap on the phone to audio coming out of it.
 *
 * The shape of the job is: find the Mac, hand it the text, and turn the
 * paragraphs it reports into a playlist. The playlist is the interesting part,
 * because it is built twice over. Whatever paragraphs exist when the read
 * starts go in immediately so playback can begin; and then, while the language
 * model is still writing, the manifest is polled and new paragraphs are added
 * to the end as they appear. The person hears the first paragraph while the
 * last one is still being written.
 *
 * Everything blocking happens on one background thread. The player is touched
 * only on the main thread, which is what Media3 requires.
 */
object ReadController {

    private const val TAG = "BreezeRead"
    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private val active = AtomicReference<Live?>(null)

    /** The one binding to the playback service, while there is a read to drive. */
    private var connection: ListenableFuture<MediaController>? = null

    private data class Live(
        val readId: String,
        val endpoint: Server.Endpoint,
        val server: Server,
        /** Set once no more paragraphs are coming, however that came about. */
        val complete: AtomicBoolean = AtomicBoolean(false),
    )

    /** What is being read: text the phone already has, or a picture of some. */
    sealed class Source {
        data class Text(val text: String) : Source()
        data class Screenshot(val image: ByteArray) : Source()
    }

    /**
     * Start reading. Returns immediately; progress arrives as short toasts.
     *
     * Toasts rather than a screen because every trigger is a single tap from
     * somewhere else on the phone -- a selection toolbar, a share sheet, a
     * notification -- and putting an activity in front of the person at that
     * moment would be taking over their phone to tell them something they can
     * hear for themselves in a second.
     */
    fun begin(context: Context, source: Source) {
        val app = context.applicationContext
        val settings = Settings(app)
        if (!settings.paired) {
            say(app, "Pair this phone with your Mac first")
            return
        }
        say(app, "Asking ${settings.name}…")

        worker.execute {
            val server = Server(app, settings)
            try {
                val endpoint = server.locate(wake = true) { progress -> say(app, progress) }
                endPrevious(server)

                val started = when (source) {
                    is Source.Text ->
                        server.startRead(endpoint, source.text, settings.useLlm)
                    is Source.Screenshot ->
                        server.startReadFromImage(endpoint, source.image, settings.useLlm)
                }

                val readId = started.optString("read_id")
                if (readId.isBlank()) throw Server.ServerError("The Mac started no read")
                noteSubstitution(app, settings, started)
                val live = Live(readId, endpoint, server)
                active.set(live)

                val title = started.optString("preview").take(60).ifBlank { "Reading" }
                play(app, server, endpoint, live, started.optInt("paragraphs", 1), title)
                try {
                    follow(app, server, endpoint, readId, started, title)
                } finally {
                    // However following ended -- the model finished, gave up, or
                    // failed -- nothing more is coming, which is what lets the
                    // player treat running out of paragraphs as the end.
                    live.complete.set(true)
                }
            } catch (error: Exception) {
                Log.w(TAG, "The read failed", error)
                say(app, error.message ?: "That did not work")
            }
        }
    }

    /**
     * The Mac read this in its own voice, because the one asked for is gone.
     *
     * Said once and then put right: a voice deleted on the Mac is not coming
     * back, so the phone stops asking for it. Repeating the message on every
     * read would be nagging about something the phone can simply fix, and
     * leaving the dead id in place is what would make it repeat.
     */
    private fun noteSubstitution(app: Context, settings: Settings, started: JSONObject) {
        if (!started.optBoolean("voice_substituted")) return
        val gone = settings.voiceName.ifBlank { settings.voiceId }.ifBlank { "That voice" }
        val used = started.optString("voice_name").ifBlank { "its own voice" }
        settings.clearVoice()
        say(app, "$gone is gone from ${settings.name} — reading in $used")
    }

    /** Stop whatever is playing and let the Mac drop the audio it was holding. */
    fun stop(context: Context) {
        val app = context.applicationContext
        val live = active.getAndSet(null) ?: return
        worker.execute { live.server.endRead(live.endpoint, live.readId) }
        main.post {
            controller(app) { player ->
                player.stop()
                player.clearMediaItems()
            }
            // Nothing left to drive: hand the binding back so the service can
            // stop and take its player with it.
            main.post { release() }
        }
    }

    private fun endPrevious(server: Server) {
        val previous = active.getAndSet(null) ?: return
        server.endRead(previous.endpoint, previous.readId)
    }

    // -- the playlist -----------------------------------------------------
    private fun play(
        app: Context,
        server: Server,
        endpoint: Server.Endpoint,
        live: Live,
        paragraphs: Int,
        title: String,
    ) {
        val items = (0 until maxOf(paragraphs, 1)).map { index ->
            item(server, endpoint, live.readId, index, title)
        }
        main.post {
            controller(app) { player ->
                player.addListener(ends(app, live))
                player.setMediaItems(items)
                player.prepare()
                player.play()
            }
        }
    }

    /**
     * Notice the read ending, and let go of the player when it does.
     *
     * The service stops itself when playback ends, but a service somebody is
     * still bound to is not destroyed -- and this app binds to its own. So
     * without this the read finishes, the audio stops, and the process stays
     * up at service priority with an ExoPlayer in it until Android needs the
     * memory for something else.
     *
     * Running out of paragraphs only means the read is over once the Mac has
     * said there are no more coming. While a language model is still writing,
     * the player can reach the end of what exists and be given more a moment
     * later, and that is not something to shut down over.
     */
    private fun ends(app: Context, live: Live) = object : Player.Listener {
        override fun onPlaybackStateChanged(state: Int) {
            if (state != Player.STATE_ENDED || !live.complete.get()) return
            active.compareAndSet(live, null)
            release()
        }
    }

    /**
     * Keep adding paragraphs while the language model is still producing them.
     *
     * Polling rather than a stream: the manifest is a few bytes, the answer
     * changes every few seconds at most, and a poll that fails is a poll that
     * simply happens again -- where a dropped stream would need reconnecting
     * logic to say the same thing.
     */
    private fun follow(
        app: Context,
        server: Server,
        endpoint: Server.Endpoint,
        readId: String,
        started: JSONObject,
        title: String,
    ) {
        var known = maxOf(started.optInt("paragraphs", 1), 1)
        var final = started.optBoolean("final", false)
        val deadline = System.currentTimeMillis() + FOLLOW_LIMIT_MS
        var wait = FIRST_POLL_MS

        while (!final && System.currentTimeMillis() < deadline) {
            Thread.sleep(wait)
            // Backing off matters on a phone: a model still writing after five
            // minutes is not going to produce a paragraph in the next second
            // either, and a fixed interval would mean thousands of requests
            // and a radio that never gets to sleep.
            wait = minOf(wait * 2, SLOWEST_POLL_MS)
            if (active.get()?.readId != readId) return  // replaced by another read
            val manifest = try {
                server.manifest(endpoint, readId)
            } catch (error: Exception) {
                Log.w(TAG, "Lost track of the read", error)
                return
            }
            manifest.optString("error").takeIf { it.isNotBlank() && it != "null" }?.let {
                say(app, it)
                return
            }
            final = manifest.optBoolean("final", false)
            val now = manifest.optInt("paragraphs", known)
            if (now > known) {
                wait = FIRST_POLL_MS  // it is producing again; look sooner
                val added = (known until now).map { index ->
                    item(server, endpoint, readId, index, title)
                }
                known = now
                main.post { controller(app) { player -> player.addMediaItems(added) } }
            }
        }
    }

    private fun item(
        server: Server,
        endpoint: Server.Endpoint,
        readId: String,
        index: Int,
        title: String,
    ): MediaItem = MediaItem.Builder()
        .setMediaId("$readId#$index")
        .setUri(server.paragraphUrl(endpoint, readId, index))
        .setMediaMetadata(
            MediaMetadata.Builder()
                .setTitle("Paragraph ${index + 1}")
                .setArtist(title)
                .build()
        )
        .build()

    /**
     * Do something with the player, connecting to the service if need be.
     *
     * One connection, kept and reused. A MediaController is a binding to the
     * playback service, and this is called for every paragraph the language
     * model adds to a growing read -- so building a fresh one each time, and
     * never releasing it, would leave a long read holding twenty bindings that
     * between them keep the service, and its ExoPlayer, alive indefinitely.
     *
     * Main thread only, which is where Media3 requires a controller to be used.
     */
    private fun controller(app: Context, action: (MediaController) -> Unit) {
        val existing = connection
        val future = if (existing != null && !existing.isCancelled) {
            existing
        } else {
            val token = SessionToken(app, ComponentName(app, PlayerService::class.java))
            MediaController.Builder(app, token).buildAsync().also { connection = it }
        }
        future.addListener({
            try {
                val player = future.get()
                if (player.isConnected) {
                    action(player)
                } else {
                    // The service went away -- after a read ended, most likely.
                    // Drop the stale binding and make a fresh one next time.
                    release()
                    Log.i(TAG, "The player had stopped; reconnecting on the next read")
                }
            } catch (error: Exception) {
                Log.w(TAG, "Could not reach the player", error)
                release()
            }
        }, ContextCompat.getMainExecutor(app))
    }

    /** Let go of the player. Main thread. */
    private fun release() {
        connection?.let { MediaController.releaseFuture(it) }
        connection = null
    }

    private fun say(app: Context, message: String) {
        main.post { Toast.makeText(app, message, Toast.LENGTH_SHORT).show() }
    }

    /** However long a document is, the model is not still writing it an hour on. */
    private const val FOLLOW_LIMIT_MS = 60 * 60 * 1000L
    private const val FIRST_POLL_MS = 1_500L
    private const val SLOWEST_POLL_MS = 15_000L
}
