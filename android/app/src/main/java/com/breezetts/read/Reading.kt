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
import androidx.media3.session.MediaController
import androidx.media3.session.SessionToken
import org.json.JSONObject
import java.util.concurrent.Executors
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

    private data class Live(
        val readId: String,
        val endpoint: Server.Endpoint,
        val server: Server,
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
                active.set(Live(readId, endpoint, server))

                val title = started.optString("preview").take(60).ifBlank { "Reading" }
                play(app, server, endpoint, readId, started.optInt("paragraphs", 1), title)
                follow(app, server, endpoint, readId, started, title)
            } catch (error: Exception) {
                Log.w(TAG, "The read failed", error)
                say(app, error.message ?: "That did not work")
            }
        }
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
        readId: String,
        paragraphs: Int,
        title: String,
    ) {
        val items = (0 until maxOf(paragraphs, 1)).map { index ->
            item(server, endpoint, readId, index, title)
        }
        main.post {
            controller(app) { player ->
                player.setMediaItems(items)
                player.prepare()
                player.play()
            }
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

        while (!final && System.currentTimeMillis() < deadline) {
            Thread.sleep(1_500)
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
     * Must run on the main thread, and the connection is cheap enough to make
     * per action -- the service and its session outlive any one of them.
     */
    private fun controller(app: Context, action: (MediaController) -> Unit) {
        val token = SessionToken(app, ComponentName(app, PlayerService::class.java))
        val future = MediaController.Builder(app, token).buildAsync()
        future.addListener({
            try {
                action(future.get())
            } catch (error: Exception) {
                Log.w(TAG, "Could not reach the player", error)
            }
        }, ContextCompat.getMainExecutor(app))
    }

    private fun say(app: Context, message: String) {
        main.post { Toast.makeText(app, message, Toast.LENGTH_SHORT).show() }
    }

    /** However long a document is, the model is not still writing it an hour on. */
    private const val FOLLOW_LIMIT_MS = 60 * 60 * 1000L
}
