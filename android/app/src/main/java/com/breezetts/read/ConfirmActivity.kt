package com.breezetts.read

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import java.util.concurrent.Executors

/**
 * A look at what was read out of a screenshot, before it is read out loud.
 *
 * The model on the Mac is asked to leave out the clock, the tab bar and the
 * buttons, and it is good at it -- but "which part of this picture did you
 * mean" is a judgement, and a wrong one costs a minute of listening to a
 * cookie banner. Two seconds of looking is cheaper, and the text is editable,
 * so a stray line goes away with a swipe rather than with a better prompt.
 *
 * Turned off in settings for anyone who would rather take the chance.
 */
class ConfirmActivity : AppCompatActivity() {

    private val worker = Executors.newSingleThreadExecutor()

    override fun onCreate(state: Bundle?) {
        super.onCreate(state)
        setContentView(R.layout.activity_confirm)

        // The screenshot travels as a file rather than inside the intent: a
        // phone screenshot is comfortably over the megabyte an intent can
        // carry, and the failure for going over it is the whole transaction
        // being dropped rather than anything legible.
        val handover = intent?.getStringExtra(EXTRA_IMAGE)?.let { java.io.File(it) }
        val image = handover?.takeIf { it.isFile }?.readBytes()
        handover?.delete()
        if (image == null || image.isEmpty()) {
            Toast.makeText(this, "No screenshot came through", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        val note = findViewById<TextView>(R.id.note)
        val field = findViewById<EditText>(R.id.text)
        val read = findViewById<Button>(R.id.read)

        note.text = "Reading the screenshot…"
        read.isEnabled = false
        findViewById<Button>(R.id.cancel).setOnClickListener { finish() }
        read.setOnClickListener {
            val text = field.text?.toString().orEmpty().trim()
            if (text.isBlank()) {
                Toast.makeText(this, "There is nothing to read", Toast.LENGTH_SHORT).show()
            } else {
                ReadController.begin(this, ReadController.Source.Text(text))
                finish()
            }
        }

        val settings = Settings(this)
        val server = Server(this, settings)
        worker.execute {
            try {
                val endpoint = server.locate(wake = true)
                val result = server.extractText(endpoint, image)
                val text = result.optString("text")
                val filtered = result.optBoolean("filtered", false)
                runOnUiThread {
                    field.setText(text)
                    read.isEnabled = text.isNotBlank()
                    note.text = when {
                        text.isBlank() ->
                            "Nothing readable was found in that screenshot."
                        filtered ->
                            "Interface text was left out. Delete anything else you " +
                                "do not want read."
                        else ->
                            "Read with the system's text recognition, which does not " +
                                "leave out the clock — tidy it up before reading."
                    }
                }
            } catch (error: Exception) {
                runOnUiThread {
                    note.text = error.message ?: "The screenshot could not be read"
                    read.isEnabled = false
                }
            }
        }
    }

    override fun onDestroy() {
        worker.shutdownNow()
        super.onDestroy()
    }

    companion object {
        private const val EXTRA_IMAGE = "image"

        fun start(context: Context, image: ByteArray) {
            val handover = java.io.File(context.cacheDir, "screenshot-handover.png")
            handover.writeBytes(image)
            context.startActivity(
                Intent(context, ConfirmActivity::class.java)
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    .putExtra(EXTRA_IMAGE, handover.absolutePath)
            )
        }
    }
}
