package com.breezetts.read

import android.app.Activity
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Toast

/**
 * The three ways a read starts, and what each one has to work around.
 *
 * All of them are activities that do one thing and finish, with no window of
 * their own: the person is in the middle of reading something else, and a flash
 * of an app is not an acceptable price for a button.
 */

/**
 * "Read on Mac" in the text selection toolbar.
 *
 * The best of the three by a distance. It appears wherever text can be
 * selected, anywhere in the operating system, needs no permission at all, and
 * the text arrives in the intent -- so there is nothing to go wrong between
 * selecting it and reading it.
 */
class ProcessTextActivity : Activity() {
    override fun onCreate(state: Bundle?) {
        super.onCreate(state)
        val text = intent?.getCharSequenceExtra(Intent.EXTRA_PROCESS_TEXT)?.toString()
            ?: intent?.getCharSequenceExtra(Intent.EXTRA_PROCESS_TEXT_READONLY)?.toString()
        if (text.isNullOrBlank()) {
            Toast.makeText(this, "Nothing was selected", Toast.LENGTH_SHORT).show()
        } else {
            ReadController.begin(this, ReadController.Source.Text(text))
        }
        finish()
    }
}

/** The share sheet: text from any app, and screenshots. */
class ShareActivity : Activity() {
    override fun onCreate(state: Bundle?) {
        super.onCreate(state)
        val type = intent?.type.orEmpty()
        when {
            type.startsWith("image/") -> shareImage()
            else -> shareText()
        }
        finish()
    }

    private fun shareText() {
        val text = intent?.getCharSequenceExtra(Intent.EXTRA_TEXT)?.toString()
        if (text.isNullOrBlank()) {
            Toast.makeText(this, "There was no text to read", Toast.LENGTH_SHORT).show()
            return
        }
        ReadController.begin(this, ReadController.Source.Text(text))
    }

    private fun shareImage() {
        @Suppress("DEPRECATION")
        val uri = intent?.getParcelableExtra<Uri>(Intent.EXTRA_STREAM)
        if (uri == null) {
            Toast.makeText(this, "There was no image to read", Toast.LENGTH_SHORT).show()
            return
        }
        val image = try {
            contentResolver.openInputStream(uri)?.use { it.readBytes() }
        } catch (error: Exception) {
            null
        }
        if (image == null || image.isEmpty()) {
            Toast.makeText(this, "That image could not be read", Toast.LENGTH_SHORT).show()
            return
        }
        // Screenshots go through a look-first step by default: a model deciding
        // which part of a picture is "the article" is a judgement, and a glance
        // is cheaper than listening to the wrong half of one.
        if (Settings(this).confirmScreenshots) {
            ConfirmActivity.start(this, image)
        } else {
            ReadController.begin(this, ReadController.Source.Screenshot(image))
        }
    }
}

/**
 * The notification button's target.
 *
 * Android 10 and later only let the app holding window focus read the
 * clipboard. A notification action runs with no focus at all, so the button
 * cannot read it directly; this activity exists to be focused for the instant
 * it takes, and then to get out of the way. It is the documented way round the
 * restriction rather than a way through it -- the person pressed a button, and
 * the read is visibly the app's doing.
 */
class ClipboardActivity : Activity() {
    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (!hasFocus) return
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
        val text = clipboard?.primaryClip
            ?.takeIf { it.itemCount > 0 }
            ?.getItemAt(0)
            ?.coerceToText(this)
            ?.toString()
        if (text.isNullOrBlank()) {
            Toast.makeText(this, "The clipboard is empty", Toast.LENGTH_SHORT).show()
        } else {
            ReadController.begin(this, ReadController.Source.Text(text))
        }
        finish()
    }

    companion object {
        fun intent(context: Context): Intent =
            Intent(context, ClipboardActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK)
    }
}
