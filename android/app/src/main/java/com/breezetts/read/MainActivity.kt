package com.breezetts.read

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.journeyapps.barcodescanner.ScanContract
import com.journeyapps.barcodescanner.ScanOptions
import java.util.concurrent.Executors

/**
 * The only screen this app has, and most of the time you never open it.
 *
 * It does four things: pair with the Mac, say whether the Mac is answering,
 * hold the handful of settings that are not worth putting anywhere else, and
 * show enough diagnostics to work out what is wrong without plugging the phone
 * into anything.
 *
 * Everything else happens from the selection toolbar, the share sheet and the
 * notification.
 */
class MainActivity : AppCompatActivity() {

    private val worker = Executors.newSingleThreadExecutor()
    private lateinit var settings: Settings
    private var voices: List<Server.Voice> = emptyList()

    private val scanner = registerForActivityResult(ScanContract()) { result ->
        val contents = result?.contents
        if (contents.isNullOrBlank()) return@registerForActivityResult
        val problem = settings.applyPairingCode(contents)
        if (problem != null) {
            Toast.makeText(this, problem, Toast.LENGTH_LONG).show()
        } else {
            Toast.makeText(this, "Paired with ${settings.name}", Toast.LENGTH_SHORT).show()
            Trigger.apply(this)
            refresh()
        }
    }

    override fun onCreate(state: Bundle?) {
        super.onCreate(state)
        setContentView(R.layout.activity_main)
        settings = Settings(this)
        askForNotifications()

        findViewById<Button>(R.id.pair).setOnClickListener {
            scanner.launch(
                ScanOptions()
                    .setPrompt("Point this at the code on the Mac's Phone panel")
                    .setBeepEnabled(false)
                    .setOrientationLocked(false)
            )
        }

        findViewById<Button>(R.id.readClipboard).setOnClickListener {
            startActivity(ClipboardActivity.intent(this))
        }

        findViewById<Button>(R.id.test).setOnClickListener { check() }

        findViewById<CheckBox>(R.id.useLlm).apply {
            isChecked = settings.useLlm
            setOnCheckedChangeListener { _, checked -> settings.useLlm = checked }
        }
        findViewById<CheckBox>(R.id.confirmOcr).apply {
            isChecked = settings.confirmScreenshots
            setOnCheckedChangeListener { _, checked ->
                settings.confirmScreenshots = checked
            }
        }
        findViewById<CheckBox>(R.id.showNotification).apply {
            isChecked = settings.showNotification
            setOnCheckedChangeListener { _, checked ->
                settings.showNotification = checked
                Trigger.apply(this@MainActivity)
            }
        }

        findViewById<EditText>(R.id.remoteHost).apply {
            setText(settings.remoteHost)
            addTextChangedListener(object : TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) = Unit
                override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) = Unit
                override fun afterTextChanged(s: Editable?) {
                    settings.remoteHost = s?.toString().orEmpty()
                }
            })
        }

        setUpPowerModes()
        showVoices()
        refresh()
    }

    override fun onResume() {
        super.onResume()
        if (settings.paired) Trigger.apply(this)
        refresh()
    }

    override fun onDestroy() {
        worker.shutdownNow()
        super.onDestroy()
    }

    // -- the voice --------------------------------------------------------
    /**
     * Which of the Mac's voices to read in.
     *
     * The list lives on the Mac, so this screen has to ask for it, and asking
     * can fail -- the Mac may be asleep, or on another network. So the spinner
     * is drawn from whatever is known right now: the saved choice on its own
     * if that is all there is, and the full library once it arrives. Opening
     * this screen out of range shows the voice you picked rather than an empty
     * list that looks like the setting was lost.
     */
    private fun showVoices(loaded: List<Server.Voice>? = null) {
        if (loaded != null) voices = loaded
        val chosen = settings.voiceId
        val known = voices.map { it.id }

        // A voice chosen on this phone but no longer on the Mac stays in the
        // list until it is changed. Dropping it silently would look like the
        // phone had forgotten it, and the Mac will refuse that read anyway.
        val missing = chosen.isNotBlank() && voices.isNotEmpty() && chosen !in known
        val extra = if (missing || (chosen.isNotBlank() && voices.isEmpty())) {
            listOf(Server.Voice(chosen, settings.voiceName.ifBlank { chosen }))
        } else {
            emptyList()
        }
        val options = extra + voices
        val labels = listOf(MAC_CHOOSES) + options.map { it.name }

        val spinner = findViewById<Spinner>(R.id.voice)
        spinner.onItemSelectedListener = null  // redrawing is not a choice
        spinner.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, labels
        )
        val at = options.indexOfFirst { it.id == chosen }
        spinner.setSelection(if (chosen.isBlank() || at < 0) 0 else at + 1)
        spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
            override fun onItemSelected(
                parent: AdapterView<*>?, view: android.view.View?, position: Int, id: Long,
            ) {
                // A spinner reports a selection when it first lays itself out,
                // and that report is the one the setSelection above asked for.
                // Without this the act of showing the screen would hand the
                // saved voice back to the Mac.
                val picked = if (position == 0) "" else options[position - 1].id
                if (picked == settings.voiceId) return
                if (picked.isBlank()) {
                    settings.clearVoice()
                } else {
                    settings.voiceId = picked
                    settings.voiceName = options[position - 1].name
                }
                describeVoice(missingOnMac = false)
            }
        }
        describeVoice(missingOnMac = missing)
    }

    private fun describeVoice(missingOnMac: Boolean) {
        findViewById<TextView>(R.id.voiceState).text = when {
            missingOnMac ->
                "${settings.voiceName.ifBlank { settings.voiceId }} is no longer on " +
                    "${settings.name}. Reads will be refused until you pick another."
            settings.voiceId.isBlank() ->
                "Reads use whatever ${settings.name} is set to."
            else ->
                "Every read from this phone asks for ${settings.voiceName}, " +
                    "whatever ${settings.name} is set to."
        }
    }

    /** Ask the Mac for its voice library, quietly; the screen works without it. */
    private fun loadVoices() {
        if (!settings.paired) return
        val server = Server(this, settings)
        worker.execute {
            val found = runCatching {
                server.voices(server.locate(wake = false))
            }.getOrNull() ?: return@execute
            runOnUiThread { showVoices(found) }
        }
    }

    // -- the pieces -------------------------------------------------------
    private fun setUpPowerModes() {
        val spinner = findViewById<Spinner>(R.id.powerMode)
        spinner.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, MODE_LABELS
        )
        spinner.setSelection(MODE_VALUES.indexOf(settings.powerMode).coerceAtLeast(0))
        spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
            override fun onItemSelected(
                parent: AdapterView<*>?, view: android.view.View?, position: Int, id: Long,
            ) {
                val mode = MODE_VALUES[position]
                if (mode == settings.powerMode) return
                settings.powerMode = mode
                sendPowerMode(mode)
            }
        }
    }

    private fun sendPowerMode(mode: String) {
        val server = Server(this, settings)
        worker.execute {
            try {
                val endpoint = server.locate(wake = true)
                val state = server.setPowerMode(endpoint, mode)
                val trouble = state.optString("error")
                runOnUiThread {
                    findViewById<TextView>(R.id.powerState).text =
                        if (trouble.isNotBlank() && trouble != "null") trouble
                        else describePower(state)
                }
            } catch (error: Exception) {
                runOnUiThread { say(error.message ?: "The Mac did not answer") }
            }
        }
    }

    private fun refresh() {
        val status = findViewById<TextView>(R.id.status)
        status.text = if (settings.paired) {
            "Paired with ${settings.name} at ${settings.host}:${settings.port}"
        } else {
            "Not paired. Open the Phone panel in the Mac's web page and scan the code."
        }
        findViewById<TextView>(R.id.diagnostics).text = buildString {
            append("fingerprint  ").append(settings.fingerprint.take(16).ifBlank { "—" })
            append("\nlast good    ").append(settings.lastGoodHost.ifBlank { "—" })
            append("\nwake to      ").append(settings.macAddresses.firstOrNull() ?: "—")
            append("\nremote host  ").append(settings.remoteHost.ifBlank { "off" })
        }
        if (settings.paired) {
            check(quiet = true)
            loadVoices()
        }
    }

    /** Find the Mac and report exactly what happened, which is the point. */
    private fun check(quiet: Boolean = false) {
        if (!settings.paired) return
        val status = findViewById<TextView>(R.id.status)
        if (!quiet) status.text = "Looking for ${settings.name}…"
        val server = Server(this, settings)
        worker.execute {
            val started = System.currentTimeMillis()
            try {
                val endpoint = server.locate(wake = !quiet) { progress ->
                    runOnUiThread { status.text = progress }
                }
                val took = System.currentTimeMillis() - started
                val power = runCatching { server.power(endpoint) }.getOrNull()
                runOnUiThread {
                    status.text = "${settings.name} answered at ${endpoint.host} in ${took} ms"
                    if (power != null) {
                        findViewById<TextView>(R.id.powerState).text = describePower(power)
                    }
                }
            } catch (error: Exception) {
                runOnUiThread { status.text = error.message ?: "No answer" }
            }
        }
    }

    private fun describePower(state: org.json.JSONObject): String {
        val battery = state.optJSONObject("battery")
        val percent = battery?.optInt("percent", -1) ?: -1
        val charging = battery?.optBoolean("charging", false) ?: false
        val holding = state.optBoolean("holding_awake", false)
        val womp = state.optInt("womp", 0) == 1
        return buildString {
            append(if (holding) "Staying awake" else "Sleeping normally")
            if (percent >= 0) {
                append(" · battery ").append(percent).append('%')
                if (charging) append(" (charging)")
            }
            append(" · wake for network ").append(if (womp) "on" else "OFF")
        }
    }

    private fun askForNotifications() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
        val granted = ContextCompat.checkSelfPermission(
            this, Manifest.permission.POST_NOTIFICATIONS
        ) == PackageManager.PERMISSION_GRANTED
        if (!granted) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1
            )
        }
    }

    private fun say(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    private companion object {
        /** The first entry in the voice list: no instruction, the old behaviour. */
        const val MAC_CHOOSES = "The voice the Mac is set to"

        val MODE_VALUES = listOf("off", "keep_on", "sleep_when_done")
        val MODE_LABELS = listOf(
            "Leave the Mac's own settings alone",
            "Keep on — stay awake with the lid closed",
            "Sleep when done — sleep after a read",
        )
    }
}
