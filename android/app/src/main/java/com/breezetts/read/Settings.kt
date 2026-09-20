package com.breezetts.read

import android.content.Context
import android.content.SharedPreferences
import org.json.JSONObject

/**
 * What this phone knows about the Mac it was paired with.
 *
 * Plain SharedPreferences rather than anything newer: every one of these is
 * read on the way into a request, often from a trigger activity that lives for
 * fifty milliseconds, and a synchronous read is the only thing that is simple
 * there. The whole thing is a dozen fields.
 *
 * The token and the certificate fingerprint are the two that matter. Together
 * they are the entire relationship with the Mac: one says "it is me asking",
 * the other says "and it really is the Mac answering". Both arrive by
 * photographing a code on the Mac's screen, which is why neither ever has to
 * cross the network that they exist to protect.
 */
class Settings(context: Context) {

    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences("breeze", Context.MODE_PRIVATE)

    var host: String
        get() = prefs.getString(HOST, "") ?: ""
        set(value) = prefs.edit().putString(HOST, value).apply()

    var port: Int
        get() = prefs.getInt(PORT, 7860)
        set(value) = prefs.edit().putInt(PORT, value).apply()

    var token: String
        get() = prefs.getString(TOKEN, "") ?: ""
        set(value) = prefs.edit().putString(TOKEN, value).apply()

    var fingerprint: String
        get() = prefs.getString(FINGERPRINT, "") ?: ""
        set(value) = prefs.edit().putString(FINGERPRINT, value).apply()

    var name: String
        get() = prefs.getString(NAME, "the Mac") ?: "the Mac"
        set(value) = prefs.edit().putString(NAME, value).apply()

    /** Hardware addresses to aim a wake packet at, likeliest first. */
    var macAddresses: List<String>
        get() = (prefs.getString(MACS, "") ?: "").split(",").filter { it.isNotBlank() }
        set(value) = prefs.edit().putString(MACS, value.joinToString(",")).apply()

    /** The last address that actually answered, tried before a new lookup. */
    var lastGoodHost: String
        get() = prefs.getString(LAST_HOST, "") ?: ""
        set(value) = prefs.edit().putString(LAST_HOST, value).apply()

    /**
     * A host to try when nothing on this network answers -- a Tailscale address,
     * say. Off by default and never sent a wake packet: a Mac reached this way
     * is one that is already awake, because nothing can wake it from outside
     * its own network.
     */
    var remoteHost: String
        get() = prefs.getString(REMOTE, "") ?: ""
        set(value) = prefs.edit().putString(REMOTE, value.trim()).apply()

    /**
     * A voice on the Mac to read in, instead of the one the Mac is set to.
     *
     * Empty means "whatever the Mac would have chosen", which is what this
     * phone asked for before the setting existed and is still the default. The
     * id is what the Mac understands; the name is kept beside it only so the
     * screen can say which voice is chosen without asking the Mac first --
     * useful precisely when the Mac is not answering.
     *
     * A voice deleted on the Mac leaves an id here that no longer resolves.
     * The Mac refuses that read rather than quietly reading in another voice,
     * and the settings screen says so the next time it manages to ask.
     */
    var voiceId: String
        get() = prefs.getString(VOICE, "") ?: ""
        set(value) = prefs.edit().putString(VOICE, value.trim()).apply()

    var voiceName: String
        get() = prefs.getString(VOICE_NAME, "") ?: ""
        set(value) = prefs.edit().putString(VOICE_NAME, value).apply()

    /** Forget the chosen voice and go back to the Mac's own. */
    fun clearVoice() {
        prefs.edit().remove(VOICE).remove(VOICE_NAME).apply()
    }

    var useLlm: Boolean
        get() = prefs.getBoolean(LLM, false)
        set(value) = prefs.edit().putBoolean(LLM, value).apply()

    var confirmScreenshots: Boolean
        get() = prefs.getBoolean(CONFIRM, true)
        set(value) = prefs.edit().putBoolean(CONFIRM, value).apply()

    /**
     * Whether the ongoing "Read the clipboard" notification is shown.
     *
     * It costs no battery -- an ongoing notification is a row in a list, not a
     * process -- but it is always there, and a permanent notification for
     * something used twice a day is a reasonable thing not to want. The
     * selection-toolbar action and the share sheet work without it.
     */
    var showNotification: Boolean
        get() = prefs.getBoolean(NOTIFY, true)
        set(value) = prefs.edit().putBoolean(NOTIFY, value).apply()

    var powerMode: String
        get() = prefs.getString(POWER, "sleep_when_done") ?: "sleep_when_done"
        set(value) = prefs.edit().putString(POWER, value).apply()

    val paired: Boolean
        get() = host.isNotBlank() && token.isNotBlank() && fingerprint.isNotBlank()

    /** Take everything out of a scanned pairing code. Returns why it failed. */
    fun applyPairingCode(scanned: String): String? {
        val pairing = PairingCode.parse(scanned) ?: return PairingCode.WHY_NOT
        host = pairing.host
        port = pairing.port
        token = pairing.token
        fingerprint = pairing.fingerprint
        name = pairing.name
        macAddresses = pairing.macAddresses
        lastGoodHost = ""
        return null
    }

    private companion object {
        const val HOST = "host"
        const val PORT = "port"
        const val TOKEN = "token"
        const val FINGERPRINT = "fingerprint"
        const val NAME = "name"
        const val MACS = "macs"
        const val LAST_HOST = "lastGoodHost"
        const val REMOTE = "remoteHost"
        const val VOICE = "voiceId"
        const val VOICE_NAME = "voiceName"
        const val LLM = "useLlm"
        const val CONFIRM = "confirmScreenshots"
        const val NOTIFY = "showNotification"
        const val POWER = "powerMode"
    }
}


/** What a pairing code says, once it has been read. */
data class Pairing(
    val host: String,
    val port: Int,
    val token: String,
    val fingerprint: String,
    val name: String,
    val macAddresses: List<String>,
)

/**
 * Reading the code photographed off the Mac's screen.
 *
 * Kept apart from where it is stored so it can be checked without a phone:
 * this is the one piece of the pairing flow that can be wrong in a way nothing
 * later would notice -- a missing fingerprint would simply mean the connection
 * trusted anything, which looks exactly like working.
 */
object PairingCode {

    const val WHY_NOT =
        "That is not a pairing code, or it is missing the address, the token " +
            "or the fingerprint"

    fun parse(scanned: String): Pairing? {
        val payload = try {
            JSONObject(scanned)
        } catch (error: Exception) {
            return null
        }
        val host = payload.optString("host")
        val token = payload.optString("token")
        val fingerprint = payload.optString("fingerprint")
        if (host.isBlank() || token.isBlank() || fingerprint.isBlank()) return null
        val macs = payload.optJSONArray("mac_addresses")
        return Pairing(
            host = host,
            port = payload.optInt("port", 7860),
            token = token,
            fingerprint = fingerprint,
            name = payload.optString("name").ifBlank { "the Mac" },
            macAddresses = (0 until (macs?.length() ?: 0))
                .mapNotNull { index -> macs?.optString(index) }
                .filter { it.isNotBlank() },
        )
    }
}
