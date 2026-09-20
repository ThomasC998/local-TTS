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

    var useLlm: Boolean
        get() = prefs.getBoolean(LLM, false)
        set(value) = prefs.edit().putBoolean(LLM, value).apply()

    var confirmScreenshots: Boolean
        get() = prefs.getBoolean(CONFIRM, true)
        set(value) = prefs.edit().putBoolean(CONFIRM, value).apply()

    var powerMode: String
        get() = prefs.getString(POWER, "sleep_when_done") ?: "sleep_when_done"
        set(value) = prefs.edit().putString(POWER, value).apply()

    val paired: Boolean
        get() = host.isNotBlank() && token.isNotBlank() && fingerprint.isNotBlank()

    /** Take everything out of a scanned pairing code. Returns why it failed. */
    fun applyPairingCode(scanned: String): String? {
        val payload = try {
            JSONObject(scanned)
        } catch (error: Exception) {
            return "That code is not a pairing code"
        }
        val scannedHost = payload.optString("host")
        val scannedToken = payload.optString("token")
        val scannedPrint = payload.optString("fingerprint")
        if (scannedHost.isBlank() || scannedToken.isBlank() || scannedPrint.isBlank()) {
            return "That pairing code is missing the address, the token or the fingerprint"
        }
        host = scannedHost
        port = payload.optInt("port", 7860)
        token = scannedToken
        fingerprint = scannedPrint
        name = payload.optString("name", "the Mac")
        lastGoodHost = ""
        val macs = payload.optJSONArray("mac_addresses")
        macAddresses = (0 until (macs?.length() ?: 0)).mapNotNull { macs?.optString(it) }
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
        const val LLM = "useLlm"
        const val CONFIRM = "confirmScreenshots"
        const val POWER = "powerMode"
    }
}
