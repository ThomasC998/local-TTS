package com.breezetts.read

import android.content.Context
import okhttp3.Call
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.security.MessageDigest
import java.security.SecureRandom
import java.security.cert.CertificateException
import java.security.cert.X509Certificate
import java.util.concurrent.TimeUnit
import javax.net.ssl.SSLContext
import javax.net.ssl.X509TrustManager

/**
 * Talking to the Mac, over a connection that can only be the Mac.
 *
 * The certificate on the other end is self-signed, which by itself proves
 * nothing at all -- anything on the same Wi-Fi could present one. What makes it
 * mean something is that the phone was told, in the pairing code, the exact
 * fingerprint to expect. From then on this client accepts that one certificate
 * and no other, which is a stronger guarantee than a public certificate
 * authority gives: there is no third party who could be persuaded to issue a
 * second one.
 *
 * Because the certificate is pinned to a single key, the hostname check is
 * turned off deliberately. The Mac's address changes with the router's mood and
 * the pin is what identity means here; verifying a name as well would only mean
 * re-pairing every time the lease changed, and would add nothing.
 */
class Server(context: Context, private val settings: Settings) {

    private val discovery = Discovery(context)

    /** Where the Mac was found, and whether that address can be woken. */
    data class Endpoint(val host: String, val port: Int, val wakeable: Boolean) {
        val base: String get() = "https://$host:$port"
    }

    class ServerError(message: String, val code: Int = 0) : IOException(message)

    private val client: OkHttpClient by lazy { buildClient(settings.fingerprint) }

    fun httpClient(): OkHttpClient = client

    // -- finding it -------------------------------------------------------
    /**
     * The Mac, waking it if it is asleep and this is its own network.
     *
     * Tried in the order that is cheapest when it works: the address that
     * answered last time, then whatever is announcing itself on this network,
     * then the paired address, then the remote one. A wake is only attempted
     * for an address on this network, because a magic packet does not route and
     * a sleeping Mac's VPN is asleep with it.
     */
    fun locate(wake: Boolean = true, onProgress: (String) -> Unit = {}): Endpoint {
        val port = settings.port
        val candidates = LinkedHashSet<String>()
        if (settings.lastGoodHost.isNotBlank()) candidates.add(settings.lastGoodHost)
        discovery.find(2_000)?.let { candidates.add(it.first) }
        if (settings.host.isNotBlank()) candidates.add(settings.host)

        for (host in candidates) {
            if (alive(host, port)) return found(host, port, true)
        }

        if (wake && candidates.isNotEmpty() && settings.macAddresses.isNotEmpty()) {
            onProgress("Waking ${settings.name}…")
            Wake.send(settings.macAddresses, candidates.first(), port)
            val deadline = System.currentTimeMillis() + WAKE_WAIT_MS
            while (System.currentTimeMillis() < deadline) {
                for (host in candidates) {
                    if (alive(host, port)) return found(host, port, true)
                }
                Thread.sleep(1_000)
            }
        }

        val remote = settings.remoteHost
        if (remote.isNotBlank()) {
            onProgress("Trying $remote…")
            val remoteHost = remote.substringBefore(":")
            val remotePort = remote.substringAfter(":", "").toIntOrNull() ?: port
            if (alive(remoteHost, remotePort)) return found(remoteHost, remotePort, false)
        }

        throw ServerError(
            if (settings.paired) {
                "${settings.name} did not answer. It may be asleep on battery, " +
                    "which a MacBook often cannot be woken from, or on another network."
            } else {
                "This phone is not paired with a Mac yet."
            }
        )
    }

    private fun found(host: String, port: Int, wakeable: Boolean): Endpoint {
        settings.lastGoodHost = host
        return Endpoint(host, port, wakeable)
    }

    /** Whether something is answering there. Health needs no token by design. */
    fun alive(host: String, port: Int, timeoutMs: Long = 1_500): Boolean = try {
        val quick = client.newBuilder()
            .connectTimeout(timeoutMs, TimeUnit.MILLISECONDS)
            .readTimeout(timeoutMs, TimeUnit.MILLISECONDS)
            .build()
        quick.newCall(Request.Builder().url("https://$host:$port/health").build())
            .execute().use { it.isSuccessful }
    } catch (error: Exception) {
        false
    }

    // -- asking it for things ---------------------------------------------
    fun startRead(endpoint: Endpoint, text: String, useLlm: Boolean): JSONObject {
        val payload = JSONObject()
            .put("text", text)
            .put("llm", useLlm)
        return call(
            Request.Builder()
                .url("${endpoint.base}/v1/read")
                .post(payload.toString().toRequestBody(JSON))
                .build()
        )
    }

    fun startReadFromImage(endpoint: Endpoint, image: ByteArray, useLlm: Boolean): JSONObject {
        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("llm", useLlm.toString())
            .addFormDataPart(
                "image", "screenshot.png", image.toRequestBody("image/png".toMediaType())
            )
            .build()
        return call(
            Request.Builder().url("${endpoint.base}/v1/read").post(body).build(),
            timeoutSeconds = 180,
        )
    }

    fun extractText(endpoint: Endpoint, image: ByteArray): JSONObject {
        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "image", "screenshot.png", image.toRequestBody("image/png".toMediaType())
            )
            .build()
        return call(
            Request.Builder().url("${endpoint.base}/v1/vision/extract").post(body).build(),
            timeoutSeconds = 180,
        )
    }

    fun manifest(endpoint: Endpoint, readId: String): JSONObject =
        call(Request.Builder().url("${endpoint.base}/v1/read/$readId/manifest").build())

    fun endRead(endpoint: Endpoint, readId: String) {
        runCatching {
            call(
                Request.Builder().url("${endpoint.base}/v1/read/$readId").delete().build()
            )
        }
    }

    fun power(endpoint: Endpoint): JSONObject =
        call(Request.Builder().url("${endpoint.base}/v1/power").build())

    fun setPowerMode(endpoint: Endpoint, mode: String): JSONObject = call(
        Request.Builder()
            .url("${endpoint.base}/v1/power/mode")
            .post(JSONObject().put("mode", mode).toString().toRequestBody(JSON))
            .build()
    )

    /**
     * Tell the Mac what went wrong here.
     *
     * A phone in a pocket has no console, and the moment you plug it in to look
     * is the moment it starts working. So failures are posted to the server,
     * where they land in the log next to whatever it was doing at the time.
     */
    fun report(endpoint: Endpoint, level: String, message: String, detail: String = "") {
        runCatching {
            call(
                Request.Builder()
                    .url("${endpoint.base}/v1/client-log")
                    .post(
                        JSONObject()
                            .put("level", level)
                            .put("message", message)
                            .put("detail", detail)
                            .toString().toRequestBody(JSON)
                    )
                    .build()
            )
        }
    }

    /** The URL a media player fetches one paragraph from. */
    fun paragraphUrl(endpoint: Endpoint, readId: String, index: Int): String =
        paragraphUrl(endpoint.base, readId, index, settings.token)

    private fun call(request: Request, timeoutSeconds: Long = 60): JSONObject {
        val authorised = request.newBuilder()
            .header("Authorization", "Bearer ${settings.token}")
            .build()
        val caller: Call = client.newBuilder()
            .readTimeout(timeoutSeconds, TimeUnit.SECONDS)
            .build()
            .newCall(authorised)
        caller.execute().use { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) {
                throw ServerError(explain(response.code, body), response.code)
            }
            return if (body.isBlank()) JSONObject() else JSONObject(body)
        }
    }

    private fun explain(code: Int, body: String): String {
        val detail = runCatching { JSONObject(body).optString("detail") }.getOrNull()
        return when (code) {
            401 -> "The Mac did not recognise this phone. Pair it again."
            429 -> "The Mac is refusing requests from this phone for a few minutes."
            404 -> detail?.takeIf { it.isNotBlank() } ?: "That read is over."
            else -> detail?.takeIf { it.isNotBlank() } ?: "The Mac answered $code"
        }
    }

    companion object {
        /**
         * Where one paragraph lives.
         *
         * The token rides in the query string because a media player fetches
         * these itself and attaches no headers of its own, and ``wait=1``
         * because a player handed audio of unknown length invents a duration
         * from it -- better to wait for the first paragraph than to show a
         * scrub bar claiming the read is twenty-four hours long.
         */
        fun paragraphUrl(base: String, readId: String, index: Int, token: String): String =
            "$base/v1/read/$readId/p$index.wav?wait=1&t=$token"

        private val JSON = "application/json; charset=utf-8".toMediaType()
        private const val WAKE_WAIT_MS = 20_000L

        /**
         * A client that trusts exactly one certificate: the paired Mac's.
         *
         * Everything else -- every public authority, every other self-signed
         * certificate on the network -- is rejected, because the only thing
         * this phone ever talks to is that one machine.
         */
        private fun buildClient(fingerprint: String): OkHttpClient {
            val trust = object : X509TrustManager {
                override fun checkClientTrusted(
                    chain: Array<out X509Certificate>?, authType: String?,
                ) = Unit

                override fun checkServerTrusted(
                    chain: Array<out X509Certificate>?, authType: String?,
                ) {
                    val leaf = chain?.firstOrNull()
                        ?: throw CertificateException("The server sent no certificate")
                    val digest = MessageDigest.getInstance("SHA-256")
                        .digest(leaf.encoded)
                        .joinToString("") { byte -> "%02x".format(byte) }
                    if (!digest.equals(fingerprint, ignoreCase = true)) {
                        throw CertificateException(
                            "This is not the Mac this phone was paired with"
                        )
                    }
                }

                override fun getAcceptedIssuers(): Array<X509Certificate> = emptyArray()
            }
            val context = SSLContext.getInstance("TLS").apply {
                init(null, arrayOf(trust), SecureRandom())
            }
            return OkHttpClient.Builder()
                .sslSocketFactory(context.socketFactory, trust)
                // Safe only because of the pin above: identity here is the key,
                // not the name, and the name is a DHCP lease.
                .hostnameVerifier { _, _ -> true }
                .connectTimeout(4, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .retryOnConnectionFailure(true)
                .build()
        }
    }
}
