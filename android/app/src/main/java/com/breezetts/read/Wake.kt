package com.breezetts.read

import android.util.Log
import java.io.IOException
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.Socket

/**
 * Waking the Mac, which is the part of this that has no guarantees.
 *
 * Two things are sent, because it is genuinely unclear which one a given Mac
 * responds to and neither costs anything:
 *
 * *A magic packet*, the traditional wake-on-LAN broadcast -- six 0xFF bytes and
 * the hardware address sixteen times over, to UDP 9 on the broadcast address.
 * Apple documents this for Ethernet.
 *
 * *A connection attempt* to the port the server listens on. Apple Silicon Macs
 * keep the Wi-Fi radio associated while asleep and commonly wake for a packet
 * addressed to a port they were listening on, which in practice works more
 * often than the magic packet does over Wi-Fi.
 *
 * Both are broadcast to a subnet, so both only work while the phone is on the
 * Mac's own network. There is no version of this that reaches a sleeping
 * machine through a VPN: the software that would answer is asleep too.
 *
 * And a MacBook running on battery may not wake for either. That is a property
 * of the machine, not of this code, which is why the app says so plainly rather
 * than retrying forever.
 */
object Wake {

    private const val TAG = "BreezeWake"
    private const val WOL_PORT = 9

    /** Send everything that might wake it. Failures are expected and ignored. */
    fun send(macAddresses: List<String>, host: String, port: Int) {
        for (mac in macAddresses.take(3)) {
            runCatching { magicPacket(mac, broadcastFor(host)) }
                .onFailure { Log.w(TAG, "Magic packet to $mac failed", it) }
        }
        runCatching { poke(host, port) }
    }

    /**
     * The subnet broadcast address for the Mac's address.
     *
     * 255.255.255.255 is filtered by many Wi-Fi access points; the directed
     * broadcast for the /24 the Mac is on gets through far more often, and a
     * home network is a /24 in all but pathological cases.
     */
    internal fun broadcastAddressFor(host: String): String {
        val parts = host.split(".")
        if (parts.size == 4 && parts.all { it.toIntOrNull() in 0..255 }) {
            return parts.take(3).joinToString(".") + ".255"
        }
        return "255.255.255.255"
    }

    /**
     * The magic packet's bytes: six 0xFF, then the hardware address sixteen
     * times. Pure, and therefore the one part of waking that can be checked
     * without a sleeping Mac to aim it at.
     */
    internal fun magicPacketBytes(mac: String): ByteArray {
        val hardware = mac.split(":", "-")
            .mapNotNull { it.trim().toIntOrNull(16)?.toByte() }
        if (hardware.size != 6) throw IOException("Not a hardware address: $mac")
        val payload = ByteArray(6 + 16 * 6)
        for (index in 0 until 6) payload[index] = 0xFF.toByte()
        val address = hardware.toByteArray()
        for (repeat in 0 until 16) {
            System.arraycopy(address, 0, payload, 6 + repeat * 6, 6)
        }
        return payload
    }

    private fun broadcastFor(host: String): InetAddress =
        InetAddress.getByName(broadcastAddressFor(host))

    private fun magicPacket(mac: String, target: InetAddress) {
        val payload = magicPacketBytes(mac)
        DatagramSocket().use { socket ->
            socket.broadcast = true
            socket.send(DatagramPacket(payload, payload.size, target, WOL_PORT))
        }
        Log.i(TAG, "Magic packet sent to $mac via $target")
    }

    /** A connection attempt, which is itself a wake signal on recent Macs. */
    private fun poke(host: String, port: Int) {
        Socket().use { socket ->
            socket.connect(InetSocketAddress(host, port), 1_200)
        }
    }
}
