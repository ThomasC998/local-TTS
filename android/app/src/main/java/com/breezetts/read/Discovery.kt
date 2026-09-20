package com.breezetts.read

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.util.Log
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Finding the Mac on this network without having been told where it is.
 *
 * The address it was paired at is a DHCP lease, and a lease is a thing that
 * expires. Rather than make the person re-pair every time the router renumbers
 * the house, the Mac announces a Bonjour service and this looks for it. What
 * pairing stores is then which machine to trust, and this answers where it is
 * right now.
 *
 * Deliberately short and deliberately blocking: it runs on a background thread
 * inside a lookup that is already going to take a second, and a discovery that
 * has not answered in two seconds is one that is not going to.
 */
class Discovery(context: Context) {

    private val manager =
        context.applicationContext.getSystemService(Context.NSD_SERVICE) as? NsdManager

    /** The first Breeze server announcing itself here, or null. */
    fun find(timeoutMs: Long): Pair<String, Int>? {
        val nsd = manager ?: return null
        val done = CountDownLatch(1)
        var result: Pair<String, Int>? = null

        val resolveListener = object : NsdManager.ResolveListener {
            override fun onResolveFailed(info: NsdServiceInfo?, errorCode: Int) {
                Log.w(TAG, "Could not resolve the service: $errorCode")
                done.countDown()
            }

            override fun onServiceResolved(info: NsdServiceInfo?) {
                val address = info?.host?.hostAddress
                if (address != null) result = address to info.port
                done.countDown()
            }
        }

        val discoveryListener = object : NsdManager.DiscoveryListener {
            override fun onStartDiscoveryFailed(type: String?, errorCode: Int) {
                done.countDown()
            }

            override fun onStopDiscoveryFailed(type: String?, errorCode: Int) = Unit
            override fun onDiscoveryStarted(type: String?) = Unit
            override fun onDiscoveryStopped(type: String?) = Unit

            override fun onServiceFound(info: NsdServiceInfo?) {
                if (info != null) {
                    @Suppress("DEPRECATION")
                    nsd.resolveService(info, resolveListener)
                }
            }

            override fun onServiceLost(info: NsdServiceInfo?) = Unit
        }

        return try {
            @Suppress("DEPRECATION")
            nsd.discoverServices(SERVICE_TYPE, NsdManager.PROTOCOL_DNS_SD, discoveryListener)
            done.await(timeoutMs, TimeUnit.MILLISECONDS)
            runCatching { nsd.stopServiceDiscovery(discoveryListener) }
            result
        } catch (error: Exception) {
            Log.w(TAG, "Discovery failed", error)
            null
        }
    }

    private companion object {
        const val SERVICE_TYPE = "_breezetts._tcp"
        const val TAG = "BreezeDiscovery"
    }
}
