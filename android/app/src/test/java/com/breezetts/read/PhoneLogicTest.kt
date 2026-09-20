package com.breezetts.read

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The parts of the phone app that can be wrong without anything noticing.
 *
 * These run on the JVM -- `gradle test`, no device, no emulator, a couple of
 * seconds -- so they run on every push alongside the build. Anything needing a
 * real Android framework is deliberately not here: the value of this file is
 * that it is cheap enough to never be skipped.
 *
 * What is worth testing is the code whose failure is silent. A pairing code
 * missing its fingerprint would leave the connection trusting any certificate,
 * which looks exactly like working. A magic packet with the address bytes in
 * the wrong order simply never wakes anything, and you would blame the Mac.
 */
class PairingCodeTest {

    private val good = """
        {"v":1,"host":"192.168.0.150","name":"Thomass-MacBook-Pro","port":7872,
         "token":"abc123","fingerprint":"e480c8","mac_addresses":["aa:f0:c9:83:26:ae"]}
    """.trimIndent()

    @Test
    fun `reads everything the Mac put in the code`() {
        val pairing = PairingCode.parse(good)!!
        assertEquals("192.168.0.150", pairing.host)
        assertEquals(7872, pairing.port)
        assertEquals("abc123", pairing.token)
        assertEquals("e480c8", pairing.fingerprint)
        assertEquals("Thomass-MacBook-Pro", pairing.name)
        assertEquals(listOf("aa:f0:c9:83:26:ae"), pairing.macAddresses)
    }

    @Test
    fun `a code with no fingerprint is refused`() {
        // The dangerous one: without a fingerprint there is nothing to pin, and
        // a connection that trusts anything behaves exactly like one that works.
        val payload = """{"host":"192.168.0.150","token":"abc123"}"""
        assertNull(PairingCode.parse(payload))
    }

    @Test
    fun `a code with no token is refused`() {
        val payload = """{"host":"192.168.0.150","fingerprint":"e480c8"}"""
        assertNull(PairingCode.parse(payload))
    }

    @Test
    fun `a code with no address is refused`() {
        val payload = """{"token":"abc123","fingerprint":"e480c8"}"""
        assertNull(PairingCode.parse(payload))
    }

    @Test
    fun `something that is not a pairing code at all is refused`() {
        assertNull(PairingCode.parse("https://example.com"))
        assertNull(PairingCode.parse(""))
    }

    @Test
    fun `the port falls back to the usual one`() {
        val payload = """{"host":"h","token":"t","fingerprint":"f"}"""
        assertEquals(7860, PairingCode.parse(payload)!!.port)
    }

    @Test
    fun `a Mac with no name still pairs`() {
        val payload = """{"host":"h","token":"t","fingerprint":"f"}"""
        assertEquals("the Mac", PairingCode.parse(payload)!!.name)
    }
}

class WakeTest {

    @Test
    fun `a magic packet is six ones then the address sixteen times`() {
        val packet = Wake.magicPacketBytes("aa:f0:c9:83:26:ae")
        assertEquals(102, packet.size)
        assertArrayEquals(ByteArray(6) { 0xFF.toByte() }, packet.copyOfRange(0, 6))

        val address = byteArrayOf(
            0xAA.toByte(), 0xF0.toByte(), 0xC9.toByte(),
            0x83.toByte(), 0x26.toByte(), 0xAE.toByte(),
        )
        for (repeat in 0 until 16) {
            val at = 6 + repeat * 6
            assertArrayEquals(
                "repetition $repeat", address, packet.copyOfRange(at, at + 6)
            )
        }
    }

    @Test
    fun `hyphens are a hardware address too`() {
        assertArrayEquals(
            Wake.magicPacketBytes("aa:f0:c9:83:26:ae"),
            Wake.magicPacketBytes("aa-f0-c9-83-26-ae"),
        )
    }

    @Test(expected = java.io.IOException::class)
    fun `something that is not an address is refused`() {
        Wake.magicPacketBytes("not-an-address")
    }

    @Test
    fun `the packet goes to the subnet, not to everywhere`() {
        // 255.255.255.255 is dropped by a lot of access points; the directed
        // broadcast for the Mac's own /24 gets through far more often.
        assertEquals("192.168.0.255", Wake.broadcastAddressFor("192.168.0.150"))
        assertEquals("10.0.1.255", Wake.broadcastAddressFor("10.0.1.7"))
    }

    @Test
    fun `an address that is not IPv4 falls back to everywhere`() {
        assertEquals("255.255.255.255", Wake.broadcastAddressFor("mac.local"))
        assertEquals("255.255.255.255", Wake.broadcastAddressFor("fe80::1"))
        assertEquals("255.255.255.255", Wake.broadcastAddressFor("999.1.1.1"))
    }
}

class ParagraphUrlTest {

    @Test
    fun `a paragraph carries its token and asks for nothing else`() {
        val url = Server.paragraphUrl("https://192.168.0.150:7861", "rd_abc", 3, "tok")
        assertEquals(
            "https://192.168.0.150:7861/v1/read/rd_abc/p3.wav?t=tok", url
        )
    }

    @Test
    fun `the token is there because a media player sends no headers`() {
        val url = Server.paragraphUrl("https://h:1", "rd", 0, "secret")
        assertTrue(url.contains("t=secret"))
    }
}
