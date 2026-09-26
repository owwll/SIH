package org.enigma.navicore.odometer

import org.junit.Assert.*
import org.junit.Test
import java.io.File

/**
 * JVM unit tests for VirtualOdometer and fallback behavior.
 */
class VirtualOdometerTest {

    private val sampleWindow = FloatArray(60) { i -> (i % 6) * 0.1f }

    @Test
    fun testLocalVirtualOdometerFallbackExecution() {
        val odometer = LocalVirtualOdometer(10)
        val output = odometer.infer(sampleWindow)

        assertNotNull(output)
        assertTrue(output.vx >= 0.0f)
        assertTrue(output.varianceVx > 0.0f)
        assertTrue(output.stoppedProb in 0.0f..1.0f)
    }

    @Test
    fun testLocalVirtualOdometerRejectsInvalidSize() {
        val odometer = LocalVirtualOdometer(10)
        val invalidWindow = FloatArray(50)

        assertThrows(IllegalArgumentException::class.java) {
            odometer.infer(invalidWindow)
        }
    }

    @Test
    fun testTfliteVirtualOdometerFileFallback() {
        // Points to non-existent file to assert graceful fallback to LocalVirtualOdometer
        val dummyFile = File("non_existent_model.tflite")
        val odometer = TfliteVirtualOdometer(dummyFile)

        assertFalse(odometer.isModelLoaded)
        val output = odometer.infer(sampleWindow)
        assertNotNull(output)
        assertTrue(output.vx >= 0.0f)
    }
}
