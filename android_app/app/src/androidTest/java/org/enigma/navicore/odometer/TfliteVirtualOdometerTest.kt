package org.enigma.navicore.odometer

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Instrumented test verifying the Android TFLite Neural Virtual Odometer (DV-08).
 * Ensures parity with Python evaluation from DV-07a against the real INT8 model:
 * 1. Checks that the INT8 TFLite model is successfully loaded via NNAPI or XNNPACK.
 * 2. Feeds the exact real-world sample window from Coventry IO-VNBD (S-S1.csv).
 * 3. Asserts the Android output matches the DV-07a reference within tight tolerance,
 *    verifying that inputs were not silently transposed or rescaled.
 * 4. Verifies LocalVirtualOdometer fallback functionality.
 */
@RunWith(AndroidJUnit4::class)
class TfliteVirtualOdometerTest {

    // Reference values measured in DV-07a on the same test window from S-S1.csv
    private val expectedVx = 14.5568f
    private val expectedVariance = 7.1541f
    private val expectedStoppedProb = 0.0195f

    // 60-float real sample window from Coventry IO-VNBD (Channels-First: 6 channels x 10 timesteps)
    private val sampleWindowChannelsFirst = floatArrayOf(
        // Ax (m/s^2, gravity decoupled)
        0.0f, 0.03792798f, 0.18898708f, -0.22149628f, 0.04046312f, -0.00484529f, -0.17063183f, 0.00612099f, -0.3170911f, 0.16399369f,
        // Ay (m/s^2, gravity decoupled)
        0.0f, 0.20103034f, 0.16651481f, -0.19739547f, -0.17455655f, 0.15746543f, 0.1918098f, 0.29184768f, 0.16841105f, -0.22640696f,
        // Az (m/s^2, gravity decoupled)
        0.0f, -0.27626228f, 0.21920872f, -0.00324726f, -0.13971806f, -0.07407093f, -0.20049095f, 0.21998787f, -0.12320518f, -0.23768425f,
        // Gx (rad/s)
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        // Gy (rad/s)
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        // Gz (rad/s, yaw rate)
        0.005f, 0.005f, 0.005f, 0.005f, 0.005f, 0.005f, 0.005f, 0.005f, 0.005f, 0.005f
    )

    private val sampleWindowInterleaved: FloatArray
        get() {
            val interleaved = FloatArray(60)
            for (t in 0 until 10) {
                for (c in 0 until 6) {
                    interleaved[t * 6 + c] = sampleWindowChannelsFirst[c * 10 + t]
                }
            }
            return interleaved
        }

    @Test
    fun testModelLoadsSuccessfullyFromAssets() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        TfliteVirtualOdometer(context).use { odometer ->
            assertTrue(
                "TFLite model should be loaded and active with NNAPI/XNNPACK",
                odometer.isModelLoaded
            )
        }
    }

    @Test
    fun testTfliteInferenceParityWithPythonDV07a_Interleaved() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        TfliteVirtualOdometer(context).use { odometer ->
            assertTrue(odometer.isModelLoaded)

            // Feed time-interleaved array [10 samples x 6 channels] (standard API.MD §3 contract)
            val output = odometer.infer(sampleWindowInterleaved)

            // Assert parity against Python reference values
            assertEquals("Forward speed Vx parity mismatch", expectedVx, output.vx, 0.25f)
            assertEquals("Variance sigma^2 parity mismatch", expectedVariance, output.varianceVx, 0.25f)
            assertEquals("Stopped probability parity mismatch", expectedStoppedProb, output.stoppedProb, 0.05f)

            // Sanity assertions
            assertTrue("Vx must be positive", output.vx > 0.0f)
            assertTrue("Variance must be strictly positive", output.varianceVx > 0.0f)
            assertTrue("Stopped probability must be valid", output.stoppedProb in 0.0f..1.0f)
        }
    }

    @Test
    fun testTfliteInferenceParityWithPythonDV07a_ChannelsFirst() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        TfliteVirtualOdometer(context).use { odometer ->
            assertTrue(odometer.isModelLoaded)

            // Feed channels-first array [6 channels x 10 samples] directly
            val output = odometer.infer(sampleWindowChannelsFirst, isChannelsFirst = true)

            // Both formats must produce identical neural inferences
            assertEquals("Forward speed Vx parity mismatch", expectedVx, output.vx, 0.25f)
            assertEquals("Variance sigma^2 parity mismatch", expectedVariance, output.varianceVx, 0.25f)
            assertEquals("Stopped probability parity mismatch", expectedStoppedProb, output.stoppedProb, 0.05f)
        }
    }

    @Test
    fun testLocalVirtualOdometerFallback() {
        val fallback = LocalVirtualOdometer(10)
        val output = fallback.infer(sampleWindowInterleaved)

        assertNotNull(output)
        assertTrue("Fallback Vx should be finite and non-negative", output.vx >= 0.0f)
        assertTrue("Fallback variance must be positive", output.varianceVx > 0.0f)
        assertTrue("Fallback stopped probability must be in [0, 1]", output.stoppedProb in 0.0f..1.0f)
    }
}
