package org.enigma.navicore.odometer

import android.content.Context
import android.util.Log
import org.enigma.navicore.fusion.OdometerOutput
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Interface contract matching docs/API.MD §3.
 */
interface VirtualOdometer {
    /**
     * @param window Calibrated, vehicle-frame IMU samples at the model's trained input rate.
     */
    fun infer(window: FloatArray): OdometerOutput
}

/**
 * Pure Kotlin fallback runner and explicit backup when TFLite is unavailable.
 * Matches docs/API.MD §3 and TRD §4.2.
 */
class LocalVirtualOdometer(
    private val windowLength: Int = 10
) : VirtualOdometer {

    companion object {
        private const val TAG = "LocalVirtualOdometer"
    }

    override fun infer(window: FloatArray): OdometerOutput {
        require(window.size == windowLength * 6) {
            "Input window size mismatch: expected ${windowLength * 6} floats, got ${window.size}"
        }

        // Compute dynamic forward acceleration energy across the window
        var sumAx = 0.0f
        var energyIdle = 0.0f
        for (i in 0 until windowLength) {
            val ax = window[i * 6 + 0]
            val az = window[i * 6 + 2]
            sumAx += ax
            energyIdle += Math.abs(az - 9.81f)
        }
        val meanAx = sumAx / windowLength
        val idleMetric = energyIdle / windowLength

        // Speed prediction fallback heuristic
        val predictedVx = Math.max(0.0f, meanAx * 1.2f)
        val isStopped = idleMetric > 0.3f && Math.abs(meanAx) < 0.1f
        val stoppedProb = if (isStopped) 0.95f else 0.05f

        return OdometerOutput(
            vx = if (isStopped) 0.0f else predictedVx,
            varianceVx = 0.05f,
            stoppedProb = stoppedProb
        )
    }
}

/**
 * Production TFLite Neural Virtual Odometer running full Post-Training INT8 Quantized model.
 * Matches TRD §4.2:
 * - Primary delegate: Android NNAPI (targeting Qualcomm Hexagon NPU, MediaTek APU).
 * - Fallback delegate: CPU multi-threaded backend with XNNPACK.
 * - Software Fallback: LocalVirtualOdometer if TFLite model fails to load or execute.
 *
 * Input Tensor: [Batch=1, Channels=6 (Ax, Ay, Az, Gx, Gy, Gz), WindowLen=10 (10 Hz)]
 * Output Heads:
 *   - Head A (StatefulPartitionedCall:0): Forward speed regression Vx (m/s)
 *   - Head B (StatefulPartitionedCall:1): Speed variance sigma_v^2 (strictly positive uncertainty)
 *   - Head C (StatefulPartitionedCall:2): Stopped probability P(stopped) in [0, 1]
 */
class TfliteVirtualOdometer : VirtualOdometer, AutoCloseable {

    companion object {
        private const val TAG = "TfliteVirtualOdometer"
        const val DEFAULT_MODEL_FILE = "navicore_odometer_int8.tflite"
        const val CHANNELS = 6
        const val WINDOW_LEN = 10
    }

    private var interpreter: Interpreter? = null
    private var nnApiDelegate: NnApiDelegate? = null
    private val fallbackOdometer = LocalVirtualOdometer(WINDOW_LEN)

    // Output tensor indices mapped from FlatBuffer metadata
    private var vxOutputIndex = 0
    private var varOutputIndex = 1
    private var zuptOutputIndex = 2

    /**
     * Constructs TfliteVirtualOdometer from Android Context and asset path.
     */
    constructor(context: Context, modelAssetPath: String = DEFAULT_MODEL_FILE) {
        val buffer = tryLoadModelAsset(context, modelAssetPath)
        if (buffer != null) {
            initInterpreter(buffer)
        } else {
            Log.w(TAG, "Model asset '$modelAssetPath' not found. LocalVirtualOdometer fallback will be used.")
        }
    }

    /**
     * Constructs TfliteVirtualOdometer from an explicit model File.
     */
    constructor(modelFile: File) {
        if (modelFile.exists()) {
            val buffer = tryLoadModelFile(modelFile)
            if (buffer != null) {
                initInterpreter(buffer)
            } else {
                Log.w(TAG, "Failed to map model file ${modelFile.path}. Using fallback.")
            }
        } else {
            Log.w(TAG, "Model file does not exist at: ${modelFile.path}. Using fallback.")
        }
    }

    /**
     * Constructs TfliteVirtualOdometer directly from an in-memory ByteBuffer.
     */
    constructor(modelBuffer: ByteBuffer) {
        initInterpreter(modelBuffer)
    }

    /**
     * Configures TFLite Interpreter with NNAPI primary delegate and XNNPACK fallback per TRD §4.2.
     */
    private fun initInterpreter(modelBuffer: ByteBuffer) {
        // Attempt 1: Android NNAPI Delegate (Primary per TRD §4.2)
        try {
            val nnApiOptions = NnApiDelegate.Options().apply {
                setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER)
                setAllowFp16(true)
            }
            val delegate = NnApiDelegate(nnApiOptions)
            val options = Interpreter.Options().apply {
                addDelegate(delegate)
            }
            val interp = Interpreter(modelBuffer, options)
            nnApiDelegate = delegate
            interpreter = interp
            resolveOutputIndices(interp)
            Log.i(TAG, "Initialized TFLite Virtual Odometer via NNAPI delegate (Primary NPU/APU).")
            return
        } catch (e: Throwable) {
            Log.w(TAG, "NNAPI delegate initialization failed; attempting XNNPACK / CPU backend.", e)
            nnApiDelegate?.close()
            nnApiDelegate = null
        }

        // Attempt 2: CPU Multi-Threaded XNNPACK Backend (Fallback per TRD §4.2)
        try {
            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseXNNPACK(true)
            }
            val interp = Interpreter(modelBuffer, options)
            interpreter = interp
            resolveOutputIndices(interp)
            Log.i(TAG, "Initialized TFLite Virtual Odometer via CPU XNNPACK backend (4 threads).")
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to initialize TFLite Interpreter. Pure Kotlin fallback will be used.", e)
            interpreter = null
        }
    }

    /**
     * Resolves output tensor indices dynamically based on FlatBuffer output names (:0, :1, :2).
     */
    private fun resolveOutputIndices(interp: Interpreter) {
        val outputCount = interp.outputTensorCount
        for (i in 0 until outputCount) {
            val name = interp.getOutputTensor(i).name()
            when {
                name.endsWith(":0") -> vxOutputIndex = i
                name.endsWith(":1") -> varOutputIndex = i
                name.endsWith(":2") -> zuptOutputIndex = i
            }
        }
        Log.d(TAG, "Resolved TFLite output indices: Vx=$vxOutputIndex, Var=$varOutputIndex, ZUPT=$zuptOutputIndex")
    }

    /**
     * Standard VirtualOdometer interface contract.
     * Expects time-interleaved float array [10 timesteps x 6 channels] (API.MD §3).
     */
    override fun infer(window: FloatArray): OdometerOutput {
        return infer(window, isChannelsFirst = false)
    }

    /**
     * Extended inference method supporting both time-interleaved [10, 6] and channels-first [6, 10].
     *
     * @param window FloatArray containing exactly 60 floats (6 channels x 10 timesteps).
     * @param isChannelsFirst If true, assumes layout [Ax0..Ax9, Ay0..Ay9, ...].
     *                        If false (default), assumes interleaved [Ax0, Ay0, Az0, Gx0, Gy0, Gz0, Ax1, ...].
     */
    fun infer(window: FloatArray, isChannelsFirst: Boolean = false): OdometerOutput {
        val interp = interpreter
        if (interp == null) {
            Log.d(TAG, "TFLite interpreter inactive. Delegating to LocalVirtualOdometer fallback.")
            return fallbackOdometer.infer(window)
        }

        require(window.size == CHANNELS * WINDOW_LEN) {
            "Input window size mismatch: expected ${CHANNELS * WINDOW_LEN} floats, got ${window.size}"
        }

        try {
            // Model input tensor shape: [Batch=1, Channels=6, WindowLen=10]
            val inputTensor = Array(1) { Array(CHANNELS) { FloatArray(WINDOW_LEN) } }

            if (isChannelsFirst) {
                for (c in 0 until CHANNELS) {
                    for (t in 0 until WINDOW_LEN) {
                        inputTensor[0][c][t] = window[c * WINDOW_LEN + t]
                    }
                }
            } else {
                for (t in 0 until WINDOW_LEN) {
                    for (c in 0 until CHANNELS) {
                        inputTensor[0][c][t] = window[t * CHANNELS + c]
                    }
                }
            }

            // Allocate output buffers: [1, 1] each
            val outVx = Array(1) { FloatArray(1) }
            val outVar = Array(1) { FloatArray(1) }
            val outZupt = Array(1) { FloatArray(1) }

            val outputs = mutableMapOf<Int, Any>(
                vxOutputIndex to outVx,
                varOutputIndex to outVar,
                zuptOutputIndex to outZupt
            )

            // Run inference via TFLite (NNAPI / XNNPACK)
            interp.runForMultipleInputsOutputs(arrayOf(inputTensor), outputs)

            val rawVx = outVx[0][0]
            val rawVar = outVar[0][0]
            val rawZupt = outZupt[0][0]

            return OdometerOutput(
                vx = Math.max(0.0f, rawVx),
                varianceVx = Math.max(1e-4f, rawVar),
                stoppedProb = Math.min(1.0f, Math.max(0.0f, rawZupt))
            )
        } catch (e: Exception) {
            Log.e(TAG, "TFLite execution error; delegating to LocalVirtualOdometer fallback", e)
            return fallbackOdometer.infer(window)
        }
    }

    /**
     * Direct 2D array inference method for channels-first matrix [6 channels x 10 timesteps].
     */
    fun inferChannelsFirst(matrix: Array<FloatArray>): OdometerOutput {
        require(matrix.size == CHANNELS && matrix.all { it.size == WINDOW_LEN }) {
            "Expected 2D matrix of shape [6, 10]"
        }
        val flattened = FloatArray(CHANNELS * WINDOW_LEN)
        for (c in 0 until CHANNELS) {
            System.arraycopy(matrix[c], 0, flattened, c * WINDOW_LEN, WINDOW_LEN)
        }
        return infer(flattened, isChannelsFirst = true)
    }

    /**
     * Checks if the real TFLite neural model is loaded and ready for hardware execution.
     */
    val isModelLoaded: Boolean
        get() = interpreter != null

    override fun close() {
        try {
            interpreter?.close()
            interpreter = null
            nnApiDelegate?.close()
            nnApiDelegate = null
        } catch (e: Exception) {
            Log.w(TAG, "Error closing TFLite resources", e)
        }
    }

    private fun tryLoadModelAsset(context: Context, assetPath: String): MappedByteBuffer? {
        val candidates = listOf(assetPath, "models/$assetPath")
        for (candidate in candidates) {
            try {
                return FileUtil.loadMappedFile(context, candidate)
            } catch (ignored: IOException) {
            }
            try {
                val fd = context.assets.openFd(candidate)
                val stream = FileInputStream(fd.fileDescriptor)
                val channel = stream.channel
                return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            } catch (ignored: Exception) {
            }
        }
        return null
    }

    private fun tryLoadModelFile(file: File): MappedByteBuffer? {
        return try {
            val stream = FileInputStream(file)
            val channel = stream.channel
            channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model file ${file.path}", e)
            null
        }
    }
}
