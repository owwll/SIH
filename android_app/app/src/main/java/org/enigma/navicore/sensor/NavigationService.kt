package org.enigma.navicore.sensor

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*
import org.enigma.navicore.fusion.FusionCore
import org.enigma.navicore.fusion.FusionMode
import org.enigma.navicore.fusion.FusionState
import org.enigma.navicore.odometer.VirtualOdometer
import org.enigma.navicore.odometer.TfliteVirtualOdometer
import org.enigma.navicore.odometer.LocalVirtualOdometer

class NavigationService : Service() {

    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private lateinit var sensorManager: ImuSensorManager
    private val fusionCore = FusionCore()
    private lateinit var virtualOdometer: VirtualOdometer
    private var wakeLock: PowerManager.WakeLock? = null

    companion object {
        const val CHANNEL_ID = "navicore_live_channel"
        const val NOTIFICATION_ID = 101
        var currentFusionState: FusionState? = null
            private set
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildNotification("Initializing NaviCore AI Engine..."))

        // Initialize real TFLite Virtual Odometer with NNAPI/XNNPACK; fallback to LocalVirtualOdometer if needed
        virtualOdometer = try {
            TfliteVirtualOdometer(this)
        } catch (e: Exception) {
            LocalVirtualOdometer()
        }

        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "NaviCore:SensorWakeLock").apply {
            acquire(10 * 60 * 1000L) // 10 minutes timeout
        }

        sensorManager = ImuSensorManager(this)
        sensorManager.startListening()

        serviceScope.launch {
            var lastTimestamp = 0L
            val windowBuffer = FloatArray(60) // 10 samples x 6 channels
            var windowIdx = 0

            sensorManager.imuFlow.collect { sample ->
                val dt = if (lastTimestamp > 0) (sample.timestampNanos - lastTimestamp) / 1e9f else 0.01f
                lastTimestamp = sample.timestampNanos

                // Ingest into C++ Kalman Core
                fusionCore.processImu(
                    sample.timestampNanos,
                    sample.ax, sample.ay, sample.az,
                    sample.gx, sample.gy, sample.gz,
                    dt
                )

                // Fill window for AI Odometer inference
                if (windowIdx < 10) {
                    val base = windowIdx * 6
                    windowBuffer[base + 0] = sample.ax
                    windowBuffer[base + 1] = sample.ay
                    windowBuffer[base + 2] = sample.az
                    windowBuffer[base + 3] = sample.gx
                    windowBuffer[base + 4] = sample.gy
                    windowBuffer[base + 5] = sample.gz
                    windowIdx++
                } else {
                    // Window full -> Run Odometer Inference
                    val odomOut = virtualOdometer.infer(windowBuffer)
                    fusionCore.updateAiOdometer(odomOut)
                    windowIdx = 0
                }

                currentFusionState = fusionCore.getState()
            }
        }
    }

    private fun buildNotification(statusText: String): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("NaviCore AI Navigation")
            .setContentText(statusText)
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "NaviCore Navigation Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Live high-frequency dead reckoning background loop"
            }
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }
    }

    override fun onDestroy() {
        (virtualOdometer as? AutoCloseable)?.close()
        sensorManager.stopListening()
        serviceScope.cancel()
        wakeLock?.let { if (it.isHeld) it.release() }
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
