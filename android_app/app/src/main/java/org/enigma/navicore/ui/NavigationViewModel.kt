package org.enigma.navicore.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.enigma.navicore.fusion.FusionCore
import org.enigma.navicore.fusion.FusionMode
import org.enigma.navicore.fusion.FusionState
import org.enigma.navicore.odometer.VirtualOdometer
import org.enigma.navicore.odometer.LocalVirtualOdometer

class NavigationViewModel(
    private val fusionCore: FusionCore = FusionCore(),
    private val virtualOdometer: VirtualOdometer = LocalVirtualOdometer()
) : ViewModel() {

    private val _fusionState = MutableStateFlow(
        FusionState(
            lat = 19.0760,
            lon = 72.8777,
            alt = 10.0,
            speedMps = 0.0f,
            headingRad = 0.0f,
            headingUncertaintyRad = 0.01f,
            mode = FusionMode.OPEN_SKY,
            blackoutDurationMs = 0,
            withinValidatedRange = true
        )
    )
    val fusionState: StateFlow<FusionState> = _fusionState.asStateFlow()

    private val _isSimulatingBlackout = MutableStateFlow(false)
    val isSimulatingBlackout: StateFlow<Boolean> = _isSimulatingBlackout.asStateFlow()

    fun toggleBlackoutSimulation(enabled: Boolean) {
        _isSimulatingBlackout.value = enabled
    }

    fun onImuReceived(timestampNanos: Long, ax: Float, ay: Float, az: Float, gx: Float, gy: Float, gz: Float, dt: Float) {
        fusionCore.processImu(timestampNanos, ax, ay, az, gx, gy, gz, dt)
        _fusionState.value = fusionCore.getState()
    }

    fun onGnssReceived(timestampNanos: Long, lat: Double, lon: Double, alt: Double, speedMps: Float, headingDeg: Float) {
        if (!_isSimulatingBlackout.value) {
            fusionCore.updateGnss(timestampNanos, lat, lon, alt, speedMps, headingDeg)
            _fusionState.value = fusionCore.getState()
        }
    }
}
