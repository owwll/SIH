# NaviCore AI — Work Chunk Prompts (Modular Structure Edition)
**Team @enigm@ (132834) | SIH 2026 | Problem Statement 260168**
**Active builders: Dhruvin Vaghasiya (ML) · Manthan (remaining)**

---

## 0. How to use this document

I read `NaviCore-Ai-main.zip` end-to-end (PRD, TRD, ROADMAP, AI_ARCHITECTURE, API.MD, and every source file in `core_cpp/`, `ml_pipeline/`, `android_app/`, `ros2_node/`, `web_visualizer/`, `sdk_python/`, `scripts/`) and diffed what's **actually implemented** against what the docs claim. This edition adds **two new chunks — DV-00 and MN-00** — that convert every flat/mixed directory into a proper **feature-wise folder structure** *before* any other work starts, and every chunk after them has its file paths rewritten to match. That ordering is the whole point: restructuring first means nobody edits a file at a path that moves out from under them mid-chunk.

Each chunk below is:

- **Self-contained** — paste the "Prompt to run" block straight into Claude Code (or this chat) on its own, in a fresh session, without re-explaining the project. It already contains the file paths, the current broken/missing state, and the target behaviour.
- **Ordered** — do them top-to-bottom within your track; later chunks assume earlier ones are done. **DV-00 and MN-00 come first, full stop, before DV-01 or MN-01.**
- **Traceable** — each maps to a specific FR/NFR in `docs/PRD.md` or `docs/TRD.md` so nothing drifts from spec.

Two chunks (**DV-07 / MN-02** and **DV-08 / MN-03**) are two-sided — one person writes the native/ML side, the other wires it into the app. Do those in the order listed and hand off the interface, don't build both halves solo.

---

## 1. Audit summary — what I actually found in the repo

The project looks far more complete than it is. The scaffolding (headers, class signatures, docs, CMake files) is essentially 100% done. But several pieces are **wired to nothing** or are **placeholder stand-ins**. Paths below are the **pre-refactor** paths as they exist in the repo today — DV-00/MN-00 move them; every chunk from DV-01/MN-01 onward already uses the post-refactor paths.

| # | Gap | Evidence (pre-refactor path) |
|---|---|---|
| 1 | The shipped TFLite model is fake | `models/exported/navicore_odometer_int8.tflite` is **37 bytes** — not a real model, just a placeholder file |
| 2 | Android never runs the real ML model | `VirtualOdometer.kt`'s `LocalVirtualOdometer` is a hand-written heuristic (`vx = meanAx * 1.2f`), explicitly commented `"Pure Kotlin fallback runner and TFLite execution placeholder"` |
| 3 | The physics-based ZUPT engine is dead code | `SpectralZuptEngine` (`nhc_zupt.cpp`) is only ever called from the unit test — `EskfFilter`/`jni_bridge.cpp` never instantiate it; ZUPT today is 100% driven by the AI classifier's `stoppedProb` |
| 4 | Adaptive NHC covariance is dead code | `NhcKinematics::ComputeAdaptiveCovariance` (FR-05, road-roughness scaling) has zero call sites anywhere in the repo |
| 5 | HMM map-matching is disconnected | `HmmMapMatcher` (full Viterbi implementation, `hmm_matcher.cpp`) exists in C++ but is **never linked into the Android build** — `android_app/.../cpp/CMakeLists.txt` only compiles `jni_bridge.cpp`, `eskf.cpp`, `auto_calib.cpp`. Kotlin's `MapMatcher.kt` → `SimpleOsmMapMatcher.match()` just `return null` unconditionally |
| 6 | Rerouting / dislodgement recovery is dead code | `DynamicReroutingEngine` (`rerouting_engine.hpp`) has zero call sites |
| 7 | Barometer and vehicle-profile modules are dead code | `BarometerTracker`, `VehicleProfileManager` — zero call sites anywhere |
| 8 | No pipeline orchestrator | TRD §6.1 documents a `navicore_pipeline.hpp` "Main Native Pipeline Orchestrator" — this file does not exist; `jni_bridge.cpp` improvises the wiring inline instead |
| 9 | ROS2 node never uses the AI odometer path | `ros2_node/src/navicore_node.cpp` only calls `eskf_->UpdateGnss()` and `Predict()` — it never subscribes to an AI-odometer topic, so the ROS2 node has no GNSS-blackout dead-reckoning at all, despite that being the entire point of the project |
| 10 | No verified training run | `ml_pipeline/` has real training/model/export code, but nothing in the repo shows an actual completed training run, logged RMSE, or ZUPT precision/recall — Phase 1 exit criteria (ROADMAP.md §2) are unmet |
| 11 | **Modularity gap** | `core_cpp/` is a flat `include/navicore/*.hpp` + `src/*.cpp` bag with no feature separation; `ros2_node/`, `web_visualizer/`, `sdk_python/`, `scripts/` are single-file-per-concern or flat directories with no grouping, which is exactly what made gaps 3–8 easy to miss and will keep causing merge conflicts as both of you touch the same flat folders |

Everything else (Kotlin UI shell, `MainActivity.kt`, `NavigationService.kt`, `ImuSensorManager.kt`, `web_visualizer/`, `sdk_python/`, most of `scripts/`) is genuinely built and mostly just needs wiring, integration-testing, and polish — not a rewrite.

---

## 2. Target modular folder structure

Feature-wise means: **one top-level folder per concern, named after what it does, not after its file type.** Below is old → new for every path referenced later in this document. Nothing here is a rewrite of logic — DV-00/MN-00 are pure `git mv` + include/import-path fixes + build-file updates.

### 2.1 `core_cpp/` (Dhruvin's exclusive scope)

| Feature | Old path | New path |
|---|---|---|
| Fusion filter | `core_cpp/include/navicore/eskf.hpp` | `core_cpp/include/navicore/fusion/eskf.hpp` |
| | `core_cpp/src/eskf.cpp` | `core_cpp/src/fusion/eskf.cpp` |
| ZUPT / NHC | `core_cpp/include/navicore/nhc_zupt.hpp` | `core_cpp/include/navicore/zupt/nhc_zupt.hpp` |
| | `core_cpp/src/nhc_zupt.cpp` | `core_cpp/src/zupt/nhc_zupt.cpp` |
| Mount calibration | `core_cpp/include/navicore/auto_calib.hpp` | `core_cpp/include/navicore/calibration/auto_calib.hpp` |
| | `core_cpp/src/auto_calib.cpp` | `core_cpp/src/calibration/auto_calib.cpp` |
| Map matching | `core_cpp/include/navicore/hmm_matcher.hpp` | `core_cpp/include/navicore/mapmatch/hmm_matcher.hpp` |
| | `core_cpp/src/hmm_matcher.cpp` | `core_cpp/src/mapmatch/hmm_matcher.cpp` |
| Rerouting | `core_cpp/include/navicore/rerouting_engine.hpp` | `core_cpp/include/navicore/routing/rerouting_engine.hpp` |
| | (rerouting_engine.cpp, if/when DV-10 adds one) | `core_cpp/src/routing/rerouting_engine.cpp` |
| Barometer | `core_cpp/include/navicore/barometer_tracker.hpp` | `core_cpp/include/navicore/sensors/barometer_tracker.hpp` |
| Vehicle profiles | `core_cpp/include/navicore/vehicle_profile.hpp` | `core_cpp/include/navicore/vehicle/vehicle_profile_manager.hpp` |
| Pipeline orchestrator (new, DV-10) | `core_cpp/include/navicore/navicore_pipeline.hpp` | `core_cpp/include/navicore/pipeline/navicore_pipeline.hpp` |
| | `core_cpp/src/navicore_pipeline.cpp` | `core_cpp/src/pipeline/navicore_pipeline.cpp` |
| Tests | `core_cpp/tests/test_eskf.cpp` | `core_cpp/tests/fusion/test_eskf.cpp` |
| Build file | `core_cpp/CMakeLists.txt` | unchanged location; `NAVICORE_SOURCES` list updated to the new subpaths |

`#include "navicore/eskf.hpp"` becomes `#include "navicore/fusion/eskf.hpp"`, and so on for every header — DV-00 must grep-and-fix every include site, not just move files.

### 2.2 `ml_pipeline/` (Dhruvin's exclusive scope)

This was already grouped by feature (`dataset/`, `models/`, `training/`) — only light cleanup needed:

| Feature | Old path | New path |
|---|---|---|
| TFLite export | `ml_pipeline/src/models/export_tflite.py` | `ml_pipeline/src/models/export/export_tflite.py` |
| Model I/O contract test | `ml_pipeline/tests/test_model_io_contract.py` | `ml_pipeline/tests/models/test_model_io_contract.py` |

Everything else in `ml_pipeline/` (`dataset/iovnbd_loader.py`, `dataset/augmentations.py`, `dataset/preprocessing.py`, `models/tcn_odometer.py`, `models/loss.py`, `training/train.py`, `config/training_config.yaml`) is already correctly modular — leave as-is.

### 2.3 `android_app/` (Manthan's exclusive scope, except that Dhruvin lands DV-08/DV-11 inside it before handing off)

| Feature | Old path | New path |
|---|---|---|
| IMU ingestion | `.../navicore/sensor/ImuSensorManager.kt` | `.../navicore/imu/ImuSensorManager.kt` |
| Foreground service | `.../navicore/sensor/NavigationService.kt` (or wherever it currently sits) | `.../navicore/service/NavigationService.kt` |
| Odometer (heuristic + TFLite) | `.../navicore/odometer/VirtualOdometer.kt` | unchanged — already its own package |
| Map matching | `.../navicore/mapmatch/MapMatcher.kt` | unchanged — already its own package |
| Fusion bridge | `.../navicore/fusion/FusionCore.kt`, `FusionState.kt` | unchanged — already its own package |
| Map UI | `.../navicore/ui/MapScreen.kt` | `.../navicore/ui/map/MapScreen.kt` |
| ViewModel | `.../navicore/ui/NavigationViewModel.kt` | `.../navicore/ui/viewmodel/NavigationViewModel.kt` |
| Telemetry overlay (new, MN-05) | — | `.../navicore/ui/telemetry/TelemetryOverlay.kt` |
| Native glue | `android_app/app/src/main/cpp/jni_bridge.cpp`, `CMakeLists.txt` | unchanged — this is meant to stay a single thin bridge file, see §3 below on why it's a shared touchpoint |

(`.../navicore/` = `android_app/app/src/main/java/org/enigma/navicore/`)

### 2.4 `ros2_node/`, `web_visualizer/`, `sdk_python/`, `scripts/` (Manthan's exclusive scope)

| Feature | Old path | New path |
|---|---|---|
| ROS2 node | `ros2_node/src/navicore_node.cpp` | `ros2_node/src/node/navicore_node.cpp` |
| ROS2 AI-odometer bridge (new, MN-07) | — | `ros2_node/src/bridge/ai_odometer_bridge.cpp` |
| ROS2 custom msg (new, MN-07) | — | `ros2_node/msg/AiOdometer.msg` |
| Visualizer entry point | `web_visualizer/index.html`, `sensor_stream.html` | unchanged (entry points stay at root) |
| Visualizer JS | `web_visualizer/app.js` (monolithic, ~4600 lines total across the folder) | split into `web_visualizer/src/js/map-renderer.js`, `web_visualizer/src/js/telemetry-panel.js`, `web_visualizer/src/js/socket-client.js` |
| Visualizer CSS | `web_visualizer/style.css` | `web_visualizer/src/css/style.css` |
| Fleet SDK | `sdk_python/navicore_sdk.py` (single 108-line file) | package `sdk_python/navicore_sdk/{__init__.py, fusion.py, state.py}` |
| Verification scripts | `scripts/test_end_to_end.py`, `verify_vehicle_kinematics.py`, `strict_verification_harness.py`, `verify_application.py` | `scripts/verification/...` |
| Simulation scripts | `scripts/simulate_drive.py`, `scripts/record_field_session.py` | `scripts/simulation/...` |
| Profiling scripts | `scripts/profile_hardware_footprint.py`, `scripts/benchmark_suite.py` | `scripts/profiling/...` |
| Demo/dev-server scripts | `scripts/realtime_sensor_server.py`, `scripts/launch_visualizer.py` | `scripts/demo/...` |

---

## 3. Sync protocol — how the two tracks avoid stepping on each other

The old flat layout was the biggest source of future merge conflicts: `core_cpp/src/*.cpp` had every engine in one directory, and `scripts/*.py` mixed verification/simulation/profiling/demo together with no ownership signal. The new layout fixes that structurally, but three things still need explicit process:

1. **DV-00 and MN-00 run in Week 0, in parallel, on separate branches (`refactor/core-cpp-modules` and `refactor/app-modules`), and both merge to `main` before DV-01 or MN-01 branch off.** They touch disjoint directories (`core_cpp/` + `ml_pipeline/` vs. `android_app/` + `ros2_node/` + `web_visualizer/` + `sdk_python/` + `scripts/`), so there's no file-level overlap — but merging both *before* anyone else branches means nobody has to rebase a half-finished feature chunk across a mass file-move later.
2. **Shared touchpoints get a lock, not a merge fight.** Three files are edited by both people at different points: `android_app/app/src/main/cpp/CMakeLists.txt` (Dhruvin adds core_cpp sources in DV-11, Manthan doesn't otherwise touch it), `android_app/app/src/main/cpp/jni_bridge.cpp` (Dhruvin's DV-08 adds nothing here, but Manthan's MN-02 rewrites it right after Dhruvin's DV-11 handoff), and `.../navicore/fusion/FusionState.kt` (Manthan's MN-02 adds fields, MN-04/MN-05 consume them). Do these strictly in the documented order — DV-11 → MN-02 — and don't start MN-02 until DV-11's `HANDOFF.md` is merged.
3. **Every chunk branches from `main` after the previous handoff chunk in its dependency chain is merged**, not from a stale local branch — this is what actually prevents path conflicts, since `main` always reflects the current folder structure.

---

## 4. Ownership split

| Track | Owner | Scope |
|---|---|---|
| **A — ML + Fusion/Physics Core** | **Dhruvin Vaghasiya** | Everything in `ml_pipeline/`, the trained model, and everything in `core_cpp/` (ESKF, calibration, NHC/ZUPT, HMM, rerouting, barometer, vehicle profiles) — because the fusion math and the odometer are the load-bearing, math-heavy pieces the ROADMAP itself treats as one connected track (MLE + SYS). Also owns getting the real model running inside the Android app (TFLite integration), since that's still "ML work." |
| **B — App, Robotics, Tooling, QA** | **Manthan** | `android_app/` UI + sensor + service layer, the JNI glue that calls into Dhruvin's native code, `ros2_node/`, `web_visualizer/`, `sdk_python/`, `scripts/`, and the Phase 4 QA/demo/documentation pass. |

This matches ROADMAP.md's own role table (MLE, SYS → Dhruvin; AND, GIS, QA → Manthan), just collapsed from 5 roles onto 2 people.

---

# TRACK A — Dhruvin Vaghasiya (ML + Fusion Core)

## DV-00 — Restructure `core_cpp/` and `ml_pipeline/` into feature modules
**Maps to**: foundational — do this before DV-01, in parallel with MN-00, per §3

**Prompt to run:**
```
core_cpp/ is currently a flat bag: include/navicore/*.hpp and src/*.cpp with
no feature separation, and ml_pipeline/ needs two small path cleanups.
Apply the exact old-path -> new-path mapping in this project's
"Target modular folder structure" doc (§2.1-2.2), specifically:

core_cpp/include/navicore/eskf.hpp            -> .../fusion/eskf.hpp
core_cpp/src/eskf.cpp                          -> src/fusion/eskf.cpp
core_cpp/include/navicore/nhc_zupt.hpp        -> .../zupt/nhc_zupt.hpp
core_cpp/src/nhc_zupt.cpp                      -> src/zupt/nhc_zupt.cpp
core_cpp/include/navicore/auto_calib.hpp      -> .../calibration/auto_calib.hpp
core_cpp/src/auto_calib.cpp                    -> src/calibration/auto_calib.cpp
core_cpp/include/navicore/hmm_matcher.hpp     -> .../mapmatch/hmm_matcher.hpp
core_cpp/src/hmm_matcher.cpp                   -> src/mapmatch/hmm_matcher.cpp
core_cpp/include/navicore/rerouting_engine.hpp -> .../routing/rerouting_engine.hpp
core_cpp/include/navicore/barometer_tracker.hpp -> .../sensors/barometer_tracker.hpp
core_cpp/include/navicore/vehicle_profile.hpp  -> .../vehicle/vehicle_profile_manager.hpp
core_cpp/tests/test_eskf.cpp                   -> tests/fusion/test_eskf.cpp
ml_pipeline/src/models/export_tflite.py        -> src/models/export/export_tflite.py
ml_pipeline/tests/test_model_io_contract.py    -> tests/models/test_model_io_contract.py

1. `git mv` every file to its new path (create the intermediate feature
   directories as you go: fusion/, zupt/, calibration/, mapmatch/,
   routing/, sensors/, vehicle/, export/).
2. Fix every `#include "navicore/..."` site across the whole repo
   (headers, .cpp files, and the unit test) to the new subpaths — grep
   for the old include paths, don't rely on memory.
3. Update core_cpp/CMakeLists.txt's NAVICORE_SOURCES list to the new
   src/ subpaths for every file currently listed, and leave clear
   commented placeholders for hmm_matcher.cpp / nhc_zupt.cpp so DV-10
   only has to uncomment, not re-derive the list.
4. Fix ml_pipeline's two import sites (export_tflite.py's new location,
   and the test file's import of it) accordingly.
5. Confirm `cmake --build .` still succeeds in core_cpp/ and
   `pytest ml_pipeline/tests/` still collects and runs cleanly — this
   chunk moves files, it must not change behaviour.

Acceptance: every file listed above lives at its new path; nothing else
in the repo still references an old path; core_cpp builds clean; existing
ml_pipeline tests still pass unchanged.
```

## DV-01 — Verify IO-VNBD dataset + document resampling strategy
**Maps to**: PRD FR-03, ROADMAP Phase 1 Week 1 Days 1–2

**Prompt to run:**
```
Open /ml_pipeline/src/dataset/iovnbd_loader.py and data/IO-VNBD_SPEC.txt,
data/sample_iovnbd/S-S1.csv and V-S1.csv.

1. Confirm the actual sample rate in the CSVs (the spec says 10 Hz; earlier
   project notes assumed 100 Hz — verify empirically from timestamp deltas,
   don't trust either number blindly).
2. Write a short RESAMPLING_STRATEGY.md in ml_pipeline/ documenting: the
   measured rate, how on-device 100 Hz phone IMU data will be
   downsampled/decimated to match this training rate, and what
   anti-aliasing filter (if any) is applied before decimation.
3. Update iovnbd_loader.py so the sample rate is read from a config value
   (ml_pipeline/config/training_config.yaml), never hardcoded, per
   docs/CODE-STYLE.md §4.
4. Confirm the loader correctly parses both S-S1.csv and V-S1.csv end to end
   with no dropped rows, and print a summary: row count, duration, speed
   range, class balance for any ZUPT/stationary label column.

Acceptance: RESAMPLING_STRATEGY.md exists and is correct; loader runs
cleanly on both sample files and prints the summary; no hardcoded rate
literals remain in iovnbd_loader.py.
```

## DV-02 — Finish and validate the augmentation pipeline
**Maps to**: PRD FR-03, ROADMAP Phase 1 Week 1 Day 4

**Prompt to run:**
```
Open /ml_pipeline/src/dataset/augmentations.py and preprocessing.py.

Verify (and complete if missing) these four augmentations, each
independently toggleable and parameterized (no magic numbers):
1. Synthetic pothole/shock injection (short high-magnitude accel spikes).
2. Engine-idle harmonic injection in the 15–30 Hz band (simulates idling
   traffic vibration so the ZUPT classifier learns to ignore it).
3. Gaussian sensor noise on all 6 IMU channels.
4. ±15° random orientation jitter (simulates imperfect/shifting phone
   mount, independent of the auto-calibration step).

Write a small script/notebook cell that applies each augmentation to one
sample window from data/sample_iovnbd/S-S1.csv and plots before/after, so
correctness is visually verifiable. Confirm preprocessing.py's gravity
decoupling (LPF fc=0.5 Hz) runs before windowing, not after.

Acceptance: all 4 augmentations produce visibly correct, bounded output on
a real sample window; none of them silently no-op.
```

## DV-03 — Implement/verify the 1D-TCN model architecture
**Maps to**: PRD FR-03, TRD §4.1, ROADMAP Phase 1 Week 2 Day 6

**Prompt to run:**
```
Open /ml_pipeline/src/models/tcn_odometer.py.

Verify the architecture exactly matches TRD.md §4.1:
Input [Batch, 6 channels, 100 timesteps] →
Stem Conv1D(32, k=7) + BatchNorm + SpatialDropout(0.1) →
3 dilated residual blocks (dilation 1, 2, 4; channels 32→64→128; each with
  two Conv1D+BatchNorm, a skip connection — 1x1 conv on blocks 2 and 3 —
  and a Squeeze-and-Excitation block before the final ReLU) →
GlobalAveragePooling1D →
Dense(64, ReLU) + Dropout(0.2) →
Head A: forward speed regression (Vx, scalar)
Head B: speed variance / uncertainty (σ_v², adaptive covariance)

Add a third output for the ZUPT/idle classifier logit if it isn't already
a model head (check whether it's currently computed separately from the
spectral engine instead — if so, add it as Head C here per TRD, sharing
the backbone).

Write a shape/contract unit test asserting: input shape
(batch, 6, window_len) in, three outputs out with correct shapes, and that
window_len matches whatever DV-01 determined the resampling target to be.

Acceptance: model builds, forward pass runs on a dummy batch, all 3 heads
present, shape test passes.
```

## DV-04 — Finalize the multi-task loss
**Maps to**: TRD §4.1, ROADMAP Phase 1 Week 2 Day 7

**Prompt to run:**
```
Open /ml_pipeline/src/models/loss.py.

Implement/verify a multi-task loss combining:
1. Gaussian NLL loss on (Vx_pred, σ²_pred) against ground-truth speed —
   this is what makes σ² a real learned uncertainty, not a fixed constant.
2. Binary cross-entropy on the ZUPT/idle head against a stationary-label
   target (derive this label from ground-truth speed ≈ 0 if IO-VNBD
   doesn't ship one directly — document exactly how in a code comment).
3. A configurable weighting between the two terms (from
   ml_pipeline/config/training_config.yaml, not hardcoded).

Add a unit test with synthetic predictions/targets verifying the loss
is finite, differentiable, and decreases when predictions move toward
ground truth.

Acceptance: loss.py has both terms correctly implemented and weighted,
with a passing unit test.
```

## DV-05 — Run a real training job and log measured metrics
**Maps to**: PRD FR-03 acceptance criteria, ROADMAP Phase 1 Week 2 Days 8–9 (this is the actual Phase 1 exit criterion — nothing in the repo currently proves this happened)

**Prompt to run:**
```
Open /ml_pipeline/src/training/train.py and ml_pipeline/config/training_config.yaml.

1. Wire together DV-01's loader, DV-02's augmentations, DV-03's model, and
   DV-04's loss into a real training loop.
2. Split data/sample_iovnbd into train/held-out (document the split logic
   and seed).
3. Run training to convergence on the available sample data. Log every
   run's exact config, seed, and dataset version to a run directory
   (per docs/CODE-STYLE.md §4 — no unlogged runs).
4. On the held-out split, compute and report:
   - Velocity RMSE (m/s) — the actual measured number.
   - ZUPT precision, recall, and false-positive rate (call out FP rate
     specifically — SECURITY.md §7 treats a false ZUPT lock as worse than
     residual drift).
5. Save the best checkpoint and write EVAL_REPORT.md in ml_pipeline/ with
   these numbers, replacing every "target" placeholder elsewhere in the
   docs that currently cites PRD §3.2's unvalidated targets.

Acceptance: EVAL_REPORT.md exists with real, reproducible numbers (not
targets); a checkpoint file exists; the run is logged with config+seed.

Note: if data/sample_iovnbd is too small for a meaningful RMSE, say so
explicitly in EVAL_REPORT.md rather than reporting a misleadingly good
number — pull the full IO-VNBD dataset per REAL_DATA_TRANSITION_PLAN.md
if needed and document that step.
```

## DV-06 — INT8 quantization and a real TFLite export
**Maps to**: TRD §4.2, ROADMAP Phase 1 Week 2 Days 9–10 — **this fixes the fake 37-byte model file**

**Prompt to run:**
```
Open /ml_pipeline/src/models/export/export_tflite.py.

The current models/exported/navicore_odometer_int8.tflite is a 37-byte
placeholder, not a real model — this is the highest-priority ML fix.

1. Export DV-05's trained checkpoint to ONNX/TorchScript, then to TFLite.
2. Apply full post-training INT8 quantization using a representative
   dataset drawn from the held-out split (per TRD §4.2).
3. Overwrite models/exported/navicore_odometer_int8.tflite with the real
   quantized model. Confirm its file size lands near the ~460 KB target
   from TRD §4.2 (report the actual size, don't just assert the target).
4. Run both the Float32 and INT8 versions on the held-out split and report
   the accuracy delta (RMSE difference) in EVAL_REPORT.md.
5. Update models/production_release_manifest.json's sha256/size entry for
   this file to match the real exported model.

Acceptance: the .tflite file is a valid, loadable TFLite model of
plausible size (not 37 bytes); Float32→INT8 accuracy delta is measured
and documented; manifest hash is regenerated and matches.
```

## DV-07a — Model I/O contract test
**Maps to**: API.MD §3, ROADMAP Phase 1 Week 2 Day 10

**Prompt to run:**
```
Create ml_pipeline/tests/models/test_model_io_contract.py.

Load models/exported/navicore_odometer_int8.tflite via the TFLite
interpreter (Python), assert its input tensor shape exactly matches
docs/API.MD §3's documented window_len/channel-count/sample-rate contract,
and assert its output tensor count/shapes match the 2–3 heads from DV-03.
Run inference on one real sample window and assert output values are
finite and in a physically plausible range (Vx ≥ 0, σ² > 0).

This test is the contract Manthan's Android TFLite integration (DV-08)
depends on — if this test passes, the model is safe to wire into the app.

Acceptance: test passes against the real exported model from DV-06.
```

## DV-08 — Wire the real TFLite model into the Android app
**Maps to**: API.MD §3, replaces the `LocalVirtualOdometer` placeholder — **do this after DV-06/DV-07a**

**Prompt to run:**
```
Open android_app/app/src/main/java/org/enigma/navicore/odometer/VirtualOdometer.kt.

Currently LocalVirtualOdometer is a hand-written heuristic
(predictedVx = meanAx * 1.2f) explicitly commented as a
"TFLite execution placeholder" — it never runs the real model.

1. Add the TFLite Android dependency (org.tensorflow:tensorflow-lite,
   +tensorflow-lite-support) to android_app/app/build.gradle.kts.
2. Copy models/exported/navicore_odometer_int8.tflite (the real one from
   DV-06) into android_app/app/src/main/assets/.
3. Implement a new TfliteVirtualOdometer : VirtualOdometer (same
   odometer/ package as VirtualOdometer.kt and LocalVirtualOdometer —
   this package is already correctly feature-scoped, don't split it)
   that loads the model via Interpreter, configures NNAPI delegate as
   primary with XNNPACK as fallback (TRD §4.2), and runs inference on
   each window, returning OdometerOutput(vx, varianceVx, stoppedProb)
   using the model's own 3 heads instead of the hand-written heuristic.
4. Keep LocalVirtualOdometer as an explicit, clearly-labeled Kotlin
   fallback (for devices/tests where TFLite fails to load), selected only
   as a fallback — never silently preferred over the real model.
5. Add a Kotlin instrumented test feeding one known sample window and
   asserting the TFLite path's output is close to the Python inference
   result from DV-07a on the same window (sanity-checks the Android
   integration didn't silently transpose/rescale inputs).

Acceptance: app runs real INT8 inference on-device via NNAPI/XNNPACK;
fallback path still exists but is clearly secondary; parity test passes.
```

## DV-09 — Wire SpectralZuptEngine and NhcKinematics into the ESKF
**Maps to**: PRD FR-05, FR-06, TRD §3.4 — **currently dead code, only touched by the unit test**

**Prompt to run:**
```
Open core_cpp/include/navicore/fusion/eskf.hpp, src/fusion/eskf.cpp, and
include/navicore/zupt/nhc_zupt.hpp.

Today EskfFilter::ApplyZupt() is only triggered externally by the AI
classifier's stoppedProb (see jni_bridge.cpp) — the native
SpectralZuptEngine (frequency-domain idle-harmonic detector) and
NhcKinematics::ComputeAdaptiveCovariance (road-roughness-adaptive NHC
noise) exist fully implemented in src/zupt/nhc_zupt.cpp but are never
instantiated outside core_cpp/tests/fusion/test_eskf.cpp.

1. Add a SpectralZuptEngine member to EskfFilter. Feed it dynamic
   acceleration magnitude every Predict() call. Combine its boolean
   output with the AI classifier's stoppedProb (e.g. ZUPT fires if either
   signals lock with high confidence, or require both to agree — decide
   and document the fusion rule in a code comment, since a false lock is
   explicitly called out as worse than residual drift per SECURITY.md §7).
2. In the AI-odometer measurement update path (UpdateAiOdometer), call
   NhcKinematics::ComputeAdaptiveCovariance using a road-roughness metric
   derived from recent vertical-acceleration variance, and use the
   returned sigma_vy/sigma_vz as the actual NHC measurement noise instead
   of any fixed constant currently in eskf.cpp.
3. Extend core_cpp/tests/fusion/test_eskf.cpp with a case proving: (a) the
   spectral engine alone can trigger ZUPT on synthetic idle-harmonic
   input even if the AI classifier disagrees, and (b) NHC covariance
   visibly widens under synthetic rough-road input vs smooth-road input.

Acceptance: both classes have real call sites inside EskfFilter; new
tests pass; existing tests still pass.
```

## DV-10 — Build the missing `navicore_pipeline` orchestrator and wire HMM, rerouting, barometer, vehicle profiles
**Maps to**: TRD §6.1 (documents this file; it doesn't exist), PRD FR-08

**Prompt to run:**
```
core_cpp currently has no "Main Native Pipeline Orchestrator" despite
TRD.md §6.1 documenting core_cpp/include/navicore/pipeline/navicore_pipeline.hpp
as the central piece — right now jni_bridge.cpp improvises the wiring
inline, and HmmMapMatcher (mapmatch/), DynamicReroutingEngine (routing/),
BarometerTracker (sensors/), and VehicleProfileManager (vehicle/) all have
ZERO call sites anywhere in the repo.

1. Create core_cpp/include/navicore/pipeline/navicore_pipeline.hpp +
   core_cpp/src/pipeline/navicore_pipeline.cpp: a NavicorePipeline class
   that owns one EskfFilter (fusion/), one MountCalibrator (calibration/),
   one HmmMapMatcher (mapmatch/), one DynamicReroutingEngine (routing/),
   and one VehicleProfileManager (vehicle/) instance, and exposes a small
   clean API: ProcessImu(), UpdateGnss(), UpdateAiOdometer(), GetState(),
   GetSnappedRoad(), IsRerouteNeeded().
2. Wire HmmMapMatcher: after every GetState() call, call Match() and
   surface the result (nullable — off-road areas like basements should
   still return null per its own contract) as part of the pipeline's
   output.
3. Wire DynamicReroutingEngine: feed it the current route polyline (stub
   an empty/placeholder route input if the routing UI itself doesn't
   exist yet — document that as a follow-up) and gyro samples for
   dislodgement detection; surface both IsRerouteNeeded() and
   IsDislodged() in the pipeline output.
4. Wire VehicleProfileManager: let the pipeline be constructed with a
   VehicleType, and use the selected profile's NHC stiffness / idle-band
   values instead of the fixed constants DV-09 used, so different vehicle
   types (2-wheeler vs sedan vs truck) actually behave differently.
5. Leave BarometerTracker wiring as a documented TODO with a one-line
   comment explaining it needs a barometer sensor input source that
   doesn't exist yet on the Android side — don't fabricate a fake input
   for it.
6. Update core_cpp/CMakeLists.txt to add src/pipeline/navicore_pipeline.cpp,
   src/mapmatch/hmm_matcher.cpp, and src/zupt/nhc_zupt.cpp to
   NAVICORE_SOURCES (the last two are currently missing from the source
   list despite being real .cpp files — DV-00 left commented placeholders
   for exactly this).

Acceptance: navicore_pipeline.hpp/.cpp exist and compile; all four
previously-dead classes have at least one real call site; core_cpp builds
clean with `cmake --build .`; test_eskf still passes.
```

## DV-11 — Update the Android native CMake + hand off the pipeline API
**Maps to**: hand-off point to Manthan's MN-02

**Prompt to run:**
```
Open android_app/app/src/main/cpp/CMakeLists.txt.

It currently only compiles jni_bridge.cpp + fusion/eskf.cpp +
calibration/auto_calib.cpp — missing zupt/nhc_zupt.cpp,
mapmatch/hmm_matcher.cpp, and (after DV-10) pipeline/navicore_pipeline.cpp,
so none of DV-09/DV-10's work is actually reachable from the Android build.

1. Add src/zupt/nhc_zupt.cpp, src/mapmatch/hmm_matcher.cpp, and
   src/pipeline/navicore_pipeline.cpp (all under core_cpp/, referenced by
   relative path) to the add_library(navicore_jni SHARED ...) source list.
2. Confirm it still builds cleanly for the Android NDK target (arm64-v8a
   at minimum).
3. Write a one-page core_cpp/HANDOFF.md describing the NavicorePipeline
   public API from DV-10 (method signatures, what each returns, threading
   assumptions — is it safe to call from a single background thread only,
   etc.) and note the new module layout (fusion/, zupt/, calibration/,
   mapmatch/, routing/, sensors/, vehicle/, pipeline/) so Manthan can
   rewrite jni_bridge.cpp against it in MN-02 without needing to read all
   of core_cpp himself.

Acceptance: android_app/app/src/main/cpp builds with all core_cpp sources
included at their post-DV-00 paths; HANDOFF.md is accurate and complete
enough that MN-02 doesn't require re-deriving anything from core_cpp
source. Per §3, do not start MN-02 until this chunk's HANDOFF.md is merged.
```

## DV-12 — Compile the final measured-metrics report
**Maps to**: PRD §3.2, ROADMAP Phase 4 Day 46 (ML portion)

**Prompt to run:**
```
Gather EVAL_REPORT.md (DV-05/DV-06), the DV-09 NHC/ZUPT test results, and
any on-device latency numbers Manthan reports from MN-06's profiling pass.

Produce ml_pipeline/FINAL_ML_RESULTS.md containing only measured numbers
(velocity RMSE, ZUPT precision/recall/FP-rate, Float32→INT8 delta, model
size, on-device inference latency) with the exact run/config each number
came from. Explicitly flag anything from PRD §3.2 that is still a target,
not yet measured, or was invalidated by a limited sample dataset size.

Acceptance: every number in this file traces to a specific logged run;
nothing is copied from PRD's target table without a "measured" or
"target — not yet validated" label next to it.
```

---

# TRACK B — Manthan (App, Robotics, Tooling, QA)

## MN-00 — Restructure `android_app/`, `ros2_node/`, `web_visualizer/`, `sdk_python/`, `scripts/` into feature modules
**Maps to**: foundational — do this before MN-01, in parallel with DV-00, per §3

**Prompt to run:**
```
Apply the exact old-path -> new-path mapping in this project's
"Target modular folder structure" doc (§2.3-2.4):

android_app: .../navicore/sensor/ImuSensorManager.kt -> .../navicore/imu/ImuSensorManager.kt
             .../navicore/sensor/NavigationService.kt -> .../navicore/service/NavigationService.kt
             .../navicore/ui/MapScreen.kt -> .../navicore/ui/map/MapScreen.kt
             .../navicore/ui/NavigationViewModel.kt -> .../navicore/ui/viewmodel/NavigationViewModel.kt
             (odometer/, mapmatch/, fusion/ packages are already correctly
             feature-scoped — leave them exactly as they are, and leave
             app/src/main/cpp/ flat, it's the native glue layer and DV-11/
             MN-02 depend on its current single-file shape)

ros2_node:   src/navicore_node.cpp -> src/node/navicore_node.cpp
             create empty src/bridge/ (MN-07 will add
             ai_odometer_bridge.cpp here) and msg/ (MN-07 will add
             AiOdometer.msg here) directories now so MN-07 doesn't also
             have to touch CMakeLists.txt's directory scan pattern later

web_visualizer: app.js -> split by concern into src/js/map-renderer.js,
                src/js/telemetry-panel.js, src/js/socket-client.js
                (index.html and sensor_stream.html stay at root as entry
                points and just update their <script> tags)
                style.css -> src/css/style.css

sdk_python:  navicore_sdk.py -> package navicore_sdk/{__init__.py,
             fusion.py, state.py}; __init__.py re-exports the public
             API so `from navicore_sdk import NavicoreSdk` (or whatever
             the current public class name is) keeps working unchanged
             for anyone importing it

scripts:     test_end_to_end.py, verify_vehicle_kinematics.py,
             strict_verification_harness.py, verify_application.py
                -> scripts/verification/
             simulate_drive.py, record_field_session.py
                -> scripts/simulation/
             profile_hardware_footprint.py, benchmark_suite.py
                -> scripts/profiling/
             realtime_sensor_server.py, launch_visualizer.py
                -> scripts/demo/

1. `git mv` every file per the mapping above.
2. Fix every import/require/<script src>/subprocess-path reference across
   the whole repo (including scripts that call other scripts, and
   launch_visualizer.py's reference to realtime_sensor_server.py) — grep
   for old paths, don't rely on memory.
3. Fix Gradle/package.json/CMake glob patterns if any of them enumerate
   files by directory rather than by explicit list.
4. Confirm the app still builds (`./gradlew assembleDebug`), the
   visualizer still loads with no console errors, `sdk_python`'s existing
   tests (if any) still pass, and every script in scripts/ still runs
   with its original CLI args unchanged.

Acceptance: every file listed above lives at its new path; nothing else
in the repo still references an old path; app build, visualizer, SDK, and
scripts all behave identically to before the move.
```

## MN-01 — Audit and harden IMU ingestion + foreground service
**Maps to**: PRD FR-01, ROADMAP Phase 3 Week 6 Days 27–29

**Prompt to run:**
```
Open android_app/app/src/main/java/org/enigma/navicore/imu/
ImuSensorManager.kt and android_app/app/src/main/java/org/enigma/navicore/
service/NavigationService.kt.

1. Confirm accelerometer + gyroscope are sampled at the highest rate the
   device sustains (target 100 Hz), using a rolling circular buffer with a
   1.0 s window and 50% overlap, per FR-01. Fix if the current window/
   overlap/rate isn't actually parameterized this way.
2. Confirm NavigationService runs as a proper foreground service with a
   persistent notification (required for sustained background sensor
   capture — Android kills background sensor listeners otherwise).
3. Add a dropped-sample counter and log a warning if drop rate exceeds 1%
   over a 10-minute session (FR-01's acceptance criterion).
4. Write a short test/manual checklist confirming sustained 10-minute
   capture on a real device with measured (not assumed) sample-rate
   consistency logged.

Acceptance: FR-01's stated acceptance criteria (≤1% drop over 10 min,
measured rate logged) are actually checked in code, not just assumed.
```

## MN-02 — Rewrite the JNI bridge against Dhruvin's `NavicorePipeline`
**Maps to**: hand-off from DV-11 — do this after DV-10/DV-11 land

**Prompt to run:**
```
Read core_cpp/HANDOFF.md (written in DV-11) for the NavicorePipeline API
and the new core_cpp module layout (fusion/, zupt/, calibration/,
mapmatch/, routing/, sensors/, vehicle/, pipeline/).

Rewrite android_app/app/src/main/cpp/jni_bridge.cpp to own one
NavicorePipeline instance instead of separate raw EskfFilter/
MountCalibrator pointers. Update
android_app/.../fusion/FusionCore.kt's external fun declarations to match
any new native methods (e.g. a native call exposing the HMM-snapped road
segment and reroute/dislodge flags that DV-10 added).

Extend FusionState (Kotlin data class, same fusion/ package) with the new
fields the pipeline now exposes (snapped road id + confidence,
isRerouteNeeded, isDislodged) so the UI layer (MN-04) can consume them.

Acceptance: jni_bridge.cpp is a thin wrapper around NavicorePipeline (no
raw engine wiring left inline); FusionState carries the new fields;
existing GNSS/AI-odometer/ZUPT flows behave identically to before the
rewrite (regression-test manually against MN-01's checklist).
```

## MN-03 — Replace `SimpleOsmMapMatcher` stub with the real HMM bridge
**Maps to**: PRD FR-08, API.MD §5 — currently `return null` unconditionally

**Prompt to run:**
```
Open android_app/app/src/main/java/org/enigma/navicore/mapmatch/
MapMatcher.kt.

SimpleOsmMapMatcher.match() currently just returns null unconditionally —
it never calls into the native HmmMapMatcher DV-10 wired into the
pipeline (now living at core_cpp/{include/navicore,src}/mapmatch/).

1. Replace SimpleOsmMapMatcher with an implementation that reads the
   snapped-road fields MN-02 added to FusionState (populated natively by
   DV-10's HmmMapMatcher) and maps them into the existing
   SnappedCoordinate(lat, lon, roadSegmentId, confidence) contract from
   API.MD §5.
2. Add offline OSM road-network loading (a new file in the same
   mapmatch/ package, e.g. OfflineRoadNetworkLoader.kt): parse a small
   offline OSM PBF/vector extract for the demo area into RoadSegment
   structs and pass them into the native LoadRoadNetwork() call at app
   startup (this is the part that was never built — decide a reasonable
   demo-area extent, e.g. a few km around the planned demo route, and
   document how to swap it for a different city).
3. Confirm match() genuinely returns null in unmapped areas (e.g.
   basement parking) per its own documented contract, rather than always
   returning null as it does today.

Acceptance: map-matching visibly snaps to real roads on a recorded test
log, and returns null (not a wrong guess) when off the loaded road
network.
```

## MN-04 — Implement the 4 UI modes + 10 Hz interpolation + blackout simulation toggle
**Maps to**: PRD §7, ROADMAP Phase 3 Week 7 Days 31–34

**Prompt to run:**
```
Open android_app/app/src/main/java/org/enigma/navicore/ui/map/MapScreen.kt
and android_app/app/src/main/java/org/enigma/navicore/ui/viewmodel/
NavigationViewModel.kt.

Implement/verify, per PRD §7:
1. OPEN SKY MODE — green beacon, shown when FusionMode == OPEN_SKY.
2. AI DEAD RECKONING MODE — cyan pulsing halo, shown during
   FusionMode.DEAD_RECKONING; must display elapsed blackout time AND a
   visible confidence indicator that visibly downgrades once
   blackoutDurationMs exceeds 120000ms (FusionState.withinValidatedRange
   == false) — never show a falsely-confident indicator past that point.
3. TRAFFIC IDLE MODE — gold anchor, shown during FusionMode.ZUPT_LOCKED,
   with a confidence/uncertainty cue, not a flat static "locked" label.
4. RE-CALIBRATION toast — subtle notification when MountCalibrator
   detects dislodgement (surfaced via MN-02's new FusionState field),
   showing real elapsed recalibration time, not a fixed number.
5. A 10 Hz coordinate interpolator smoothing the marker between raw
   FusionCore.getState() polls, so the UI never snap-jumps.
6. A GNSS-blackout simulation toggle in the UI, with the UI copy
   explicitly labeling it "SIMULATED BLACKOUT" wherever it's visible —
   never presented as a real GNSS outage.

Acceptance: all 4 modes are visually distinct and driven by real
FusionState fields (no hardcoded mode); blackout toggle is unambiguously
labeled as simulated; interpolation is smooth on a real device.
```

## MN-05 — Live telemetry overlay with real computed numbers
**Maps to**: ROADMAP Phase 3 Week 8 Day 37

**Prompt to run:**
```
Create android_app/app/src/main/java/org/enigma/navicore/ui/telemetry/
TelemetryOverlay.kt (new file, new ui/telemetry/ package alongside
ui/map/ and ui/viewmodel/) and mount it on MapScreen.kt, showing, live and
computed (never hardcoded): current drift estimate, blackout duration,
confidence/uncertainty value, current FusionMode, and (once MN-03 lands)
the current map-matched road name/id if any.

Source every value directly from FusionState — if a number isn't
available yet from the native layer, show "—" rather than a fabricated
placeholder number.

Acceptance: overlay updates live during a test drive/simulated blackout
and every displayed number is traceable to a FusionState field.
```

## MN-06 — Run the on-device profiling pass
**Maps to**: PRD §3.2 (CPU/battery/latency targets), ROADMAP Phase 3 Day 38

**Prompt to run:**
```
Open scripts/profiling/profile_hardware_footprint.py and
scripts/profiling/benchmark_suite.py — these already exist and look
complete; the task is to actually run them against a real or emulator
build and report results, not to rewrite them.

1. Run the app on at least 2 real Android devices (or clearly document if
   only an emulator was available and why).
2. Measure and record: CPU %, battery drain %/hr, model inference latency
   (DV-08's TFLite path), and end-to-end latency (IMU sample → UI update).
3. Write PERFORMANCE_REPORT.md with the real measured numbers next to
   PRD §3.2's target table, clearly marked measured vs target, and hand
   the inference-latency number to Dhruvin for DV-12.

Acceptance: PERFORMANCE_REPORT.md exists with real numbers from an actual
run on real (or documented emulator) hardware — never numbers adjusted to
match the PRD targets.
```

## MN-07 — Complete the ROS2 node's GNSS-blackout / AI-odometer path
**Maps to**: TRD §6, PRD "Autonomous/Defense AGVs" persona — currently GNSS+IMU only, no blackout handling at all

**Prompt to run:**
```
Open ros2_node/src/node/navicore_node.cpp.

Today this node only calls eskf_->Predict() and UpdateGnss() — it has NO
AI-odometer subscription and therefore no GNSS-blackout dead-reckoning
path at all, despite that being the entire point of NaviCore. It also
never calls ApplyZupt() or any NHC/HMM logic.

1. Create ros2_node/msg/AiOdometer.msg (a small custom msg carrying
   vx/variance/stopped_prob) and ros2_node/src/bridge/
   ai_odometer_bridge.cpp — a subscriber to a new topic
   (/navicore/ai_odometer) that wires the message into
   eskf_->UpdateAiOdometer() / ApplyZupt(), mirroring the Android
   jni_bridge.cpp logic (post MN-02's pipeline rewrite, prefer using
   core_cpp's NavicorePipeline here too instead of a bare EskfFilter, for
   consistency — same pipeline/ module Dhruvin built in DV-10).
2. Publish a GNSS-blackout status string on the existing pub_status_
   topic (OPEN_SKY / DEAD_RECKONING / ZUPT_LOCKED), matching Android's
   FusionMode enum values.
3. Fix PublishState() in navicore_node.cpp: it currently writes
   state.latitude_deg/longitude_deg straight into
   odom.pose.pose.position.x/y, which is wrong for a nav_msgs/Odometry
   message (that expects a local Cartesian frame, not raw lat/lon) —
   either convert to a local ENU frame first, or additionally publish a
   proper sensor_msgs/NavSatFix on a separate topic so downstream ROS
   consumers get valid geographic coordinates.
4. Update ros2_node's CMakeLists.txt/package.xml to build the new
   src/bridge/ source and generate the new msg/AiOdometer.msg.

Acceptance: the ROS2 node has a working GNSS-blackout dead-reckoning path
end-to-end (verifiable by publishing synthetic IMU+GNSS+odometer test
messages and observing mode switch + odom output); odometry message
fields are semantically correct.

Note: docs/ROADMAP.md and PRD.md both explicitly mark ROS2 as "explicitly
not SIH scope" / post-hackathon. Confirm with the team whether this chunk
is actually needed for the submission before spending time on it, or
whether it's safe to defer.
```

## MN-08 — Audit + integration-test the web visualizer and realtime sensor server
**Maps to**: internal demo tooling, not part of the Android deliverable but used for development/judging demos

**Prompt to run:**
```
Open web_visualizer/index.html, sensor_stream.html,
web_visualizer/src/js/{map-renderer.js, telemetry-panel.js,
socket-client.js}, web_visualizer/src/css/style.css, and
scripts/demo/realtime_sensor_server.py, scripts/demo/launch_visualizer.py.

This code already exists and looks substantial (~4600 lines total) — the
task is verification and integration, not a rewrite:
1. Run scripts/demo/launch_visualizer.py end-to-end and confirm it serves
   the visualizer and connects to realtime_sensor_server.py without
   errors.
2. Feed it a recorded IMU/GNSS log (e.g. data/field_logs/live_test_run.csv)
   and confirm the visualizer renders a plausible trajectory and mode
   transitions.
3. Fix any broken data-format mismatches between what the Kotlin/native
   pipeline now emits (post MN-02's FusionState changes) and what
   socket-client.js expects to receive over its socket/HTTP interface.
4. Document how to launch it for a live judge demo in a short section of
   README.md.

Acceptance: visualizer runs against a real recorded log or a live
realtime_sensor_server feed with no console errors, and correctly reflects
mode changes.
```

## MN-09 — Audit and finish the Python Fleet SDK
**Maps to**: PRD §6 competitive table ("post-hackathon" commercial SDK), sdk_python/navicore_sdk/

**Prompt to run:**
```
Open sdk_python/navicore_sdk/{__init__.py, fusion.py, state.py} (post
MN-00 restructure — a Python-only re-implementation of the fusion loop,
not a binding to core_cpp).

1. Confirm its feed_gnss/feed_imu-style API surface (now in fusion.py)
   matches API.MD's documented public contract (state fields, mode names,
   method names).
2. Decide and document explicitly (in a docstring at the top of
   fusion.py) whether this is meant to be (a) a lightweight pure-Python
   reference implementation for partners without native bindings, or (b)
   meant to eventually call into core_cpp via a Python C-extension/
   pybind11 wrapper. Right now it silently reimplements simplified fusion
   logic in Python with its own separate bias/confidence numbers, which
   will silently diverge from the real ESKF's behaviour — pick one and
   make it explicit rather than leaving it ambiguous.
3. Add sdk_python/tests/test_fusion.py exercising feed_gnss/feed_imu with
   a synthetic drive and asserting mode transitions (OPEN_SKY →
   DEAD_RECKONING → back) behave sanely.

Acceptance: the SDK's intended scope is explicit and documented; basic
pytest coverage exists and passes.
```

## MN-10 — QA pass: run the verification harnesses and compile Phase 4 docs
**Maps to**: ROADMAP Phase 4, docs/FINALE_SUBMISSION_DOSSIER.md, JUDGE_QA_DEFENSE_GUIDE.md

**Prompt to run:**
```
Open scripts/verification/{test_end_to_end.py, verify_vehicle_kinematics.py,
strict_verification_harness.py, verify_application.py} and
scripts/simulation/{simulate_drive.py, record_field_session.py}.

1. Run all of them against the current build (post MN-02/MN-03/MN-07 and
   Dhruvin's DV-06 real model) and fix whatever breaks — these were
   likely written/last passing against an earlier state of the code.
2. Run scripts/simulation/simulate_drive.py to generate synthetic
   30s/60s/120s/300s blackout stress tests at 20–80 km/h (TRD §8 point 3)
   and report actual drift % for each — compare against PRD §3.2's
   "single-digit % of distance, ≤60s" target and be honest in the report
   if it doesn't hold at 120s/300s (it isn't supposed to, per PRD's
   stated validated range).
3. Compile RESULTS_SUMMARY.md (ROADMAP Phase 4 Day 46) merging: DV-12's
   ML results, MN-06's performance report, and this chunk's drift-test
   results — every number sourced, nothing copied from a target table
   without a measured/target label.
4. Do the Day 49 fact-check pass: go through docs/FINALE_SUBMISSION_DOSSIER.md
   and JUDGE_QA_DEFENSE_GUIDE.md line by line and flag any claim that
   isn't backed by something in RESULTS_SUMMARY.md — mark it as future
   work instead of shipping it as a claim.

Acceptance: all verification scripts run clean against the current build;
RESULTS_SUMMARY.md exists with sourced numbers; the finale dossier has no
unsupported claims left in it.
```

---

## 5. Suggested sequencing (maps back to ROADMAP.md's phases)

```
Week 0     DV-00 (core_cpp + ml_pipeline restructure)   \  run in parallel,
           MN-00 (app/ros2/visualizer/sdk/scripts        > both merge to
                  restructure)                          /  main before Week 1

Week 1–2   DV-01 → DV-02 → DV-03 → DV-04 → DV-05 → DV-06 → DV-07a   (Track A, Phase 1)
           MN-01 (Track B, can start immediately, independent)

Week 3–5   DV-09 → DV-10 → DV-11                                    (Track A, Phase 2)
           MN-02 (blocked on DV-11 handoff) → MN-03 (blocked on DV-10)

Week 6–8   DV-08 (blocked on DV-06/DV-07a)                          (Track A, Phase 3 support)
           MN-04 → MN-05 → MN-06 → MN-07 → MN-08                    (Track B, Phase 3)

Week 9–10  DV-12                                                     (Track A, Phase 4)
           MN-09 → MN-10                                             (Track B, Phase 4)
```

Hand-off points to watch: **DV-00/MN-00 → everything** (nothing else branches until both merge), **DV-11 → MN-02**, **DV-10 → MN-03**, **DV-06/DV-07a → DV-08**. Everything else in each track can run in parallel with the other track.