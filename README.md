# 🖐️ Connected Rehabilitation Glove — Adaptive Hand Motor Recovery

> **ESIEA — 3rd-year engineering student**
> Solo project · Hardware + Embedded AI · Portfolio for biomedical/robotics engineering internship

[![Python](https://img.shields.io/badge/Python-3-blue?logo=python)](https://python.org)
[![Arduino](https://img.shields.io/badge/Arduino-ESP32-teal?logo=arduino)](https://arduino.cc)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-K--NN-orange)](https://scikit-learn.org)

> ⚠️ Research / portfolio prototype. Not a medical device, not clinically validated. See [Known limitations](#️-known-limitations--roadmap).

---

## 📌 Overview

Hand motor impairment affects millions of patients across a wide spectrum of conditions: **stroke** (hemiplegia, spasticity), **Parkinson's disease** (resting tremor, bradykinesia), **Guillain-Barré syndrome** (peripheral motor nerve damage), **post-traumatic rehabilitation** (hand/wrist fractures, tendon and crush injuries) and **post-surgical orthopedic recovery**.

Motor re-learning relies on high-intensity, repetitive, task-specific practice (Langhorne et al., 2009). Sustaining high repetition volumes without structured, adaptive feedback is a persistent clinical challenge. This prototype does **not** enforce a fixed clinical volume: target repetitions per series range from 5 to 40 (see [Session structure](#session-structure)).

This project is a **low-cost connected glove** that:
- Measures finger flexion (5 flex sensors) and grip pressure (3 FSR) in real time
- Classifies the patient's motor profile with a **k-NN classifier** (4 profiles) from 4 standardized tests
- Adapts exercise difficulty between series with a **rule-based adaptive policy** (clinical-style rules, safety bounds, reward shaping computed and logged)
- Estimates the amplitude trend with a **linear regression** on session history
- Guides the patient with an OLED display, LED, buzzer and a single button

The pipeline has four layers (0–3). Layer 0 is signal processing; layers 1 and 3 are classical machine-learning models (k-NN, linear regression); layer 2 is a rule-based policy. The motor-assessment features are generic biomechanical quantities (amplitude, speed, tremor proxy, force, asymmetry); no condition-specific logic is implemented. The system has not been validated on any patient population.

Hardware acquisition, filtering, repetition counting and the user interface run on an **ESP32**. All classification, adaptation and prediction logic runs on a **Raspberry Pi 5** (Flask server).

> **Naming note.** Some source files, classes, comments and JSON fields keep historical names from an earlier design iteration: `couche2_rl_env.py`, the class `AgentRL`, the field `message_rl`, "RL" in docstrings and log lines, and "AVC" (stroke) in the firmware header and simulator docstring. Despite these names, **layer 2 is a rule-based policy, not reinforcement learning** (no learning, no Q-table, no value function), and the assessment logic is not stroke-specific.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        GLOVE (ESP32)                        │
│  5× Velostat flex sensors (via CD74HC4067 mux, ch. C0–C4)   │
│  3× FSR pressure sensors (direct ADC: GPIO34/35/36)         │
│                                                             │
│  Layer 0: EMA filter (α = 0.2)                              │
│  State machine · rep counting · OLED / LED / buzzer / button│
│  HTTP (WiFi): GET /status · POST /evaluation · POST /data   │
└────────────────────────────┬────────────────────────────────┘
                             │  WiFi / HTTP JSON
┌────────────────────────────▼────────────────────────────────┐
│                  BRAIN (Raspberry Pi 5, Flask)              │
│                                                             │
│  Layer 1: Motor profile classification                      │
│           k-NN · 5 features · 4 profiles                    │
│  Layer 2: Adaptive difficulty policy (rule-based)           │
│           + Safety Envelope + exercise library              │
│  Layer 3: Amplitude trend prediction (linear regression)    │
│  Messages returned to the glove (coach text, adaptation     │
│  message, prediction) displayed on the OLED                 │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline — layer by layer

| Layer | Role | Method | Location |
|-------|------|--------|----------|
| **0 — Signal filtering** | Smooth sensor noise | EMA filter (α = 0.2) | ESP32 (C++) |
| **1 — Motor profile classification** | Identify the motor profile from 5 features | k-NN (scikit-learn) | RPi (Python) |
| **2 — Adaptive difficulty** | Adjust exercise level / switch exercise after each series | Rule-based policy · Safety Envelope · reward shaping (logged) | RPi (Python) |
| **3 — Progress prediction** | Estimate amplitude trend over 2 weeks | Linear regression | RPi (Python) |

---

## 🔧 Hardware

| Component | Role |
|-----------|------|
| ESP32 DevKit | Sensor reading, EMA filtering, state machine, WiFi/HTTP |
| CD74HC4067 | 16-channel analog multiplexer (channels C0–C4 used for the 5 flex sensors) on one analog GPIO |
| 5× Velostat flex sensors (handmade) | Per-finger flexion, mapped to 0°–90° |
| 3× FSR pressure sensors | Grip pressure — thumb, index, middle finger (mapped to 0–100 %) |
| OLED SSD1306 128×64 (I²C, address 0x3C) | Instructions, feedback, messages |
| LED + buzzer + button | Multimodal feedback and single-button control |
| Raspberry Pi 5 | Flask server, k-NN, adaptive policy, regression |

### Wiring

```
CD74HC4067:
  SIG → GPIO39    S0 → GPIO33    S1 → GPIO32
  S2  → GPIO17    S3 → GPIO16    EN  → GND    VCC → 3.3V
  C0–C4 → Velostat flex sensors (fingers 1–5)
  (other channels not used)

FSR pressure sensors (direct ADC):
  Thumb → GPIO34    Index → GPIO35    Middle → GPIO36

OLED SSD1306 (I2C):  SDA → GPIO21    SCL → GPIO22

Other:  LED → GPIO4    Buzzer (passive) → GPIO14    Button → GPIO18
```

Each multiplexer channel is read as the average of 8 ADC samples (12-bit, 11 dB attenuation), then EMA-filtered.

---

## 🎮 User interaction (firmware)

**Single button**

| Action | Detection | Use |
|--------|-----------|-----|
| Long press | > 1500 ms | Wake up (sleep screen) / shut down (final screen) |
| Short press | < 1500 ms, not followed by a second press within 400 ms | Next / confirm; during an exercise: declare the series failed |
| Double press | two short presses; the second release occurs < 400 ms after the first release | Validate / start |

A short press is therefore reported with a ~400 ms delay (the firmware waits to rule out a double press).

**Feedback**
- LED: on while the movement is detected as correct during an exercise, off otherwise; blinks at wake-up, on WiFi connection, on series success and at the end of the session.
- Buzzer: one short beep per validated repetition; a double beep when a series is **successfully** completed (not after a manual failure); 3 long beeps when the end-of-session screen is reached.

**State machine:** sleep → welcome → (`GET /status`) → [hardware screen → evaluation (4 tests) → profile] → exercise → (series success screen) → adaptation → coach message → exercise … → end of session → prediction → sleep.

**Repetition counting (on the ESP32):** a movement is considered correct when the mean flexion of the 5 fingers is > 45° and the mean deviation of the fingers from that mean is < 25°. A repetition is counted about 1 s after the correct posture is first detected. The timer is **not** reset when the posture is lost, and while the posture is held continuously one repetition is counted roughly every second; there is no requirement to reopen the hand between repetitions. A series is successful when the target number of repetitions is reached; a short press during an exercise declares a voluntary failure. The same criterion is applied to every exercise (see limitations).

**Series buffer:** during an exercise the glove records 5 angles + 3 pressures every 300 ms, up to 200 samples (≈ 60 s); later samples of a longer series are not recorded. The buffer is sent to the server at the end of the series.

**Offline:** if WiFi is unavailable at boot the glove displays "Mode hors-ligne"; the server is required for evaluation and exercises (nothing is sent without WiFi, and there is no retry).

---

## 🤖 AI Pipeline — Technical Details

### Layer 0 — EMA filter (ESP32)

`y[n] = α·x[n] + (1−α)·y[n−1]`, α = 0.2, applied to the 5 flex channels and the 3 pressure channels (once per main-loop iteration). Flex readings are converted to angles by a linear map between per-finger `flexMin`/`flexMax` constants hard-coded in the firmware (and duplicated in `couche1_knn.py`, where they are not used for computation). For all five fingers `flexMax` is 4094 (ADC full scale); `flexMin` values were measured on the prototype and differ per finger.

### Layer 1 — Motor profile classification (k-NN)

**Assessment protocol — 4 tests** (sampled every 200 ms, at most 30 samples = 6 s per test; instructions displayed on the OLED, in French; the patient moves to the next test with a short press):

| Test | Instruction | Used to compute |
|------|-------------|-----------------|
| Test 1 | Close the hand slowly, 5× | Amplitude, speed |
| Test 2 | Hold the hand open, still, 10 s | Tremor proxy |
| Test 3 | Squeeze the ball hard, 5× (flex + FSR) | Force, speed, asymmetry |
| Test 4 | Repeat opening quickly, 20× | Asymmetry |

The tests are inspired by clinical assessments (Fugl-Meyer FMA-UE, ARAT, UPDRS tremor items); **no clinical score is computed or reproduced**.

**5 extracted features** (`ExtracteurMetriques`, scale 0–100):

| Feature | Computation |
|---------|-------------|
| Amplitude | Mean over fingers of the maximum angle in Test 1, divided by 90° |
| Speed | Largest mean inter-sample angle change (across fingers) over Tests 1 and 3, normalised by 45° per sample interval, capped at 100 |
| Force | Mean FSR pressure (%) in Test 3 |
| Tremor (proxy) | Mean temporal variance of the angles in Test 2, normalised by 50, capped at 100. This is a variance measure, **not** a spectral/FFT analysis (the sampling rate is too low for that) |
| Asymmetry | Standard deviation across fingers of the maximum angle, averaged over Tests 3 and 4, normalised by 45°, capped at 100 |

An empty test buffer is replaced by zeros.

**4 motor profiles (output classes):**

| Profile | Description | Exercises (in order) |
|---------|-------------|----------------------|
| `deficit_mobilite` | Reduced range of motion, low force and speed | E1, E2, E3, E7 |
| `deficit_controle` | High tremor proxy, moderate amplitude | E3, E8, E9, E4 |
| `deficit_coordination` | High inter-finger asymmetry | E2, E4, E5, E6 |
| `recuperation` | High amplitude/speed/force, low tremor and asymmetry | E6, E9, E10, E7 |

**Training data (simulated):** the model is trained at server start-up on a synthetic dataset (`generer_dataset`: 80 samples per class, fixed seed 42, Gaussian, clipped to 0–100). Means (σ) per feature `[amplitude, speed, force, tremor, asymmetry]`:

| Profile | amplitude | speed | force | tremor | asymmetry |
|---------|-----------|-------|-------|--------|-----------|
| mobilité | 18 (8) | 22 (7) | 14 (7) | 12 (5) | 20 (8) |
| contrôle | 55 (10) | 42 (15) | 50 (12) | 78 (10) | 30 (10) |
| coordination | 45 (12) | 38 (10) | 40 (10) | 18 (8) | 80 (8) |
| récupération | 83 (7) | 76 (8) | 79 (8) | 10 (5) | 14 (6) |

**Classifier:** `StandardScaler` + `KNeighborsClassifier` (k = 5, Euclidean), 5-fold cross-validation accuracy printed at training. At inference the 20 nearest neighbours are combined with inverse-distance weighting to produce a percentage per profile; the highest is the dominant profile (its percentage is logged as confidence). Because the data are simulated, the cross-validation accuracy measures separability of the synthetic classes, **not clinical performance**. `modele_knn.json` is a serialized copy (scaler parameters + standardized training data) written by `couche1_knn.py`; the server does not load it and retrains at start-up.

### Exercise library (`exercises.py`)

10 exercises × 5 difficulty levels. Each level defines target repetitions, hold duration, speed label, success threshold and the OLED instruction. Per-exercise reward weights `(amplitude, force, speed, tremor, asymmetry)`:

| ID | Exercise | Main metric | Reps L1→L5 | Weights |
|----|----------|-------------|-----------|---------|
| E1 | Ouverture/Fermeture | amplitude | 10, 15, 20, 30, 40 | .5 .1 .2 .1 .1 |
| E2 | Flexion par doigt | asymmetry | 5, 10, 15, 20, 25 | .2 .1 .1 .2 .4 |
| E3 | Flexion contrôlée | tremor | 5, 8, 10, 12, 15 | .3 .1 .1 .4 .1 |
| E4 | Opposition Pouce-Index | force | 8, 12, 15, 20, 25 | .2 .4 .1 .2 .1 |
| E5 | Séquence pouce-doigts | asymmetry | 5, 8, 10, 15, 20 | .1 .1 .2 .2 .4 |
| E6 | Séquence digitale | asymmetry | 5, 8, 12, 15, 20 | .2 .1 .2 .1 .4 |
| E7 | Pression progressive | force | 8, 10, 12, 15, 20 | .2 .5 .1 .1 .1 |
| E8 | Maintien isométrique | tremor | 5, 6, 8, 10, 12 | .1 .3 0 .5 .1 |
| E9 | Cible angulaire | tremor | 5, 6, 8, 10, 12 | .3 .1 .1 .4 .1 |
| E10 | Tâche fonctionnelle | force | 5, 8, 10, 12, 15 | .2 .3 .1 .2 .2 |

**What is used at runtime.** The server/firmware use the target repetitions and the OLED instruction of the current level, the exercise name (in the adaptation message when switching exercise) and the reward weights (reward is computed and logged). Success thresholds and main metrics are used by the offline simulator only. Hold durations (`duree_s`), speed labels, exercise descriptions, `profils_cibles` and the per-exercise `MAX_SERIES` attribute are currently informational and not used by any code (the series cap is the module constant `MAX_SERIES = 5` in `couche2_rl_env.py`).

### Layer 2 — Adaptive difficulty policy (rule-based)

A **rule-based adaptive policy** decides, after each completed series, how to change the difficulty. It is not a learned policy: there is no Q-table, value function or training. The decision depends only on the binary outcome of the series (target repetitions reached or not), the recent history and counters.

**State per patient** (saved in `progress_patient.json`): profile, exercise list and index, difficulty level [1–5], session number, last 3 series results, series count on the current exercise, consecutive successes at level 5, fatigue index [0–1] (+0.04 per series, reset on exercise change).

**Actions (evaluated in this order after each series):**

| Action | Effect | Condition |
|--------|--------|-----------|
| A3 — Next exercise | Exercise index + 1, level reset to 1, counters and fatigue reset (end of session if it was the last exercise) | 3 consecutive successes at level 5, **or** 5 series done on the current exercise |
| A1 — Increase | Level + 1 (max 5), history reset | Last 3 series all successful |
| A2 — Decrease | Level − 1 (min 1), history reset | Last 2 series failed, or fatigue > 0.7 with a failed series |
| A0 — Maintain | Level unchanged | Otherwise |

With the current constants, three of these conditions cannot be reached in practice (see [limitations](#️-known-limitations--roadmap)): the "3 successes at level 5" condition, the "fatigue > 0.7" condition, and consequently levels 4 and 5 in normal use.

**Reward shaping (computed and logged, not used for decisions):**

`R = w_amp·amplitude + w_force·force + w_speed·speed + w_tremor·(1 − tremor) + w_asym·(1 − asymmetry) − 0.2·[series failed] − 0.15·[fatigue > 0.7]`

with the exercise-specific weights above and metrics in [0, 1] computed by the server from the series buffer (same definitions as layer 1, on a single continuous buffer; the tremor term is the variance of the whole series).

**Safety Envelope:** difficulty always clamped to [1, 5]; a maximum of 5 series per exercise before automatic progression (prevents spending a whole session on one exercise).

### Session management

| Session | Behaviour |
|---------|-----------|
| 1 | Mandatory k-NN assessment (4 tests) |
| 2–4, 6–9, … | No assessment; state reloaded from `progress_patient.json`; restarts from the first exercise of the profile; the difficulty carried over is the one the patient had on the last exercise of the previous session; series counters, history and fatigue reset |
| 5, 10, 15, … (multiples of 5) | New k-NN assessment; a new patient state is created (level reset to 1) |

Behaviour to be aware of:
- The server decides that a new session starts when the glove calls `GET /status` after the previous session has ended. If the glove is switched off **mid-session**, the next `GET /status` resumes the current exercise with the saved state (no new session, no re-assessment).
- The "session ended" flag is cleared as soon as `GET /status` answers. If a session that requires an assessment (1, 5, 10…) is interrupted before `POST /evaluation` is received, the next `GET /status` resumes the previous saved state without assessment.
- The per-session amplitude list used for the prediction is held in server memory and is lost if the server restarts mid-session.

### Session structure

A session runs through the 4 exercises of the profile. Each *series* is one attempt at the target repetitions of the current level (5–40 depending on exercise and level), at most 5 series per exercise (so at most 20 series per session). Adaptation takes place **between series**, not within one.

### Layer 3 — Progress prediction (linear regression)

At the end of each session the mean amplitude (degrees, from the maximum angle per finger over each series) over its series is appended to `historique_patient.json`. The message returned at the end of the session is built as follows:
- before session 4, or with fewer than 2 history points, a generic message "Progression en cours : quelques seances de plus sont necessaires pour une estimation chiffree." is returned (no numerical estimate);
- if the history spans less than 0.5 day, only the last mean amplitude is reported, with a recommendation to continue sessions daily;
- otherwise a linear regression of amplitude versus time (days) is fitted and evaluated 14 days after the last point; the prediction is clamped to 0–90° and the difference with the **last measured amplitude** is limited to ±20°;
- the resulting message is French: positive trend (≥ +1°), trend to monitor (≤ −1°), or stable amplitude (|Δ| < 1°). The positive and the "to monitor" messages recommend confirming with a physiotherapist; the "stable" message recommends regularity.

The message is shown on the end-of-session screen; the OLED shows at most 3 lines of 21 characters, so long messages are truncated.

### Coach / messages

After each series the server returns a short status message (series number and the adaptation decision; "Bravo, seance complete !" at session end), displayed on the OLED coach and adaptation screens. LLM-generated coaching is **not implemented** in this version of the repository.

---

## 🌐 Server API (`rpi/pipeline_bridge.py`, port 5000)

| Route | Sent by the glove | Main response fields |
|-------|-------------------|----------------------|
| `GET /status` | — | `evaluation_requise`, `seance`, `materiel_requis`; if no evaluation: current exercise (`exercice_id`, `consigne_oled`, `repetitions`, `difficulte`, `profil`, `exercice_position`, `exercice_total`) |
| `POST /evaluation` | `test1_angles`, `test2_angles`, `test3_angles`, `test3_pression`, `test4_angles` | `profil`, `profil_fr`, current exercise fields |
| `POST /data` | `serie_angles`, `serie_pression`, `succes`, `echec_manuel` (also `angles`, `pressure_pct` snapshots, ignored by the server) | during a session: `fin_seance` (false), `coach`, `message_rl`, current exercise fields. At end of session: `fin_seance` (true), `coach`, `message_rl`, `prediction` only (no exercise fields) |

`POST /data` returns HTTP 400 if the server has no patient state (no evaluation performed since start-up and no saved `progress_patient.json`). The server is single-patient / single-glove.

---

## 🗂️ Repository Structure

```
rehab-glove-ai/
├── esp32/
│   └── gant_esp32_v2.ino          # Firmware: EMA, state machine, rep counting, HTTP
├── ia/
│   ├── couche1_knn.py             # Feature extractor + k-NN classifier
│   ├── couche2_rl_env.py          # Adaptive policy + session manager (historical filename)
│   ├── exercises.py               # Exercise library (10 exercises × 5 levels)
│   ├── patient_sim.py             # Patient simulator (offline tests)
│   └── modele_knn.json            # Serialized k-NN scaler / training data
├── rpi/
│   ├── pipeline_bridge.py         # Flask server
│   ├── couche1_knn.py             # (mirror)
│   ├── couche2_rl_env.py          # (mirror)
│   ├── exercises.py               # (mirror)
│   ├── modele_knn.json            # (mirror)
│   └── historique_patient_exemple.json   # Example session history
├── docs/
│   ├── schema_kicad.pdf           # KiCad schematic export
│   └── (link to glove demo video)
├── .gitignore
└── README.md
```

The Flask server imports the modules located next to it, i.e. the `rpi/` copies; the `ia/` copies are used for offline tests. The mirrored files are maintained by hand and must be kept identical.

Files created at runtime in the working directory of the server: `progress_patient.json`, `historique_patient.json`, `session_state.json`.

---

## 🚀 Setup & Run

### ESP32
1. Open `esp32/gant_esp32_v2.ino` in Arduino IDE 2.x (ESP32 board package installed).
2. Install `ArduinoJson` (the firmware is written for the v6 API: `StaticJsonDocument` / `DynamicJsonDocument`), `Adafruit SSD1306`, `Adafruit GFX`.
3. Set `WIFI_SSID`, `WIFI_PASSWORD` and `SERVER_BASE` (e.g. `http://<RPi-IP>:5000`) in the configuration block (never commit real credentials).
4. Flash the ESP32.

### Raspberry Pi 5
```bash
git clone https://github.com/elena484-r/rehab-glove-ai
cd rehab-glove-ai/rpi
pip install flask numpy scikit-learn --break-system-packages
python3 pipeline_bridge.py    # trains the k-NN, then starts Flask on port 5000
```

The server listens on all interfaces (`0.0.0.0`) in Flask debug mode, without authentication or encryption: use it only on a trusted local network.

### Offline tests (no hardware)
```bash
cd rehab-glove-ai/ia
python3 couche1_knn.py     # trains the k-NN, prints CV accuracy, tests 4 cases, writes modele_knn.json
python3 couche2_rl_env.py  # simulates 8 series with the adaptive policy
python3 patient_sim.py     # simulates 3 sessions (one try per exercise at level 2) for deficit_mobilite
python3 exercises.py       # lists the exercise library per profile
```

---

## ⚠️ Known Limitations & Roadmap

| Limitation | Context |
|-----------|---------|
| Handmade Velostat sensors, non-linear | Per-finger `flexMin`/`flexMax` constants are hard-coded (`flexMax` = ADC full scale for all fingers); no calibration at start-up. Absolute calibration is not possible with this material. Spectra Symbol flex sensors are the natural next step. |
| Classifier trained on simulated data | Synthetic Gaussian distributions. Validation on a real cohort (stroke, Parkinson's, post-traumatic…), with a physiotherapist or neurologist, is required before any clinical use. |
| Tremor is a variance proxy | Sampling at 200–300 ms (≈ 3–5 Hz) gives a Nyquist limit of ≈ 1.7–2.5 Hz, which does not allow spectral tremor analysis. |
| Generic repetition criterion | The ESP32 validates repetitions with the same posture criterion for all exercises; exercise-specific validation (single finger, hold duration, target angle) is not implemented. The hold timer is not reset when the posture is lost, and a continuously held posture is counted about once per second. |
| Reward is logged only | Decisions are rule-based; the reward is not used for learning. |
| Levels 4–5 are not reachable in practice | Three consecutive successes are needed to go up one level and an exercise is left after 5 series, so within one exercise the difficulty level increases at series 3 or 4. Every new exercise restarts at level 1, hence only the first exercise of a session (which inherits the previous difficulty) can reach level 3, and the "3 successes at level 5" rule never triggers. Lowering the number of successes required, or raising the series cap, is future work. |
| Fatigue rule inactive | Fatigue grows by 0.04 per series and is reset when the exercise changes (max 5 series), so it never exceeds 0.2: the "fatigue > 0.7" decrease rule and the 0.15 reward penalty are never triggered. |
| Fixed evaluation buffers | 30 samples (6 s) per test, whereas the instructions mention a 10 s hold (Test 2) and 20 repetitions (Test 4); later samples are ignored. Series buffers are capped at 200 samples (≈ 60 s). |
| JSON payload sizes on the ESP32 | `envoyerEvaluation` serialises up to ~690 values into a `StaticJsonDocument<2048>` and `envoyerDonnees` into a `DynamicJsonDocument(24576)` (≈ 150 samples). These capacities are small compared with the worst-case payloads and must be checked (`overflowed()`, server-side logging of array lengths) and enlarged if data are truncated. |
| No retry on network failure | If an HTTP request fails, the series buffer is already cleared and the data of that series are lost. |
| Single patient, no security | One patient/one glove per server, no authentication, plain HTTP, Flask debug mode enabled. |
| No LLM coach | Messages are generated by fixed rules. |
| No pain detection / emergency stop | A between-session pain questionnaire and an emergency stop are future work. |
| No CE marking | A full MDR 2017/745 conformity study would be required for medical use. |

---

## 📚 References

- **Langhorne P., Coupar F., Pollock A. (2009).** *Motor recovery after stroke: a systematic review.* Lancet Neurology, 8(8), 741–754.
- **Fugl-Meyer AR et al. (1975).** *The post-stroke hemiplegic patient: a method for evaluation of physical performance.* Scand J Rehabil Med.
- **Lyle RC (1981).** *A performance test for assessment of upper limb function in physical rehabilitation treatment and research.* Int J Rehabil Res (ARAT).
- **Unified Parkinson's Disease Rating Scale (UPDRS)** — tremor and bradykinesia items.
- **Hoehn MM & Yahr MD (1967).** *Parkinsonism: onset, progression and mortality.* Neurology.
- **Hughes RA & Cornblath DR (2005).** *Guillain-Barré syndrome.* Lancet, 366(9497), 1653–1666.
- **Colditz JC.** *Therapist's management of the stiff hand.* In: Rehabilitation of the Hand and Upper Extremity.

The scales above inspired the assessment design and provide clinical context; they are not implemented as scoring systems.

---

## 👩‍💻 Author

**Elena Reca** — ESIEA, 3rd-year engineering student, seeking an internship from September 2027F
Target specialization: Biomedical Engineering · Medical Robotics · Embedded AI
System design and hardware integration carried out independently; software pipeline developed with the support of AI tools — all code fully reviewed, manually tested on hardware, and mastered by me.

*Portfolio project — June–September 2026*

---

*This is a student research prototype. It is not a certified medical device and must not be used for clinical diagnosis or treatment.*
