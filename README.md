# 🖐️ Connected Rehabilitation Glove — Adaptive Hand Motor Recovery

> **ESIEA — 2nd year preparatory cycle**  
> Solo project · Hardware + Embedded AI · Portfolio for biomedical/robotics engineering internship

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Arduino](https://img.shields.io/badge/Arduino-ESP32-teal?logo=arduino)](https://arduino.cc)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-K--NN-orange)](https://scikit-learn.org)
[![Claude API](https://img.shields.io/badge/Claude-API-blueviolet)](https://anthropic.com)

---

## 📌 Overview

Hand motor impairment affects millions of patients across a wide spectrum of pathologies: **stroke** (hemiplegia, spasticity), **Parkinson's disease** (resting tremor, bradykinesia), **Guillain-Barré syndrome** (peripheral motor nerve damage), **post-traumatic rehabilitation** (hand/wrist fractures, tendon injuries, crush injuries), and **post-surgical orthopedic recovery**.

In all these contexts, evidence-based rehabilitation requires **300–400 repetitions per session** to drive neuroplasticity and motor re-learning (Langhorne et al., 2011), yet sustaining this intensity without real-time adaptive feedback is a persistent clinical challenge.

This project is a **low-cost connected glove** designed for any patient requiring structured hand motor rehabilitation, that:
- Measures finger flexion and grip force in real time
- Classifies the patient's motor deficit profile using a **k-NN classifier**
- Adapts exercise difficulty dynamically via a **Q-Learning RL agent**
- Predicts motor recovery trajectory via **linear regression**
- Generates personalized French coaching messages via the **Claude API**

All AI processing runs on a Raspberry Pi 5. The ESP32 handles only signal acquisition and EMA filtering, keeping the embedded layer lightweight and auditable.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        GLOVE  (ESP32)                       │
│   5× Velostat flex sensors  +  3× FSR pressure sensors     │
│   CD74HC4067 multiplexer  →  16-ch analog on 1 GPIO        │
│                                                             │
│   Layer 0 : EMA filter  (α = 0.2, C++)                     │
│   → JSON payload via WiFi HTTP POST every 500 ms           │
└────────────────────────────┬────────────────────────────────┘
                             │  WiFi / HTTP
┌────────────────────────────▼────────────────────────────────┐
│                    BRAIN  (Raspberry Pi 5)                  │
│                                                             │
│   Layer 1 — Motor Profile Classification                    │
│             k-NN  ·  5 biomechanical features  ·  4 classes │
│                                                             │
│   Layer 2 — Adaptive Difficulty (RL agent)                  │
│             Q-Learning · Bellman equation · ε-greedy        │
│             Safety Envelope (clinical constraints)          │
│                                                             │
│   Layer 3 — Recovery Prediction                             │
│             Linear regression on session history            │
│             → "＋12° in ~2 weeks" displayed on OLED        │
│                                                             │
│   Coach — Claude API                                        │
│             Personalized motivational messages in French    │
│             Displayed on OLED SSD1306                       │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline — Layer by Layer

| Layer | Role | Method | Location |
|-------|------|--------|----------|
| **0 — Signal filtering** | Smooth Velostat sensor noise | EMA filter (α = 0.2) | ESP32 C++ |
| **1 — Motor profile classification** | Identify deficit type from 5 extracted features | k-NN (k=5, scikit-learn) | RPi Python |
| **2 — Adaptive difficulty** | Adjust exercise level in real time | Q-Learning · Bellman · ε-greedy | RPi Python |
| **3 — Recovery prediction** | Predict AROM trajectory over weeks | Linear regression | RPi Python |
| **Coach** | Transform metrics into natural language | Claude API (claude-3-haiku) | RPi Python |

---

## 🤖 AI Pipeline — Technical Details

### Layer 1 — Motor Profile Classification (k-NN)

The classifier maps **5 biomechanical features** to **4 motor deficit profiles**, following the clinical framework of the **Fugl-Meyer Assessment for Upper Extremity (FMA-UE)** and the **Action Research Arm Test (ARAT)**.

**Assessment protocol — 4 standardized tests:**

| Test | Clinical Name | Reference Scale | Feature extracted |
|------|--------------|-----------------|-------------------|
| Test 1 — Full flexion | Dynamic Goniometry · Active Range of Motion (AROM) | Fugl-Meyer (FMA-UE) | Amplitude [°] |
| Test 2 — Sustained hold | Static Postural Control · Tremor Analysis (FFT) | UPDRS | Tremor power [Hz] |
| Test 3 — Repeated grips | Inter-finger Coordination · Fine Dexterity Kinematics | Box & Block Test · ARAT | Asymmetry index |
| Test 4 — Repeated series | Muscular Fatigability · Motor Endurance Index | Isokinetic protocols | Fatigue decay rate |

**5 extracted features (input vector to k-NN):**

| Feature | Biomechanical term | Unit |
|---------|-------------------|------|
| Amplitude | Active Range of Motion (AROM) | degrees |
| Speed | Peak Angular Velocity (ω_max) | °/s |
| Tremor | Postural Tremor Power Spectrum (FFT 3–12 Hz) | normalized |
| Coordination | Inter-finger Phase Lag / Synergy Index | normalized |
| Fatigue | Force/Amplitude Decay Rate | normalized |

**4 motor deficit profiles (output classes):**

| Profile | Clinical description | Exercises assigned |
|---------|---------------------|-------------------|
| `deficit_mobilite` | Reduced AROM — limited finger extension/flexion amplitude | E1, E2, E3, E7 |
| `deficit_controle` | Postural instability — high tremor, poor dosing | E3, E4, E8, E9 |
| `deficit_coordination` | Inter-finger dyssynergia — asymmetric activation | E2, E4, E5, E6 |
| `recuperation` | Functional recovery phase — strength and speed training | E6, E7, E9, E10 |

---

### Layer 2 — Adaptive RL Agent (Q-Learning)

The difficulty adaptation is modelled as a **Markov Decision Process (MDP)**:

**State space** `S = (AROM_bin, force_bin, tremor_bin, level, fatigue_bin)` — 5 dimensions, 108 discrete states.

**Action space:**

| Action | Effect |
|--------|--------|
| A0 — Decrease | Difficulty level − 1 (min 1) |
| A1 — Maintain | No change |
| A2 — Increase | Difficulty level + 1 (max 5) |
| A3 — Next exercise | Advance to next exercise in the profile's sequence |

**Reward function:**

$$R_t = w_1 \cdot \text{AROM} + w_2 \cdot \Delta\text{ROM} - w_3 \cdot \text{Tremor} - w_4 \cdot (1 - \text{success}) - w_5 \cdot \text{Fatigue}$$

Weights $w_i$ are exercise-specific (defined in `exercises.py`).

**Bellman update (Q-Learning):**

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]$$

with α = 0.2, γ = 0.85, ε = 0.15.

**Design choices:**
- **Optimistic initialization** (`Q_init = +2.0`): forces exploration of all actions before exploitation, preventing premature convergence to "Maintain".
- **Sliding window memory** (last 3 results): erases early-session failure effect; agent reacts to current patient state.
- **Acclimatization period**: first 2 series of a new exercise apply 70% reduced failure penalty — mirroring the clinical discovery phase.
- **Safety Envelope**: clinical rules override Q-Learning when the signal is unambiguous (3 consecutive successes → always increase; tremor > 0.65 → never increase).
- **Persistent Q-table**: saved in `progress_patient.json` between sessions — the agent improves across sessions, not just within one.

**Session management:**

| Session | Behaviour |
|---------|-----------|
| Session 1 | Mandatory k-NN assessment (4 tests) |
| Sessions 2–4 | Resume from `progress_patient.json` (Q-table included) |
| Session 5+ | Automatic k-NN re-assessment |

---

### Layer 3 — Recovery Prediction (Linear Regression)

Linear regression on AROM scores from the last 10 sessions.  
Output displayed on OLED: `"+12° in ~2 weeks"` / `"Stable progression"` / `"Consult your physio"`.  
Minimum 3 completed sessions required before prediction is shown.

---

### Coach — Claude API

On each completed series, a 1–2 sentence motivational message is generated in French, conditioned on:
- Motor deficit profile
- Current AROM and tremor values
- RL agent decision

Displayed on the OLED SSD1306. Fallback messages used if API is unavailable.

---

## 🔧 Hardware

| Component | Role |
|-----------|------|
| ESP32 DevKit | MCU: sensor reading, EMA filtering, WiFi JSON transmission |
| CD74HC4067 | 16-channel analog multiplexer — 8 sensors on 1 analog GPIO |
| 5× Velostat flex sensors (handmade) | Per-finger flexion measurement (0°–90°) |
| 3× FSR pressure sensors | Grip force — thumb, index, middle finger |
| OLED SSD1306 128×64 | Real-time feedback + coach messages |
| LED + passive buzzer + button | Multimodal feedback and session control |
| Raspberry Pi 5 | AI pipeline, Flask server, Claude API calls |

### Multiplexer wiring

```
CD74HC4067:
  SIG → GPIO39    S0 → GPIO33    S1 → GPIO32
  S2  → GPIO17    S3 → GPIO16    EN  → GND    VCC → 3.3V

  C0–C4  → Velostat flex sensors (fingers 1–5)
  C5–C7  → not used (reserved)

FSR pressure sensors (direct ADC1):
  Thumb → GPIO34    Index → GPIO35    Middle → GPIO36

OLED SSD1306 (I2C):
  SDA → GPIO21    SCL → GPIO22

Other:
  LED    → GPIO4     Buzzer (passive) → GPIO14
  Button → GPIO18
```

---

## 🗂️ Repository Structure

```
rehab-glove-ai/
├── esp32/
│   └── gant_esp32_v2.ino          # ESP32 firmware: EMA filter, state machine, WiFi
├── ia/
│   ├── couche1_knn.py             # k-NN classifier + feature extractor
│   ├── couche2_rl_env.py          # Q-Learning RL agent + session manager
│   ├── exercises.py               # Exercise library (10 exercises × 5 levels)
│   ├── patient_sim.py             # Patient simulator for offline RL testing
│   └── modele_knn.json            # Serialized trained k-NN model
├── rpi/
│   ├── pipeline_bridge.py         # Flask server: /status /evaluation /data /health
│   ├── couche1_knn.py             # (mirror) k-NN used at runtime
│   ├── couche2_rl_env.py          # (mirror) RL agent used at runtime
│   ├── exercises.py               # (mirror) exercise library
│   ├── modele_knn.json            # (mirror) trained model
│   ├── progress_patient.json      # Persistent patient state + Q-table
│   └── historique_patient.json    # Session history for regression prediction
├── docs/
│   ├── schema_cablage.pdf         # KiCad schematic export
│   └── photos/                    # Annotated hardware photos
├── .gitignore
└── README.md
```

---

## 🚀 Setup & Run

### ESP32

1. Open `esp32/gant_esp32_v2.ino` in Arduino IDE 2.x
2. Install libraries: `ArduinoJson`, `Adafruit SSD1306`, `Adafruit GFX`
3. Set your WiFi credentials and RPi IP address in the constants block
4. Flash to ESP32

### Raspberry Pi 5

```bash
git clone https://github.com/elena484-r/rehab-glove-ai
cd rehab-glove-ai/rpi

pip install flask anthropic python-dotenv scikit-learn numpy --break-system-packages

cp .env.exemple .env          # add your ANTHROPIC_API_KEY
python3 pipeline_bridge.py    # starts Flask on port 5000
```

### Offline simulation (no hardware required)

```bash
cd rehab-glove-ai/ia
python3 patient_sim.py        # runs 4 simulated sessions with OLED output preview
```

---

## ⚠️ Known Limitations & Roadmap

| Current limitation | Context |
|-------------------|---------|
| Velostat sensors are handmade and non-linear | Relative software calibration (per-finger min/max at startup) is applied. Documented honestly — absolute calibration is not possible with this material. Replacement by Spectra Symbol flex sensors is the natural next step. |
| AI trained on simulated data | The noise profile was characterized on real sensors. Validation on a real patient cohort across multiple pathologies (stroke, Parkinson's, post-traumatic), in partnership with a physiotherapist or neurologist, is the required next step before any clinical use. |
| No CE marking | A full MDR 2017/745 conformity study is required before any medical use. This is a research/portfolio prototype. |
| No pain detection | Emergency stop via long button press is implemented. A between-session pain questionnaire is planned. |

---

## 📚 Clinical References

- **Langhorne P. et al. (2011).** *Motor recovery after stroke: a systematic review.* Lancet Neurology, 10(9), 861–872.
- **Fugl-Meyer AR et al. (1975).** *The post-stroke hemiplegic patient — a method for evaluation of physical performance.* Scandinavian Journal of Rehabilitation Medicine.
- **Lyle RC (1981).** *A performance test for assessment of upper limb function in physical rehabilitation.* (ARAT) International Journal of Rehabilitation Research.
- **Unified Parkinson's Disease Rating Scale (UPDRS)** — tremor and bradykinesia sub-score methodology.
- **Hoehn MM & Yahr MD (1967).** *Parkinsonism: onset, progression and mortality.* Neurology — staging scale for motor impairment in Parkinson's disease.
- **Hughes RA & Cornblath DR (2005).** *Guillain-Barré syndrome.* Lancet, 366(9497), 1653–1666 — peripheral motor nerve rehabilitation protocols.
- **Colditz JC (2000).** *Therapist's management of the stiff hand.* Rehabilitation of the Hand and Upper Extremity — post-traumatic and post-surgical hand rehabilitation.

---

## 👩‍💻 Author

**Elena R.** — ESIEA, 2nd year preparatory cycle  
Target specialization: Biomedical Engineering · Medical Robotics · Embedded AI  
Project built entirely solo — hardware fabrication, embedded firmware, AI pipeline, documentation.

*Portfolio project — June–September 2026*

---

*Note: This is a student research prototype. It is not a certified medical device and must not be used for clinical diagnosis or treatment. The system is designed to be pathology-agnostic at the motor assessment level — the k-NN classifier operates on biomechanical features (AROM, tremor, coordination, fatigue) that are relevant across stroke, Parkinson's disease, Guillain-Barré syndrome, post-traumatic and post-surgical hand rehabilitation.*
