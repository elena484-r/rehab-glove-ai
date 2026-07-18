# 🖐️ Gant Connecté de Rééducation Post-AVC

> Projet étudiant — ESIEA 2e année cycle préparatoire  
> Objectif : concevoir un dispositif médical embarqué intelligent pour la rééducation de la main après un AVC, combinant hardware fait main, traitement du signal temps réel, et pipeline d'IA en 4 couches.

---

## 🎯 Contexte médical

Après un AVC, 80% des patients présentent des séquelles motrices de la main. La rééducation intensive (300–400 répétitions/séance selon Langhorne, 2011) est prouvée efficace mais difficile à maintenir sans feedback en temps réel.

Ce projet propose un gant connecté low-cost qui :
- mesure les mouvements de chaque doigt en temps réel
- adapte automatiquement la difficulté des exercices
- génère des messages de coaching personnalisés via IA
- prédit la progression du patient sur plusieurs semaines

---

## 🏗️ Architecture technique

```
┌─────────────────────────────────────────────────────┐
│                   GANT (ESP32)                      │
│  5 capteurs flexion + 3 capteurs pression           │
│  → Multiplexeur CD74HC4067                          │
│  → Couche 1 : Filtre EMA (C++)                      │
│  → Envoi JSON WiFi toutes les 500ms                 │
└──────────────────────┬──────────────────────────────┘
                       │ WiFi / HTTP POST
┌──────────────────────▼──────────────────────────────┐
│              CERVEAU (Raspberry Pi 5)               │
│  Couche 2 : Analyse IA du profil moteur(k-NN Python)│
│  Couche 3 RL : adapte le programme selon le profil  │
│  Couche 4 : Prédiction progression (régression)     │
│  + Appel Claude API → messages coaching français    │
└─────────────────────────────────────────────────────┘
Flexion complèteTous les doigts > 80°Flexion partielleTous les doigts entre 20° et 60°Mouvement asymétriqueCertains doigts bougent, d'autres pas — ça révèle une compensation inter-doigtsTremblementOscillations rapides détectées sur la dérivée temporelle
```

### Pipeline IA en 4 couches

| Couche | Rôle | Technologie | Où |
|--------|------|-------------|-----|
| 1 — Filtrage signal | Lisse le bruit des capteurs Velostat | Filtre EMA (α=0.2) | ESP32 C++ |
| 2 — Classification | Reconnaît le type de mouvement | k-NN / arbre de décision | scikit-learn → C++ |
| 3 — Décision adaptative | Ajuste la difficulté automatiquement | Politique RL simple | Python RPi |
| 4 — Prédiction | Prédit l'amplitude dans X semaines | Régression linéaire | scikit-learn |
| Coach IA | Transforme les métriques en phrases | Claude API (Anthropic) | Python RPi |

---

## 🔧 Hardware

| Composant | Rôle |
|-----------|------|
| ESP32 | MCU embarqué : lecture capteurs, filtrage, envoi WiFi |
| CD74HC4067 | Multiplexeur 16 canaux : 8 capteurs sur 1 broche analogique |
| 5× capteurs flexion (Velostat fait main) | Mesure de flexion par doigt (0°–90°) |
| 3× capteurs pression (Velostat fait main) | Détection de force (pouce, index, majeur) |
| Écran OLED SSD1306 | Affichage feedback + messages coach |
| LED + Buzzer + Bouton | Feedback multimodal |
| Raspberry Pi 5 | Traitement IA, serveur Flask, appels Claude API |

### Câblage multiplexeur
```
MUX CD74HC4067 :
  SIG → GPIO39   S0 → GPIO33   S1 → GPIO32
  S2  → GPIO17   S3 → GPIO16   EN  → GND
  C0–C4 → capteurs flexion     C5–C7 → capteurs pression

Capteurs pression directs (ADC1) :
  Pouce → GPIO34    Index → GPIO35    Majeur → GPIO36
```

---

## 🤖 Pipeline IA — détail

### Couche 1 : Filtre EMA (embarqué ESP32)
```cpp
// Moyenne mobile exponentielle — lisse le bruit Velostat
valeur_filtree = 0.2 * lecture_brute + 0.8 * valeur_filtree_precedente;
```

### Couche 2 : Classification de mouvement
Entraîné sur données simulées (bruit caractérisé sur capteurs réels) + données collectées.
Classes : flexion complète / flexion partielle / tremblement / compensation poignet.

### Couche 3 : Politique adaptative
Règle RL simple : 3 succès consécutifs → difficulté +1 / 2 échecs → difficulté -1.
Implémente le concept de "reward shaping" en RL.

### Couche 4 : Prédiction de progression
Régression linéaire sur historique de sessions → prédit l'amplitude atteinte dans X semaines.
*(Validé sur données simulées — à confirmer sur cohorte réelle)*

---

## 📋 Programmes de rééducation

Basés sur les protocoles validés en rééducation neurologique (Langhorne 2011) :

| Score initial /20 | Niveau | Programme | Durée séance | Séances/jour |
|-------------------|--------|-----------|--------------|--------------|
| 0–5 | Très sévère | A — Stimulation passive | 15 min | 2 |
| 6–10 | Sévère | B — Mobilisation active | 20 min | 2 |
| 11–15 | Modéré | C — Renforcement | 25 min | 2 |
| 16–20 | Léger | D — Performance | 30 min | 1 |

---

## 🗂️ Structure du projet

```
rehab-glove-ai/
├── esp32/
│   └── gant_esp32_lecture_wifi.ino   # Code principal ESP32
├── rpi/
│   ├── serveur_gant.py               # Serveur Flask réception données
│   ├── claude_coach.py               # Intégration Claude API
│   └── .env                          # Clé API (non commitée)
├── ia/
│   ├── generate_data.py              # Génération données simulées
│   ├── train_classifier.py           # Entraînement k-NN
│   ├── adaptive_policy.py            # Politique RL
│   └── predict_progression.py        # Régression linéaire
└── docs/
    ├── schema_cablage.png
    └── fiche_technique.pdf
```

---

## 🚀 Installation

### ESP32
1. Ouvrir `esp32/gant_esp32_lecture_wifi.ino` dans Arduino IDE
2. Installer les bibliothèques : `ArduinoJson`, `Adafruit SSD1306`, `Adafruit GFX`
3. Modifier `WIFI_SSID`, `WIFI_PASSWORD`, `SERVER_URL` avec vos infos
4. Téléverser sur l'ESP32

### Raspberry Pi 5
```bash
git clone https://github.com/elena484-r/rehab-glove-ai
cd rehab-glove-ai/rpi
pip install flask anthropic python-dotenv scikit-learn --break-system-packages
cp .env.exemple .env   # puis renseigner la clé Anthropic
python3 serveur_gant.py
```

---

## ⚠️ Limites connues & perspectives

| Limite actuelle | Perspective d'amélioration |
|-----------------|---------------------------|
| Capteurs Velostat fait main : non-linéaires, sensibles à la pression de couture | Remplacement par FSR industriels (Interlink 402) ou flex sensors Spectra Symbol |
| Données IA entraînées sur simulation | Validation sur cohorte de patients réels en partenariat avec kinésithérapeute |
| Pas de marquage CE | Étude de conformité MDR 2017/745 à prévoir avant tout usage clinique |
| Pas de détection de douleur | Ajout bouton d'arrêt d'urgence + questionnaire entre séances |

---

## 📚 Références médicales

- Langhorne P. et al. (2011). *Motor recovery after stroke: a systematic review*. Lancet Neurology.
- Protocoles de rééducation neurologique — principes de répétition intensive et neuroplasticité.

---

## 👩‍💻 Auteure

**Elena** — Étudiante ESIEA, 2e année cycle préparatoire  
Spécialisation visée : Ingénierie biomédicale & Robotique médicale  
Projet réalisé en autonomie complète — hardware, software, IA, documentation.

---

*Projet réalisé dans le cadre d'un portfolio technique — juin/août 2026*
