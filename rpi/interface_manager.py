"""
============================================================
INTERFACE MANAGER — RASPBERRY PI 5
============================================================
Ce fichier remplace serveur_gant.py.
Il reçoit les données de l'ESP32, fait tourner le pipeline IA,
et renvoie dans la réponse HTTP :
  - le profil moteur détecté
  - le message coach Claude
  - la décision RL
  - la prédiction de progression
  - le niveau de difficulté actuel

L'ESP32 lit cette réponse et met à jour l'affichage OLED.

Usage :
  python3 interface_manager.py

Prérequis :
  pip install flask anthropic python-dotenv numpy --break-system-packages
  modele_knn.json dans le même dossier
  .env avec ANTHROPIC_API_KEY
============================================================
"""

import json
import os
import time
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime
from collections import deque
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

app = Flask(__name__)

# ============================================================
# CHARGEMENT DU CLASSIFIEUR k-NN (depuis JSON)
# ============================================================
class ClassifieurProfilMoteur:
    FEATURES = ["amplitude", "vitesse", "force", "tremblement", "asymetrie"]
    LABELS_FR = {
        "deficit_mobilite":     "Mobilite reduite",
        "deficit_controle":     "Controle a ameliorer",
        "deficit_coordination": "Coordination",
        "recuperation":         "Bonne progression",
    }

    def __init__(self, chemin="modele_knn.json"):
        with open(chemin) as f:
            m = json.load(f)
        self.k           = m["k"]
        self.classes_map = m["training_data"]["classes_map"]
        self.X_train     = np.array(m["training_data"]["X"])
        self.y_train     = np.array(m["training_data"]["y"])
        self.mean        = np.array(m["scaler"]["mean"])
        self.std         = np.array(m["scaler"]["std"])
        print(f"[IA] Classifieur chargé — k={self.k}")

    def predire(self, metriques: dict) -> dict:
        x = np.array([metriques[f] for f in self.FEATURES]).reshape(1, -1)
        x_s = (x - self.mean) / self.std
        dist = np.sqrt(np.sum((self.X_train - x_s) ** 2, axis=1))
        idx = np.argsort(dist)[:self.k]
        votes = {}
        for i in idx:
            c = self.classes_map[str(self.y_train[i])]
            votes[c] = votes.get(c, 0) + 1
        profil = max(votes, key=votes.get)
        return {
            "profil":    profil,
            "label_fr":  self.LABELS_FR.get(profil, profil),
            "confiance": round(votes[profil] / self.k, 2),
        }


# ============================================================
# POLITIQUE ADAPTATIVE RL
# ============================================================
class PolitiqueRL:
    def __init__(self):
        self.difficulte  = 2
        self.historique  = []

    def evaluer(self, metriques: dict, profil: str) -> dict:
        succes = metriques["amplitude"] > 0.5 and metriques["tremblement"] < 0.4
        self.historique.append(succes)
        decision = "maintien"
        message  = f"Niveau {self.difficulte}/5"

        if len(self.historique) >= 3 and all(self.historique[-3:]):
            if self.difficulte < 5:
                self.difficulte += 1
                decision = "augmentation"
                message  = f"3 succes ! Niveau {self.difficulte}/5"
                self.historique = []

        elif len(self.historique) >= 2 and not any(self.historique[-2:]):
            if self.difficulte > 1:
                self.difficulte -= 1
                decision = "reduction"
                message  = f"Seance ajustee. Niveau {self.difficulte}/5"
                self.historique = []

        if profil == "deficit_controle" and self.difficulte > 1:
            self.difficulte -= 1
            decision = "reduction_tremblement"
            message  = f"Pause conseiller. Niveau {self.difficulte}/5"

        return {
            "decision":   decision,
            "difficulte": self.difficulte,
            "message_rl": message,
        }


# ============================================================
# CALCUL DES MÉTRIQUES
# ============================================================
def calculer_metriques(historique_angles, historique_pression) -> dict:
    if len(historique_angles) < 2:
        return {f: 0.0 for f in ["amplitude","vitesse","force","tremblement","asymetrie"]}

    A = np.array(historique_angles)
    P = np.array(historique_pression)

    amplitude_max = np.max(A, axis=0)
    amplitude  = float(np.mean(amplitude_max) / 90.0)
    delta      = np.abs(np.diff(A, axis=0))
    vitesse    = min(1.0, float(np.max(np.mean(delta, axis=1))) / 45.0)
    force      = float(np.mean(P) / 100.0)
    tremblement = min(1.0, float(np.mean(np.var(A, axis=0))) / 400.0)
    asymetrie  = min(1.0, float(np.std(amplitude_max)) / 45.0)

    return {
        "amplitude":   round(amplitude, 4),
        "vitesse":     round(vitesse, 4),
        "force":       round(force, 4),
        "tremblement": round(tremblement, 4),
        "asymetrie":   round(asymetrie, 4),
    }


# ============================================================
# MESSAGE COACH CLAUDE
# ============================================================
def generer_message_coach(metriques, profil, decision_rl) -> str:
    if not ANTHROPIC_API_KEY:
        messages_defaut = {
            "deficit_mobilite":     "Prenez votre temps. Chaque mouvement compte.",
            "deficit_controle":     "Respirez lentement. La precision prime sur la vitesse.",
            "deficit_coordination": "Concentrez-vous sur chaque doigt un par un.",
            "recuperation":         "Excellente progression ! Continuez ainsi.",
        }
        return messages_defaut.get(profil["profil"], "Continuez vos efforts !")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        contexte = f"""
Profil : {profil['label_fr']} (confiance {profil['confiance']*100:.0f}%)
Amplitude : {metriques['amplitude']:.2f}/1
Tremblement : {metriques['tremblement']:.2f}/1
Asymetrie : {metriques['asymetrie']:.2f}/1
Niveau difficulte : {decision_rl['difficulte']}/5
Decision : {decision_rl['decision']}
"""
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=80,
            system=(
                "Tu es un coach de reeducation de la main post-AVC. "
                "Reponds en francais, 1-2 phrases maximum tres courtes. "
                "Ton chaleureux et encourageant. Pas de diagnostic medical."
            ),
            messages=[{"role": "user", "content": contexte}]
        )
        return response.content[0].text.strip()
    except Exception:
        return "Continuez vos efforts, chaque seance compte !"


# ============================================================
# PRÉDICTION PROGRESSION
# ============================================================
def generer_prediction(historique_sessions) -> str:
    if len(historique_sessions) < 3:
        return ""

    scores = [s.get("scores", {}).get("rom", 0) for s in historique_sessions[-10:]]
    if len(scores) < 3:
        return ""

    x = np.arange(len(scores), dtype=float)
    y = np.array(scores)
    pente = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)

    amplitude_predite = min(90, (np.mean(scores) + pente * 14) * 18)

    if pente > 0.01:
        return f"+{amplitude_predite:.0f}deg dans ~2 sem."
    elif pente > -0.01:
        return "Progression stable."
    else:
        return "Consultez votre kine."


# ============================================================
# INITIALISATION DES COMPOSANTS
# ============================================================
classifieur   = ClassifieurProfilMoteur("modele_knn.json")
politique_rl  = PolitiqueRL()

historique_angles   = deque(maxlen=20)
historique_pression = deque(maxlen=20)
historique_sessions = []

LOG_PATH = "historique_sessions.jsonl"
if os.path.exists(LOG_PATH):
    with open(LOG_PATH) as f:
        for line in f:
            try:
                historique_sessions.append(json.loads(line.strip()))
            except Exception:
                pass

print(f"[OK] {len(historique_sessions)} sessions chargées depuis l'historique")


# ============================================================
# ENDPOINT PRINCIPAL
# ============================================================
@app.route('/data', methods=['POST'])
def recevoir_donnees():
    data = request.get_json()
    if not data:
        return jsonify({"status": "erreur"}), 400

    angles   = data.get("angles",       [0]*5)
    pression = data.get("pressure_pct", [0]*3)

    if len(angles) != 5 or len(pression) != 3:
        return jsonify({"status": "erreur", "message": "format invalide"}), 400

    # Mise à jour historique glissant
    historique_angles.append(angles)
    historique_pression.append(pression)

    # Pipeline IA
    metriques    = calculer_metriques(list(historique_angles), list(historique_pression))
    profil       = classifieur.predire(metriques)
    decision_rl  = politique_rl.evaluer(metriques, profil["profil"])
    message      = generer_message_coach(metriques, profil, decision_rl)
    prediction   = generer_prediction(historique_sessions)

    # Log session
    session = {
        "timestamp": datetime.now().isoformat(),
        "profil":    profil["profil"],
        "metriques": metriques,
        "difficulte": decision_rl["difficulte"],
        "scores":    {"rom": metriques["amplitude"] * 5},
    }
    historique_sessions.append(session)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(session) + "\n")

    # Affichage console
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {profil['label_fr']} | niv.{decision_rl['difficulte']} | {message[:40]}")

    # Réponse vers ESP32
    return jsonify({
        "status":     "ok",
        "profil":     profil["profil"],
        "label_fr":   profil["label_fr"],
        "confiance":  profil["confiance"],
        "coach":      message,
        "rl":         decision_rl["message_rl"],
        "prediction": prediction,
        "difficulte": decision_rl["difficulte"],
    }), 200


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "actif", "sessions": len(historique_sessions)})


# ============================================================
# LANCEMENT
# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("INTERFACE MANAGER — GANT REEDUCATION")
    print(f"Sessions en memoire : {len(historique_sessions)}")
    print("En attente sur le port 5000...")
    print("=" * 55)
    app.run(host='0.0.0.0', port=5000, debug=False)
