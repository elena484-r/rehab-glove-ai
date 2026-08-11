"""
============================================================
PIPELINE_BRIDGE.PY — Integration complete des couches
============================================================
Demontre l'enchainement exact :

  Capteurs (EMA deja fait sur ESP32)
       |
  Couche 1 : K-NN -> profil + metriques
       |
  Couche 2 : Agent RL -> exercice + difficulte + reward
       |
  Couche 3 : Prediction (non touchee)
       |
  Claude API -> message coach
       |
  Reponse HTTP vers ESP32

Ce fichier remplace interface_manager.py et orchestre tout.
============================================================
"""

import json
import os
from datetime import datetime
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from couche1_knn import ClassifieurKNN, ExtracteurMetriques
from exercises import get_exercices_pour_profil, BIBLIOTHEQUE
from couche2_rl_env import (
    AgentRL, EtatPatient, GestionnaireSession,
    PROGRESS_FILE
)

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LOG_PATH          = "historique_sessions.jsonl"

app = Flask(__name__)

# ============================================================
# INITIALISATION DES COUCHES
# ============================================================
print("[BRIDGE] Chargement du classifieur K-NN...")
classifieur = ClassifieurKNN(k=5)
classifieur.entrainer(n_par_classe=80)
classifieur.sauvegarder("modele_knn.json")

extracteur = ExtracteurMetriques()

# Agent RL : charge la session existante ou attend evaluation
etat_patient = GestionnaireSession.charger_progression()
agent_rl     = AgentRL(etat_patient) if etat_patient else None

print(f"[BRIDGE] Session : {'reprise' if etat_patient else 'premiere seance'}")
print("[BRIDGE] Serveur pret sur port 5000")


# ============================================================
# ENDPOINT : EVALUATION INITIALE (4 tests -> profil K-NN)
# ============================================================
@app.route('/evaluation', methods=['POST'])
def recevoir_evaluation():
    """
    Recoit les 4 tests d'evaluation depuis l'ESP32.
    Lance le K-NN, cree l'agent RL, retourne le profil.
    """
    global etat_patient, agent_rl
    data = request.get_json()
    if not data:
        return jsonify({"status": "erreur", "message": "JSON invalide"}), 400

    # Extraction des metriques depuis les 4 tests
    metriques = extracteur.extraire(
        test1_angles   = data.get("test1_angles",    []),
        test2_angles   = data.get("test2_angles",    []),
        test3_angles   = data.get("test3_angles",    []),
        test3_pression = data.get("test3_pression",  []),
        test4_angles   = data.get("test4_angles",    []),
    )

    # Couche 1 : K-NN
    resultat_knn = classifieur.predire(metriques)

    # Couche 2 : initialisation agent RL
    etat_patient = GestionnaireSession.initialiser_depuis_profil(resultat_knn)
    agent_rl     = AgentRL(etat_patient)
    GestionnaireSession.sauvegarder_progression(etat_patient)

    # Exercice initial
    ex = BIBLIOTHEQUE[etat_patient.exercice_actuel_id]
    nv = ex.niveaux[etat_patient.difficulte - 1]

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] EVALUATION -> {resultat_knn['dominant_profile_fr']} "
          f"({resultat_knn['confidence']:.1f}%)")

    return jsonify({
        "status":          "ok",
        "profil":          resultat_knn["dominant_profile"],
        "profil_fr":       resultat_knn["dominant_profile_fr"],
        "confidence":      resultat_knn["confidence"],
        "profile_scores":  resultat_knn["profile_scores"],
        "metriques":       metriques,
        "exercice_id":     ex.id,
        "exercice_nom":    ex.nom,
        "difficulte":      etat_patient.difficulte,
        "consigne_oled":   nv.consigne_oled,
        "repetitions":     nv.repetitions,
    }), 200


# ============================================================
# ENDPOINT : DONNEES EN TEMPS REEL (pendant exercice)
# ============================================================
@app.route('/data', methods=['POST'])
def recevoir_donnees():
    """
    Recoit les donnees capteurs en temps reel.
    Retourne le feedback RL + message coach.
    """
    global etat_patient, agent_rl
    data = request.get_json()
    if not data:
        return jsonify({"status": "erreur"}), 400

    angles   = data.get("angles",       [0]*5)
    pression = data.get("pressure_pct", [0]*3)
    succes   = data.get("succes",       False)

    if len(angles) != 5 or len(pression) != 3:
        return jsonify({"status": "erreur", "message": "format invalide"}), 400

    # Calcul metriques temps reel simples
    import numpy as np
    amplitude = float(np.mean(angles) / 90.0)
    force     = float(np.mean(pression) / 100.0)
    metriques_obs = {
        "amplitude":   amplitude,
        "force":       force,
        "vitesse":     min(1.0, amplitude * 0.8),
        "tremblement": max(0.0, 1.0 - amplitude),
        "asymetrie":   float(np.std(angles) / 45.0),
    }

    # Si pas encore d'agent (premiere seance sans evaluation)
    if agent_rl is None:
        return jsonify({
            "status":        "attente_evaluation",
            "message_oled":  "Evaluation requise",
        }), 200

    # Couche 2 : step RL
    reward, action, msg_rl, fin_seance, etat_dict = agent_rl.step(metriques_obs, succes)

    # Recuperer parametres exercice courant
    ex = BIBLIOTHEQUE[etat_patient.exercice_actuel_id]
    nv = ex.niveaux[etat_patient.difficulte - 1]

    # Message coach
    message_coach = generer_message_coach(metriques_obs, etat_patient.profil, msg_rl)

    # Sauvegarder progression
    GestionnaireSession.sauvegarder_progression(etat_patient)

    # Log
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {ex.nom} niv{etat_patient.difficulte} | "
          f"succes={succes} | action={action} | R={reward:.3f} | {msg_rl}")

    return jsonify({
        "status":          "ok",
        "action":          action,
        "message_rl":      msg_rl,
        "difficulte":      etat_patient.difficulte,
        "exercice_id":     ex.id,
        "exercice_nom":    ex.nom,
        "consigne_oled":   nv.consigne_oled,
        "repetitions":     nv.repetitions,
        "fin_seance":      fin_seance,
        "coach":           message_coach,
        "reward":          reward,
    }), 200


# ============================================================
# COACH CLAUDE
# ============================================================
def generer_message_coach(metriques, profil, msg_rl):
    """Genere un message coach via Claude API ou message par defaut."""
    messages_defaut = {
        "deficit_mobilite":     "Prenez votre temps. Chaque mouvement compte.",
        "deficit_controle":     "Respirez lentement. La stabilite prime.",
        "deficit_coordination": "Concentrez-vous sur chaque doigt.",
        "recuperation":         "Excellente progression ! Continuez.",
    }

    if not ANTHROPIC_API_KEY:
        return messages_defaut.get(profil, "Continuez vos efforts !")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        contexte = (
            f"Profil : {profil} | "
            f"Amplitude : {metriques['amplitude']:.2f} | "
            f"Tremblement : {metriques['tremblement']:.2f} | "
            f"Decision : {msg_rl}"
        )
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=80,
            system=(
                "Coach reeducation main post-AVC. "
                "1-2 phrases courtes en francais. "
                "Encourageant, precis, pas de diagnostic."
            ),
            messages=[{"role": "user", "content": contexte}]
        )
        return response.content[0].text.strip()
    except Exception:
        return messages_defaut.get(profil, "Continuez vos efforts !")


# ============================================================
# HEALTH CHECK
# ============================================================
@app.route('/health', methods=['GET'])
def health():
    seance = etat_patient.seance if etat_patient else 0
    profil = etat_patient.profil if etat_patient else "non_evalue"
    return jsonify({
        "status":  "actif",
        "seance":  seance,
        "profil":  profil,
    })


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("PIPELINE BRIDGE — GANT REEDUCATION")
    print("Couche 1 (K-NN) + Couche 2 (RL) integrees")
    print("En attente sur le port 5000...")
    print("=" * 55)
    app.run(host='0.0.0.0', port=5000, debug=False)
