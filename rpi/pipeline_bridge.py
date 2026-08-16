"""
============================================================
PIPELINE BRIDGE V2 — GANT REEDUCATION AVC
============================================================
Architecture pipeline (toutes les couches sur RPi) :

  Couche 0 (ESP32) : EMA alpha=0.2 — NE PAS TOUCHER
  Couche 1 (RPi)   : K-NN -> profil moteur dominant
  Couche 2 (RPi)   : Agent RL -> adaptation difficulte + exercice
  Couche 3 (RPi)   : Regression lineaire -> prediction progression
  Coach    (RPi)   : Claude API -> message coach francais -> OLED

Endpoints Flask :
  GET  /status     : ESP32 demande au demarrage si evaluation requise
  POST /evaluation : 4 tests -> profil K-NN + premier exercice
  POST /data       : boucle 500ms -> decision RL + coach + prediction
  GET  /health     : statut serveur

Gestion des seances :
  Seance 1   : evaluation K-NN obligatoire (4 tests)
  Seances 2-4: reprise depuis progress_patient.json
  Seance 5+  : reevaluation K-NN automatique

Usage :
  cd ~/rehab-glove-ai/rpi && python3 pipeline_bridge_v2.py

Prerequis :
  pip install flask scikit-learn numpy anthropic python-dotenv --break-system-packages
  modele_knn.json, .env (ANTHROPIC_API_KEY), couche1_knn.py,
  couche2_rl_env.py, exercises.py dans le meme dossier
============================================================
"""

import json, os, numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from couche1_knn import ClassifieurKNN, ExtracteurMetriques
from exercises import get_exercices_pour_profil, BIBLIOTHEQUE
from couche2_rl_env import AgentRL, EtatPatient, GestionnaireSession

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LOG_PATH          = "historique_sessions.jsonl"
app               = Flask(__name__)

# ============================================================
# INITIALISATION K-NN
# ============================================================
print("[BRIDGE] Chargement K-NN...")
classifieur = ClassifieurKNN(k=5)
classifieur.entrainer(n_par_classe=80)
classifieur.sauvegarder("modele_knn.json")
extracteur  = ExtracteurMetriques()

etat_patient = GestionnaireSession.charger_progression()
agent_rl     = AgentRL(etat_patient) if etat_patient else None
print(f"[BRIDGE] Session : {'reprise seance ' + str(etat_patient.seance+1) if etat_patient else 'premiere seance'}")

# Chargement historique sessions pour la couche prediction (regression lineaire)
historique_sessions = []
if os.path.exists(LOG_PATH):
    with open(LOG_PATH) as _f:
        for _line in _f:
            try:
                historique_sessions.append(json.loads(_line.strip()))
            except Exception:
                pass
print(f"[BRIDGE] {len(historique_sessions)} sessions chargees depuis l'historique")
print("[BRIDGE] Pret sur port 5000")


# ============================================================
# GET /status — VERIFIE SI EVALUATION REQUISE AU DEMARRAGE
# ============================================================
@app.route('/status', methods=['GET'])
def get_status():
    """
    Appele par l'ESP32 au demarrage (apres double appui accueil).
    Retourne si evaluation K-NN requise ou si on reprend directement.
    """
    etat = GestionnaireSession.charger_progression()

    if etat is None:
        # Premiere seance : evaluation obligatoire
        return jsonify({
            "evaluation_requise": True,
            "seance":             1,
            "profil":             None,
            "exercice_nom":       None,
            "consigne_oled":      None,
            "repetitions":        0,
            "difficulte":         1,
            "materiel_requis":    "Prevoyez une balle souple.",
        }), 200

    numero_seance     = etat.seance + 1
    evaluation_req    = GestionnaireSession.doit_reevaluer(numero_seance)
    ex                = BIBLIOTHEQUE[etat.exercice_actuel_id]
    nv                = ex.niveaux[etat.difficulte - 1]

    return jsonify({
        "evaluation_requise": evaluation_req,
        "seance":             numero_seance,
        "profil":             etat.profil,
        "exercice_id":        ex.id,
        "exercice_nom":       ex.nom,
        "consigne_oled":      nv.consigne_oled,
        "repetitions":        nv.repetitions,
        "difficulte":         etat.difficulte,
        "materiel_requis":    "Prevoyez une balle souple." if evaluation_req else "",
    }), 200


# ============================================================
# POST /evaluation — 4 TESTS -> PROFIL K-NN
# ============================================================
@app.route('/evaluation', methods=['POST'])
def recevoir_evaluation():
    global etat_patient, agent_rl
    data = request.get_json()
    if not data:
        return jsonify({"status": "erreur"}), 400

    metriques = extracteur.extraire(
        test1_angles   = data.get("test1_angles",   []),
        test2_angles   = data.get("test2_angles",   []),
        test3_angles   = data.get("test3_angles",   []),
        test3_pression = data.get("test3_pression", []),
        test4_angles   = data.get("test4_angles",   []),
    )

    resultat_knn = classifieur.predire(metriques)
    etat_patient = GestionnaireSession.initialiser_depuis_profil(resultat_knn)
    agent_rl     = AgentRL(etat_patient)
    GestionnaireSession.sauvegarder_progression(etat_patient)

    ex = BIBLIOTHEQUE[etat_patient.exercice_actuel_id]
    nv = ex.niveaux[etat_patient.difficulte - 1]

    print(f"[EVAL] {resultat_knn['dominant_profile_fr']} ({resultat_knn['confidence']:.1f}%)")

    return jsonify({
        "status":        "ok",
        "profil":        resultat_knn["dominant_profile"],
        "profil_fr":     resultat_knn["dominant_profile_fr"],
        "confidence":    resultat_knn["confidence"],
        "metriques":     metriques,
        "exercice_id":   ex.id,
        "exercice_nom":  ex.nom,
        "consigne_oled": nv.consigne_oled,
        "repetitions":   nv.repetitions,
        "difficulte":    etat_patient.difficulte,
    }), 200


# ============================================================
# POST /data — DONNEES TEMPS REEL PENDANT EXERCICE
# ============================================================
@app.route('/data', methods=['POST'])
def recevoir_donnees():
    global etat_patient, agent_rl
    data = request.get_json()
    if not data:
        return jsonify({"status": "erreur"}), 400

    angles   = data.get("angles",       [0]*5)
    pression = data.get("pressure_pct", [0]*3)
    succes   = data.get("succes",       False)

    if len(angles) != 5 or len(pression) != 3:
        return jsonify({"status": "erreur"}), 400

    if agent_rl is None:
        return jsonify({
            "status":       "attente_evaluation",
            "message_oled": "Evaluation requise",
            "fin_seance":   False,
        }), 200

    amplitude = float(np.mean(angles) / 90.0)
    metriques_obs = {
        "amplitude":   amplitude,
        "force":       float(np.mean(pression) / 100.0),
        "vitesse":     min(1.0, amplitude * 0.8),
        "tremblement": max(0.0, 1.0 - amplitude),
        "asymetrie":   float(np.std(angles) / 45.0),
    }

    reward, action, msg_rl, fin_seance, _ = agent_rl.step(metriques_obs, succes)
    ex = BIBLIOTHEQUE[etat_patient.exercice_actuel_id]
    nv = ex.niveaux[etat_patient.difficulte - 1]

    coach = generer_message_coach(metriques_obs, etat_patient.profil, msg_rl)
    GestionnaireSession.sauvegarder_progression(etat_patient)

    # Log session — mise a jour en memoire ET sur disque
    session_log = {
        "timestamp":    datetime.now().isoformat(),
        "profil":       etat_patient.profil,
        "scores":       {"rom": metriques_obs["amplitude"] * 5},
        "difficulte":   etat_patient.difficulte,
        "decision_rl":  msg_rl,
    }
    historique_sessions.append(session_log)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(session_log) + "\n")

    # Couche 3 : prediction de progression
    prediction = generer_prediction(historique_sessions)

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {ex.nom} niv{etat_patient.difficulte} | succes={succes} | {msg_rl}"
          + (f" | pred: {prediction}" if prediction else ""))

    return jsonify({
        "status":        "ok",
        "action":        action,
        "message_rl":    msg_rl,
        "difficulte":    etat_patient.difficulte,
        "exercice_nom":  ex.nom,
        "consigne_oled": nv.consigne_oled,
        "repetitions":   nv.repetitions,
        "fin_seance":    fin_seance,
        "coach":         coach,
        "prediction":    prediction,
        "reward":        reward,
    }), 200


# ============================================================
# COUCHE 3 — PREDICTION DE PROGRESSION (regression lineaire)
# ============================================================
def generer_prediction(sessions: list) -> str:
    """
    Regression lineaire sur les scores ROM des dernieres sessions.
    Predit l'amplitude dans ~2 semaines et retourne une chaine
    courte lisible sur l'OLED de l'ESP32 (max ~20 chars).

    Exemples de retour :
      "+12deg dans ~2 sem."
      "Progression stable."
      "Consultez votre kine."
      ""  (historique insuffisant — < 3 sessions)
    """
    if len(sessions) < 3:
        return ""

    # Prendre les 10 dernieres sessions pour la regression
    scores = [s.get("scores", {}).get("rom", None) for s in sessions[-10:]]
    scores = [s for s in scores if s is not None]

    if len(scores) < 3:
        return ""

    x = np.arange(len(scores), dtype=float)
    y = np.array(scores, dtype=float)
    x_mean, y_mean = np.mean(x), np.mean(y)
    denom = np.sum((x - x_mean) ** 2)

    if denom == 0:
        return "Progression stable."

    pente = np.sum((x - x_mean) * (y - y_mean)) / denom

    # Predire dans 14 seances (~2 semaines a 1 seance/jour)
    score_futur = float(np.mean(scores)) + pente * 14
    score_futur = max(0.0, min(5.0, score_futur))

    # Conversion score ROM [0-5] -> amplitude en degres [0-90]
    amplitude_deg = score_futur * 18.0

    if pente > 0.02:
        return f"+{amplitude_deg:.0f}deg dans ~2 sem."
    elif pente > -0.02:
        return "Progression stable."
    else:
        return "Consultez votre kine."


# ============================================================
# COACH CLAUDE
# ============================================================
def generer_message_coach(metriques, profil, msg_rl):
    defaut = {
        "deficit_mobilite":     "Prenez votre temps. Chaque mouvement compte.",
        "deficit_controle":     "Respirez lentement. La stabilite prime.",
        "deficit_coordination": "Concentrez-vous sur chaque doigt.",
        "recuperation":         "Excellente progression ! Continuez.",
    }
    if not ANTHROPIC_API_KEY:
        return defaut.get(profil, "Continuez vos efforts !")
    try:
        import anthropic
        client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        contexte = (f"Profil: {profil} | Amplitude: {metriques['amplitude']:.2f} | "
                    f"Tremblement: {metriques['tremblement']:.2f} | Decision: {msg_rl}")
        r = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=80,
            system="Coach reeducation main post-AVC. 1-2 phrases courtes en francais. Encourageant, pas de diagnostic.",
            messages=[{"role": "user", "content": contexte}]
        )
        return r.content[0].text.strip()
    except Exception:
        return defaut.get(profil, "Continuez vos efforts !")


@app.route('/health', methods=['GET'])
def health():
    s = etat_patient.seance if etat_patient else 0
    p = etat_patient.profil if etat_patient else "non_evalue"
    return jsonify({"status": "actif", "seance": s, "profil": p})


if __name__ == '__main__':
    print("=" * 55)
    print("PIPELINE BRIDGE V2 — GANT REEDUCATION")
    print("=" * 55)
    app.run(host='0.0.0.0', port=5000, debug=False)
