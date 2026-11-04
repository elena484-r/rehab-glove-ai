```python
"""
============================================================
PIPELINE BRIDGE — REHABILITATION GLOVE
============================================================
Acts as an HTTP/JSON bridge between the ESP32 firmware and the 3 layers:
  - couche1_knn.py      : K-NN diagnosis (motor profile)
  - exercises.py         : library of 10 exercises (5 levels each)
  - couche2_rl_env.py    : adaptive rule-based agent (difficulty, reward shaping)

Exposed routes (must match EXACTLY what esp32.cpp sends):
  GET  /status      -> the ESP32 calls this at startup (double press) to
                        determine whether it should perform a K-NN assessment
                        or directly resume an ongoing session.
  POST /evaluation   -> the ESP32 sends the 4 raw assessment tests (test1_angles,
                        test2_angles, test3_angles, test3_pression,
                        test4_angles) at the end of the assessment.
  POST /data          -> the ESP32 sends, at the end of EACH SET (= one
                        complete attempt to reach repetitions_cible at the
                        current difficulty level, or a voluntary abandonment):
                        serie_angles, serie_pression (complete buffer
                        sampled every 300ms), succes, echec_manuel.
                        One set = one call to AgentRL.step().
============================================================
"""

from flask import Flask, request, jsonify
import numpy as np
import json
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

from couche1_knn import ClassifieurKNN, ExtracteurMetriques
from couche2_rl_env import EtatPatient, AgentRL, GestionnaireSession
from exercises import BIBLIOTHEQUE

app = Flask(__name__)

HISTORIQUE_FILE = "historique_patient.json"   # average amplitude log per session (for regression)
SESSION_FLAG_FILE = "session_state.json"      # persists seance_terminee_flag across server restarts
SEANCE_MIN_PREDICTION = 4  # no numerical prediction before session 4 (insufficient data)


def _lire_flag_fin_seance():
    """Reads from disk whether the last known session has ended (persists across restarts)."""
    if not os.path.exists(SESSION_FLAG_FILE):
        return False
    try:
        with open(SESSION_FLAG_FILE) as f:
            return bool(json.load(f).get("seance_terminee", False))
    except (json.JSONDecodeError, OSError):
        return False


def _ecrire_flag_fin_seance(valeur):
    with open(SESSION_FLAG_FILE, "w") as f:
        json.dump({"seance_terminee": bool(valeur)}, f)

# ============================================================================
# GLOBAL SERVER STATE (single-patient, single-glove setup — one connected device)
# ============================================================================
knn_model = None          # K-NN classifier trained at startup
etat_courant = None       # EtatPatient | None
agent_courant = None      # AgentRL | None
seance_terminee_flag = False   # set to True when AgentRL.step() returns fin_seance
amplitudes_seance_courante = []  # amplitudes (degrees) from each set in the current session


def _charger_etat_sauvegarde():
    """Reloads the patient state from progress_patient.json, if present."""
    global etat_courant, agent_courant
    etat = GestionnaireSession.charger_progression()
    if etat is not None:
        etat_courant = etat
        agent_courant = AgentRL(etat_courant)
    return etat


def _reponse_exercice_courant(extra=None):
    """Builds the JSON payload describing the current exercise and difficulty level for the ESP32."""
    ex = BIBLIOTHEQUE[etat_courant.exercice_actuel_id]
    niveau = ex.niveaux[etat_courant.difficulte - 1]  # levels 1..5 -> indices 0..4
    payload = {
        "exercice_id": ex.id,                                   # raw library ID (1-10), for internal use/logging
        "exercice_position": etat_courant.exercice_idx + 1,      # position 1..N in the profile's exercise sequence -> displayed on screen
        "exercice_total": len(etat_courant.exercices_ids),       # N = number of exercises assigned to the profile (displayed as "X / N")
        "consigne_oled": niveau.consigne_oled,
        "repetitions": niveau.repetitions,
        "difficulte": etat_courant.difficulte,
        "profil": etat_courant.profil,
    }
    if extra:
        payload.update(extra)
    return payload


def _extraire_metriques_serie(serie_angles, serie_pression):
    """
    Computes the 5 normalized metrics [0-1] expected by AgentRL.calculer_reward
    from the raw buffer of one set (same logic as ExtracteurMetriques,
    but applied to a single continuous buffer instead of 4 separate tests,
    and without the *100 factor because couche2_rl_env expects a [0-1] scale).
    """
    if not serie_angles:
        return {"amplitude": 0.0, "force": 0.0, "vitesse": 0.0,
                "tremblement": 0.0, "asymetrie": 0.0}

    angles = np.array(serie_angles)      # (N, 5)
    pression = np.array(serie_pression) if serie_pression else np.zeros((1, 3))

    # Amplitude: mean of the maximum values per finger, normalized to 90 degrees
    amplitude = float(np.mean(np.max(angles, axis=0)) / 90.0)
    amplitude = max(0.0, min(1.0, amplitude))

    # Speed: maximum mean inter-sample variation (degrees/sample), normalized
    if len(angles) > 1:
        delta = np.abs(np.diff(angles, axis=0))
        vitesse = float(np.max(np.mean(delta, axis=1))) / 45.0
    else:
        vitesse = 0.0
    vitesse = max(0.0, min(1.0, vitesse))

    # Force: mean pressure (already expressed as a percentage from 0 to 100 by the firmware), normalized
    force = float(np.mean(pression)) / 100.0 if pression.size > 0 else 0.0
    force = max(0.0, min(1.0, force))

    # Tremor: mean temporal variance of the angles, normalized
    if len(angles) > 1:
        tremblement = float(np.mean(np.var(angles, axis=0))) / 50.0
    else:
        tremblement = 0.0
    tremblement = max(0.0, min(1.0, tremblement))

    # Asymmetry: dispersion between the maximum values of the different fingers, normalized
    asymetrie = float(np.std(np.max(angles, axis=0))) / 45.0
    asymetrie = max(0.0, min(1.0, asymetrie))

    return {
        "amplitude": round(amplitude, 4),
        "force": round(force, 4),
        "vitesse": round(vitesse, 4),
        "tremblement": round(tremblement, 4),
        "asymetrie": round(asymetrie, 4),
    }


def _lire_historique():
    if not os.path.exists(HISTORIQUE_FILE):
        return []
    with open(HISTORIQUE_FILE) as f:
        return json.load(f)


def _enregistrer_seance_dans_historique(numero_seance, amplitude_moy_deg):
    """Adds one (date, mean amplitude in degrees) entry to the progress log."""
    historique = _lire_historique()
    historique.append({
        "seance": numero_seance,
        "timestamp": datetime.now().isoformat(),
        "amplitude_moy_deg": round(amplitude_moy_deg, 2),
    })
    with open(HISTORIQUE_FILE, "w") as f:
        json.dump(historique, f, indent=2)
    return historique


def _construire_prediction(etat: EtatPatient, historique):
    """
    Builds the prediction message displayed at the end of a session (ETAT_PREDICTION screen).
    """
    if etat.seance < SEANCE_MIN_PREDICTION or len(historique) < 2:
        return "Progression en cours : quelques seances de plus sont necessaires pour une estimation chiffree."

    dates = [datetime.fromisoformat(h["timestamp"]) for h in historique]
    amplitudes = [h["amplitude_moy_deg"] for h in historique]
    
    t0 = dates[0]
    temps_total_jours = (dates[-1] - t0).total_seconds() / 86400.0

    # 1. If tests are performed on the same day (e.g. quick tests), the slope in days is distorted
    if temps_total_jours < 0.5:
        return f"Derniere amplitude moyenne : {amplitudes[-1]:.1f} deg. Poursuivez les seances quotidiennement pour activer la prediction."

    jours = np.array([[ (d - t0).total_seconds() / 86400.0 ] for d in dates])
    y = np.array(amplitudes)

    modele = LinearRegression()
    modele.fit(jours, y)

    jour_actuel = jours[-1][0]
    amplitude_actuelle = amplitudes[-1]
    
    # Raw prediction
    amplitude_predite_raw = float(modele.predict([[jour_actuel + 14]])[0])
    
    # 2. Safety: bound the prediction between 0° and 90° (anatomical limits)
    amplitude_predite_14j = max(0.0, min(90.0, amplitude_predite_raw))
    delta = amplitude_predite_14j - amplitude_actuelle

    # 3. Safety: maximum realistic variation (+/- 20° over two weeks)
    delta = max(-20.0, min(20.0, delta))

    if delta >= 1.0:
        return (f"Tendance positive : amplitude estimee a +{delta:.0f} deg "
                f"dans les 2 prochaines semaines. "
                f"Estimation indicative, a confirmer avec votre kinesitherapeute.")
    elif delta <= -1.0:
        return (f"Tendance a surveiller : amplitude stable ou en leger recul "
                f"({delta:+.0f} deg estimes sur 2 semaines). Parlez-en a votre kinesitherapeute.")
    else:
        return (f"Amplitude stable sur les {len(historique)} dernieres seances "
                f"(evolution estimee < 1 deg sur 2 semaines). Regularite recommandee.")

# ============================================================================
# GET /status
# ============================================================================
@app.route('/status', methods=['GET'])
def get_status():
    global etat_courant, agent_courant, seance_terminee_flag

    if etat_courant is None:
        _charger_etat_sauvegarde()

    if etat_courant is None or seance_terminee_flag:
        prochaine_seance = (etat_courant.seance + 1) if etat_courant else 1
        seance_terminee_flag = False
        _ecrire_flag_fin_seance(False)

        if GestionnaireSession.doit_reevaluer(prochaine_seance):
            print(f"[BRIDGE] Seance {prochaine_seance} -> evaluation K-NN requise")
            return jsonify({
                "evaluation_requise": True,
                "seance": prochaine_seance,
                "materiel_requis": "Balle souple",
            })

        # Sessions 2-4: resume directly without a new assessment.
        # Restart from the first exercise in the profile (the previous round is complete);
        # the achieved difficulty level is preserved, but set counters are reset to zero.
        etat_courant.seance = prochaine_seance
        etat_courant.exercice_idx = 0
        etat_courant.series_faites = 0
        etat_courant.succes_nv5 = 0
        etat_courant.historique = []
        etat_courant.fatigue = 0.0
        agent_courant = AgentRL(etat_courant)
        GestionnaireSession.sauvegarder_progression(etat_courant)
        print(f"[BRIDGE] Seance {prochaine_seance} -> reprise (profil={etat_courant.profil}, "
              f"retour a l'exercice {etat_courant.exercice_actuel_id}, difficulte conservee={etat_courant.difficulte})")
        return jsonify(_reponse_exercice_courant({
            "evaluation_requise": False,
            "seance": prochaine_seance,
            "materiel_requis": "Balle souple",
        }))

    # Session already in progress (screen refresh; no state change)
    return jsonify(_reponse_exercice_courant({
        "evaluation_requise": False,
        "seance": etat_courant.seance,
        "materiel_requis": "Balle souple",
    }))


# ============================================================================
# POST /evaluation — receives the 4 raw assessment tests and computes the K-NN profile
# ============================================================================
@app.route('/evaluation', methods=['POST'])
def evaluation():
    global etat_courant, agent_courant

    data = request.json or {}
    extracteur = ExtracteurMetriques()
    metriques = extracteur.extraire(
        data.get("test1_angles", []),
        data.get("test2_angles", []),
        data.get("test3_angles", []),
        data.get("test3_pression", []),
        data.get("test4_angles", []),
    )

    resultat = knn_model.predire(metriques)
    print(f"[KNN] Profil detecte : {resultat['dominant_profile_fr']} "
          f"({resultat['confidence']:.1f}%) | metriques={metriques}")

    etat_courant = GestionnaireSession.initialiser_depuis_profil(resultat)
    agent_courant = AgentRL(etat_courant)
    GestionnaireSession.sauvegarder_progression(etat_courant)

    return jsonify(_reponse_exercice_courant({
        "profil": resultat["dominant_profile"],
        "profil_fr": resultat["dominant_profile_fr"],
    }))


# ============================================================================
# POST /data — receives one complete set (success or voluntary abandonment)
# ============================================================================
@app.route('/data', methods=['POST'])
def receive_data():
    global etat_courant, agent_courant, seance_terminee_flag, amplitudes_seance_courante

    if etat_courant is None or agent_courant is None:
        return jsonify({"error": "Aucune evaluation en cours. Appelez /evaluation d'abord."}), 400

    data = request.json or {}
    succes = bool(data.get("succes", False))
    echec_manuel = bool(data.get("echec_manuel", False))
    serie_angles = data.get("serie_angles", [])
    serie_pression = data.get("serie_pression", [])

    metriques_obs = _extraire_metriques_serie(serie_angles, serie_pression)
    succes_reel = succes and not echec_manuel

    # Actual amplitude in degrees (normalized 0-1 -> 0-90 degrees), used for progress regression
    if serie_angles:
        amplitudes_seance_courante.append(metriques_obs["amplitude"] * 90.0)

    reward, action, msg, fin_seance, _ = agent_courant.step(metriques_obs, succes_reel)
    GestionnaireSession.sauvegarder_progression(agent_courant.etat)

    print(f"[RL] Exercice {etat_courant.exercice_actuel_id} | Serie {etat_courant.series_faites} "
          f"| succes={succes_reel} | reward={reward:.3f} | action={action} | {msg}")

    if fin_seance:
        seance_terminee_flag = True
        _ecrire_flag_fin_seance(True)
        print(f"[BRIDGE] SEANCE N°{etat_courant.seance} COMPLETEE ! Profil: {etat_courant.profil}")

        amplitude_moy = (sum(amplitudes_seance_courante) / len(amplitudes_seance_courante)
                          if amplitudes_seance_courante else 0.0)
        historique = _enregistrer_seance_dans_historique(etat_courant.seance, amplitude_moy)
        amplitudes_seance_courante = []

        return jsonify({
            "fin_seance": True,
            "coach": "Bravo, seance complete !",
            "prediction": _construire_prediction(etat_courant, historique),
            "message_rl": msg,
        })

    return jsonify(_reponse_exercice_courant({
        "fin_seance": False,
        "coach": f"Serie {etat_courant.series_faites}: {msg}",
        "message_rl": msg,
    }))


# ============================================================================
# STARTUP
# ============================================================================
if __name__ == '__main__':
    print("[BRIDGE] Chargement K-NN...")
    knn_model = ClassifieurKNN(k=5)
    knn_model.entrainer(80)

    _charger_etat_sauvegarde()
    seance_terminee_flag = _lire_flag_fin_seance()
    if etat_courant:
        print(f"[BRIDGE] Progression rechargee : profil={etat_courant.profil}, "
              f"seance={etat_courant.seance}, exercice={etat_courant.exercice_actuel_id}, "
              f"seance_terminee={seance_terminee_flag}")

    print("=======================================================")
    print("PIPELINE BRIDGE — GANT REEDUCATION (K-NN + RL branches)")
    print("=======================================================")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
```
