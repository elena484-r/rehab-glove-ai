"""
============================================================
PIPELINE BRIDGE — GANT DE REEDUCATION AVC
============================================================
Sert de pont HTTP/JSON entre le firmware ESP32 et les 3 couches :
  - couche1_knn.py      : diagnostic K-NN (profil moteur)
  - exercises.py         : bibliotheque des 10 exercices (5 niveaux chacun)
  - couche2_rl_env.py    : agent RL adaptatif (difficulte, reward shaping)

Routes exposees (doivent matcher EXACTEMENT ce que envoie esp32.cpp) :
  GET  /status      -> l'ESP32 l'appelle a l'accueil (double appui) pour
                        savoir s'il doit passer par l'evaluation K-NN ou
                        reprendre directement une seance en cours.
  POST /evaluation   -> l'ESP32 envoie les 4 tests bruts (test1_angles,
                        test2_angles, test3_angles, test3_pression,
                        test4_angles) a la fin de l'evaluation.
  POST /data          -> l'ESP32 envoie, a la fin de CHAQUE SERIE (= une
                        tentative complete d'atteindre repetitions_cible a
                        la difficulte courante, ou un abandon volontaire) :
                        serie_angles, serie_pression (buffer complet
                        echantillonne a 300ms), succes, echec_manuel.
                        Une serie = un appel a AgentRL.step().
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

HISTORIQUE_FILE = "historique_patient.json"   # journal amplitude moyenne par seance (pour la regression)
SESSION_FLAG_FILE = "session_state.json"      # persiste seance_terminee_flag entre 2 redemarrages du serveur
SEANCE_MIN_PREDICTION = 4  # pas de prediction chiffree avant la seance 4 (pas assez de donnees)


def _lire_flag_fin_seance():
    """Relit sur disque si la derniere seance connue s'est terminee (survit a un redemarrage)."""
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
# ETAT GLOBAL DU SERVEUR (mono-patient, mono-gant — un seul device connecte)
# ============================================================================
knn_model = None          # ClassifieurKNN entraine au demarrage
etat_courant = None       # EtatPatient | None
agent_courant = None      # AgentRL | None
seance_terminee_flag = False   # mis a True quand AgentRL.step() renvoie fin_seance
amplitudes_seance_courante = []  # amplitudes (deg) de chaque serie de la seance en cours


def _charger_etat_sauvegarde():
    """Recharge l'etat patient depuis progress_patient.json si present."""
    global etat_courant, agent_courant
    etat = GestionnaireSession.charger_progression()
    if etat is not None:
        etat_courant = etat
        agent_courant = AgentRL(etat_courant)
    return etat


def _reponse_exercice_courant(extra=None):
    """Construit la portion JSON decrivant l'exercice/niveau courant pour l'ESP32."""
    ex = BIBLIOTHEQUE[etat_courant.exercice_actuel_id]
    niveau = ex.niveaux[etat_courant.difficulte - 1]  # niveaux 1..5 -> index 0..4
    payload = {
        "exercice_id": ex.id,                                   # id brut bibliotheque (1-10), usage interne/logs
        "exercice_position": etat_courant.exercice_idx + 1,      # position 1..N dans le parcours du profil -> a afficher
        "exercice_total": len(etat_courant.exercices_ids),       # N = nb d'exercices du profil (affichage "X / N")
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
    Calcule les 5 metriques normalisees [0-1] attendues par AgentRL.calculer_reward
    a partir du buffer brut d'une serie (meme logique que ExtracteurMetriques,
    mais sur un seul buffer continu au lieu des 4 tests separes, et sans le
    facteur *100 puisque couche2_rl_env attend une echelle [0-1]).
    """
    if not serie_angles:
        return {"amplitude": 0.0, "force": 0.0, "vitesse": 0.0,
                "tremblement": 0.0, "asymetrie": 0.0}

    angles = np.array(serie_angles)      # (N, 5)
    pression = np.array(serie_pression) if serie_pression else np.zeros((1, 3))

    # Amplitude : moyenne des maxima par doigt, normalisee sur 90 degres
    amplitude = float(np.mean(np.max(angles, axis=0)) / 90.0)
    amplitude = max(0.0, min(1.0, amplitude))

    # Vitesse : plus grande variation moyenne inter-echantillon (deg/echantillon), normalisee
    if len(angles) > 1:
        delta = np.abs(np.diff(angles, axis=0))
        vitesse = float(np.max(np.mean(delta, axis=1))) / 45.0
    else:
        vitesse = 0.0
    vitesse = max(0.0, min(1.0, vitesse))

    # Force : pression moyenne (deja en % 0-100 cote firmware), normalisee
    force = float(np.mean(pression)) / 100.0 if pression.size > 0 else 0.0
    force = max(0.0, min(1.0, force))

    # Tremblement : variance temporelle moyenne des angles, normalisee
    if len(angles) > 1:
        tremblement = float(np.mean(np.var(angles, axis=0))) / 50.0
    else:
        tremblement = 0.0
    tremblement = max(0.0, min(1.0, tremblement))

    # Asymetrie : dispersion entre les maxima des differents doigts, normalisee
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
    """Ajoute un point (date, amplitude moyenne en degres) au journal de progression."""
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
    Message de prediction affiche en fin de seance (ecran ETAT_PREDICTION).

    - Seances 1 a 3 : pas assez de points pour une regression fiable -> message
      generique qualitatif seulement.
    - Seance 4 et suivantes : regression lineaire (scikit-learn) de l'amplitude
      moyenne (degres) en fonction du temps ecoule (jours), extrapolee a +14
      jours, sur l'historique reellement enregistre.
    """
    if etat.seance < SEANCE_MIN_PREDICTION or len(historique) < 2:
        return "Progression en cours : quelques seances de plus sont necessaires pour une estimation chiffree."

    dates = [datetime.fromisoformat(h["timestamp"]) for h in historique]
    amplitudes = [h["amplitude_moy_deg"] for h in historique]
    t0 = dates[0]
    jours = np.array([[(d - t0).total_seconds() / 86400.0] for d in dates])
    y = np.array(amplitudes)

    modele = LinearRegression()
    modele.fit(jours, y)

    jour_actuel = jours[-1][0]
    amplitude_actuelle = amplitudes[-1]
    amplitude_predite_14j = float(modele.predict([[jour_actuel + 14]])[0])
    delta = amplitude_predite_14j - amplitude_actuelle
    pente_par_jour = float(modele.coef_[0])

    if delta >= 1.0:
        return (f"Tendance positive : amplitude estimee a +{delta:.0f} deg "
                f"dans les 2 prochaines semaines (regression sur {len(historique)} seances). "
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

        # Seances 2-4 : reprise directe, pas de nouvelle evaluation.
        # On reboucle sur le 1er exercice du profil (le round precedent est termine) ;
        # la difficulte gagnee est conservee, mais les compteurs de serie repartent a zero.
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

    # Seance deja en cours (rafraichissement d'ecran, pas de changement d'etat)
    return jsonify(_reponse_exercice_courant({
        "evaluation_requise": False,
        "seance": etat_courant.seance,
        "materiel_requis": "Balle souple",
    }))


# ============================================================================
# POST /evaluation — recoit les 4 tests bruts, calcule le profil K-NN
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
# POST /data — une serie complete (succes ou abandon volontaire)
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

    # Amplitude reelle en degres (0-1 normalise -> 0-90 deg), utilisee pour la regression de progression
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
        "coach": f"Serie {etat_courant.series_faites} - {msg}",
        "message_rl": msg,
    }))


# ============================================================================
# DEMARRAGE
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
