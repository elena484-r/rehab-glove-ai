"""
============================================================
PIPELINE PRINCIPAL RPi — GANT DE RÉÉDUCATION AVC
============================================================
"""

import json
import os
import time
import numpy as np
from datetime import datetime
from collections import deque
from dotenv import load_dotenv

# ============================================================
# CHARGEMENT CONFIGURATION
# ============================================================
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

MODELE_KNN_PATH   = "modele_knn.json"
LOG_SESSIONS_PATH = "historique_sessions.jsonl"
LOG_DONNEES_PATH  = "donnees_capteurs.jsonl"

FENETRE_HISTORIQUE = 20  # 20 * 500ms = 10 secondes de données


# ============================================================
# COUCHE 2A — CALCUL DES 5 MÉTRIQUES OBJECTIVES
# ============================================================

def calculer_metriques(historique_angles: list, historique_pression: list) -> dict:
    if len(historique_angles) < 2:
        return {f: 0.0 for f in ["amplitude", "vitesse", "force", "tremblement", "asymetrie"]}

    angles   = np.array(historique_angles)    # (T, 5)
    pression = np.array(historique_pression)  # (T, 3)

    # 1. Amplitude
    amplitude_max = np.max(angles, axis=0)
    amplitude = float(np.mean(amplitude_max) / 90.0)

    # 2. Vitesse
    delta = np.abs(np.diff(angles, axis=0))
    vitesse_max = float(np.max(np.mean(delta, axis=1)))
    vitesse = min(1.0, vitesse_max / 45.0)

    # 3. Force
    force = float(np.mean(pression) / 100.0)

    # 4. Tremblement
    tremblement = min(1.0, float(np.mean(np.var(angles, axis=0))) / 400.0)

    # 5. Asymétrie
    asymetrie = min(1.0, float(np.std(amplitude_max)) / 45.0)

    return {
        "amplitude":   round(amplitude, 4),
        "vitesse":     round(vitesse, 4),
        "force":       round(force, 4),
        "tremblement": round(tremblement, 4),
        "asymetrie":   round(asymetrie, 4),
    }


# ============================================================
# COUCHE 2B — CLASSIFIEUR PROFIL MOTEUR (k-NN)
# ============================================================

class ClassifieurProfilMoteur:
    FEATURES = ["amplitude", "vitesse", "force", "tremblement", "asymetrie"]
    DESCRIPTIONS = {
        "deficit_mobilite":     "Déficit de mobilité — amplitude et force réduites",
        "deficit_controle":     "Déficit de contrôle — tremblements dominants",
        "deficit_coordination": "Déficit de coordination — asymétrie inter-doigts",
        "recuperation":         "Récupération satisfaisante — métriques proches de la normale",
    }

    def __init__(self, chemin: str = MODELE_KNN_PATH):
        with open(chemin) as f:
            m = json.load(f)
        self.k            = m["k"]
        self.classes_map  = m["training_data"]["classes_map"]
        self.X_train      = np.array(m["training_data"]["X"])
        self.y_train      = np.array(m["training_data"]["y"])
        self.scaler_mean  = np.array(m["scaler"]["mean"])
        self.scaler_std   = np.array(m["scaler"]["std"])
        print(f"[IA] Classifieur chargé — k={self.k}, {len(self.X_train)} exemples d'entraînement")

    def predire(self, metriques: dict) -> dict:
        x = np.array([metriques[f] for f in self.FEATURES]).reshape(1, -1)
        x_scaled = (x - self.scaler_mean) / self.scaler_std
        distances = np.sqrt(np.sum((self.X_train - x_scaled) ** 2, axis=1))
        idx = np.argsort(distances)[:self.k]
        votes = {}
        for i in idx:
            c = self.classes_map[str(self.y_train[i])]
            votes[c] = votes.get(c, 0) + 1
        profil = max(votes, key=votes.get)
        return {
            "profil":      profil,
            "description": self.DESCRIPTIONS.get(profil, profil),
            "confiance":   round(votes[profil] / self.k, 2),
            "votes":       votes,
        }


# ============================================================
# COUCHE 3 — POLITIQUE ADAPTATIVE (RL simple)
# ============================================================

class PolitiqueAdaptative:
    def __init__(self, difficulte_initiale: int = 2):
        self.difficulte   = difficulte_initiale
        self.historique   = []
        self.nb_decisions = 0

    def enregistrer_resultat(self, succes: bool, profil: str) -> dict:
        self.historique.append(succes)
        ancienne_diff = self.difficulte
        decision = "maintien"

        if len(self.historique) >= 3 and all(self.historique[-3:]):
            if self.difficulte < 5:
                self.difficulte += 1
                decision = "augmentation"
                self.historique = []

        elif len(self.historique) >= 2 and not any(self.historique[-2:]):
            if self.difficulte > 1:
                self.difficulte -= 1
                decision = "reduction"
                self.historique = []

        if profil == "deficit_controle" and self.difficulte > 1:
            self.difficulte -= 1
            decision = "reduction_tremblement"

        self.nb_decisions += 1

        return {
            "decision":         decision,
            "difficulte_avant": ancienne_diff,
            "difficulte_apres": self.difficulte,
            "succes_recent":    succes,
            "profil_patient":   profil,
        }

    def get_parametres_exercice(self) -> dict:
        params = {
            1: {"repetitions": 10, "vitesse_cible": "lente",   "resistance": "faible"},
            2: {"repetitions": 20, "vitesse_cible": "lente",   "resistance": "faible"},
            3: {"repetitions": 30, "vitesse_cible": "normale", "resistance": "moyenne"},
            4: {"repetitions": 40, "vitesse_cible": "normale", "resistance": "forte"},
            5: {"repetitions": 50, "vitesse_cible": "rapide",  "resistance": "maximale"},
        }
        return params[self.difficulte]


# ============================================================
# COUCHE 4 — PRÉDICTION DE PROGRESSION
# ============================================================

class PredicteurProgression:
    def __init__(self, chemin_historique: str = LOG_SESSIONS_PATH):
        self.chemin = chemin_historique
        self.sessions = self._charger_sessions()

    def _charger_sessions(self) -> list:
        sessions = []
        if not os.path.exists(self.chemin):
            return sessions
        with open(self.chemin) as f:
            for line in f:
                try:
                    sessions.append(json.loads(line.strip()))
                except Exception:
                    pass
        return sessions

    def predire(self) -> dict:
        if len(self.sessions) < 3:
            return {
                "disponible": False,
                "message": "Historique insuffisant (min 3 sessions)",
                "nb_sessions": len(self.sessions)
            }

        scores = [s["scores"]["rom"] for s in self.sessions if "scores" in s]
        if len(scores) < 3:
            return {"disponible": False, "message": "Données ROM manquantes"}

        x = np.arange(len(scores), dtype=float)
        y = np.array(scores, dtype=float)

        x_mean, y_mean = np.mean(x), np.mean(y)
        pente = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        intercept = y_mean - pente * x_mean

        x_futur = len(scores) + 14
        score_predit = max(0.0, min(5.0, pente * x_futur + intercept))
        amplitude_predite = score_predit * 18

        return {
            "disponible":       True,
            "score_rom_actuel": round(scores[-1], 2),
            "score_rom_predit": round(score_predit, 2),
            "amplitude_predite_deg": round(amplitude_predite, 1),
            "dans_semaines":    2,
            "tendance":         "hausse" if pente > 0.01 else "stable" if pente > -0.01 else "baisse",
            "nb_sessions":      len(scores),
            "simule":           self.sessions[0].get("simule", False) if self.sessions else False,
        }


# ============================================================
# CLAUDE API — MESSAGE COACH
# ============================================================

def generer_message_coach(metriques: dict, profil: dict,
                          decision_rl: dict, prediction: dict) -> str:
    if not ANTHROPIC_API_KEY:
        return "Mode hors-ligne : continuez vos exercices régulièrement."

    try:
        import anthropic
        # Timeout court (3s) pour ne jamais bloquer le programme
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=3.0)

        contexte = f"""
Profil moteur détecté : {profil['description']} (confiance {profil['confiance']*100:.0f}%)
Métriques actuelles : amplitude={metriques['amplitude']:.2f}/1, 
  force={metriques['force']:.2f}/1, tremblement={metriques['tremblement']:.2f}/1,
  asymétrie={metriques['asymetrie']:.2f}/1
Décision système : {decision_rl['decision']} 
  (difficulté {decision_rl['difficulte_avant']}→{decision_rl['difficulte_apres']}/5)
"""
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=120,
            system="Tu es un coach de rééducation. Réponds en français en 2 phrases max.",
            messages=[{"role": "user", "content": contexte}]
        )
        return response.content[0].text.strip()

    except Exception:
        # Fallback hors-ligne instantané
        return "Excellent travail ! Continuez vos mouvements en douceur."


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

class PipelineGant:
    def __init__(self):
        print("=" * 60)
        print("PIPELINE GANT RÉÉDUCATION — DÉMARRAGE")
        print("=" * 60)

        self.classifieur  = ClassifieurProfilMoteur(MODELE_KNN_PATH)
        self.politique_rl = PolitiqueAdaptative(difficulte_initiale=2)
        self.predicteur   = PredicteurProgression(LOG_SESSIONS_PATH)

        self.historique_angles   = deque(maxlen=FENETRE_HISTORIQUE)
        self.historique_pression = deque(maxlen=FENETRE_HISTORIQUE)

        self.derniere_ligne_lue = 0
        self.derniere_analyse   = time.time()
        self.intervalle_analyse = 5.0

    def lire_nouvelles_donnees(self) -> int:
        if not os.path.exists(LOG_DONNEES_PATH):
            return 0

        nouvelles = 0
        try:
            with open(LOG_DONNEES_PATH, "r") as f:
                lignes = f.readlines()

            for ligne in lignes[self.derniere_ligne_lue:]:
                try:
                    data = json.loads(ligne.strip())
                    angles   = data.get("angles", [])
                    pression = data.get("pressure_pct", data.get("pressure", []))

                    if pression and max(pression) > 100:
                        pression = [min(100.0, p / 35.0) for p in pression]

                    if len(angles) == 5 and len(pression) == 3:
                        self.historique_angles.append(angles)
                        self.historique_pression.append(pression)
                        nouvelles += 1
                except Exception:
                    pass

            self.derniere_ligne_lue = len(lignes)
        except Exception:
            pass  # Évite les erreurs en cas de conflit de lecture/écriture avec serveur_gant.py

        return nouvelles

    def analyser(self) -> dict | None:
        if len(self.historique_angles) < 5:
            return None

        angles   = list(self.historique_angles)
        pression = list(self.historique_pression)

        metriques = calculer_metriques(angles, pression)
        profil = self.classifieur.predire(metriques)

        succes = metriques["amplitude"] > 0.5 and metriques["tremblement"] < 0.4
        decision_rl = self.politique_rl.enregistrer_resultat(
            succes=succes, profil=profil["profil"]
        )
        params = self.politique_rl.get_parametres_exercice()
        prediction = self.predicteur.predire()

        message_coach = generer_message_coach(metriques, profil, decision_rl, prediction)

        resultat = {
            "timestamp":     datetime.now().isoformat(),
            "metriques":     metriques,
            "profil":        profil,
            "decision_rl":   decision_rl,
            "parametres":    params,
            "prediction":    prediction,
            "message_coach": message_coach,
        }

        self._afficher(resultat)
        self._sauvegarder_session(resultat)
        return resultat

    def _afficher(self, r: dict):
        ts = r["timestamp"][11:19]
        m  = r["metriques"]
        p  = r["profil"]
        rl = r["decision_rl"]

        print(f"\n{'─'*60}\n[{ts}] ANALYSE PIPELINE\n{'─'*60}")
        print(f"  MÉTRIQUES    amp={m['amplitude']:.2f}  vit={m['vitesse']:.2f}  force={m['force']:.2f}")
        print(f"  PROFIL       {p['profil']} ({p['confiance']*100:.0f}%)")
        print(f"  RL           difficulte {rl['difficulte_apres']}/5")
        print(f"  COACH        {r['message_coach']}")

    def _sauvegarder_session(self, r: dict):
        try:
            with open(LOG_SESSIONS_PATH, "a") as f:
                f.write(json.dumps({
                    "timestamp":    r["timestamp"],
                    "metriques":    r["metriques"],
                    "profil":       r["profil"]["profil"],
                    "difficulte":   r["decision_rl"]["difficulte_apres"],
                    "scores":       {"rom": r["metriques"]["amplitude"] * 5, "force": r["metriques"]["force"] * 5},
                    "message_coach": r["message_coach"]
                }) + "\n")
        except Exception:
            pass

    def run(self):
        print("En écoute des données... (Ctrl+C pour arrêter)\n")
        try:
            while True:
                nouvelles = self.lire_nouvelles_donnees()
                if time.time() - self.derniere_analyse >= self.intervalle_analyse:
                    self.analyser()
                    self.derniere_analyse = time.time()

                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n[PIPELINE] Arrêt propre.")


if __name__ == "__main__":
    if not os.path.exists(LOG_DONNEES_PATH):
        with open(LOG_DONNEES_PATH, "w") as f:
            for i in range(30):
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "angles": [20.0]*5,
                    "pressure_pct": [30.0]*3
                }) + "\n")

    pipeline = PipelineGant()
    pipeline.run()
