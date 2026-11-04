"""
============================================================
LAYER 2 — RULE-BASED ADAPTIVE AGENT
============================================================
Exact specified rules:

Session 1          : K-NN assessment -> profile -> exercises
Sessions 2-4       : loads progress_patient.json and resumes
Session 5 (or later): K-NN re-assessment if progress is detected

Action 3 (move to the next exercise) is triggered if:
  1. Patient succeeds 3 times in a row at level 5 (complete mastery)
  2. Maximum reached: MAX_SERIES attempts without reaching level 5

Safety Envelope: difficulty clamped to [1-5], never outside bounds.

Reward Shaping:
  R = sum(w_i * performance_i) - penalties
  The weights correspond to the current exercise.
============================================================
"""

import json
import os
import numpy as np
from exercises import get_exercices_pour_profil, BIBLIOTHEQUE


PROGRESS_FILE = "progress_patient.json"
MAX_SERIES    = 5    # safety limit before automatic progression
SUCCES_NV5    = 3    # number of level 5 successes required to validate the exercise


class EtatPatient:
    """Complete patient state saved between sessions."""

    def __init__(self, profil, exercices_ids, difficulte=1, seance=1):
        self.profil        = profil
        self.exercices_ids = exercices_ids   # list of exercise IDs
        self.exercice_idx  = 0               # index in the exercise list
        self.difficulte    = difficulte      # current level [1-5]
        self.seance        = seance
        self.historique    = []              # bool: True=success, False=failure (last 3)
        self.series_faites = 0              # number of sets completed for the current exercise
        self.succes_nv5    = 0              # number of successes at level 5
        self.fatigue       = 0.0

    @property
    def exercice_actuel_id(self):
        return self.exercices_ids[self.exercice_idx]

    def vers_dict(self):
        return {
            "profil":        self.profil,
            "exercices_ids": self.exercices_ids,
            "exercice_idx":  self.exercice_idx,
            "difficulte":    self.difficulte,
            "seance":        self.seance,
            "historique":    self.historique,
            "series_faites": self.series_faites,
            "succes_nv5":    self.succes_nv5,
            "fatigue":       self.fatigue,
        }

    @classmethod
    def depuis_dict(cls, d):
        e = cls(d["profil"], d["exercices_ids"], d["difficulte"], d["seance"])
        e.exercice_idx  = d["exercice_idx"]
        e.historique    = d["historique"]
        e.series_faites = d["series_faites"]
        e.succes_nv5    = d["succes_nv5"]
        e.fatigue       = d["fatigue"]
        return e


class AgentRL:
    """
    Rule-based adaptive agent for rehabilitation management,
    inspired by reinforcement learning concepts.

    Actions:
      0 : Maintain difficulty
      1 : Increase difficulty (+1, max 5)
      2 : Decrease difficulty (-1, min 1)
      3 : Move to the next exercise

    Safety Envelope: every action is validated before being applied.
    """

    def __init__(self, etat: EtatPatient):
        self.etat = etat

    # ----------------------------------------------------------
    # SAFETY ENVELOPE
    # ----------------------------------------------------------
    def _clamp_difficulte(self, d):
        """Ensures that difficulty remains within [1, 5]."""
        return max(1, min(5, d))

    # ----------------------------------------------------------
    # REWARD SHAPING
    # ----------------------------------------------------------
    def calculer_reward(self, metriques_obs, succes, exercice):
        """
        R = sum(w_i * perf_i) - penalties
        Adaptive weights based on the current exercise.
        """
        w = exercice.poids_reward
        perf = metriques_obs

        # Positive contribution of each metric
        # For tremor: lower is better, so the value is inverted
        r_amp   = w["amplitude"]   * perf.get("amplitude", 0)
        r_force = w["force"]       * perf.get("force", 0)
        r_vit   = w["vitesse"]     * perf.get("vitesse", 0)
        r_trem  = w["tremblement"] * (1.0 - perf.get("tremblement", 0))
        r_asym  = w["asymetrie"]   * (1.0 - perf.get("asymetrie", 0))

        reward = r_amp + r_force + r_vit + r_trem + r_asym

        # Penalties
        if not succes:
            reward -= 0.2
        if self.etat.fatigue > 0.7:
            reward -= 0.15  # high fatigue penalty

        return round(reward, 4)

    # ----------------------------------------------------------
    # ACTION SELECTION
    # ----------------------------------------------------------
    def choisir_action(self, metriques_obs, succes):
        """
        Selects an action according to predefined adaptive rules.

        Returns (action, justification)
        """
        etat = self.etat
        hist = etat.historique
        d    = etat.difficulte

        # Update history (keep the last 3 results)
        hist.append(succes)
        if len(hist) > 3:
            hist.pop(0)

        # Update counters
        etat.series_faites += 1
        if succes and d == 5:
            etat.succes_nv5 += 1
        else:
            etat.succes_nv5 = 0

        # Fatigue increases with each set
        etat.fatigue = min(1.0, etat.fatigue + 0.04)

        # --- ACTION 3 RULE: move to the next exercise ---
        # Case 1: complete mastery (3 successes at level 5)
        if etat.succes_nv5 >= SUCCES_NV5:
            return 3, f"Maitrise niveau 5 ! Exercice suivant."

        # Case 2: safety limit (MAX_SERIES without reaching level 5)
        if etat.series_faites >= MAX_SERIES:
            return 3, f"Plafond {MAX_SERIES} series. Exercice suivant."

        # --- DIFFICULTY RULES ---
        # 3 consecutive successes -> increase difficulty
        if len(hist) == 3 and all(hist):
            return 1, f"3 succes ! Niveau {d}->{min(5,d+1)}"

        # 2 consecutive failures -> decrease difficulty
        if len(hist) >= 2 and not any(hist[-2:]):
            return 2, f"2 echecs. Niveau {d}->{max(1,d-1)}"

        # High fatigue + failure -> decrease difficulty
        if etat.fatigue > 0.7 and not succes:
            return 2, f"Fatigue elevee. Niveau {d}->{max(1,d-1)}"

        # Maintain difficulty
        return 0, f"Maintien niveau {d}"

    # ----------------------------------------------------------
    # APPLY ACTION
    # ----------------------------------------------------------
    def appliquer_action(self, action, justification):
        """Applies the decision using the safety envelope."""
        etat = self.etat
        exercice_actuel = BIBLIOTHEQUE[etat.exercice_actuel_id]
        msg = justification

        if action == 0:
            # Maintain difficulty
            pass

        elif action == 1:
            # Increase difficulty
            etat.difficulte = self._clamp_difficulte(etat.difficulte + 1)
            etat.historique = []   # reset consecutive results after level change

        elif action == 2:
            # Decrease difficulty
            etat.difficulte = self._clamp_difficulte(etat.difficulte - 1)
            etat.historique = []   # reset consecutive results after level change

        elif action == 3:
            # Move to the next exercise
            if etat.exercice_idx < len(etat.exercices_ids) - 1:
                etat.exercice_idx += 1
                etat.series_faites = 0
                etat.succes_nv5    = 0
                etat.historique    = []       # reset consecutive result counters
                etat.difficulte    = 1        # reset difficulty to level 1
                etat.fatigue       = 0.0      # reset fatigue for the new exercise
                nouvel_ex = BIBLIOTHEQUE[etat.exercice_actuel_id]
                msg = f"{justification} -> {nouvel_ex.nom}"
            else:
                msg = "Seance terminee ! Tous les exercices completes."
                return action, msg, True  # fin_seance=True

        return action, msg, False  # fin_seance=False

    # ----------------------------------------------------------
    # COMPLETE STEP
    # ----------------------------------------------------------
    def step(self, metriques_obs, succes):
        """
        Performs one complete agent step.
        Returns: (reward, action, message, fin_seance, etat_dict)
        """
        exercice = BIBLIOTHEQUE[self.etat.exercice_actuel_id]
        reward   = self.calculer_reward(metriques_obs, succes, exercice)
        action, justif = self.choisir_action(metriques_obs, succes)
        action, msg, fin_seance = self.appliquer_action(action, justif)

        return reward, action, msg, fin_seance, self.etat.vers_dict()


# ============================================================
# SESSION MANAGEMENT
# ============================================================
class GestionnaireSession:
    """
    Manages session logic:
      Session 1   : K-NN assessment required
      Sessions 2-4: resume from progress_patient.json
      Session 5+  : K-NN re-assessment
    """

    @staticmethod
    def doit_reevaluer(numero_seance):
        """Returns True if a K-NN assessment is required."""
        return numero_seance == 1 or numero_seance % 5 == 0

    @staticmethod
    def charger_progression():
        """Loads the state from the JSON file. Returns None for the first session."""
        if not os.path.exists(PROGRESS_FILE):
            return None
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
        return EtatPatient.depuis_dict(data)

    @staticmethod
    def sauvegarder_progression(etat: EtatPatient, chemin=PROGRESS_FILE):
        def convertir_types_numpy(obj):
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(chemin, "w") as f:
            json.dump(etat.vers_dict(), f, indent=2, default=convertir_types_numpy)

    @staticmethod
    def initialiser_depuis_profil(resultat_knn):
        """Creates a new EtatPatient from the K-NN result."""
        profil    = resultat_knn["dominant_profile"]
        exercices = get_exercices_pour_profil(profil)
        ids       = [ex.id for ex in exercices]

        etat_existant = GestionnaireSession.charger_progression()
        numero_seance = (etat_existant.seance + 1) if etat_existant else 1

        return EtatPatient(
            profil=profil,
            exercices_ids=ids,
            difficulte=1,
            seance=numero_seance,
        )


if __name__ == "__main__":
    from exercises import BIBLIOTHEQUE

    print("=== TEST AGENT  ===")

    # Simulate a session
    etat = EtatPatient(
        profil="deficit_mobilite",
        exercices_ids=[1, 2, 3, 7],
        difficulte=2,
        seance=1,
    )
    agent = AgentRL(etat)

    print(f"Exercice : {BIBLIOTHEQUE[etat.exercice_actuel_id].nom}")
    print(f"Difficulte initiale : {etat.difficulte}")

    # Simulate 8 steps
    for i in range(8):
        obs     = {"amplitude":0.6,"force":0.5,"vitesse":0.4,"tremblement":0.2,"asymetrie":0.3}
        succes  = i % 3 != 2  # failure every 3 iterations
        reward, action, msg, fin, _ = agent.step(obs, succes)
        print(f"  Step {i+1} | succes={succes} | action={action} | {msg} | R={reward:.3f}")
        if fin:
            print("  -> SEANCE TERMINEE")
            break
