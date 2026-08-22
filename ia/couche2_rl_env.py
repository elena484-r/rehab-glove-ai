"""
============================================================
COUCHE 2 — AGENT RL ADAPTATIF
============================================================
Regles exactes specifiees :

Seance 1          : evaluation K-NN -> profil -> exercices
Seances 2-4       : charge progress_patient.json, reprend
Seance 5 (ou +)   : re-evaluation K-NN si progression

Action 3 (passer a l'exercice suivant) declenchee si :
  1. Patient reussi 3x de suite au niveau 5 (maitrise complete)
  2. Plafond atteint : MAX_SERIES tentatives sans atteindre niv 5

Safety Envelope : difficulte clampee [1-5], jamais hors bornes.

Reward Shaping :
  R = somme(w_i * performance_i) - penalites
  Les poids sont ceux de l'exercice en cours.
============================================================
"""

import json
import os
import numpy as np
from exercises import get_exercices_pour_profil, BIBLIOTHEQUE


PROGRESS_FILE = "progress_patient.json"
MAX_SERIES    = 5    # plafond securite avant passage auto
SUCCES_NV5    = 3    # nombre de succes niv5 pour valider l'exercice


class EtatPatient:
    """Etat complet du patient sauvegarde entre les seances."""

    def __init__(self, profil, exercices_ids, difficulte=1, seance=1):
        self.profil        = profil
        self.exercices_ids = exercices_ids   # liste des IDs exercices
        self.exercice_idx  = 0               # index dans la liste
        self.difficulte    = difficulte      # niveau actuel [1-5]
        self.seance        = seance
        self.historique    = []              # bool: True=succes, False=echec (3 derniers)
        self.series_faites = 0              # compteur series exercice actuel
        self.succes_nv5    = 0              # compteur succes au niveau 5
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
    Agent RL adaptatif pour la gestion de la reeducation.

    Actions :
      0 : Maintenir la difficulte
      1 : Augmenter la difficulte (+1, max 5)
      2 : Diminuer la difficulte (-1, min 1)
      3 : Passer a l'exercice suivant

    Safety Envelope : toute action est validee avant application.
    """

    def __init__(self, etat: EtatPatient):
        self.etat = etat

    # ----------------------------------------------------------
    # SAFETY ENVELOPE
    # ----------------------------------------------------------
    def _clamp_difficulte(self, d):
        """Garantit que la difficulte reste dans [1, 5]."""
        return max(1, min(5, d))

    # ----------------------------------------------------------
    # REWARD SHAPING
    # ----------------------------------------------------------
    def calculer_reward(self, metriques_obs, succes, exercice):
        """
        R = somme(w_i * perf_i) - penalites
        Poids adaptatifs selon l'exercice en cours.
        """
        w = exercice.poids_reward
        perf = metriques_obs

        # Contribution positive de chaque metrique
        # Pour le tremblement : moins = mieux, donc on inverse
        r_amp   = w["amplitude"]   * perf.get("amplitude", 0)
        r_force = w["force"]       * perf.get("force", 0)
        r_vit   = w["vitesse"]     * perf.get("vitesse", 0)
        r_trem  = w["tremblement"] * (1.0 - perf.get("tremblement", 0))
        r_asym  = w["asymetrie"]   * (1.0 - perf.get("asymetrie", 0))

        reward = r_amp + r_force + r_vit + r_trem + r_asym

        # Penalites
        if not succes:
            reward -= 0.2
        if self.etat.fatigue > 0.7:
            reward -= 0.15  # penalite fatigue elevee

        return round(reward, 4)

    # ----------------------------------------------------------
    # SELECTION ACTION
    # ----------------------------------------------------------
    def choisir_action(self, metriques_obs, succes):
        """
        Choisit l'action RL selon les regles specifiees.

        Retourne (action, justification)
        """
        etat = self.etat
        hist = etat.historique
        d    = etat.difficulte

        # Mise a jour historique (garder 3 derniers)
        hist.append(succes)
        if len(hist) > 3:
            hist.pop(0)

        # Mise a jour compteurs
        etat.series_faites += 1
        if succes and d == 5:
            etat.succes_nv5 += 1
        else:
            etat.succes_nv5 = 0

        # Fatigue augmente avec les series
        etat.fatigue = min(1.0, etat.fatigue + 0.04)

        # --- REGLE ACTION 3 : passer a l'exercice suivant ---
        # Cas 1 : maitrise complete (3 succes au niveau 5)
        if etat.succes_nv5 >= SUCCES_NV5:
            return 3, f"Maitrise niveau 5 ! Exercice suivant."

        # Cas 2 : plafond de securite (MAX_SERIES sans atteindre niv 5)
        if etat.series_faites >= MAX_SERIES:
            return 3, f"Plafond {MAX_SERIES} series. Exercice suivant."

        # --- REGLES DIFFICULTE ---
        # 3 succes consecutifs -> augmenter
        if len(hist) == 3 and all(hist):
            return 1, f"3 succes ! Niveau {d}->{min(5,d+1)}"

        # 2 echecs consecutifs -> diminuer
        if len(hist) >= 2 and not any(hist[-2:]):
            return 2, f"2 echecs. Niveau {d}->{max(1,d-1)}"

        # Fatigue elevee + echec -> reduire
        if etat.fatigue > 0.7 and not succes:
            return 2, f"Fatigue elevee. Niveau {d}->{max(1,d-1)}"

        # Maintenir
        return 0, f"Maintien niveau {d}"

    # ----------------------------------------------------------
    # APPLIQUER ACTION
    # ----------------------------------------------------------
    def appliquer_action(self, action, justification):
        """Applique la decision RL avec safety envelope."""
        etat = self.etat
        exercice_actuel = BIBLIOTHEQUE[etat.exercice_actuel_id]
        msg = justification

        if action == 0:
            # Maintien
            pass

        elif action == 1:
            # Augmenter difficulte
            etat.difficulte = self._clamp_difficulte(etat.difficulte + 1)
            etat.historique = []   # CORRECTION BUG 1 : reset consecutifs apres changement niveau

        elif action == 2:
            # Diminuer difficulte
            etat.difficulte = self._clamp_difficulte(etat.difficulte - 1)
            etat.historique = []   # CORRECTION BUG 1 : reset consecutifs apres changement niveau

        elif action == 3:
            # Passer a l'exercice suivant
            if etat.exercice_idx < len(etat.exercices_ids) - 1:
                etat.exercice_idx += 1
                etat.series_faites = 0
                etat.succes_nv5    = 0
                etat.historique    = []       # reset compteurs consecutifs
                etat.difficulte    = 1        # CORRECTION BUG 2 : reset niveau a 1
                etat.fatigue       = 0.0      # reset fatigue pour nouvel exercice
                nouvel_ex = BIBLIOTHEQUE[etat.exercice_actuel_id]
                msg = f"{justification} -> {nouvel_ex.nom}"
            else:
                msg = "Seance terminee ! Tous les exercices completes."
                return action, msg, True  # fin_seance=True

        return action, msg, False  # fin_seance=False

    # ----------------------------------------------------------
    # STEP COMPLET
    # ----------------------------------------------------------
    def step(self, metriques_obs, succes):
        """
        Un pas complet de l'agent RL.
        Retourne : (reward, action, message, fin_seance, etat_dict)
        """
        exercice = BIBLIOTHEQUE[self.etat.exercice_actuel_id]
        reward   = self.calculer_reward(metriques_obs, succes, exercice)
        action, justif = self.choisir_action(metriques_obs, succes)
        action, msg, fin_seance = self.appliquer_action(action, justif)

        return reward, action, msg, fin_seance, self.etat.vers_dict()


# ============================================================
# GESTIONNAIRE DE SESSIONS
# ============================================================
class GestionnaireSession:
    """
    Gere la logique de sessions :
      Seance 1   : evaluation K-NN requise
      Seances 2-4: reprise depuis progress_patient.json
      Seance 5+  : re-evaluation K-NN
    """

    @staticmethod
    def doit_reevaluer(numero_seance):
        """Retourne True si une evaluation K-NN est necessaire."""
        return numero_seance == 1 or numero_seance % 5 == 0

    @staticmethod
    def charger_progression():
        """Charge l'etat depuis le fichier JSON. None si premiere seance."""
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
        """Cree un nouvel EtatPatient depuis le resultat K-NN."""
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

    print("=== TEST AGENT RL ===")

    # Simulation d'une session
    etat = EtatPatient(
        profil="deficit_mobilite",
        exercices_ids=[1, 2, 3, 7],
        difficulte=2,
        seance=1,
    )
    agent = AgentRL(etat)

    print(f"Exercice : {BIBLIOTHEQUE[etat.exercice_actuel_id].nom}")
    print(f"Difficulte initiale : {etat.difficulte}")

    # Simuler 8 steps
    for i in range(8):
        obs     = {"amplitude":0.6,"force":0.5,"vitesse":0.4,"tremblement":0.2,"asymetrie":0.3}
        succes  = i % 3 != 2  # echec toutes les 3 iterations
        reward, action, msg, fin, _ = agent.step(obs, succes)
        print(f"  Step {i+1} | succes={succes} | action={action} | {msg} | R={reward:.3f}")
        if fin:
            print("  -> SEANCE TERMINEE")
            break
