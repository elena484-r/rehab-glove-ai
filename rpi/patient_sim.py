"""
============================================================
PATIENT_SIM.PY — Simulateur de patient pour tests RL
============================================================
Simule la main d'un patient avec :
  - Raideur initiale selon le profil
  - Tremblement variable
  - Fatigue progressive dans la seance
  - Progression lente entre les seances

Utilise uniquement pour tester la boucle RL
sans hardware physique connecte.
============================================================
"""

import numpy as np


class PatientSimulator:
    """
    Simule les performances d'un patient post-AVC.
    Retourne des metriques realistes selon l'exercice
    et l'etat interne du patient.
    """

    PROFILS_PARAMS = {
        "deficit_mobilite": {
            "amplitude_base": 0.20, "force_base": 0.15,
            "vitesse_base":   0.22, "tremblement_base": 0.12,
            "asymetrie_base": 0.20, "taux_progression": 0.008,
        },
        "deficit_controle": {
            "amplitude_base": 0.55, "force_base": 0.50,
            "vitesse_base":   0.42, "tremblement_base": 0.78,
            "asymetrie_base": 0.30, "taux_progression": 0.006,
        },
        "deficit_coordination": {
            "amplitude_base": 0.45, "force_base": 0.40,
            "vitesse_base":   0.38, "tremblement_base": 0.18,
            "asymetrie_base": 0.80, "taux_progression": 0.007,
        },
        "recuperation": {
            "amplitude_base": 0.83, "force_base": 0.79,
            "vitesse_base":   0.76, "tremblement_base": 0.10,
            "asymetrie_base": 0.14, "taux_progression": 0.003,
        },
    }

    def __init__(self, profil="deficit_mobilite", seed=42):
        self.profil = profil
        self.rng    = np.random.default_rng(seed)
        params      = self.PROFILS_PARAMS[profil]

        # Etat interne
        self.amplitude   = params["amplitude_base"]
        self.force       = params["force_base"]
        self.vitesse     = params["vitesse_base"]
        self.tremblement = params["tremblement_base"]
        self.asymetrie   = params["asymetrie_base"]
        self.fatigue     = 0.0          # 0 = repose, 1 = epuise
        self.seance      = 0
        self.progression = params["taux_progression"]

    def simuler_exercice(self, exercice, niveau):
        """
        Simule la performance du patient sur un exercice a un niveau donne.
        Retourne les metriques observees [0-1] et si l'exercice est reussi.
        """
        niv_params = exercice.niveaux[niveau - 1]
        seuil      = niv_params.seuil_succes

        # Difficulte augmente la fatigue
        self.fatigue = min(1.0, self.fatigue + 0.05 * niveau)

        # Bruit gaussien realiste
        bruit = lambda: self.rng.normal(0, 0.04)

        # Performance observee (penalisee par la fatigue)
        facteur_fatigue = 1.0 - 0.3 * self.fatigue

        obs = {
            "amplitude":   np.clip(self.amplitude   * facteur_fatigue + bruit(), 0, 1),
            "force":       np.clip(self.force        * facteur_fatigue + bruit(), 0, 1),
            "vitesse":     np.clip(self.vitesse      * facteur_fatigue + bruit(), 0, 1),
            "tremblement": np.clip(self.tremblement  * (1 + 0.2 * self.fatigue) + bruit(), 0, 1),
            "asymetrie":   np.clip(self.asymetrie    + bruit(), 0, 1),
        }

        # Succes : metrique principale depasse le seuil
        m_princ = obs[exercice.metrique_principale]
        # Pour tremblement : succes si FAIBLE (inverser)
        if exercice.metrique_principale == "tremblement":
            succes = (1.0 - m_princ) >= seuil
        else:
            succes = m_princ >= seuil

        return obs, succes

    def nouvelle_seance(self):
        """Reinitialise la fatigue et applique la progression inter-seances."""
        self.fatigue  = 0.0
        self.seance  += 1
        # Progression lente mais reelle
        self.amplitude   = min(1.0, self.amplitude   + self.progression)
        self.force       = min(1.0, self.force        + self.progression * 0.8)
        self.vitesse     = min(1.0, self.vitesse      + self.progression * 0.7)
        self.tremblement = max(0.0, self.tremblement  - self.progression * 0.5)
        self.asymetrie   = max(0.0, self.asymetrie    - self.progression * 0.6)

    def etat(self):
        """Retourne l'etat actuel du patient."""
        return {
            "profil":      self.profil,
            "seance":      self.seance,
            "fatigue":     round(self.fatigue, 3),
            "amplitude":   round(self.amplitude, 3),
            "tremblement": round(self.tremblement, 3),
        }


if __name__ == "__main__":
    from exercises import get_exercices_pour_profil

    patient   = PatientSimulator("deficit_mobilite")
    exercices = get_exercices_pour_profil("deficit_mobilite")

    print("=== SIMULATION PATIENT ===")
    print(f"Profil : {patient.profil}")

    for seance in range(3):
        patient.nouvelle_seance()
        print(f"\n--- Seance {seance+1} ---")
        for ex in exercices:
            obs, succes = patient.simuler_exercice(ex, niveau=2)
            print(f"  {ex.nom}: {'REUSSI' if succes else 'ECHEC'} | "
                  f"amp={obs['amplitude']:.2f} trem={obs['tremblement']:.2f}")
