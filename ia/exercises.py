"""
============================================================
EXERCISES.PY — Bibliotheque de 10 exercices instrumentes
============================================================
Chaque exercice a :
  - 5 niveaux de difficulte (1=facile, 5=expert)
  - Des poids de reward adaptes a son objectif
  - Une liste de consignes affichees sur l'OLED
  - Une assignation aux profils moteurs concernes

Attribution exercices -> profils :
  deficit_mobilite    : E1, E2, E3, E7
  deficit_controle    : E3, E8, E9, E4
  deficit_coordination: E2, E4, E5, E6
  recuperation        : E6, E9, E10, E7
============================================================
"""

from dataclasses import dataclass, field


# ============================================================
# PARAMETRES PAR NIVEAU DE DIFFICULTE
# ============================================================
@dataclass
class NiveauDifficulte:
    niveau: int               # 1 a 5
    repetitions: int          # nombre de repetitions cibles
    duree_s: int              # duree maintien en secondes (si applicable)
    vitesse: str              # "lente" | "normale" | "rapide"
    seuil_succes: float       # seuil de succes metrique principale [0-1]
    consigne_oled: str        # texte affiche sur l'ecran OLED


# ============================================================
# CLASSE DE BASE EXERCICE
# ============================================================
@dataclass
class Exercice:
    id: int
    nom: str
    description: str
    profils_cibles: list      # profils pour lesquels cet exercice est prescrit
    metrique_principale: str  # metrique evaluee en priorite
    poids_reward: dict        # w_amplitude, w_force, w_vitesse, w_tremblement, w_asymetrie
    niveaux: list             # liste de 5 NiveauDifficulte
    MAX_SERIES: int = 5       # plafond de securite : max series avant passage auto


# ============================================================
# DEFINITION DES 10 EXERCICES
# ============================================================
def creer_bibliotheque():
    """Retourne la liste des 10 exercices avec leurs 5 niveaux."""

    exercices = [

        # E1 — Ouverture/Fermeture globale
        Exercice(
            id=1, nom="Ouverture/Fermeture",
            description="Ouvrez et fermez la main completement.",
            profils_cibles=["deficit_mobilite"],
            metrique_principale="amplitude",
            poids_reward={"amplitude":0.5,"force":0.1,"vitesse":0.2,"tremblement":0.1,"asymetrie":0.1},
            niveaux=[
                NiveauDifficulte(1, 10, 0, "lente",  0.30, "Fermez la main doucement. 10x"),
                NiveauDifficulte(2, 15, 0, "lente",  0.45, "Fermez la main doucement. 15x"),
                NiveauDifficulte(3, 20, 0, "normale",0.55, "Fermez/ouvrez regulierement. 20x"),
                NiveauDifficulte(4, 30, 0, "normale",0.65, "Amplitude maximale. 30x"),
                NiveauDifficulte(5, 40, 0, "rapide", 0.75, "Vitesse + amplitude max. 40x"),
            ]
        ),

        # E2 — Flexion doigt par doigt
        Exercice(
            id=2, nom="Flexion par doigt",
            description="Pliez chaque doigt un par un.",
            profils_cibles=["deficit_mobilite", "deficit_coordination"],
            metrique_principale="asymetrie",
            poids_reward={"amplitude":0.2,"force":0.1,"vitesse":0.1,"tremblement":0.2,"asymetrie":0.4},
            niveaux=[
                NiveauDifficulte(1,  5, 0, "lente",  0.20, "Pliez le pouce seul. 5x"),
                NiveauDifficulte(2, 10, 0, "lente",  0.35, "Chaque doigt, lentement. 10x"),
                NiveauDifficulte(3, 15, 0, "lente",  0.50, "Isolez chaque doigt. 15x"),
                NiveauDifficulte(4, 20, 0, "normale",0.60, "Independance digitale. 20x"),
                NiveauDifficulte(5, 25, 0, "normale",0.70, "Controle parfait. 25x"),
            ]
        ),

        # E3 — Flexion progressive controlee
        Exercice(
            id=3, nom="Flexion controlee",
            description="Fermez la main tres lentement, maintenez.",
            profils_cibles=["deficit_mobilite", "deficit_controle"],
            metrique_principale="tremblement",
            poids_reward={"amplitude":0.3,"force":0.1,"vitesse":0.1,"tremblement":0.4,"asymetrie":0.1},
            niveaux=[
                NiveauDifficulte(1,  5, 3, "lente",  0.20, "Fermez lentement. Maintenez 3s. 5x"),
                NiveauDifficulte(2,  8, 4, "lente",  0.30, "Lent + maintien 4s. 8x"),
                NiveauDifficulte(3, 10, 5, "lente",  0.45, "Tres lent, stable. 10x"),
                NiveauDifficulte(4, 12, 6, "lente",  0.55, "Stabilite maximale. 12x"),
                NiveauDifficulte(5, 15, 8, "lente",  0.65, "Controle parfait. 15x"),
            ]
        ),

        # E4 — Opposition Pouce-Index
        Exercice(
            id=4, nom="Opposition Pouce-Index",
            description="Pincez avec le pouce et l'index.",
            profils_cibles=["deficit_controle", "deficit_coordination"],
            metrique_principale="force",
            poids_reward={"amplitude":0.2,"force":0.4,"vitesse":0.1,"tremblement":0.2,"asymetrie":0.1},
            niveaux=[
                NiveauDifficulte(1,  8, 0, "lente",  0.20, "Pincez pouce+index. 8x"),
                NiveauDifficulte(2, 12, 0, "lente",  0.35, "Pincez fermement. 12x"),
                NiveauDifficulte(3, 15, 2, "lente",  0.45, "Pince + maintien 2s. 15x"),
                NiveauDifficulte(4, 20, 3, "normale",0.55, "Force + precision. 20x"),
                NiveauDifficulte(5, 25, 4, "normale",0.65, "Pince maximale. 25x"),
            ]
        ),

        # E5 — Sequence Pouce -> chaque doigt
        Exercice(
            id=5, nom="Sequence pouce-doigts",
            description="Touchez chaque doigt avec le pouce en sequence.",
            profils_cibles=["deficit_coordination"],
            metrique_principale="asymetrie",
            poids_reward={"amplitude":0.1,"force":0.1,"vitesse":0.2,"tremblement":0.2,"asymetrie":0.4},
            niveaux=[
                NiveauDifficulte(1,  5, 0, "lente",  0.20, "Pouce->index. Lent. 5x"),
                NiveauDifficulte(2,  8, 0, "lente",  0.30, "Pouce -> chaque doigt. 8x"),
                NiveauDifficulte(3, 10, 0, "lente",  0.45, "Sequence complete. 10x"),
                NiveauDifficulte(4, 15, 0, "normale",0.55, "Sequence fluide. 15x"),
                NiveauDifficulte(5, 20, 0, "rapide", 0.65, "Sequence rapide. 20x"),
            ]
        ),

        # E6 — Sequence digitale 1-2-3-4-5
        Exercice(
            id=6, nom="Sequence digitale",
            description="Pliez les doigts un par un en sequence 1-2-3-4-5.",
            profils_cibles=["deficit_coordination", "recuperation"],
            metrique_principale="asymetrie",
            poids_reward={"amplitude":0.2,"force":0.1,"vitesse":0.2,"tremblement":0.1,"asymetrie":0.4},
            niveaux=[
                NiveauDifficulte(1,  5, 0, "lente",  0.25, "1-2-3-4-5 lentement. 5x"),
                NiveauDifficulte(2,  8, 0, "lente",  0.35, "Sequence 1->5. 8x"),
                NiveauDifficulte(3, 12, 0, "normale",0.50, "Sequence reguliere. 12x"),
                NiveauDifficulte(4, 15, 0, "normale",0.60, "Vitesse + regularite. 15x"),
                NiveauDifficulte(5, 20, 0, "rapide", 0.70, "Sequence expert. 20x"),
            ]
        ),

        # E7 — Pression progressive
        Exercice(
            id=7, nom="Pression progressive",
            description="Serrez progressivement de faible a fort.",
            profils_cibles=["deficit_mobilite", "recuperation"],
            metrique_principale="force",
            poids_reward={"amplitude":0.2,"force":0.5,"vitesse":0.1,"tremblement":0.1,"asymetrie":0.1},
            niveaux=[
                NiveauDifficulte(1,  8, 0, "lente",  0.15, "Serrez doucement. 8x"),
                NiveauDifficulte(2, 10, 0, "lente",  0.25, "Force moderee. 10x"),
                NiveauDifficulte(3, 12, 0, "normale",0.40, "Force progressive. 12x"),
                NiveauDifficulte(4, 15, 0, "normale",0.55, "Force importante. 15x"),
                NiveauDifficulte(5, 20, 0, "rapide", 0.65, "Force maximale. 20x"),
            ]
        ),

        # E8 — Maintien de pression isometrique
        Exercice(
            id=8, nom="Maintien isometrique",
            description="Serrez et maintenez la pression constante.",
            profils_cibles=["deficit_controle"],
            metrique_principale="tremblement",
            poids_reward={"amplitude":0.1,"force":0.3,"vitesse":0.0,"tremblement":0.5,"asymetrie":0.1},
            niveaux=[
                NiveauDifficulte(1,  5, 3, "lente",  0.20, "Serrez, maintenez 3s. 5x"),
                NiveauDifficulte(2,  6, 4, "lente",  0.30, "Maintenez 4s stable. 6x"),
                NiveauDifficulte(3,  8, 5, "lente",  0.45, "Stabilite 5s. 8x"),
                NiveauDifficulte(4, 10, 6, "lente",  0.55, "Constance 6s. 10x"),
                NiveauDifficulte(5, 12, 8, "lente",  0.65, "Maitrise totale. 12x"),
            ]
        ),

        # E9 — Atteindre + maintenir cible
        Exercice(
            id=9, nom="Cible angulaire",
            description="Atteignez un angle precis et maintenez.",
            profils_cibles=["deficit_controle", "recuperation"],
            metrique_principale="tremblement",
            poids_reward={"amplitude":0.3,"force":0.1,"vitesse":0.1,"tremblement":0.4,"asymetrie":0.1},
            niveaux=[
                NiveauDifficulte(1,  5, 3, "lente",  0.25, "Pliez a 30deg. Maintenez. 5x"),
                NiveauDifficulte(2,  6, 3, "lente",  0.35, "Cible 45deg. 6x"),
                NiveauDifficulte(3,  8, 4, "lente",  0.50, "Cible 60deg. 8x"),
                NiveauDifficulte(4, 10, 4, "normale",0.60, "Precision + maintien. 10x"),
                NiveauDifficulte(5, 12, 5, "normale",0.70, "Cible exacte. 12x"),
            ]
        ),

        # E10 — Tache fonctionnelle : Saisir -> Maintenir -> Relacher
        Exercice(
            id=10, nom="Tache fonctionnelle",
            description="Saisissez, maintenez, relacher. Simuler un objet.",
            profils_cibles=["recuperation"],
            metrique_principale="force",
            poids_reward={"amplitude":0.2,"force":0.3,"vitesse":0.1,"tremblement":0.2,"asymetrie":0.2},
            niveaux=[
                NiveauDifficulte(1,  5, 3, "lente",  0.30, "Saisir+maintenir 3s. 5x"),
                NiveauDifficulte(2,  8, 4, "lente",  0.40, "Objet leger. 8x"),
                NiveauDifficulte(3, 10, 5, "normale",0.50, "Objet moyen. 10x"),
                NiveauDifficulte(4, 12, 5, "normale",0.60, "Precision + force. 12x"),
                NiveauDifficulte(5, 15, 6, "rapide", 0.70, "Tache complete. 15x"),
            ]
        ),
    ]

    return {ex.id: ex for ex in exercices}


# Attribution profil -> exercices (IDs)
PROFIL_EXERCICES = {
    "deficit_mobilite":     [1, 2, 3, 7],
    "deficit_controle":     [3, 8, 9, 4],
    "deficit_coordination": [2, 4, 5, 6],
    "recuperation":         [6, 9, 10, 7],
}

BIBLIOTHEQUE = creer_bibliotheque()


def get_exercices_pour_profil(profil):
    """Retourne la liste ordonnee des exercices pour un profil donne."""
    ids = PROFIL_EXERCICES.get(profil, [1, 2, 3, 7])
    return [BIBLIOTHEQUE[i] for i in ids]


if __name__ == "__main__":
    print("=== BIBLIOTHEQUE DES EXERCICES ===")
    for profil, ids in PROFIL_EXERCICES.items():
        print(f"\n{profil}:")
        for eid in ids:
            ex = BIBLIOTHEQUE[eid]
            print(f"  E{eid}: {ex.nom} | metrique: {ex.metrique_principale}")
