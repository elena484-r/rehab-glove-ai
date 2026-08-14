"""
============================================================================
MEDGLOVE-AI — BIBLIOTHÈQUE DES 10 EXERCICES INSTRUMENTÉS & CLINIKING
============================================================================
Définit la matrice des 10 exercices de rééducation neuromusculaire avec :
 - Nom clinique et description
 - Métriques cibles (angle AROM 0-90°, force FSR 0-100%)
 - Consigne OLED formatée pour écran SSD1306 (21 chars max/ligne)
 - Vecteurs de pondération pour le Reward Shaping de l'agent RL
============================================================================
"""

from typing import Dict, List


class Exercice:

    def __init__(
        self,
        id_ex: int,
        nom: str,
        nom_clinique: str,
        consigne_oled: str,
        type_mouvement: str,  # 'flexion', 'pression', 'coordination', 'maintien'
        cible_valeur: float,  # Angle cible (deg) ou force cible (%)
        poids_reward: Dict[str, float],
    ):
        self.id = id_ex
        self.nom = nom
        self.nom_clinique = nom_clinique
        self.consigne_oled = consigne_oled
        self.type_mouvement = type_mouvement
        self.cible_valeur = cible_valeur
        self.poids_reward = poids_reward


# ============================================================================
# MATRICE COMPLÈTE DES 10 EXERCICES INSTRUMENTÉS
# ============================================================================
BIBLIOTHEQUE: Dict[int, Exercice] = {
    1: Exercice(
        id_ex=1,
        nom="Ouverture / Fermeture",
        nom_clinique="Mobilisation Globale Active (FMA-UE)",
        consigne_oled="Fermez puis ouvrez\nla main au maximum",
        type_mouvement="flexion",
        cible_valeur=70.0,  # 70° AROM
        poids_reward={
            "amplitude": 0.5,
            "force": 0.1,
            "vitesse": 0.2,
            "tremblement": 0.1,
            "asymetrie": 0.1,
        },
    ),
    2: Exercice(
        id_ex=2,
        nom="Dissociation Doigt par Doigt",
        nom_clinique="Isolation Moteur Interdigitale",
        consigne_oled="Pliez les doigts\nun par un lentement",
        type_mouvement="coordination",
        cible_valeur=55.0,
        poids_reward={
            "amplitude": 0.2,
            "force": 0.1,
            "vitesse": 0.1,
            "tremblement": 0.2,
            "asymetrie": 0.4,
        },
    ),
    3: Exercice(
        id_ex=3,
        nom="Flexion Progressive",
        nom_clinique="Glissement Tendineux Controle",
        consigne_oled="Fermez la main\ntres lentement",
        type_mouvement="flexion",
        cible_valeur=60.0,
        poids_reward={
            "amplitude": 0.3,
            "force": 0.1,
            "vitesse": 0.4,
            "tremblement": 0.1,
            "asymetrie": 0.1,
        },
    ),
    4: Exercice(
        id_ex=4,
        nom="Pince Pouce-Index",
        nom_clinique="Pince Pollicidigitale Fine (Kapandji)",
        consigne_oled="Serrez l index\ncontre le pouce",
        type_mouvement="pression",
        cible_valeur=40.0,  # 40% force FSR
        poids_reward={
            "amplitude": 0.1,
            "force": 0.5,
            "vitesse": 0.1,
            "tremblement": 0.2,
            "asymetrie": 0.1,
        },
    ),
    5: Exercice(
        id_ex=5,
        nom="Opposition Sequentielle",
        nom_clinique="Sequence d Opposition de Kapandji",
        consigne_oled="Touchez chaque doigt\navec le pouce",
        type_mouvement="coordination",
        cible_valeur=50.0,
        poids_reward={
            "amplitude": 0.2,
            "force": 0.1,
            "vitesse": 0.2,
            "tremblement": 0.1,
            "asymetrie": 0.4,
        },
    ),
    6: Exercice(
        id_ex=6,
        nom="Sequence Digitale Complexe",
        nom_clinique="Reprogrammation Praxique (1->5)",
        consigne_oled="Sequence doigts:\n1 -> 2 -> 3 -> 4 -> 5",
        type_mouvement="coordination",
        cible_valeur=60.0,
        poids_reward={
            "amplitude": 0.1,
            "force": 0.1,
            "vitesse": 0.3,
            "tremblement": 0.1,
            "asymetrie": 0.4,
        },
    ),
    7: Exercice(
        id_ex=7,
        nom="Pression Progressive FSR",
        nom_clinique="Graduation Isometrique de Force",
        consigne_oled="Pressez la balle\nprogressivement",
        type_mouvement="pression",
        cible_valeur=60.0,
        poids_reward={
            "amplitude": 0.1,
            "force": 0.6,
            "vitesse": 0.1,
            "tremblement": 0.1,
            "asymetrie": 0.1,
        },
    ),
    8: Exercice(
        id_ex=8,
        nom="Maintien Isometrique",
        nom_clinique="Tenue de Grip Contre-Resistance",
        consigne_oled="Maintenez la pression\nsans trembler (5s)",
        type_mouvement="maintien",
        cible_valeur=50.0,
        poids_reward={
            "amplitude": 0.1,
            "force": 0.3,
            "vitesse": 0.1,
            "tremblement": 0.4,
            "asymetrie": 0.1,
        },
    ),
    9: Exercice(
        id_ex=9,
        nom="Cible Angulaire & Maintien",
        nom_clinique="Biofeedback Goniometrique Cible",
        consigne_oled="Atteignez l angle\net bloquez 3 sec",
        type_mouvement="maintien",
        cible_valeur=65.0,
        poids_reward={
            "amplitude": 0.3,
            "force": 0.1,
            "vitesse": 0.1,
            "tremblement": 0.4,
            "asymetrie": 0.1,
        },
    ),
    10: Exercice(
        id_ex=10,
        nom="Tache Fonctionnelle TOT",
        nom_clinique="Task-Oriented Training Complete",
        consigne_oled="Saisissez la balle\nMaintenez - Relachez",
        type_mouvement="flexion",
        cible_valeur=75.0,
        poids_reward={
            "amplitude": 0.3,
            "force": 0.3,
            "vitesse": 0.1,
            "tremblement": 0.1,
            "asymetrie": 0.2,
        },
    ),
}

# Mapping automatique des 4 exercices prescrits par profil moteur K-NN
PROFIL_EXERCICES_MAP: Dict[str, List[int]] = {
    "deficit_mobilite": [1, 2, 3, 7],
    "deficit_controle": [3, 8, 9, 4],
    "deficit_coordination": [2, 4, 5, 6],
    "recuperation": [6, 9, 10, 7],
}


def get_exercices_pour_profil(profil_id: str) -> List[Exercice]:
    """Retourne la liste des 4 objets Exercice prescrits pour un profil K-NN."""
    ids = PROFIL_EXERCICES_MAP.get(profil_id, [1, 2, 3, 7])
    return [BIBLIOTHEQUE[i] for i in ids]
