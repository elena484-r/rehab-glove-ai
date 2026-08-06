"""
============================================================
GANT REEDUCATION AVC - SCORING MEDICAL + SIMULATION DONNEES
============================================================
Ce fichier fait deux choses :
  1. Calcule les 4 scores d'évaluation (OBJ 0) depuis les données du gant
  2. Génère un dataset simulé réaliste basé sur le bruit réel mesuré

Usage :
  python3 scoring_et_simulation.py

Dépendances :
  pip install numpy --break-system-packages
"""

import numpy as np
import json
import os
from datetime import datetime, timedelta
import random

# ============================================================
# PARTIE 1 — LES 4 SCORES D'EVALUATION (OBJ 0)
# ============================================================

# Valeurs de calibration réelles mesurées sur le gant
FLEX_MIN = [699,  0,    2260, 15,   1426]  # main ouverte
FLEX_MAX = [4094, 4094, 4094, 4094, 4094]  # poing fermé
NOMS_DOIGTS = ["pouce", "index", "majeur", "annulaire", "auriculaire"]

PRESSION_MIN = [0,    0,    0   ]
PRESSION_MAX = [3500, 3500, 3500]
NOMS_PRESSION = ["pouce", "index", "majeur"]


def brut_vers_angle(valeur_brute: float, index_doigt: int) -> float:
    """Convertit une valeur brute ADC en angle (0°-90°) pour un doigt donné."""
    vmin = FLEX_MIN[index_doigt]
    vmax = FLEX_MAX[index_doigt]
    if vmax == vmin:
        return 0.0
    angle = (valeur_brute - vmin) / (vmax - vmin) * 90.0
    return max(0.0, min(90.0, angle))


def brut_vers_pression(valeur_brute: float, index_capteur: int) -> float:
    """Convertit une valeur brute ADC en pourcentage de pression (0-100%)."""
    vmin = PRESSION_MIN[index_capteur]
    vmax = PRESSION_MAX[index_capteur]
    if vmax == vmin:
        return 0.0
    pct = (valeur_brute - vmin) / (vmax - vmin) * 100.0
    return max(0.0, min(100.0, pct))


def calculer_score_rom(flex_brut: list) -> float:
    """
    Score ROM (Range of Motion) — Couverture de l'amplitude articulaire.
    Mesure si le patient peut plier chaque doigt sur toute son amplitude.
    
    Retourne un score 0-5 basé sur l'amplitude moyenne des 5 doigts.
    Référence : amplitude normale = 90° pour une flexion complète.
    """
    angles = [brut_vers_angle(flex_brut[i], i) for i in range(5)]
    amplitude_moyenne = sum(angles) / 5.0
    
    # Score linéaire : 0° = 0/5, 90° = 5/5
    score = (amplitude_moyenne / 90.0) * 5.0
    return round(min(5.0, score), 2)


def calculer_score_force(pression_brut: list) -> float:
    """
    Score Force — Capacité de préhension.
    Mesure la force exercée sur les 3 capteurs de pression.
    
    Retourne un score 0-5.
    Référence : force normale adulte = 20-40 kg selon âge/sexe.
    """
    pressions_pct = [brut_vers_pression(pression_brut[i], i) for i in range(3)]
    force_moyenne = sum(pressions_pct) / 3.0
    
    score = (force_moyenne / 100.0) * 5.0
    return round(min(5.0, score), 2)


def calculer_score_vitesse(historique_angles: list, dt_secondes: float = 0.5) -> float:
    """
    Score Vitesse — Rapidité d'exécution du mouvement.
    Calcule la dérivée angulaire (ΔAngle/Δt) sur une séquence de mesures.
    
    historique_angles : liste de listes d'angles [t0, t1, t2, ...] où
                        chaque élément est une liste de 5 angles (un par doigt)
    dt_secondes : intervalle de temps entre chaque mesure (0.5s par défaut)
    
    Retourne un score 0-5.
    Référence : vitesse normale = ouvrir/fermer la main 10 fois en < 10 secondes.
    """
    if len(historique_angles) < 2:
        return 0.0
    
    vitesses = []
    for i in range(1, len(historique_angles)):
        angles_t0 = historique_angles[i-1]
        angles_t1 = historique_angles[i]
        # Variation angulaire moyenne sur tous les doigts
        delta_angle = sum(abs(angles_t1[j] - angles_t0[j]) for j in range(5)) / 5.0
        vitesse = delta_angle / dt_secondes  # degrés par seconde
        vitesses.append(vitesse)
    
    vitesse_max = max(vitesses) if vitesses else 0.0
    
    # Vitesse de référence : 90° en 1 seconde = 90°/s = score 5/5
    score = (vitesse_max / 90.0) * 5.0
    return round(min(5.0, score), 2)


def calculer_score_coordination(angles_actuels: list, angles_cible: list) -> float:
    """
    Score Coordination — Précision à reproduire une position cible.
    Mesure l'écart entre la position demandée et la position réelle.
    
    Retourne un score 0-5 (5 = précision parfaite).
    Référence : écart < 5° = excellent, > 30° = insuffisant.
    """
    if len(angles_actuels) != 5 or len(angles_cible) != 5:
        return 0.0
    
    ecart_moyen = sum(abs(angles_actuels[i] - angles_cible[i]) for i in range(5)) / 5.0
    
    # Score inversé : 0° d'écart = 5/5, 30°+ d'écart = 0/5
    score = max(0.0, 5.0 - (ecart_moyen / 30.0) * 5.0)
    return round(score, 2)


def determiner_programme(score_total: float) -> str:
    """Détermine le programme de rééducation selon le score total /20."""
    if score_total <= 5:
        return "A"  # Très sévère — stimulation passive
    elif score_total <= 10:
        return "B"  # Sévère — mobilisation active
    elif score_total <= 15:
        return "C"  # Modéré — renforcement
    else:
        return "D"  # Léger — performance


def evaluer_patient(flex_brut: list, pression_brut: list,
                    historique_angles: list, angles_cible: list) -> dict:
    """
    Evaluation complète OBJ 0 : calcule les 4 scores et assigne un programme.
    
    Retourne un dictionnaire complet du bilan patient.
    """
    s_rom   = calculer_score_rom(flex_brut)
    s_force = calculer_score_force(pression_brut)
    s_vitesse = calculer_score_vitesse(historique_angles)
    
    # Calcul des angles actuels pour la coordination
    angles_actuels = [brut_vers_angle(flex_brut[i], i) for i in range(5)]
    s_coord = calculer_score_coordination(angles_actuels, angles_cible)
    
    score_total = s_rom + s_force + s_vitesse + s_coord
    programme = determiner_programme(score_total)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "scores": {
            "rom":          s_rom,
            "force":        s_force,
            "vitesse":      s_vitesse,
            "coordination": s_coord,
            "total":        round(score_total, 2)
        },
        "programme_assigne": programme,
        "angles_mesures": [round(brut_vers_angle(flex_brut[i], i), 1) for i in range(5)],
        "pressions_pct":  [round(brut_vers_pression(pression_brut[i], i), 1) for i in range(3)]
    }


# ============================================================
# PARTIE 2 — GENERATION DE DONNEES SIMULEES REALISTES
# ============================================================
# Basé sur le bruit réel mesuré sur les capteurs Velostat :
# - Instabilité au repos : ±200 à ±500 points selon le doigt
# - Non-linéarité : réponse plus rapide dans la première moitié
# - Valeurs min mesurées : [699, 0, 2260, 15, 1426]

BRUIT_REPOS = [300, 150, 200, 100, 250]  # bruit mesuré au repos par doigt


def simuler_flexion(index_doigt: int, angle_cible: float,
                    niveau_bruit: float = 1.0) -> float:
    """
    Simule une valeur ADC brute pour un doigt à un angle donné.
    Inclut le bruit réel caractérisé sur les capteurs Velostat.
    """
    vmin = FLEX_MIN[index_doigt]
    vmax = FLEX_MAX[index_doigt]
    
    # Valeur théorique linéaire
    valeur_theorique = vmin + (angle_cible / 90.0) * (vmax - vmin)
    
    # Bruit gaussien proportionnel au bruit mesuré
    bruit = np.random.normal(0, BRUIT_REPOS[index_doigt] * niveau_bruit)
    
    valeur_simulee = valeur_theorique + bruit
    return float(np.clip(valeur_simulee, 0, 4095))


def simuler_pression(index_capteur: int, force_pct: float) -> float:
    """Simule une valeur ADC brute pour un capteur de pression."""
    vmin = PRESSION_MIN[index_capteur]
    vmax = PRESSION_MAX[index_capteur]
    valeur = vmin + (force_pct / 100.0) * (vmax - vmin)
    bruit = np.random.normal(0, 100)
    return float(np.clip(valeur + bruit, 0, 4095))


def generer_sequence_mouvement(classe: str, nb_points: int = 20) -> list:
    """
    Génère une séquence temporelle d'angles simulant un mouvement classifié.
    
    Classes :
    - 'flexion_complete'  : fermeture progressive des 5 doigts (0° → 90°)
    - 'flexion_partielle' : fermeture à mi-course (0° → 45°)
    - 'tremblement'       : oscillations rapides à faible amplitude
    - 'compensation'      : un ou deux doigts bougent, les autres pas
    """
    sequences = []
    
    for t in range(nb_points):
        progress = t / (nb_points - 1)
        
        if classe == "flexion_complete":
            angles = [90.0 * progress + np.random.normal(0, 3) for _ in range(5)]
            
        elif classe == "flexion_partielle":
            angles = [45.0 * progress + np.random.normal(0, 3) for _ in range(5)]
            
        elif classe == "tremblement":
            # Oscillations rapides autour d'une valeur moyenne
            base = 30.0
            freq = 8.0  # Hz simulé
            angles = [base + 15 * np.sin(2 * np.pi * freq * t / nb_points)
                      + np.random.normal(0, 5) for _ in range(5)]
            
        elif classe == "compensation":
            # Seulement index et majeur bougent (compensation poignet)
            angles = []
            for i in range(5):
                if i in [1, 2]:  # index et majeur bougent normalement
                    angles.append(70.0 * progress + np.random.normal(0, 3))
                else:  # les autres restent quasi-immobiles
                    angles.append(5.0 + np.random.normal(0, 5))
        else:
            angles = [0.0] * 5
        
        # Convertir les angles en valeurs ADC brutes simulées
        flex_brut = [simuler_flexion(i, max(0, min(90, angles[i])))
                     for i in range(5)]
        sequences.append({
            "t": t,
            "flex_brut": [round(v, 1) for v in flex_brut],
            "angles":    [round(max(0, min(90, a)), 1) for a in angles]
        })
    
    return sequences


def generer_dataset_complet(nb_par_classe: int = 40,
                            fichier_sortie: str = "dataset_mouvements.jsonl") -> None:
    """
    Génère le dataset complet d'entraînement pour le classifieur k-NN.
    
    Produit un fichier JSONL avec nb_par_classe exemples pour chacune
    des 4 classes de mouvement.
    
    IMPORTANT : Ces données sont simulées à partir du bruit réel mesuré
    sur les capteurs Velostat. À valider sur données réelles si possible.
    """
    classes = ["flexion_complete", "flexion_partielle", "tremblement", "compensation"]
    total = 0
    
    print(f"Génération du dataset — {nb_par_classe} exemples × {len(classes)} classes")
    print(f"ATTENTION : données simulées basées sur bruit réel mesuré (Velostat)")
    print("-" * 60)
    
    with open(fichier_sortie, "w") as f:
        for classe in classes:
            for i in range(nb_par_classe):
                sequence = generer_sequence_mouvement(classe, nb_points=20)
                
                # Feature extraction : angles moyens + variance + vitesse max
                tous_angles = [pt["angles"] for pt in sequence]
                angles_array = np.array(tous_angles)
                
                features = {
                    "angles_moyens":   np.mean(angles_array, axis=0).tolist(),
                    "angles_variance": np.var(angles_array, axis=0).tolist(),
                    "angle_final":     tous_angles[-1],
                    "vitesse_max":     float(np.max(np.abs(np.diff(angles_array, axis=0))))
                }
                
                enregistrement = {
                    "classe":   classe,
                    "features": features,
                    "sequence": sequence,
                    "simule":   True,   # toujours indiquer que c'est simulé
                    "source":   "Velostat_bruit_caracterise"
                }
                f.write(json.dumps(enregistrement) + "\n")
                total += 1
            
            print(f"  ✓ {classe} : {nb_par_classe} exemples générés")
    
    print(f"\nDataset complet : {total} exemples → {fichier_sortie}")


def generer_historique_sessions(nb_sessions: int = 20,
                                fichier_sortie: str = "historique_sessions.jsonl") -> None:
    """
    Génère un historique de sessions pour la couche 4 (prédiction).
    Simule une progression réaliste d'un patient post-AVC sur 4 semaines.
    """
    date_debut = datetime.now() - timedelta(days=nb_sessions)
    
    with open(fichier_sortie, "w") as f:
        for i in range(nb_sessions):
            # Progression réaliste : amélioration lente avec bruit
            progress = i / nb_sessions
            
            score_rom   = min(5, 0.5 + 3.5 * progress + np.random.normal(0, 0.3))
            score_force = min(5, 0.3 + 3.0 * progress + np.random.normal(0, 0.4))
            score_vit   = min(5, 0.2 + 2.5 * progress + np.random.normal(0, 0.3))
            score_coord = min(5, 0.4 + 3.2 * progress + np.random.normal(0, 0.35))
            score_total = score_rom + score_force + score_vit + score_coord
            
            session = {
                "session_id": i + 1,
                "date": (date_debut + timedelta(days=i)).isoformat(),
                "programme": determiner_programme(score_total),
                "scores": {
                    "rom":          round(max(0, score_rom), 2),
                    "force":        round(max(0, score_force), 2),
                    "vitesse":      round(max(0, score_vit), 2),
                    "coordination": round(max(0, score_coord), 2),
                    "total":        round(max(0, score_total), 2)
                },
                "repetitions_realisees": int(20 + 15 * progress + np.random.normal(0, 3)),
                "simule": True
            }
            f.write(json.dumps(session) + "\n")
    
    print(f"Historique : {nb_sessions} sessions → {fichier_sortie}")


# ============================================================
# TESTS ET DEMONSTRATION
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GANT REEDUCATION — SCORING + SIMULATION")
    print("=" * 60)
    
    # --- Test 1 : Patient avec mobilité très réduite ---
    print("\n[TEST 1] Patient sévère (score attendu : bas)")
    flex_severe   = [800, 100, 2300, 50, 1500]   # quasi main ouverte
    pression_sev  = [100, 50,  80]                # pression très faible
    historique_sev = [[brut_vers_angle(flex_severe[i], i) for i in range(5)]] * 3
    cible = [45, 45, 45, 45, 45]
    
    bilan = evaluer_patient(flex_severe, pression_sev, historique_sev, cible)
    print(f"  Scores : {bilan['scores']}")
    print(f"  Programme assigné : {bilan['programme_assigne']}")
    
    # --- Test 2 : Patient avec bonne récupération ---
    print("\n[TEST 2] Patient modéré (score attendu : moyen)")
    flex_moyen   = [2500, 2000, 3500, 2000, 2800]
    pression_moy = [1800, 1500, 1600]
    historique_moy = [
        [20, 20, 20, 20, 20],
        [40, 40, 40, 40, 40],
        [60, 60, 60, 60, 60]
    ]
    
    bilan2 = evaluer_patient(flex_moyen, pression_moy, historique_moy, cible)
    print(f"  Scores : {bilan2['scores']}")
    print(f"  Programme assigné : {bilan2['programme_assigne']}")
    
    # --- Test 3 : Patient quasi rétabli ---
    print("\n[TEST 3] Patient léger (score attendu : élevé)")
    flex_leger   = [3800, 3900, 4000, 3900, 3800]
    pression_leg = [3000, 2900, 3100]
    historique_leg = [
        [70, 72, 68, 71, 70],
        [80, 82, 79, 81, 80],
        [88, 89, 87, 90, 88]
    ]
    
    bilan3 = evaluer_patient(flex_leger, pression_leg, historique_leg, cible)
    print(f"  Scores : {bilan3['scores']}")
    print(f"  Programme assigné : {bilan3['programme_assigne']}")
    
    # --- Génération dataset ---
    print("\n" + "=" * 60)
    print("GENERATION DES DONNEES SIMULEES")
    print("=" * 60)
    generer_dataset_complet(nb_par_classe=40, fichier_sortie="dataset_mouvements.jsonl")
    generer_historique_sessions(nb_sessions=20, fichier_sortie="historique_sessions.jsonl")
    
    print("\n✓ Tout est prêt pour l'entraînement du classifieur (couche 2)")
    print("✓ Historique prêt pour la régression de prédiction (couche 4)")
    print("\nProchaine étape : python3 train_classifier.py")
