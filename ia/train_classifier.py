"""
============================================================
GANT REEDUCATION — CLASSIFIEUR PROFIL MOTEUR (Couche 2 IA)
============================================================
Entrée  : 5 métriques objectives calculées depuis les capteurs
            amplitude · vitesse · force · tremblement · asymétrie
Sortie  : profil moteur du patient parmi 4 classes :
            - deficit_mobilite    : amplitude et force réduites
            - deficit_controle    : tremblement dominant
            - deficit_coordination: asymétrie inter-doigts marquée
            - recuperation        : toutes métriques proches du normal

Pourquoi k-NN et pas deep learning :
  Les 5 métriques sont des nombres propres et interprétables.
  k-NN est explicable en entretien ("le profil le plus proche
  de ce patient dans notre base d'exemples"), entraînable sur
  peu de données, et ne nécessite aucune librairie lourde.

Données : simulées avec distributions réalistes par profil.
  Le bruit gaussien reproduit la variabilité inter-patient.
  À valider sur données réelles (cohorte clinique).
============================================================
"""

import numpy as np
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# DÉFINITION DES 5 MÉTRIQUES (features)
# ============================================================
# Toutes normalisées entre 0 et 1 pour comparaison entre patients
#
# amplitude  : amplitude moyenne des 5 doigts (0=aucun mvt, 1=90°)
# vitesse    : vitesse max d'exécution (0=immobile, 1=très rapide)
# force      : pression moyenne capteurs (0=aucune, 1=maximale)
# tremblement: intensité des oscillations (0=stable, 1=très instable)
# asymetrie  : écart-type des angles inter-doigts (0=symétrique, 1=très asymétrique)

FEATURES = ["amplitude", "vitesse", "force", "tremblement", "asymetrie"]
CLASSES  = ["deficit_mobilite", "deficit_controle", "deficit_coordination", "recuperation"]


# ============================================================
# GÉNÉRATION DES DONNÉES SIMULÉES PAR PROFIL
# ============================================================
def simuler_profil(profil: str, n: int = 60) -> np.ndarray:
    """
    Génère n exemples simulés pour un profil moteur donné.
    Les distributions sont basées sur la littérature clinique
    en rééducation post-AVC (Langhorne 2011).
    
    Retourne un array (n, 5) de features normalisées [0-1].
    """
    np.random.seed(None)
    
    if profil == "deficit_mobilite":
        # Amplitude et force très réduites
        # Vitesse lente mais stable (peu de tremblement)
        # Mouvement symétrique mais limité
        return np.column_stack([
            np.clip(np.random.normal(0.20, 0.08, n), 0, 1),  # amplitude faible
            np.clip(np.random.normal(0.25, 0.07, n), 0, 1),  # vitesse lente
            np.clip(np.random.normal(0.15, 0.07, n), 0, 1),  # force faible
            np.clip(np.random.normal(0.15, 0.06, n), 0, 1),  # peu de tremblement
            np.clip(np.random.normal(0.20, 0.08, n), 0, 1),  # asymétrie modérée
        ])
    
    elif profil == "deficit_controle":
        # Amplitude correcte mais tremblement dominant
        # Force variable, vitesse irrégulière
        return np.column_stack([
            np.clip(np.random.normal(0.55, 0.10, n), 0, 1),  # amplitude correcte
            np.clip(np.random.normal(0.45, 0.15, n), 0, 1),  # vitesse irrégulière
            np.clip(np.random.normal(0.50, 0.12, n), 0, 1),  # force correcte
            np.clip(np.random.normal(0.75, 0.10, n), 0, 1),  # TREMBLEMENT fort
            np.clip(np.random.normal(0.30, 0.10, n), 0, 1),  # asymétrie modérée
        ])
    
    elif profil == "deficit_coordination":
        # Certains doigts fonctionnent, d'autres pas → asymétrie forte
        # Amplitude globale moyenne, mais très variable entre doigts
        return np.column_stack([
            np.clip(np.random.normal(0.45, 0.12, n), 0, 1),  # amplitude moyenne
            np.clip(np.random.normal(0.40, 0.10, n), 0, 1),  # vitesse moyenne
            np.clip(np.random.normal(0.40, 0.10, n), 0, 1),  # force moyenne
            np.clip(np.random.normal(0.20, 0.08, n), 0, 1),  # peu de tremblement
            np.clip(np.random.normal(0.78, 0.08, n), 0, 1),  # ASYMÉTRIE forte
        ])
    
    elif profil == "recuperation":
        # Toutes les métriques proches de la normale
        # Légère variabilité résiduelle, normale en fin de rééducation
        return np.column_stack([
            np.clip(np.random.normal(0.82, 0.07, n), 0, 1),  # amplitude élevée
            np.clip(np.random.normal(0.75, 0.08, n), 0, 1),  # vitesse bonne
            np.clip(np.random.normal(0.78, 0.08, n), 0, 1),  # force bonne
            np.clip(np.random.normal(0.12, 0.05, n), 0, 1),  # tremblement résiduel
            np.clip(np.random.normal(0.15, 0.06, n), 0, 1),  # bonne symétrie
        ])
    
    raise ValueError(f"Profil inconnu : {profil}")


def generer_dataset(n_par_classe: int = 60):
    """Génère le dataset complet et retourne X, y."""
    X_list, y_list = [], []
    for classe in CLASSES:
        X_list.append(simuler_profil(classe, n_par_classe))
        y_list.extend([classe] * n_par_classe)
    return np.vstack(X_list), np.array(y_list)


# ============================================================
# FONCTION DE CALCUL DES MÉTRIQUES DEPUIS LES DONNÉES GANT
# ============================================================
def calculer_metriques(historique_angles: list, historique_pression: list) -> dict:
    """
    Calcule les 5 métriques normalisées depuis un historique de mesures.
    
    historique_angles   : liste de listes [[a0,a1,a2,a3,a4], ...] en degrés
    historique_pression : liste de listes [[p0,p1,p2], ...] en %
    
    Retourne un dict avec les 5 features normalisées [0-1].
    """
    if not historique_angles or len(historique_angles) < 2:
        return {f: 0.0 for f in FEATURES}
    
    angles_array = np.array(historique_angles)      # shape (T, 5)
    pression_array = np.array(historique_pression)  # shape (T, 3)
    
    # 1. Amplitude : moyenne des amplitudes max par doigt, normalisée sur 90°
    amplitude_max = np.max(angles_array, axis=0)
    amplitude = float(np.mean(amplitude_max) / 90.0)
    
    # 2. Vitesse : variation angulaire maximale entre deux instants consécutifs
    delta_angles = np.abs(np.diff(angles_array, axis=0))  # shape (T-1, 5)
    vitesse_max = float(np.max(np.mean(delta_angles, axis=1)))
    vitesse = min(1.0, vitesse_max / 45.0)  # 45°/step = vitesse max de référence
    
    # 3. Force : pression moyenne normalisée
    force = float(np.mean(pression_array) / 100.0)
    
    # 4. Tremblement : variance temporelle des angles (oscillations rapides)
    variance_temporelle = np.var(angles_array, axis=0)
    tremblement = min(1.0, float(np.mean(variance_temporelle)) / 400.0)
    
    # 5. Asymétrie : écart-type des amplitudes entre doigts (inter-doigts)
    ecart_type_doigts = float(np.std(amplitude_max))
    asymetrie = min(1.0, ecart_type_doigts / 45.0)
    
    return {
        "amplitude":   round(amplitude, 4),
        "vitesse":     round(vitesse, 4),
        "force":       round(force, 4),
        "tremblement": round(tremblement, 4),
        "asymetrie":   round(asymetrie, 4),
    }


# ============================================================
# ENTRAÎNEMENT ET ÉVALUATION DU CLASSIFIEUR
# ============================================================
def entrainer_classifieur(n_par_classe: int = 60, k: int = 5):
    """
    Génère les données, entraîne le k-NN, évalue et exporte le modèle.
    """
    print("=" * 60)
    print("ENTRAÎNEMENT DU CLASSIFIEUR PROFIL MOTEUR")
    print("=" * 60)
    
    # Génération des données
    print(f"\n▶ Génération dataset : {n_par_classe} exemples × {len(CLASSES)} classes")
    X, y = generer_dataset(n_par_classe)
    print(f"  Dataset : {X.shape[0]} exemples, {X.shape[1]} features")
    
    # Normalisation (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entraînement k-NN
    print(f"\n▶ Entraînement k-NN (k={k})...")
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_train, y_train)
    
    # Évaluation
    y_pred = knn.predict(X_test)
    precision_test = knn.score(X_test, y_test)
    
    # Validation croisée (plus fiable que test unique)
    cv_scores = cross_val_score(knn, X_scaled, y, cv=5)
    
    print(f"\n▶ Résultats :")
    print(f"  Précision test       : {precision_test*100:.1f}%")
    print(f"  Précision CV (5-fold): {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
    
    print(f"\n▶ Rapport de classification :")
    print(classification_report(y_test, y_pred, target_names=CLASSES))
    
    print("▶ Matrice de confusion :")
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    print(f"  {'':25s}", end="")
    for c in CLASSES:
        print(f"{c[:10]:>12}", end="")
    print()
    for i, classe in enumerate(CLASSES):
        print(f"  {classe[:25]:25s}", end="")
        for val in cm[i]:
            print(f"{val:>12}", end="")
        print()
    
    # Export du modèle en JSON (pour le RPi, sans pickle)
    exporter_modele(knn, scaler, cv_scores.mean())
    
    return knn, scaler


def exporter_modele(knn, scaler, precision_cv: float):
    """
    Exporte le modèle k-NN en JSON pur — pas de pickle.
    Le RPi peut charger ce fichier sans scikit-learn installé.
    Permet aussi de versionner le modèle sur GitHub proprement.
    """
    modele = {
        "type": "knn",
        "k": knn.n_neighbors,
        "classes": CLASSES,
        "features": FEATURES,
        "precision_cv": round(precision_cv, 4),
        "note": "Données simulées — à valider sur cohorte réelle",
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "std":  scaler.scale_.tolist()
        },
        "training_data": {
            "X": knn._fit_X.tolist(),
            "y": knn._y.tolist(),
            "classes_map": {str(i): c for i, c in enumerate(knn.classes_)}
        }
    }
    
    with open("modele_knn.json", "w") as f:
        json.dump(modele, f, indent=2)
    
    print(f"\n▶ Modèle exporté → modele_knn.json")
    print(f"  Taille : {len(json.dumps(modele)) // 1024} KB")
    print(f"  Chargeable sur RPi sans scikit-learn")


# ============================================================
# INFÉRENCE TEMPS RÉEL (utilisé par le pipeline RPi)
# ============================================================
class ClassifieurProfilMoteur:
    """
    Classe légère pour l'inférence temps réel sur le RPi.
    Charge le modèle JSON exporté — ne nécessite que numpy.
    """
    
    def __init__(self, chemin_modele: str = "modele_knn.json"):
        with open(chemin_modele, "r") as f:
            self.modele = json.load(f)
        
        self.k = self.modele["k"]
        self.classes_map = self.modele["training_data"]["classes_map"]
        self.classes = list(self.modele["training_data"]["classes_map"].values())
        self.X_train = np.array(self.modele["training_data"]["X"])
        self.y_train = np.array(self.modele["training_data"]["y"])
        self.scaler_mean = np.array(self.modele["scaler"]["mean"])
        self.scaler_std  = np.array(self.modele["scaler"]["std"])
        self.features    = self.modele["features"]
    
    def normaliser(self, x: np.ndarray) -> np.ndarray:
        return (x - self.scaler_mean) / self.scaler_std
    
    def predire(self, metriques: dict) -> dict:
        """
        Prédit le profil moteur depuis un dict de métriques.
        
        metriques : {"amplitude":0.4, "vitesse":0.3, "force":0.2,
                     "tremblement":0.7, "asymetrie":0.2}
        
        Retourne : {"profil": "deficit_controle", "confiance": 0.8,
                    "distances": {...}, "metriques_recues": {...}}
        """
        x = np.array([metriques[f] for f in self.features]).reshape(1, -1)
        x_scaled = self.normaliser(x)
        
        # Calcul des distances euclidiennes à tous les points d'entraînement
        distances = np.sqrt(np.sum((self.X_train - x_scaled) ** 2, axis=1))
        
        # k voisins les plus proches
        idx_voisins = np.argsort(distances)[:self.k]
        classes_voisins = [self.classes_map[str(self.y_train[i])] for i in idx_voisins]
        
        # Vote majoritaire
        votes = {}
        for c in classes_voisins:
            votes[c] = votes.get(c, 0) + 1
        
        profil_predit = max(votes, key=votes.get)
        confiance = votes[profil_predit] / self.k
        
        return {
            "profil":           profil_predit,
            "confiance":        round(confiance, 2),
            "votes":            votes,
            "metriques_recues": metriques
        }


# ============================================================
# TEST DE L'INFÉRENCE TEMPS RÉEL
# ============================================================
def tester_inference():
    """Teste le classifieur chargé depuis JSON sur des cas réels."""
    print("\n" + "=" * 60)
    print("TEST INFÉRENCE TEMPS RÉEL (chargement depuis JSON)")
    print("=" * 60)
    
    clf = ClassifieurProfilMoteur("modele_knn.json")
    
    cas_test = [
        {
            "nom": "Patient avec mobilité très réduite",
            "metriques": {"amplitude":0.18, "vitesse":0.22, "force":0.12,
                          "tremblement":0.10, "asymetrie":0.18},
            "attendu": "deficit_mobilite"
        },
        {
            "nom": "Patient avec tremblements importants",
            "metriques": {"amplitude":0.58, "vitesse":0.42, "force":0.51,
                          "tremblement":0.80, "asymetrie":0.25},
            "attendu": "deficit_controle"
        },
        {
            "nom": "Patient avec forte asymétrie inter-doigts",
            "metriques": {"amplitude":0.47, "vitesse":0.38, "force":0.41,
                          "tremblement":0.18, "asymetrie":0.82},
            "attendu": "deficit_coordination"
        },
        {
            "nom": "Patient en bonne récupération",
            "metriques": {"amplitude":0.85, "vitesse":0.78, "force":0.80,
                          "tremblement":0.10, "asymetrie":0.12},
            "attendu": "recuperation"
        },
    ]
    
    nb_correct = 0
    for cas in cas_test:
        resultat = clf.predire(cas["metriques"])
        correct = resultat["profil"] == cas["attendu"]
        if correct:
            nb_correct += 1
        status = "✓" if correct else "✗"
        print(f"\n{status} {cas['nom']}")
        print(f"  Profil prédit : {resultat['profil']} (confiance {resultat['confiance']*100:.0f}%)")
        print(f"  Attendu       : {cas['attendu']}")
        print(f"  Votes k-NN    : {resultat['votes']}")
    
    print(f"\nPrécision sur cas test manuels : {nb_correct}/{len(cas_test)}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # 1. Entraînement
    knn, scaler = entrainer_classifieur(n_par_classe=60, k=5)
    
    # 2. Test inférence depuis JSON
    tester_inference()
    
    print("\n" + "=" * 60)
    print("FICHIERS GÉNÉRÉS")
    print("=" * 60)
    print("  modele_knn.json     → modèle exporté pour le RPi")
    print("\nProchaine étape : intégrer ClassifieurProfilMoteur")
    print("dans le pipeline RPi (pipeline_rpi.py)")
    print("=" * 60)
