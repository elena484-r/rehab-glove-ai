"""
============================================================
COUCHE 1 — DIAGNOSTIC MOTEUR K-NN
============================================================
Pipeline :
  4 tests d'évaluation -> 5 métriques -> profil dominant

Métriques normalisées [0-100] :
  - Amplitude  : ROM actif (flexion maximale)
  - Force      : pression relative FSR
  - Vitesse    : vitesse angulaire dynamique
  - Tremblement: micro-oscillations (variance temporelle)
  - Asymétrie  : désynchronisation interdigitale

Profils :
  - deficit_mobilite
  - deficit_controle
  - deficit_coordination
  - recuperation
============================================================
"""

import numpy as np
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

PROFILS = ["deficit_mobilite", "deficit_controle", "deficit_coordination", "recuperation"]
PROFILS_FR = {
    "deficit_mobilite":     "Deficit de Mobilite",
    "deficit_controle":     "Deficit de Controle",
    "deficit_coordination": "Deficit de Coordination",
    "recuperation":         "Recuperation Satisfaisante",
}
FEATURES = ["amplitude", "vitesse", "force", "tremblement", "asymetrie"]

FLEX_MIN = [699, 0, 2260, 15, 1426]
FLEX_MAX = [4094, 4094, 4094, 4094, 4094]
PRES_MIN = [0, 0, 0]
PRES_MAX = [3500, 3500, 3500]


class ExtracteurMetriques:
    """Extrait les 5 metriques depuis les 4 tests d'evaluation."""

    def extraire(self, test1_angles, test2_angles, test3_angles,
                 test3_pression, test4_angles):
        t1 = np.array(test1_angles) if test1_angles else np.zeros((1, 5))
        t2 = np.array(test2_angles) if test2_angles else np.zeros((1, 5))
        t3 = np.array(test3_angles) if test3_angles else np.zeros((1, 5))
        t4 = np.array(test4_angles) if test4_angles else np.zeros((1, 5))
        p3 = np.array(test3_pression) if test3_pression else np.zeros((1, 3))

        # 1. Amplitude (Test 1 - flexion max)
        amplitude_max = np.max(t1, axis=0)
        amplitude = float(np.mean(amplitude_max) / 90.0 * 100.0)

        # 2. Vitesse (Tests 1 et 3)
        vmax = 0.0
        for arr in [t1, t3]:
            if len(arr) > 1:
                delta = np.abs(np.diff(arr, axis=0))
                vmax = max(vmax, float(np.max(np.mean(delta, axis=1))))
        vitesse = min(100.0, vmax / 45.0 * 100.0)

        # 3. Force (Test 3 - pression FSR)
        force = float(np.mean(p3) / 100.0 * 100.0) if p3.size > 0 else 0.0

        # 4. Tremblement (Test 2 - maintien statique = variance)
        if len(t2) > 1:
            tremblement = min(100.0, float(np.mean(np.var(t2, axis=0))) / 50.0 * 100.0)
        else:
            tremblement = 0.0

        # 5. Asymetrie (Tests 3 et 4 - ecart entre doigts)
        asym_vals = []
        for arr in [t3, t4]:
            if len(arr) > 0:
                asym_vals.append(float(np.std(np.max(arr, axis=0))))
        asymetrie = min(100.0, (np.mean(asym_vals) if asym_vals else 0.0) / 45.0 * 100.0)

        return {
            "amplitude":   round(amplitude, 2),
            "vitesse":     round(vitesse, 2),
            "force":       round(force, 2),
            "tremblement": round(tremblement, 2),
            "asymetrie":   round(asymetrie, 2),
        }


def generer_dataset(n_par_classe=80):
    """Genere donnees simulees avec distributions realistes par profil."""
    rng = np.random.default_rng(42)
    distributions = {
        "deficit_mobilite":
            [(18,8),(22,7),(14,7),(12,5),(20,8)],
        "deficit_controle":
            [(55,10),(42,15),(50,12),(78,10),(30,10)],
        "deficit_coordination":
            [(45,12),(38,10),(40,10),(18,8),(80,8)],
        "recuperation":
            [(83,7),(76,8),(79,8),(10,5),(14,6)],
    }
    X_list, y_list = [], []
    for profil, dists in distributions.items():
        cols = [np.clip(rng.normal(m, s, n_par_classe), 0, 100) for m, s in dists]
        X_list.append(np.column_stack(cols))
        y_list.extend([profil] * n_par_classe)
    return np.vstack(X_list), np.array(y_list)


class ClassifieurKNN:
    """Classifieur k-NN pour le diagnostic du profil moteur."""

    def __init__(self, k=5):
        self.k = k
        self.knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        self.scaler = StandardScaler()
        self.entraine = False

    def entrainer(self, n_par_classe=80):
        X, y = generer_dataset(n_par_classe)
        X_s = self.scaler.fit_transform(X)
        self.knn.fit(X_s, y)
        self.entraine = True
        scores = cross_val_score(self.knn, X_s, y, cv=5)
        print(f"[KNN] Precision CV 5-fold : {scores.mean()*100:.1f}% +/- {scores.std()*100:.1f}%")
        return float(scores.mean())

    def predire(self, metriques):
        if not self.entraine:
            raise RuntimeError("Classifieur non entraine.")
        x = np.array([metriques[f] for f in FEATURES]).reshape(1, -1)
        x_s = self.scaler.transform(x)

        distances, indices = self.knn.kneighbors(x_s, n_neighbors=min(20, len(self.knn._fit_X)))
        distances, indices = distances[0], indices[0]

        profile_scores = {p: 0.0 for p in PROFILS}
        poids_total = 0.0
        for dist, idx in zip(distances, indices):
            poids = 1.0 / (dist + 1e-6)
            classe = self.knn.classes_[self.knn._y[idx]]
            profile_scores[classe] += poids
            poids_total += poids

        for p in PROFILS:
            profile_scores[p] = round(profile_scores[p] / poids_total * 100, 1)

        dominant = max(profile_scores, key=profile_scores.get)
        return {
            "metrics": metriques,
            "dominant_profile": dominant,
            "dominant_profile_fr": PROFILS_FR[dominant],
            "confidence": profile_scores[dominant],
            "profile_scores": {
                "Mobilite":     profile_scores["deficit_mobilite"],
                "Controle":     profile_scores["deficit_controle"],
                "Coordination": profile_scores["deficit_coordination"],
                "Recuperation": profile_scores["recuperation"],
            }
        }

    def sauvegarder(self, chemin="modele_knn.json"):
        modele = {
            "k": self.k,
            "features": FEATURES,
            "profils": PROFILS,
            "scaler": {
                "mean": self.scaler.mean_.tolist(),
                "std":  self.scaler.scale_.tolist(),
            },
            "training_data": {
                "X": self.knn._fit_X.tolist(),
                "y": self.knn._y.tolist(),
                "classes_map": {str(i): c for i, c in enumerate(self.knn.classes_)},
            }
        }
        with open(chemin, "w") as f:
            json.dump(modele, f, indent=2)
        print(f"[KNN] Modele sauvegarde -> {chemin}")


if __name__ == "__main__":
    clf = ClassifieurKNN(k=5)
    clf.entrainer(80)
    clf.sauvegarder("modele_knn.json")
    cas = [
        {"amplitude":18,"vitesse":20,"force":12,"tremblement":10,"asymetrie":18},
        {"amplitude":58,"vitesse":44,"force":52,"tremblement":80,"asymetrie":28},
        {"amplitude":47,"vitesse":38,"force":42,"tremblement":16,"asymetrie":82},
        {"amplitude":84,"vitesse":77,"force":80,"tremblement": 9,"asymetrie":13},
    ]
    attendus = PROFILS
    for m, a in zip(cas, attendus):
        res = clf.predire(m)
        ok = "OK" if res["dominant_profile"] == a else "ECHEC"
        print(f"[{ok}] {res['dominant_profile_fr']} ({res['confidence']:.1f}%)")
