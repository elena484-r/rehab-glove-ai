"""
SERVEUR FLASK - RECEPTION DONNEES DU GANT (v2)
------------------------------------------------
Reçoit le JSON de l'ESP32 :
{
  "flex": [v0,v1,v2,v3,v4],          <- valeurs brutes filtrées EMA
  "pressure": [v0,v1,v2],            <- valeurs brutes filtrées EMA
  "angles": [a0,a1,a2,a3,a4],        <- angles en degrés (0-90)
  "pressure_pct": [p0,p1,p2]         <- pression en % (0-100)
}

Lancer : python3 serveur_gant_v2.py
"""

from flask import Flask, request, jsonify
from datetime import datetime
import json, os

app = Flask(__name__)
LOG_FILE = "sessions/donnees_capteurs.jsonl"

DOIGTS = ["pouce", "index", "majeur", "annulaire", "auriculaire"]
PRESSIONS = ["pouce", "index", "majeur"]

os.makedirs("sessions", exist_ok=True)


def valider(data):
    for champ, taille in [("flex", 5), ("pressure", 3), ("angles", 5), ("pressure_pct", 3)]:
        if champ not in data:
            return False, f"Champ '{champ}' manquant"
        if len(data[champ]) != taille:
            return False, f"'{champ}' : {taille} valeurs attendues, {len(data[champ])} reçues"
    return True, None


@app.route('/data', methods=['POST'])
def recevoir_donnees():
    data = request.get_json()
    if data is None:
        return jsonify({"status": "erreur", "message": "JSON invalide"}), 400

    valide, erreur = valider(data)
    if not valide:
        print(f"[ERREUR] {erreur}")
        return jsonify({"status": "erreur", "message": erreur}), 400

    ts = datetime.now().strftime("%H:%M:%S")

    # Affichage console clair avec angles
    print(f"\n[{ts}] ── ANGLES (degrés) ──────────────────────")
    for i, doigt in enumerate(DOIGTS):
        barre = "█" * (data["angles"][i] // 9)  # barre visuelle sur 10 chars
        print(f"  {doigt:<12} {data['angles'][i]:>3}°  {barre}")

    print(f"[{ts}] ── PRESSION (%) ──────────────────────────")
    for i, doigt in enumerate(PRESSIONS):
        barre = "█" * (data["pressure_pct"][i] // 10)
        print(f"  {doigt:<12} {data['pressure_pct'][i]:>3}%  {barre}")

    # Sauvegarde
    enregistrement = {
        "timestamp": datetime.now().isoformat(),
        **data
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(enregistrement) + "\n")

    return jsonify({"status": "ok"}), 200


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "actif", "log": os.path.abspath(LOG_FILE)})


if __name__ == '__main__':
    print("=" * 55)
    print("  SERVEUR GANT DE REEDUCATION AVC - v2")
    print(f"  Log : {os.path.abspath(LOG_FILE)}")
    print("  En attente sur le port 5000...")
    print("=" * 55)
    app.run(host='0.0.0.0', port=5000, debug=False)
