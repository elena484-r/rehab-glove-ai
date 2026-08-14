"""
simuler_sessions.py
Genere 3 sessions fictives dans historique_sessions.jsonl
pour tester la prediction de progression (couche 4).
Usage : python3 simuler_sessions.py
"""
import json
from datetime import datetime, timedelta

sessions = [
    {"timestamp": (datetime.now()-timedelta(days=2)).isoformat(),
     "profil": "deficit_mobilite", "scores": {"rom": 1.2}, "difficulte": 1},
    {"timestamp": (datetime.now()-timedelta(days=1)).isoformat(),
     "profil": "deficit_mobilite", "scores": {"rom": 1.8}, "difficulte": 2},
    {"timestamp": datetime.now().isoformat(),
     "profil": "deficit_mobilite", "scores": {"rom": 2.5}, "difficulte": 2},
]
with open("historique_sessions.jsonl", "w") as f:
    for s in sessions:
        f.write(json.dumps(s) + "\n")
print("3 sessions simulees -> historique_sessions.jsonl")

