"""
CLAUDE COACH - PREMIER APPEL API
----------------------------------
Test independant du gant : envoie des donnees simulees a Claude
et recoit un message de coach de reeducation en francais.

Installer les dependances :
  pip install anthropic python-dotenv --break-system-packages

Lancer :
  python3 claude_coach.py
"""

import anthropic
import os
from dotenv import load_dotenv

# Charge la cle API depuis le fichier .env
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("ERREUR : cle API introuvable. Verifie ton fichier .env")
    exit(1)

client = anthropic.Anthropic(api_key=api_key)

# -------------------------------------------------------
# PROMPT SYSTEME : definit le role de Claude
# -------------------------------------------------------
PROMPT_SYSTEME = """
Tu es un coach de rééducation de la main spécialisé dans la récupération post-AVC.
Tu accompagnes un patient qui utilise un gant connecté pour faire ses exercices.

Tes règles absolues :
- Réponds TOUJOURS en français
- Maximum 2-3 phrases courtes (l'écran est petit)
- Sois encourageant, bienveillant, précis
- Adapte ton message aux données reçues (score, progression, programme)
- Ne donne jamais de conseil médical dangereux
- Varie tes formulations pour ne pas te répéter
"""

# -------------------------------------------------------
# FONCTION PRINCIPALE : genere un message coach
# -------------------------------------------------------
def generer_message_coach(donnees_seance: dict) -> str:
    """
    donnees_seance : dictionnaire avec les infos de la seance en cours
    Retourne un message court de coaching en francais
    """

    # Construction du message utilisateur avec les donnees du gant
    message_utilisateur = f"""
Voici les données de la séance en cours :
- Programme : {donnees_seance.get('programme', 'A')}
- Score ROM (flexion) : {donnees_seance.get('score_rom', 0)}/5
- Score force (pression) : {donnees_seance.get('score_force', 0)}/5
- Score vitesse : {donnees_seance.get('score_vitesse', 0)}/5
- Répétitions réalisées : {donnees_seance.get('repetitions', 0)}/{donnees_seance.get('repetitions_cible', 30)}
- Progression vs séance précédente : {donnees_seance.get('progression', 'stable')}
- Contexte : {donnees_seance.get('contexte', 'exercice en cours')}

Génère un message de coaching court et motivant adapté à cette situation.
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,  # court pour l'ecran OLED
        system=PROMPT_SYSTEME,
        messages=[
            {"role": "user", "content": message_utilisateur}
        ]
    )

    return response.content[0].text


# -------------------------------------------------------
# TEST : simule differentes situations de seance
# -------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("TEST CLAUDE COACH - GANT DE REEDUCATION")
    print("=" * 55)

    scenarios = [
        {
            "nom": "Debut de seance, patient programme A",
            "donnees": {
                "programme": "A",
                "score_rom": 1,
                "score_force": 1,
                "score_vitesse": 0,
                "repetitions": 0,
                "repetitions_cible": 30,
                "progression": "premier jour",
                "contexte": "debut de seance"
            }
        },
        {
            "nom": "Bonne progression, 3 succes consecutifs",
            "donnees": {
                "programme": "B",
                "score_rom": 3,
                "score_force": 2,
                "score_vitesse": 2,
                "repetitions": 25,
                "repetitions_cible": 40,
                "progression": "en hausse",
                "contexte": "3 seances reussies consecutives, difficulte augmentee"
            }
        },
        {
            "nom": "Patient en difficulte, mouvement trop lent",
            "donnees": {
                "programme": "C",
                "score_rom": 2,
                "score_force": 1,
                "score_vitesse": 1,
                "repetitions": 10,
                "repetitions_cible": 30,
                "progression": "en baisse",
                "contexte": "mouvement trop lent detecte par le capteur"
            }
        }
    ]

    for scenario in scenarios:
        print(f"\nScenario : {scenario['nom']}")
        print("-" * 40)
        message = generer_message_coach(scenario["donnees"])
        print(f"Claude : {message}")
        print()

    print("=" * 55)
    print("Si tu vois des messages ci-dessus -> API OK, tache validee !")
    print("=" * 55)
