# Gant de Reeducation Intelligent

Projet de gant connecte pour la reeducation de la main (post-AVC, post-fracture, etc.)
combinant capteurs de flexion/pression, ESP32, Raspberry Pi 5, et un pipeline d'IA
en 4 couches (filtrage signal, classification de mouvement, decision adaptative, prediction).

## Structure du projet

- `esp32/` : code embarque ESP32 (lecture capteurs, filtrage, envoi WiFi)
- `rpi/` : serveur Flask, scoring medical, orchestration
- `ia/` : modules IA (classification, politique adaptative, prediction, prompt Claude)
- `docs/` : schemas, images, fiche technique

## Statut

Projet en cours de developpement (objectif : 9 aout 2026).

