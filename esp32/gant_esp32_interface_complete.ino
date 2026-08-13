/*
  ============================================================================
  GANT REEDUCATION AVC - INTERFACE UTILISATEUR COMPLETE
  ============================================================================
  Fichier : gant_esp32_lecture_wifi.ino
  
  Parcours utilisateur :
    1. Démarrage (appui long)
    2. Évaluation initiale (4 tests) -> POST /evaluation
    3. Affichage profil moteur (couche 1 K-NN)
    4. Séance d'exercices avec feedback LED/buzzer -> POST /data
    5. Adaptation RL affichée (couche 2, répétitions gérées par le RPi)
    6. Fin de séance + prédiction progression

  Communication reseau :
    POST /evaluation : envoye une seule fois, a la fin des 4 tests.
                        Retourne le profil K-NN + le premier exercice.
    POST /data        : envoye en boucle pendant les exercices.
                        Retourne la decision RL + le message coach.

  Bouton unique :
    Appui long  (>1.5s) : allumer / démarrer
    Appui court (<1.5s) : confirmer / avancer
    Double appui        : valider / sélectionner

  LED :
    Verte allumée  : mouvement correct pendant exercice
    Verte éteinte  : mouvement incorrect / en attente
    Clignotement   : transition / analyse

  Buzzer :
    1 bip court  : succès mouvement
    2 bips       : fin d'exercice
    3 bips longs : fin de séance

  Câblage :
    MUX : SIG→GPIO39 S0→GPIO33 S1→GPIO32 S2→GPIO17 S3→GPIO16
    Pression : Pouce→GPIO34 Index→GPIO35 Majeur→GPIO36
    LED→GPIO4  Buzzer→GPIO14  Bouton→GPIO18
    OLED I2C : SDA=21 SCL=22
  ============================================================================
*/

#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// Declare tot pour eviter le bug classique Arduino : l'IDE genere automatiquement
// les prototypes de fonctions et les place juste apres les #include, AVANT le reste
// du fichier. Si un type custom (enum) utilise comme retour/parametre de fonction
// est defini plus bas, le prototype auto-genere ne le connait pas encore -> erreur
// "'ActionBouton' does not name a type". On definit donc l'enum ici, en tout premier.
enum ActionBouton { AUCUNE, APPUI_COURT, APPUI_LONG, DOUBLE_APPUI };

// ============================================================================
// CONFIGURATION RESEAU
// ============================================================================
const char* WIFI_SSID     = "Elena's Galaxy A54 5G";
const char* WIFI_PASSWORD = "elena234";
const char* SERVER_HOST   = "http://10.254.218.206:5000";

// Deux endpoints distincts sur le pipeline_bridge.py :
//   /evaluation -> une fois, apres les 4 tests initiaux
//   /data       -> en boucle, pendant les exercices
String urlEvaluation() { return String(SERVER_HOST) + "/evaluation"; }
String urlData()       { return String(SERVER_HOST) + "/data"; }

// ============================================================================
// PINS
// ============================================================================
#define MUX_SIG  39
#define MUX_S0   33
#define MUX_S1   32
#define MUX_S2   17
#define MUX_S3   16

#define PIN_PRESSION_POUCE   34
#define PIN_PRESSION_INDEX   35
#define PIN_PRESSION_MAJEUR  36

#define PIN_LED     4
#define PIN_BUZZER  14
#define PIN_BOUTON  18

// ============================================================================
// OLED
// ============================================================================
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// ============================================================================
// CALIBRATION (valeurs réelles mesurées)
// ============================================================================
int flexMin[5] = {699,  0,    2260, 15,   1426};
int flexMax[5] = {4094, 4094, 4094, 4094, 4094};

int pressionMin[3] = {0,    0,    0   };
int pressionMax[3] = {3500, 3500, 3500};

// ============================================================================
// FILTRE EMA — Couche 1 pipeline IA
// ============================================================================
const float ALPHA = 0.2;
float flexFiltre[5]      = {0, 0, 0, 0, 0};
float pressionFiltree[3] = {0, 0, 0};

// ============================================================================
// ETATS DU PARCOURS UTILISATEUR
// ============================================================================
enum EtatGant {
  ETAT_VEILLE,          // Attente appui long
  ETAT_ACCUEIL,         // Écran bienvenue
  ETAT_EVALUATION,      // Tests OBJ 0
  ETAT_PROFIL,          // Affichage profil moteur
  ETAT_EXERCICE,        // Séance en cours
  ETAT_ADAPTATION,      // Affichage ajustement RL
  ETAT_COACH,           // Message coach IA
  ETAT_FIN_SEANCE,      // Bilan séance
  ETAT_PREDICTION       // Prédiction progression
};

EtatGant etatActuel = ETAT_VEILLE;

// ============================================================================
// DONNÉES SESSION
// ============================================================================
String profilMoteur    = "";  // reçu depuis RPi (/evaluation)
String profilMoteurFr  = "";  // libelle FR pret a afficher, reçu depuis RPi (/evaluation)
float  confianceProfil = 0;   // confiance du K-NN (%), reçue depuis RPi
String messageCoach    = "";  // reçu depuis RPi (/data)
String messageRL       = "";  // reçu depuis RPi (/data -> message_rl)
String messagePred     = "";  // couche 3 (prédiction) non branchée côté RPi pour l'instant
int    difficulteNiv   = 1;   // niveau difficulté actuel (mis à jour par le RPi)
int    repetitionsOK   = 0;   // répétitions réussies (comptées localement pour le feedback immédiat)
int    repetitionsCible = 15; // cible : desormais fixee par le RPi (champ "repetitions")
int    exerciceActuel  = 1;   // NUMERO D'AFFICHAGE local uniquement (1,2,3... pour l'ecran) — jamais utilise pour decider la fin de seance
bool   mouvementBon    = false; // feedback LED temps réel
bool   repetitionVientDeFinir = false; // true un seul cycle, au moment ou repetitionsOK s'incremente

String exerciceNomServeur  = ""; // nom lisible de l'exercice, reçu du RPi (exercice_nom)
String consigneOledServeur = ""; // consigne à afficher, reçue du RPi (consigne_oled)
int    exerciceIdServeur   = 0;  // id exercice cote RPi (exercice_id) — NE PILOTE PLUS la fin de seance
bool   finSeanceServeur    = false; // le RPi indique que la séance/exercice doit s'arrêter (fin_seance) — SEUL declencheur de fin
float  rewardDernier       = 0.0;   // dernier reward RL reçu (debug/affichage optionnel)

// --- Echantillons temps-reel captures pendant les 4 tests d'evaluation ---
// Confirme par couche1_knn.py (ExtracteurMetriques) : chaque testX_angles doit
// etre une SERIE de mesures [doigt0..doigt4] dans le temps (pas un instantane),
// car le K-NN calcule max/diff/variance/std le long de l'axe temporel (axis=0).
#define MAX_ECH_TEST 40
float bufAngles1[MAX_ECH_TEST][5]; int nEch1 = 0; // test1 : fermeture lente (-> amplitude, vitesse)
float bufAngles2[MAX_ECH_TEST][5]; int nEch2 = 0; // test2 : ouverture complete (-> tremblement)
float bufAngles3[MAX_ECH_TEST][5]; int nEch3 = 0; // test3 : serrage fort (-> vitesse, asymetrie)
float bufAngles4[MAX_ECH_TEST][5]; int nEch4 = 0; // test4 : ecartement doigts (-> asymetrie)
float bufPression3[MAX_ECH_TEST][3]; int nEchP3 = 0; // test3 : pression (-> force)
unsigned long dernierEchTest = 0;
const unsigned long PERIODE_ECH_TEST = 120; // ms entre deux echantillons pendant un test

// ============================================================================
// GESTION BOUTON
// ============================================================================
unsigned long tempsAppui        = 0;
unsigned long dernierLacher     = 0;
bool          boutonPrecedent   = false;
bool          attendDouble      = false;
unsigned long tempsLacher1      = 0;
const unsigned long SEUIL_LONG  = 1500; // ms pour appui long
const unsigned long SEUIL_DBL   = 400;  // ms entre deux appuis pour double

ActionBouton lireBouton() {
  bool appuye = (digitalRead(PIN_BOUTON) == LOW);
  ActionBouton action = AUCUNE;
  unsigned long maintenant = millis();

  if (appuye && !boutonPrecedent) {
    tempsAppui = maintenant;
  }

  if (!appuye && boutonPrecedent) {
    unsigned long duree = maintenant - tempsAppui;

    if (duree >= SEUIL_LONG) {
      action = APPUI_LONG;
      attendDouble = false;
    } else {
      if (attendDouble && (maintenant - tempsLacher1 < SEUIL_DBL)) {
        action = DOUBLE_APPUI;
        attendDouble = false;
      } else {
        attendDouble = true;
        tempsLacher1 = maintenant;
      }
    }
  }

  if (attendDouble && !appuye && (maintenant - tempsLacher1 > SEUIL_DBL)) {
    action = APPUI_COURT;
    attendDouble = false;
  }

  boutonPrecedent = appuye;
  return action;
}

// ============================================================================
// FEEDBACK LED ET BUZZER
// ============================================================================
void bipCourt() {
  digitalWrite(PIN_BUZZER, HIGH); delay(80);
  digitalWrite(PIN_BUZZER, LOW);
}

void bipDouble() {
  bipCourt(); delay(100); bipCourt();
}

void bipFinSeance() {
  for (int i = 0; i < 3; i++) {
    digitalWrite(PIN_BUZZER, HIGH); delay(200);
    digitalWrite(PIN_BUZZER, LOW);  delay(150);
  }
}

void ledVerte(bool allumee) {
  digitalWrite(PIN_LED, allumee ? HIGH : LOW);
}

void clignoterLED(int fois) {
  for (int i = 0; i < fois; i++) {
    digitalWrite(PIN_LED, HIGH); delay(150);
    digitalWrite(PIN_LED, LOW);  delay(150);
  }
}

// ============================================================================
// AFFICHAGE OLED — fonctions par écran
// ============================================================================
void oledEffacer() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
}

void oledLigne(int y, const char* texte, bool gras = false) {
  display.setTextSize(gras ? 2 : 1);
  display.setCursor(0, y);
  display.println(texte);
  display.setTextSize(1);
}

void oledSeparateur(int y) {
  display.drawFastHLine(0, y, 128, SSD1306_WHITE);
}

void afficherVeille() {
  oledEffacer();
  display.setTextSize(1);
  display.setCursor(20, 10);
  display.println("Gant reeducation");
  oledSeparateur(22);
  display.setCursor(10, 30);
  display.println("Appui long = ON");
  display.setCursor(25, 45);
  display.println("Bienvenue !");
  display.display();
}

void afficherAccueil() {
  oledEffacer();
  display.setTextSize(1);
  display.setCursor(25, 2);
  display.println("Bonjour !");
  oledSeparateur(13);
  display.setCursor(0, 17);
  display.println("Je suis votre");
  display.println("assistant de");
  display.println("reeducation.");
  oledSeparateur(52);
  display.setCursor(0, 55);
  display.println("2x appui = debut");
  display.display();
}

void afficherEvaluation(int etape) {
  // etape 0..3 : les 4 tests reels. etape 4 : ecran d'analyse (envoi /evaluation).
  oledEffacer();
  const char* titres[]  = {"Test 1/4", "Test 2/4", "Test 3/4", "Test 4/4", "Analyse..."};
  const char* lignes1[] = {"Fermez la main", "Ouvrez la main", "Serrez fort", "Ecartez les", "Traitement de"};
  const char* lignes2[] = {"lentement.", "completement.", "5 secondes.", "doigts.", "vos resultats."};
  const char* lignes3[] = {"Repetez : 5x", "Repetez : 5x", "Maintenez...", "Repetez : 5x", ""};

  display.setCursor(0, 0);
  display.println(titres[etape]);
  oledSeparateur(10);
  display.setCursor(0, 14);
  display.println(lignes1[etape]);
  display.println(lignes2[etape]);
  display.setCursor(0, 36);
  display.println(lignes3[etape]);
  oledSeparateur(52);
  display.setCursor(0, 55);
  if (etape < 4) display.println("Appui = suivant");
  else           display.println("Patientez...");
  display.display();
}

void afficherProfil() {
  oledEffacer();
  display.setCursor(0, 0);
  display.println("Analyse terminee");
  oledSeparateur(10);
  display.setCursor(0, 14);
  display.println("Votre profil :");

  // profil_fr est deja le libelle pret a afficher, envoye par le RPi (evite la duplication)
  display.setCursor(0, 26);
  if (profilMoteurFr.length() > 0) display.println(profilMoteurFr.c_str());
  else                             display.println(profilMoteur.c_str());

  display.setCursor(0, 40);
  display.print("Niveau : ");
  display.print(difficulteNiv);
  display.print("  (");
  display.print((int)confianceProfil);
  display.println("%)");
  oledSeparateur(52);
  display.setCursor(0, 55);
  display.println("2x appui = suite");
  display.display();
}

void afficherExercice() {
  oledEffacer();
  display.setCursor(0, 0);
  if (exerciceNomServeur.length() > 0) {
    display.println(exerciceNomServeur);
  } else {
    display.print("Exercice ");
    display.print(exerciceActuel);
    display.print(" / 4");
  }
  oledSeparateur(10);
  display.setCursor(0, 14);
  if (consigneOledServeur.length() > 0) {
    // Consigne envoyee par le RPi, repartie sur 2 lignes (~21 car/ligne)
    String c = consigneOledServeur;
    int fin = min((int)c.length(), 21);
    if (fin < (int)c.length()) {
      int espace = c.lastIndexOf(' ', fin);
      if (espace > 0) fin = espace;
    }
    display.println(c.substring(0, fin));
    if (fin < (int)c.length()) display.println(c.substring(fin + 1));
  } else {
    display.println("Fermez la main");
    display.println("doucement.");
  }
  display.setCursor(0, 36);
  display.print("Rep: ");
  display.print(repetitionsOK);
  display.print(" / ");
  display.println(repetitionsCible);

  // Barre de progression
  int largeur = map(repetitionsOK, 0, repetitionsCible, 0, 100);
  display.drawRect(0, 48, 100, 5, SSD1306_WHITE);
  display.fillRect(0, 48, largeur, 5, SSD1306_WHITE);

  display.setCursor(0, 56);
  display.println("Appui = echec");
  display.display();
}

void afficherSuccesExercice() {
  oledEffacer();
  display.setTextSize(2);
  display.setCursor(25, 5);
  display.println("OK !");
  display.setTextSize(1);
  oledSeparateur(28);
  display.setCursor(0, 32);
  display.println("Tres bien !");
  display.println("Mouvement reussi.");
  oledSeparateur(52);
  display.setCursor(0, 55);
  display.println("2x appui = suite");
  display.display();
}

void afficherCoach() {
  oledEffacer();
  display.setCursor(0, 0);
  display.println("Votre coach :");
  oledSeparateur(10);
  display.setCursor(0, 14);

  // Affichage du message sur plusieurs lignes (max 21 chars/ligne)
  String msg = messageCoach;
  if (msg.length() == 0) msg = "Continuez vos efforts !";
  int start = 0, ligne = 0;
  while (start < (int)msg.length() && ligne < 3) {
    int fin = min(start + 21, (int)msg.length());
    if (fin < (int)msg.length()) {
      int espace = msg.lastIndexOf(' ', fin);
      if (espace > start) fin = espace;
    }
    display.setCursor(0, 14 + ligne * 10);
    display.println(msg.substring(start, fin));
    start = fin + 1;
    ligne++;
  }

  oledSeparateur(52);
  display.setCursor(0, 55);
  display.println("Appui = suite");
  display.display();
}

void afficherAdaptation() {
  oledEffacer();
  display.setCursor(0, 0);
  display.println("Adaptation...");
  oledSeparateur(10);
  display.setCursor(0, 14);
  if (messageRL.length() > 0) {
    display.println(messageRL.c_str());
  } else {
    display.println("Seance ajustee");
    display.print("Niveau : ");
    display.println(difficulteNiv);
  }
  display.setCursor(0, 45);
  display.println("Bonne progression !");
  oledSeparateur(52);
  display.setCursor(0, 55);
  display.println("Appui = suite");
  display.display();
}

void afficherFinSeance() {
  oledEffacer();
  display.setTextSize(2);
  display.setCursor(18, 2);
  display.println("Bravo !");
  display.setTextSize(1);
  oledSeparateur(22);
  display.setCursor(0, 26);
  display.println("Seance terminee.");
  display.print("Reussite : ");
  int pct = repetitionsCible > 0 ? (repetitionsOK * 100 / repetitionsCible) : 0;
  display.print(pct);
  display.println("%");
  oledSeparateur(50);
  display.setCursor(0, 53);
  display.println("2x appui = bilan");
  display.display();
}

void afficherPrediction() {
  oledEffacer();
  display.setCursor(0, 0);
  display.println("Votre progression");
  oledSeparateur(10);
  display.setCursor(0, 14);
  if (messagePred.length() > 0) {
    display.println(messagePred.c_str());
  } else {
    display.println("Selon vos seances:");
    display.println("Estimez +X deg");
    display.println("dans ~2 semaines.");
  }
  display.setCursor(0, 46);
  display.println("Estimation selon");
  display.println("votre rythme.");
  display.display();
}

// ============================================================================
// MULTIPLEXEUR ET FILTRES EMA
// ============================================================================
void muxSelect(uint8_t canal) {
  digitalWrite(MUX_S0, (canal >> 0) & 0x01);
  digitalWrite(MUX_S1, (canal >> 1) & 0x01);
  digitalWrite(MUX_S2, (canal >> 2) & 0x01);
  digitalWrite(MUX_S3, (canal >> 3) & 0x01);
  delayMicroseconds(80);
}

int lireCanalMux(uint8_t canal) {
  muxSelect(canal);
  long somme = 0;
  for (int i = 0; i < 8; i++) {
    somme += analogRead(MUX_SIG);
    delayMicroseconds(30);
  }
  return somme / 8;
}

void mettreAJourFiltres() {
  for (int i = 0; i < 5; i++) {
    int brut = lireCanalMux(i);
    flexFiltre[i] = ALPHA * brut + (1 - ALPHA) * flexFiltre[i];
  }
  pressionFiltree[0] = ALPHA * analogRead(PIN_PRESSION_POUCE)  + (1 - ALPHA) * pressionFiltree[0];
  pressionFiltree[1] = ALPHA * analogRead(PIN_PRESSION_INDEX)  + (1 - ALPHA) * pressionFiltree[1];
  pressionFiltree[2] = ALPHA * analogRead(PIN_PRESSION_MAJEUR) + (1 - ALPHA) * pressionFiltree[2];
}

int flexEnAngle(int i) {
  return constrain(map((int)flexFiltre[i], flexMin[i], flexMax[i], 0, 90), 0, 90);
}

int pressionEnPct(int i) {
  return constrain(map((int)pressionFiltree[i], pressionMin[i], pressionMax[i], 0, 100), 0, 100);
}

// ============================================================================
// DETECTION MOUVEMENT BON (feedback LED temps réel)
// ============================================================================
bool detecterMouvementBon() {
  // Un mouvement est "bon" si l'amplitude moyenne dépasse 50°
  // et la variance entre doigts n'est pas trop grande (pas asymétrique)
  int angles[5];
  int somme = 0;
  for (int i = 0; i < 5; i++) {
    angles[i] = flexEnAngle(i);
    somme += angles[i];
  }
  int moyenne = somme / 5;

  // Calcul variance simple
  int variance = 0;
  for (int i = 0; i < 5; i++) {
    variance += abs(angles[i] - moyenne);
  }
  variance /= 5;

  // Bon mouvement : amplitude > 45° ET symétrique (variance < 25°)
  return (moyenne > 45 && variance < 25);
}

// ============================================================================
// ECHANTILLONNAGE PENDANT LES 4 TESTS D'EVALUATION
// ============================================================================
// Appele en continu (throttle PERIODE_ECH_TEST) tant qu'on est dans un test.
// etape : 0=test1, 1=test2, 2=test3 (+pression), 3=test4
void echantillonnerTest(int etape) {
  unsigned long maintenant = millis();
  if (maintenant - dernierEchTest < PERIODE_ECH_TEST) return;
  dernierEchTest = maintenant;

  switch (etape) {
    case 0:
      if (nEch1 < MAX_ECH_TEST) {
        for (int i = 0; i < 5; i++) bufAngles1[nEch1][i] = flexEnAngle(i);
        nEch1++;
      }
      break;
    case 1:
      if (nEch2 < MAX_ECH_TEST) {
        for (int i = 0; i < 5; i++) bufAngles2[nEch2][i] = flexEnAngle(i);
        nEch2++;
      }
      break;
    case 2:
      if (nEch3 < MAX_ECH_TEST) {
        for (int i = 0; i < 5; i++) bufAngles3[nEch3][i] = flexEnAngle(i);
        nEch3++;
      }
      if (nEchP3 < MAX_ECH_TEST) {
        for (int i = 0; i < 3; i++) bufPression3[nEchP3][i] = pressionEnPct(i);
        nEchP3++;
      }
      break;
    case 3:
      if (nEch4 < MAX_ECH_TEST) {
        for (int i = 0; i < 5; i++) bufAngles4[nEch4][i] = flexEnAngle(i);
        nEch4++;
      }
      break;
  }
}

void reinitialiserBuffersTest() {
  nEch1 = 0; nEch2 = 0; nEch3 = 0; nEch4 = 0; nEchP3 = 0;
}

// ============================================================================
// ENVOI DONNEES VERS RPI
// ============================================================================
// ----------------------------------------------------------------------------
// /evaluation : envoyee UNE FOIS, apres les 4 tests initiaux.
// Chaque testX_angles est une SERIE temporelle [[d0..d4], [d0..d4], ...],
// conforme a ExtracteurMetriques.extraire() (couche1_knn.py) qui calcule
// max/diff/variance/std le long de l'axe temporel.
// ----------------------------------------------------------------------------
void envoyerEvaluation() {
  if (WiFi.status() != WL_CONNECTED) return;

  DynamicJsonDocument doc(8192);
  JsonArray t1  = doc.createNestedArray("test1_angles");
  for (int s = 0; s < nEch1; s++) {
    JsonArray ligne = t1.createNestedArray();
    for (int i = 0; i < 5; i++) ligne.add(bufAngles1[s][i]);
  }
  JsonArray t2  = doc.createNestedArray("test2_angles");
  for (int s = 0; s < nEch2; s++) {
    JsonArray ligne = t2.createNestedArray();
    for (int i = 0; i < 5; i++) ligne.add(bufAngles2[s][i]);
  }
  JsonArray t3  = doc.createNestedArray("test3_angles");
  for (int s = 0; s < nEch3; s++) {
    JsonArray ligne = t3.createNestedArray();
    for (int i = 0; i < 5; i++) ligne.add(bufAngles3[s][i]);
  }
  JsonArray t3p = doc.createNestedArray("test3_pression");
  for (int s = 0; s < nEchP3; s++) {
    JsonArray ligne = t3p.createNestedArray();
    for (int i = 0; i < 3; i++) ligne.add(bufPression3[s][i]);
  }
  JsonArray t4  = doc.createNestedArray("test4_angles");
  for (int s = 0; s < nEch4; s++) {
    JsonArray ligne = t4.createNestedArray();
    for (int i = 0; i < 5; i++) ligne.add(bufAngles4[s][i]);
  }

  String json;
  serializeJson(doc, json);

  HTTPClient http;
  http.begin(urlEvaluation());
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(json);

  if (code == 200) {
    String reponse = http.getString();
    DynamicJsonDocument rep(1024);
    if (!deserializeJson(rep, reponse)) {
      if (rep.containsKey("profil"))         profilMoteur        = rep["profil"].as<String>();
      if (rep.containsKey("profil_fr"))      profilMoteurFr      = rep["profil_fr"].as<String>();
      if (rep.containsKey("confidence"))     confianceProfil     = rep["confidence"].as<float>();
      if (rep.containsKey("difficulte"))     difficulteNiv       = rep["difficulte"].as<int>();
      if (rep.containsKey("exercice_id"))    exerciceIdServeur   = rep["exercice_id"].as<int>();
      if (rep.containsKey("exercice_nom"))   exerciceNomServeur  = rep["exercice_nom"].as<String>();
      if (rep.containsKey("consigne_oled"))  consigneOledServeur = rep["consigne_oled"].as<String>();
      if (rep.containsKey("repetitions"))    repetitionsCible    = rep["repetitions"].as<int>();
    }
  } else {
    Serial.printf("[EVAL] Erreur HTTP %d\n", code);
  }
  http.end();

  reinitialiserBuffersTest();
}

// ----------------------------------------------------------------------------
// /data : envoyee en boucle pendant les exercices.
// Retourne la decision RL (couche 2) + le message coach (Claude API).
// ----------------------------------------------------------------------------
void envoyerDonnees() {
  if (WiFi.status() != WL_CONNECTED) return;

  StaticJsonDocument<512> doc;
  JsonArray angles       = doc.createNestedArray("angles");
  JsonArray pressionPct  = doc.createNestedArray("pressure_pct");

  for (int i = 0; i < 5; i++) angles.add(flexEnAngle(i));
  for (int i = 0; i < 3; i++) pressionPct.add(pressionEnPct(i));
  // succes = une repetition vient d'etre validee depuis le dernier envoi (evenement ponctuel),
  // pas l'etat instantane du mouvement (sinon le RL recoit un "succes" toutes les 500ms).
  doc["succes"] = repetitionVientDeFinir;
  repetitionVientDeFinir = false; // consomme l'evenement

  String json;
  serializeJson(doc, json);

  HTTPClient http;
  http.begin(urlData());
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(json);

  if (code == 200) {
    String reponse = http.getString();
    StaticJsonDocument<512> rep;
    if (!deserializeJson(rep, reponse)) {
      String status = rep["status"] | "";
      if (status == "attente_evaluation") {
        // Le RPi n'a pas encore d'agent RL (evaluation pas encore faite cote serveur)
        if (rep.containsKey("message_oled")) consigneOledServeur = rep["message_oled"].as<String>();
      } else {
        if (rep.containsKey("message_rl"))     messageRL           = rep["message_rl"].as<String>();
        if (rep.containsKey("coach"))          messageCoach        = rep["coach"].as<String>();
        if (rep.containsKey("difficulte"))     difficulteNiv       = rep["difficulte"].as<int>();
        if (rep.containsKey("exercice_id"))    exerciceIdServeur   = rep["exercice_id"].as<int>();
        if (rep.containsKey("exercice_nom"))   exerciceNomServeur  = rep["exercice_nom"].as<String>();
        if (rep.containsKey("consigne_oled"))  consigneOledServeur = rep["consigne_oled"].as<String>();
        if (rep.containsKey("repetitions"))    repetitionsCible    = rep["repetitions"].as<int>();
        if (rep.containsKey("fin_seance"))     finSeanceServeur    = rep["fin_seance"].as<bool>();
        if (rep.containsKey("reward"))         rewardDernier       = rep["reward"].as<float>();
      }
    }
  }
  http.end();
}

// ============================================================================
// WIFI
// ============================================================================
void connecterWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  oledEffacer();
  display.setCursor(0, 20);
  display.println("Connexion WiFi...");
  display.display();

  unsigned long debut = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - debut < 10000) {
    delay(300);
    display.print(".");
    display.display();
  }

  oledEffacer();
  display.setCursor(0, 20);
  if (WiFi.status() == WL_CONNECTED) {
    display.println("WiFi OK !");
    clignoterLED(2);
  } else {
    display.println("WiFi echoue.");
    display.println("Mode hors-ligne.");
  }
  display.display();
  delay(1500);
}

// ============================================================================
// SETUP
// ============================================================================
void setup() {
  Serial.begin(115200);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  pinMode(MUX_S0, OUTPUT); pinMode(MUX_S1, OUTPUT);
  pinMode(MUX_S2, OUTPUT); pinMode(MUX_S3, OUTPUT);
  pinMode(MUX_SIG, INPUT);
  pinMode(PIN_PRESSION_POUCE, INPUT);
  pinMode(PIN_PRESSION_INDEX, INPUT);
  pinMode(PIN_PRESSION_MAJEUR, INPUT);
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_BOUTON, INPUT_PULLUP);

  // Initialisation filtres EMA
  for (int i = 0; i < 5; i++) flexFiltre[i] = lireCanalMux(i);
  pressionFiltree[0] = analogRead(PIN_PRESSION_POUCE);
  pressionFiltree[1] = analogRead(PIN_PRESSION_INDEX);
  pressionFiltree[2] = analogRead(PIN_PRESSION_MAJEUR);

  Wire.begin(21, 22);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("ERREUR OLED");
  }

  connecterWifi();
  afficherVeille();
  Serial.println("=== GANT PRET ===");
}

// ============================================================================
// LOOP PRINCIPALE
// ============================================================================
unsigned long dernierEnvoi    = 0;
unsigned long dernierAffichage = 0;
int etapeEvaluation = 0;

void loop() {
  mettreAJourFiltres();
  ActionBouton action = lireBouton();

  // --- Feedback LED temps réel pendant exercice ---
  if (etatActuel == ETAT_EXERCICE) {
    mouvementBon = detecterMouvementBon();
    ledVerte(mouvementBon);
  } else {
    ledVerte(false);
  }

  // --- Machine à états du parcours utilisateur ---
  switch (etatActuel) {

    case ETAT_VEILLE:
      if (action == APPUI_LONG) {
        bipCourt();
        clignoterLED(1);
        etatActuel = ETAT_ACCUEIL;
        afficherAccueil();
      }
      break;

    case ETAT_ACCUEIL:
      if (action == DOUBLE_APPUI) {
        bipCourt();
        etapeEvaluation = 0;
        etatActuel = ETAT_EVALUATION;
        afficherEvaluation(0);
      }
      break;

    case ETAT_EVALUATION:
      // Echantillonne en continu pendant le test en cours (etapes 0..3)
      if (etapeEvaluation < 4) {
        echantillonnerTest(etapeEvaluation);
      }

      if (action == APPUI_COURT) {
        etapeEvaluation++;
        if (etapeEvaluation >= 4) {
          // Les 4 tests sont faits -> envoyer les series temporelles vers /evaluation (pas /data)
          afficherEvaluation(4);
          envoyerEvaluation();
          delay(1500);
          etatActuel = ETAT_PROFIL;
          afficherProfil();
        } else {
          afficherEvaluation(etapeEvaluation);
        }
      }
      break;

    case ETAT_PROFIL:
      if (action == DOUBLE_APPUI) {
        bipCourt();
        repetitionsOK = 0;
        exerciceActuel = 1;
        etatActuel = ETAT_EXERCICE;
        afficherExercice();
      }
      break;

    case ETAT_EXERCICE:
      // Mise à jour affichage toutes les 300ms
      if (millis() - dernierAffichage > 300) {
        afficherExercice();
        dernierAffichage = millis();
      }

      // Mouvement bon détecté → compter répétition localement (feedback immédiat)
      if (mouvementBon) {
        static bool comptabilise = false;
        static unsigned long tempsDebut = 0;
        if (!comptabilise) {
          tempsDebut = millis();
          comptabilise = true;
        }
        // Comptabiliser après 1 seconde de bon mouvement
        if (millis() - tempsDebut > 1000 && comptabilise) {
          repetitionsOK++;
          repetitionVientDeFinir = true; // signale au prochain envoi /data qu'une repetition vient d'etre validee
          bipCourt();
          comptabilise = false;

          if (repetitionsOK >= repetitionsCible) {
            bipDouble();
            clignoterLED(3);
            afficherSuccesExercice();
            delay(1500);
            // On ne decide PAS ici si la seance/l'exercice est termine : c'est le RPi
            // (couche RL, via "fin_seance" dans la reponse /data) qui pilote la suite.
            // On notifie juste immediatement le serveur du succes, sans attendre le tick 500ms.
            envoyerDonnees();
            if (finSeanceServeur) {
              finSeanceServeur = false;
              etatActuel = ETAT_ADAPTATION;
              afficherAdaptation();
            } else {
              // Le RPi peut avoir change d'exercice / de niveau : on resynchronise l'affichage
              exerciceActuel++;
              repetitionsOK = 0;
              afficherExercice();
            }
          }
        }
      }

      // Appui court = déclarer échec de l'exercice
      if (action == APPUI_COURT) {
        etatActuel = ETAT_ADAPTATION;
        envoyerDonnees();
        afficherAdaptation();
      }
      break;

    case ETAT_ADAPTATION:
      if (action == APPUI_COURT) {
        bipCourt();
        etatActuel = ETAT_COACH;
        afficherCoach();
      }
      break;

    case ETAT_COACH:
      if (action == APPUI_COURT) {
        bipCourt();
        etatActuel = ETAT_FIN_SEANCE;
        bipFinSeance();
        afficherFinSeance();
      }
      break;

    case ETAT_FIN_SEANCE:
      if (action == DOUBLE_APPUI) {
        bipCourt();
        etatActuel = ETAT_PREDICTION;
        afficherPrediction();
      }
      break;

    case ETAT_PREDICTION:
      if (action == APPUI_LONG) {
        // Retour en veille pour nouvelle séance
        etatActuel = ETAT_VEILLE;
        repetitionsOK = 0;
        exerciceActuel = 1;
        afficherVeille();
      }
      break;
  }

  // Envoi données WiFi toutes les 500ms (sauf en veille et en évaluation,
  // qui a son propre envoi ponctuel vers /evaluation)
  if (etatActuel != ETAT_VEILLE && etatActuel != ETAT_EVALUATION &&
      millis() - dernierEnvoi > 500) {
    envoyerDonnees();
    dernierEnvoi = millis();

    // Le RPi (couche RL) peut décider de terminer l'exercice/la séance
    // même si le compteur local de répétitions n'est pas encore atteint.
    if (etatActuel == ETAT_EXERCICE && finSeanceServeur) {
      finSeanceServeur = false;
      etatActuel = ETAT_ADAPTATION;
      afficherAdaptation();
    }
  }

  delay(10);
}
