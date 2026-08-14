/*
  ============================================================================
  GANT REEDUCATION AVC — CODE FINAL COMPLET V2
  ============================================================================
  Nouveautes v2 :
    - Verification /status au demarrage → saute l'evaluation si seance 2-4
    - Ecran "Materiel requis" avant evaluation (balle souple)
    - Consigne OLED recue du RPi (plus de texte en dur)
    - Extinction propre : appui long depuis ETAT_PREDICTION → ETAT_VEILLE

  Parcours utilisateur :
    SEANCE 1 :
      Appui long → Accueil → GET /status → evaluation_requise=true
      → Ecran materiel → 4 tests → Profil → Exercices → Bilan → Prediction
      → Appui long = retour veille + "A bientot !"

    SEANCES 2-4 :
      Appui long → Accueil → GET /status → evaluation_requise=false
      → Directement Exercices (reprend ou patient s'etait arrete)
      → Bilan → Prediction → Appui long = retour veille

    SEANCE 5 :
      Appui long → Accueil → GET /status → evaluation_requise=true
      → Reevaluation K-NN → nouveaux exercices si profil change

  Bouton unique :
    Appui long  (>1500ms) : allumer / eteindre (retour veille)
    Appui court (<1500ms) : avancer / confirmer
    Double appui          : valider / selectionner

  LED :
    Allumee  : mouvement correct pendant exercice
    Eteinte  : mouvement incorrect ou hors exercice
    Clignote : transition / analyse

  Buzzer :
    1 bip court  : succes mouvement
    2 bips       : fin exercice
    3 bips longs : fin seance

  Cablage :
    MUX : SIG=39 S0=33 S1=32 S2=17 S3=16 EN=GND VCC=3.3V
    Pression : Pouce=34 Index=35 Majeur=36
    LED=4  Buzzer=14  Bouton=18
    OLED I2C : SDA=21 SCL=22
  ============================================================================
*/

#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ============================================================================
// CONFIGURATION RESEAU
// ============================================================================
const char* WIFI_SSID     = "Elena's Galaxy A54 5G";
const char* WIFI_PASSWORD = "elena234";
const char* SERVER_BASE   = "http://10.254.218.206:5000";

// ============================================================================
// PINS
// ============================================================================
#define MUX_SIG 39
#define MUX_S0  33
#define MUX_S1  32
#define MUX_S2  17
#define MUX_S3  16
#define PIN_PRESSION_POUCE  34
#define PIN_PRESSION_INDEX  35
#define PIN_PRESSION_MAJEUR 36
#define PIN_LED    4
#define PIN_BUZZER 14
#define PIN_BOUTON 18

// ============================================================================
// OLED
// ============================================================================
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// ============================================================================
// CALIBRATION (valeurs reelles mesurees)
// ============================================================================
int flexMin[5] = {699,  0,    2260, 15,   1426};
int flexMax[5] = {4094, 4094, 4094, 4094, 4094};
int pressionMin[3] = {0,    0,    0   };
int pressionMax[3] = {3500, 3500, 3500};

// ============================================================================
// FILTRE EMA — Couche 0 pipeline IA (NE PAS MODIFIER)
// ============================================================================
const float ALPHA = 0.2;
float flexFiltre[5]      = {0,0,0,0,0};
float pressionFiltree[3] = {0,0,0};

// ============================================================================
// MACHINE A ETATS
// ============================================================================
enum EtatGant {
  ETAT_VEILLE,
  ETAT_ACCUEIL,
  ETAT_MATERIEL,      // Ecran "prevoyez une balle souple"
  ETAT_EVALUATION,    // 4 tests K-NN
  ETAT_PROFIL,
  ETAT_EXERCICE,
  ETAT_ADAPTATION,
  ETAT_COACH,
  ETAT_FIN_SEANCE,
  ETAT_PREDICTION
};
EtatGant etatActuel = ETAT_VEILLE;

// ============================================================================
// DONNEES SESSION
// ============================================================================
String profilMoteur     = "";
String profilMoteurFR   = "";
String consigneOled     = "";   // consigne recue du RPi
String messageCoach     = "";
String messageRL        = "";
String messagePred      = "";
String materielRequis   = "";   // message materiel avant evaluation
int    difficulteNiv    = 1;
int    repetitionsCible = 15;
int    repetitionsOK    = 0;
int    exerciceActuel   = 1;
bool   evaluationRequise = true;  // par defaut true, mis a jour par /status
bool   finSeanceServeur  = false;
bool   mouvementBon      = false;
int    numeroSeance      = 1;

// ============================================================================
// BUFFERS EVALUATION (4 tests)
// ============================================================================
const int BUF_SIZE = 30;
float bufAngles1[BUF_SIZE][5];
float bufAngles2[BUF_SIZE][5];
float bufAngles3[BUF_SIZE][5];
float bufPression3[BUF_SIZE][3];
float bufAngles4[BUF_SIZE][5];
int   buf1Cnt=0, buf2Cnt=0, buf3Cnt=0, buf4Cnt=0;
int   etapeEval = 0;

// ============================================================================
// GESTION BOUTON UNIQUE
// ============================================================================
unsigned long tempsAppui    = 0;
unsigned long tempsLacher1  = 0;
bool boutonPrecedent        = false;
bool attendDouble           = false;
const unsigned long SEUIL_LONG = 1500;
const unsigned long SEUIL_DBL  = 400;

enum ActionBouton { AUCUNE, APPUI_COURT, APPUI_LONG, DOUBLE_APPUI };

ActionBouton lireBouton() {
  bool appuye = (digitalRead(PIN_BOUTON) == LOW);
  ActionBouton action = AUCUNE;
  unsigned long now = millis();

  if (appuye && !boutonPrecedent) tempsAppui = now;

  if (!appuye && boutonPrecedent) {
    unsigned long duree = now - tempsAppui;
    if (duree >= SEUIL_LONG) {
      action = APPUI_LONG;
      attendDouble = false;
    } else {
      if (attendDouble && (now - tempsLacher1 < SEUIL_DBL)) {
        action = DOUBLE_APPUI;
        attendDouble = false;
      } else {
        attendDouble = true;
        tempsLacher1 = now;
      }
    }
  }
  if (attendDouble && !appuye && (now - tempsLacher1 > SEUIL_DBL)) {
    action = APPUI_COURT;
    attendDouble = false;
  }
  boutonPrecedent = appuye;
  return action;
}

// ============================================================================
// FEEDBACK LED ET BUZZER
// ============================================================================
void bipCourt()    { digitalWrite(PIN_BUZZER,HIGH); delay(80);  digitalWrite(PIN_BUZZER,LOW); }
void bipDouble()   { bipCourt(); delay(100); bipCourt(); }
void bipFinSeance(){ for(int i=0;i<3;i++){digitalWrite(PIN_BUZZER,HIGH);delay(200);digitalWrite(PIN_BUZZER,LOW);delay(150);} }
void ledVerte(bool on){ digitalWrite(PIN_LED, on?HIGH:LOW); }
void clignoterLED(int n){ for(int i=0;i<n;i++){digitalWrite(PIN_LED,HIGH);delay(150);digitalWrite(PIN_LED,LOW);delay(150);} }

// ============================================================================
// AFFICHAGE OLED — utilitaires
// ============================================================================
void oledEffacer(){ display.clearDisplay(); display.setTextColor(SSD1306_WHITE); display.setTextSize(1); }
void oledSep(int y){ display.drawFastHLine(0,y,128,SSD1306_WHITE); }

// Affiche un texte long sur plusieurs lignes (21 chars/ligne max)
void oledTexteMultiligne(String txt, int yDepart, int nbLignesMax) {
  int start=0, ligne=0;
  while(start < (int)txt.length() && ligne < nbLignesMax) {
    int fin = min(start+21, (int)txt.length());
    if(fin < (int)txt.length()) {
      int esp = txt.lastIndexOf(' ', fin);
      if(esp > start) fin = esp;
    }
    display.setCursor(0, yDepart + ligne*10);
    display.println(txt.substring(start, fin));
    start = fin+1;
    ligne++;
  }
}

// ============================================================================
// ECRANS OLED
// ============================================================================
void afficherVeille() {
  oledEffacer();
  display.setCursor(18,2); display.println("Gant Reeducation");
  oledSep(13);
  display.setCursor(10,22); display.println("Appui long = ON");
  display.setCursor(15,38); display.println("Bonne sante !");
  display.display();
}

void afficherAuRevoir() {
  oledEffacer();
  display.setTextSize(2);
  display.setCursor(20,10); display.println("A bientot");
  display.setTextSize(1);
  display.setCursor(15,40); display.println("Bonne recuperation");
  display.display();
  delay(2500);
}

void afficherAccueil() {
  oledEffacer();
  display.setCursor(28,0); display.println("Bonjour !");
  oledSep(11);
  display.setCursor(0,15); display.println("Je suis votre");
  display.println("assistant de");
  display.println("reeducation.");
  oledSep(52);
  display.setCursor(0,55); display.println("2x appui = debuter");
  display.display();
}

void afficherChargement() {
  oledEffacer();
  display.setCursor(20,20); display.println("Connexion RPi...");
  display.setCursor(25,38); display.println("Patientez...");
  display.display();
}

void afficherMateriel() {
  oledEffacer();
  display.setCursor(0,0); display.println("Avant de commencer");
  oledSep(11);
  display.setCursor(0,15);
  if(materielRequis.length() > 0) {
    oledTexteMultiligne(materielRequis, 15, 3);
  } else {
    display.println("Prevoyez une");
    display.println("balle souple.");
  }
  oledSep(50);
  display.setCursor(0,53); display.println("Appui = continuer");
  display.display();
}

void afficherEvaluation(int etape) {
  oledEffacer();
  const char* titres[] = {"Test 1/4","Test 2/4","Test 3/4","Test 4/4","Analyse..."};
  const char* l1[]     = {"Fermez la main","Maintenez ouvert","Serrez la balle","Repetez vite","Calcul en cours"};
  const char* l2[]     = {"lentement. 5x","immobile. 10s","fort. 5x","ouverture. 20x",""};
  const char* aide[]   = {"Appui = suivant","Appui = suivant","Appui = suivant","Appui = suivant","Patientez..."};

  display.setCursor(0,0); display.println(titres[min(etape,4)]);
  oledSep(10);
  display.setCursor(0,14); display.println(l1[min(etape,4)]);
  display.setCursor(0,26); display.println(l2[min(etape,4)]);
  oledSep(52);
  display.setCursor(0,55); display.println(aide[min(etape,4)]);
  display.display();
}

void afficherProfil() {
  oledEffacer();
  display.setCursor(0,0); display.println("Analyse terminee");
  oledSep(10);
  display.setCursor(0,14); display.println("Votre profil :");
  display.setCursor(0,26);
  if     (profilMoteur=="deficit_mobilite")     display.println("Mobilite reduite");
  else if(profilMoteur=="deficit_controle")     display.println("Controle ameliorer");
  else if(profilMoteur=="deficit_coordination") display.println("Coordination");
  else if(profilMoteur=="recuperation")         display.println("Bonne progression");
  else                                          display.println(profilMoteurFR.length()>0?profilMoteurFR:profilMoteur);
  display.setCursor(0,40); display.print("Niveau initial : "); display.println(difficulteNiv);
  oledSep(52);
  display.setCursor(0,55); display.println("2x appui = suite");
  display.display();
}

void afficherExercice() {
  oledEffacer();
  display.setCursor(0,0);
  display.print("Exercice "); display.print(exerciceActuel); display.print(" / 4");
  oledSep(10);
  display.setCursor(0,14);
  // Affiche la consigne recue du RPi (plus de texte en dur)
  if(consigneOled.length() > 0) {
    oledTexteMultiligne(consigneOled, 14, 2);
  } else {
    display.println("Suivez le mouvement");
  }
  display.setCursor(0,36);
  display.print("Rep: "); display.print(repetitionsOK);
  display.print(" / "); display.println(repetitionsCible);
  // Barre de progression
  int larg = map(repetitionsOK, 0, repetitionsCible, 0, 100);
  display.drawRect(0,48,100,5,SSD1306_WHITE);
  display.fillRect(0,48,larg,5,SSD1306_WHITE);
  display.setCursor(0,56); display.println("Appui = echec");
  display.display();
}

void afficherSuccesExercice() {
  oledEffacer();
  display.setTextSize(2); display.setCursor(25,5); display.println("OK !");
  display.setTextSize(1);
  oledSep(28);
  display.setCursor(0,32); display.println("Tres bien !");
  display.println("Mouvement reussi.");
  oledSep(52);
  display.setCursor(0,55); display.println("2x appui = suite");
  display.display();
}

void afficherCoach() {
  oledEffacer();
  display.setCursor(0,0); display.println("Votre coach :");
  oledSep(10);
  String msg = messageCoach.length()>0 ? messageCoach : "Continuez vos efforts !";
  oledTexteMultiligne(msg, 14, 3);
  oledSep(52);
  display.setCursor(0,55); display.println("Appui = suite");
  display.display();
}

void afficherAdaptation() {
  oledEffacer();
  display.setCursor(0,0); display.println("Adaptation...");
  oledSep(10);
  display.setCursor(0,14);
  String msg = messageRL.length()>0 ? messageRL : "Seance ajustee.";
  oledTexteMultiligne(msg, 14, 2);
  display.setCursor(0,40); display.print("Niveau : "); display.println(difficulteNiv);
  oledSep(52);
  display.setCursor(0,55); display.println("Appui = suite");
  display.display();
}

void afficherFinSeance() {
  oledEffacer();
  display.setTextSize(2); display.setCursor(18,2); display.println("Bravo !");
  display.setTextSize(1);
  oledSep(22);
  display.setCursor(0,26); display.println("Seance terminee !");
  int pct = repetitionsCible>0 ? (repetitionsOK*100/repetitionsCible) : 0;
  display.print("Reussite : "); display.print(pct); display.println("%");
  display.print("Seance n"); display.println(numeroSeance);
  oledSep(52);
  display.setCursor(0,55); display.println("2x appui = bilan");
  display.display();
}

void afficherPrediction() {
  oledEffacer();
  display.setCursor(0,0); display.println("Votre progression");
  oledSep(10);
  display.setCursor(0,14);
  if(messagePred.length()>0) {
    oledTexteMultiligne(messagePred, 14, 3);
  } else {
    display.println("Continuez vos");
    display.println("seances regulieres.");
    display.println("Progression en cours");
  }
  oledSep(52);
  display.setCursor(0,55); display.println("Appui long = fin");
  display.display();
}

// ============================================================================
// MUX ET FILTRES EMA
// ============================================================================
void muxSelect(uint8_t c) {
  digitalWrite(MUX_S0,(c>>0)&1); digitalWrite(MUX_S1,(c>>1)&1);
  digitalWrite(MUX_S2,(c>>2)&1); digitalWrite(MUX_S3,(c>>3)&1);
  delayMicroseconds(80);
}
int lireCanalMux(uint8_t c) {
  muxSelect(c); long s=0;
  for(int i=0;i<8;i++){s+=analogRead(MUX_SIG);delayMicroseconds(30);}
  return s/8;
}
void mettreAJourFiltres() {
  for(int i=0;i<5;i++){int b=lireCanalMux(i);flexFiltre[i]=ALPHA*b+(1-ALPHA)*flexFiltre[i];}
  pressionFiltree[0]=ALPHA*analogRead(PIN_PRESSION_POUCE)+(1-ALPHA)*pressionFiltree[0];
  pressionFiltree[1]=ALPHA*analogRead(PIN_PRESSION_INDEX)+(1-ALPHA)*pressionFiltree[1];
  pressionFiltree[2]=ALPHA*analogRead(PIN_PRESSION_MAJEUR)+(1-ALPHA)*pressionFiltree[2];
}
int flexEnAngle(int i){return constrain(map((int)flexFiltre[i],flexMin[i],flexMax[i],0,90),0,90);}
int pressionEnPct(int i){return constrain(map((int)pressionFiltree[i],pressionMin[i],pressionMax[i],0,100),0,100);}

bool detecterMouvementBon() {
  int s=0; int a[5];
  for(int i=0;i<5;i++){a[i]=flexEnAngle(i);s+=a[i];}
  int moy=s/5, v=0;
  for(int i=0;i<5;i++) v+=abs(a[i]-moy);
  return (moy>45 && v/5<25);
}

// ============================================================================
// ECHANTILLONNER UN TEST (capture serie temporelle)
// ============================================================================
void echantillonnerTest(int testNum) {
  mettreAJourFiltres();
  if(testNum==1 && buf1Cnt<BUF_SIZE) {
    for(int i=0;i<5;i++) bufAngles1[buf1Cnt][i]=flexEnAngle(i);
    buf1Cnt++;
  } else if(testNum==2 && buf2Cnt<BUF_SIZE) {
    for(int i=0;i<5;i++) bufAngles2[buf2Cnt][i]=flexEnAngle(i);
    buf2Cnt++;
  } else if(testNum==3 && buf3Cnt<BUF_SIZE) {
    for(int i=0;i<5;i++) bufAngles3[buf3Cnt][i]=flexEnAngle(i);
    for(int i=0;i<3;i++) bufPression3[buf3Cnt][i]=pressionEnPct(i);
    buf3Cnt++;
  } else if(testNum==4 && buf4Cnt<BUF_SIZE) {
    for(int i=0;i<5;i++) bufAngles4[buf4Cnt][i]=flexEnAngle(i);
    buf4Cnt++;
  }
}

// ============================================================================
// WIFI
// ============================================================================
void connecterWifi() {
  WiFi.mode(WIFI_STA); WiFi.begin(WIFI_SSID,WIFI_PASSWORD);
  oledEffacer(); display.setCursor(0,20); display.println("Connexion WiFi...");
  display.display();
  unsigned long d=millis();
  while(WiFi.status()!=WL_CONNECTED && millis()-d<10000){delay(300);display.print(".");display.display();}
  oledEffacer(); display.setCursor(0,20);
  if(WiFi.status()==WL_CONNECTED){display.println("WiFi OK !"); clignoterLED(2);}
  else{display.println("WiFi echoue."); display.println("Mode hors-ligne.");}
  display.display(); delay(1200);
}

// ============================================================================
// REQUETE GET /status — VERIFIE SI EVALUATION REQUISE
// ============================================================================
bool verifierStatutSession() {
  if(WiFi.status()!=WL_CONNECTED) return true;
  HTTPClient http;
  String url = String(SERVER_BASE) + "/status";
  http.begin(url);
  int code = http.GET();
  if(code==200) {
    String body = http.getString();
    StaticJsonDocument<512> doc;
    if(!deserializeJson(doc,body)) {
      evaluationRequise = doc["evaluation_requise"].as<bool>();
      numeroSeance      = doc["seance"].as<int>();
      materielRequis    = doc["materiel_requis"].as<String>();
      if(!evaluationRequise) {
        // Charger les donnees de la session existante
        consigneOled      = doc["consigne_oled"].as<String>();
        repetitionsCible  = doc["repetitions"].as<int>();
        difficulteNiv     = doc["difficulte"].as<int>();
        profilMoteur      = doc["profil"].as<String>();
      }
    }
  }
  http.end();
  return evaluationRequise;
}

// ============================================================================
// POST /evaluation — ENVOIE LES 4 TESTS
// ============================================================================
void envoyerEvaluation() {
  if(WiFi.status()!=WL_CONNECTED) return;
  StaticJsonDocument<2048> doc;

  // Serialiser les 4 buffers de tests
  JsonArray t1=doc.createNestedArray("test1_angles");
  for(int i=0;i<buf1Cnt;i++){JsonArray r=t1.createNestedArray();for(int j=0;j<5;j++)r.add(bufAngles1[i][j]);}

  JsonArray t2=doc.createNestedArray("test2_angles");
  for(int i=0;i<buf2Cnt;i++){JsonArray r=t2.createNestedArray();for(int j=0;j<5;j++)r.add(bufAngles2[i][j]);}

  JsonArray t3=doc.createNestedArray("test3_angles");
  for(int i=0;i<buf3Cnt;i++){JsonArray r=t3.createNestedArray();for(int j=0;j<5;j++)r.add(bufAngles3[i][j]);}

  JsonArray p3=doc.createNestedArray("test3_pression");
  for(int i=0;i<buf3Cnt;i++){JsonArray r=p3.createNestedArray();for(int j=0;j<3;j++)r.add(bufPression3[i][j]);}

  JsonArray t4=doc.createNestedArray("test4_angles");
  for(int i=0;i<buf4Cnt;i++){JsonArray r=t4.createNestedArray();for(int j=0;j<5;j++)r.add(bufAngles4[i][j]);}

  String json; serializeJson(doc,json);
  HTTPClient http;
  http.begin(String(SERVER_BASE)+"/evaluation");
  http.addHeader("Content-Type","application/json");
  int code = http.POST(json);
  if(code==200){
    String body=http.getString();
    StaticJsonDocument<512> rep;
    if(!deserializeJson(rep,body)){
      profilMoteur     = rep["profil"].as<String>();
      profilMoteurFR   = rep["profil_fr"].as<String>();
      difficulteNiv    = rep["difficulte"].as<int>();
      consigneOled     = rep["consigne_oled"].as<String>();
      repetitionsCible = rep["repetitions"].as<int>();
    }
  }
  http.end();
}

// ============================================================================
// POST /data — DONNEES TEMPS REEL
// ============================================================================
void envoyerDonnees(bool succes) {
  if(WiFi.status()!=WL_CONNECTED) return;
  StaticJsonDocument<512> doc;
  JsonArray angles=doc.createNestedArray("angles");
  JsonArray pct=doc.createNestedArray("pressure_pct");
  for(int i=0;i<5;i++) angles.add(flexEnAngle(i));
  for(int i=0;i<3;i++) pct.add(pressionEnPct(i));
  doc["succes"]=succes;

  String json; serializeJson(doc,json);
  HTTPClient http;
  http.begin(String(SERVER_BASE)+"/data");
  http.addHeader("Content-Type","application/json");
  int code=http.POST(json);
  if(code==200){
    String body=http.getString();
    StaticJsonDocument<512> rep;
    if(!deserializeJson(rep,body)){
      if(rep.containsKey("coach"))        messageCoach     = rep["coach"].as<String>();
      if(rep.containsKey("message_rl"))   messageRL        = rep["message_rl"].as<String>();
      if(rep.containsKey("consigne_oled"))consigneOled     = rep["consigne_oled"].as<String>();
      if(rep.containsKey("repetitions"))  repetitionsCible = rep["repetitions"].as<int>();
      if(rep.containsKey("difficulte"))   difficulteNiv    = rep["difficulte"].as<int>();
      if(rep.containsKey("fin_seance"))   finSeanceServeur = rep["fin_seance"].as<bool>();
      if(rep.containsKey("prediction"))   messagePred      = rep["prediction"].as<String>();
    }
  }
  http.end();
}

// ============================================================================
// SETUP
// ============================================================================
void setup() {
  Serial.begin(115200);
  analogReadResolution(12); analogSetAttenuation(ADC_11db);
  pinMode(MUX_S0,OUTPUT); pinMode(MUX_S1,OUTPUT);
  pinMode(MUX_S2,OUTPUT); pinMode(MUX_S3,OUTPUT);
  pinMode(MUX_SIG,INPUT);
  pinMode(PIN_PRESSION_POUCE,INPUT); pinMode(PIN_PRESSION_INDEX,INPUT); pinMode(PIN_PRESSION_MAJEUR,INPUT);
  pinMode(PIN_LED,OUTPUT); pinMode(PIN_BUZZER,OUTPUT); pinMode(PIN_BOUTON,INPUT_PULLUP);

  for(int i=0;i<5;i++) flexFiltre[i]=lireCanalMux(i);
  pressionFiltree[0]=analogRead(PIN_PRESSION_POUCE);
  pressionFiltree[1]=analogRead(PIN_PRESSION_INDEX);
  pressionFiltree[2]=analogRead(PIN_PRESSION_MAJEUR);

  Wire.begin(21,22);
  if(!display.begin(SSD1306_SWITCHCAPVCC,0x3C)) Serial.println("ERREUR OLED");
  else { oledEffacer(); display.setCursor(0,20); display.println("Gant AVC"); display.println("Demarrage..."); display.display(); }

  connecterWifi();

  digitalWrite(PIN_LED,HIGH); bipCourt(); delay(200); digitalWrite(PIN_LED,LOW);
  afficherVeille();
  Serial.println("=== GANT PRET ===");
}

// ============================================================================
// LOOP
// ============================================================================
unsigned long dernierEnvoi    = 0;
unsigned long dernierAffichage = 0;
unsigned long dernierEchant   = 0;

void loop() {
  mettreAJourFiltres();
  ActionBouton action = lireBouton();

  // LED feedback pendant exercice
  if(etatActuel==ETAT_EXERCICE) {
    mouvementBon = detecterMouvementBon();
    ledVerte(mouvementBon);
  } else { ledVerte(false); }

  switch(etatActuel) {

    // ----------------------------------------------------------
    case ETAT_VEILLE:
      if(action==APPUI_LONG) {
        bipCourt(); clignoterLED(1);
        etatActuel=ETAT_ACCUEIL;
        afficherAccueil();
      }
      break;

    // ----------------------------------------------------------
    case ETAT_ACCUEIL:
      if(action==DOUBLE_APPUI) {
        afficherChargement();
        verifierStatutSession();  // GET /status → sait si eval requise
        bipCourt();
        if(evaluationRequise) {
          // Seance 1 ou seance 5 → afficher materiel avant eval
          etatActuel = ETAT_MATERIEL;
          afficherMateriel();
        } else {
          // Seances 2-4 → reprendre directement les exercices
          repetitionsOK  = 0;
          exerciceActuel = 1;
          finSeanceServeur = false;
          etatActuel = ETAT_EXERCICE;
          afficherExercice();
        }
      }
      break;

    // ----------------------------------------------------------
    case ETAT_MATERIEL:
      // Ecran "prevoyez une balle souple" avant evaluation
      if(action==APPUI_COURT || action==DOUBLE_APPUI) {
        bipCourt();
        etapeEval = 0;
        buf1Cnt=0; buf2Cnt=0; buf3Cnt=0; buf4Cnt=0;
        etatActuel = ETAT_EVALUATION;
        afficherEvaluation(0);
      }
      break;

    // ----------------------------------------------------------
    case ETAT_EVALUATION:
      // Echantillonner en continu le test en cours
      if(millis()-dernierEchant > 200) {
        echantillonnerTest(etapeEval+1);
        dernierEchant=millis();
      }
      if(action==APPUI_COURT) {
        etapeEval++;
        if(etapeEval>=4) {
          // Tous les tests faits → envoyer au RPi
          afficherEvaluation(4); // "Analyse..."
          envoyerEvaluation();
          delay(1500);
          etatActuel=ETAT_PROFIL;
          afficherProfil();
        } else {
          afficherEvaluation(etapeEval);
        }
      }
      break;

    // ----------------------------------------------------------
    case ETAT_PROFIL:
      if(action==DOUBLE_APPUI) {
        bipCourt();
        repetitionsOK=0; exerciceActuel=1;
        finSeanceServeur=false;
        etatActuel=ETAT_EXERCICE;
        afficherExercice();
      }
      break;

    // ----------------------------------------------------------
    case ETAT_EXERCICE:
      // Affichage toutes les 300ms
      if(millis()-dernierAffichage>300) {
        afficherExercice();
        dernierAffichage=millis();
      }

      // Mouvement bon → compter repetition
      if(mouvementBon) {
        static bool comptabilise=false;
        static unsigned long tDebut=0;
        if(!comptabilise){tDebut=millis();comptabilise=true;}
        if(millis()-tDebut>1000 && comptabilise) {
          repetitionsOK++;
          bipCourt();
          comptabilise=false;
          if(repetitionsOK>=repetitionsCible) {
            bipDouble(); clignoterLED(3);
            afficherSuccesExercice();
            envoyerDonnees(true);
            delay(1500);
            etatActuel=ETAT_ADAPTATION;
            afficherAdaptation();
          }
        }
      }

      // Appui court = declarer echec
      if(action==APPUI_COURT) {
        envoyerDonnees(false);
        etatActuel=ETAT_ADAPTATION;
        afficherAdaptation();
      }

      // Le RPi decide la fin de seance
      if(finSeanceServeur) {
        bipFinSeance(); clignoterLED(3);
        etatActuel=ETAT_FIN_SEANCE;
        afficherFinSeance();
        finSeanceServeur=false;
      }
      break;

    // ----------------------------------------------------------
    case ETAT_ADAPTATION:
      if(action==APPUI_COURT) {
        bipCourt();
        exerciceActuel++;
        repetitionsOK=0;
        etatActuel=ETAT_COACH;
        afficherCoach();
      }
      break;

    // ----------------------------------------------------------
    case ETAT_COACH:
      if(action==APPUI_COURT) {
        bipCourt();
        if(exerciceActuel>4 || finSeanceServeur) {
          bipFinSeance();
          etatActuel=ETAT_FIN_SEANCE;
          afficherFinSeance();
        } else {
          etatActuel=ETAT_EXERCICE;
          afficherExercice();
        }
      }
      break;

    // ----------------------------------------------------------
    case ETAT_FIN_SEANCE:
      if(action==DOUBLE_APPUI) {
        bipCourt();
        etatActuel=ETAT_PREDICTION;
        afficherPrediction();
      }
      break;

    // ----------------------------------------------------------
    case ETAT_PREDICTION:
      // APPUI LONG = eteindre / retour veille
      if(action==APPUI_LONG) {
        afficherAuRevoir();
        // Reset session locale
        repetitionsOK=0; exerciceActuel=1;
        profilMoteur=""; consigneOled="";
        messageCoach=""; messagePred="";
        finSeanceServeur=false;
        etatActuel=ETAT_VEILLE;
        afficherVeille();
      }
      break;
  }

  // Envoi WiFi toutes les 500ms (hors veille et evaluation)
  if(etatActuel==ETAT_EXERCICE && millis()-dernierEnvoi>500) {
    envoyerDonnees(mouvementBon);
    dernierEnvoi=millis();
  }

  delay(10);
}
