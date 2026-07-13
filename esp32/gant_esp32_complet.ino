/*
  ============================================================================
  GANT DE REEDUCATION AVC - CODE PRINCIPAL ESP32
  ============================================================================
  Basé sur le câblage Wokwi réel (DOS PCB) :

  MUX CD74HC4067 :
    SIG -> GPIO39   S0 -> GPIO33   S1 -> GPIO32   S2 -> GPIO17   S3 -> GPIO16
    EN  -> GND      VCC -> 3.3V
    C0..C4 -> 5 capteurs de flexion (pouce, index, majeur, annulaire, auriculaire)

  Capteurs de pression (directs, sans MUX) :
    Pouce  -> GPIO34    Index -> GPIO35    Majeur -> GPIO36

  Autres :
    LED    -> GPIO4     Buzzer -> GPIO14
    Bouton -> GPIO18    OLED I2C -> SDA=21, SCL=22

  Pourquoi GPIO39 pour SIG du MUX :
    Le WiFi ESP32 utilise le bloc ADC2 (GPIO 25,26,27,12,13,14...).
    Quand le WiFi est actif, ADC2 devient peu fiable.
    GPIO34,35,36,39 sont sur ADC1 -> toujours fiables avec WiFi actif.

  Couche 1 du pipeline IA : filtre EMA (moyenne mobile exponentielle)
    valeur_filtree = alpha * lecture_brute + (1-alpha) * valeur_filtree_precedente
    alpha = 0.2 -> bon compromis réactivité / lissage pour Velostat fait main
  ============================================================================
*/

#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ============================================================================
// A MODIFIER avec tes infos
// ============================================================================
const char* WIFI_SSID     = "TON_SSID";
const char* WIFI_PASSWORD = "TON_MOT_DE_PASSE";
const char* SERVER_URL    = "http://192.168.1.XX:5000/data"; // IP du RPi 5
// ============================================================================

// --- Pins MUX ---
#define MUX_SIG  39
#define MUX_S0   33
#define MUX_S1   32
#define MUX_S2   17
#define MUX_S3   16

// --- Pins pression directs ---
#define PIN_PRESSION_POUCE   34
#define PIN_PRESSION_INDEX   35
#define PIN_PRESSION_MAJEUR  36

// --- Autres ---
#define PIN_LED     4
#define PIN_BUZZER  14
#define PIN_BOUTON  18

// --- OLED ---
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ============================================================================
// CALIBRATION (à remplir après avoir mesuré tes capteurs réels)
// Procédure : main ouverte = noter la valeur -> valMin
//             main fermée  = noter la valeur -> valMax
// ============================================================================
int flexMin[5] = {0,    0,    0,    0,    0   }; // à mesurer main ouverte
int flexMax[5] = {3000, 3000, 4095, 3100, 3000}; // à mesurer main fermée
                 // pouce  index  majeur  annulaire  auriculaire

int pressionMin[3] = {0,   0,   0  }; // sans appui
int pressionMax[3] = {3500, 3500, 3500}; // appui fort
                     // pouce  index  majeur

// ============================================================================
// FILTRE EMA (Couche 1 pipeline IA)
// ============================================================================
const float ALPHA = 0.2;
float flexFiltre[5]     = {0, 0, 0, 0, 0};
float pressionFiltree[3] = {0, 0, 0};

// ============================================================================
// TIMING
// ============================================================================
unsigned long dernierEnvoi    = 0;
unsigned long dernierAffichage = 0;
const unsigned long INTERVALLE_ENVOI     = 500;  // ms
const unsigned long INTERVALLE_AFFICHAGE = 300;  // ms

// ============================================================================
// FONCTIONS MUX
// ============================================================================
void muxSelect(uint8_t canal) {
  digitalWrite(MUX_S0, (canal >> 0) & 0x01);
  digitalWrite(MUX_S1, (canal >> 1) & 0x01);
  digitalWrite(MUX_S2, (canal >> 2) & 0x01);
  digitalWrite(MUX_S3, (canal >> 3) & 0x01);
  delayMicroseconds(80); // stabilisation anti-diaphonie
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

// ============================================================================
// MISE A JOUR DES FILTRES EMA
// ============================================================================
void mettreAJourFiltres() {
  // Flexion via MUX
  for (int i = 0; i < 5; i++) {
    int brut = lireCanalMux(i);
    flexFiltre[i] = ALPHA * brut + (1 - ALPHA) * flexFiltre[i];
  }
  // Pression directs
  pressionFiltree[0] = ALPHA * analogRead(PIN_PRESSION_POUCE)  + (1 - ALPHA) * pressionFiltree[0];
  pressionFiltree[1] = ALPHA * analogRead(PIN_PRESSION_INDEX)  + (1 - ALPHA) * pressionFiltree[1];
  pressionFiltree[2] = ALPHA * analogRead(PIN_PRESSION_MAJEUR) + (1 - ALPHA) * pressionFiltree[2];
}

// ============================================================================
// CONVERSION EN ANGLE (0 a 90 degres) et PRESSION (0 a 100%)
// ============================================================================
int flexEnAngle(int indexDoigt) {
  return constrain(
    map((int)flexFiltre[indexDoigt], flexMin[indexDoigt], flexMax[indexDoigt], 0, 90),
    0, 90
  );
}

int pressionEnPourcentage(int indexDoigt) {
  return constrain(
    map((int)pressionFiltree[indexDoigt], pressionMin[indexDoigt], pressionMax[indexDoigt], 0, 100),
    0, 100
  );
}

// ============================================================================
// WIFI
// ============================================================================
void connecterWifi() {
  Serial.print("Connexion WiFi...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long debut = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - debut < 8000) {
    delay(300);
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi OK - IP: " + WiFi.localIP().toString());
  } else {
    Serial.println("\nWiFi echoue (mode sans WiFi actif)");
  }
}

// ============================================================================
// ENVOI JSON VERS RPI 5
// ============================================================================
void envoyerDonnees() {
  if (WiFi.status() != WL_CONNECTED) return;

  StaticJsonDocument<512> doc;

  // Valeurs brutes filtrées
  JsonArray flex = doc.createNestedArray("flex");
  JsonArray pression = doc.createNestedArray("pressure");
  for (int i = 0; i < 5; i++) flex.add((int)flexFiltre[i]);
  for (int i = 0; i < 3; i++) pression.add((int)pressionFiltree[i]);

  // Valeurs converties (angles et pourcentages)
  JsonArray angles = doc.createNestedArray("angles");
  JsonArray pressionPct = doc.createNestedArray("pressure_pct");
  for (int i = 0; i < 5; i++) angles.add(flexEnAngle(i));
  for (int i = 0; i < 3; i++) pressionPct.add(pressionEnPourcentage(i));

  String json;
  serializeJson(doc, json);

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(json);
  if (code > 0) {
    Serial.println("Envoye OK (" + String(code) + ")");
  } else {
    Serial.println("Erreur envoi: " + http.errorToString(code));
  }
  http.end();
}

// ============================================================================
// AFFICHAGE OLED
// ============================================================================
void afficherOLED() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);

  // Ligne 1 : angles des doigts
  display.println("FLEX (degres):");
  display.print("P:"); display.print(flexEnAngle(0));
  display.print(" I:"); display.print(flexEnAngle(1));
  display.print(" M:"); display.print(flexEnAngle(2));
  display.print(" A:"); display.print(flexEnAngle(3));
  display.print(" Au:"); display.println(flexEnAngle(4));

  // Ligne 2 : pression
  display.println("PRESSION (%):");
  display.print("P:"); display.print(pressionEnPourcentage(0));
  display.print(" I:"); display.print(pressionEnPourcentage(1));
  display.print(" M:"); display.println(pressionEnPourcentage(2));

  // Ligne 3 : statut WiFi
  display.print("WiFi: ");
  display.println(WiFi.status() == WL_CONNECTED ? "OK" : "--");

  display.display();
}

// ============================================================================
// AFFICHAGE SERIAL DEBUG
// ============================================================================
void afficherSerial() {
  const char* doigts[] = {"Pouce", "Index", "Majeur", "Annulaire", "Auriculaire"};
  Serial.println("--- CAPTEURS ---");
  for (int i = 0; i < 5; i++) {
    Serial.print(doigts[i]);
    Serial.print(": brut="); Serial.print((int)flexFiltre[i]);
    Serial.print(" angle="); Serial.print(flexEnAngle(i)); Serial.println("deg");
  }
  Serial.println("Pression (pouce/index/majeur): "
    + String(pressionEnPourcentage(0)) + "% "
    + String(pressionEnPourcentage(1)) + "% "
    + String(pressionEnPourcentage(2)) + "%");
  Serial.println();
}

// ============================================================================
// SETUP
// ============================================================================
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("=== GANT REEDUCATION AVC - DEMARRAGE ===");

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

  // Initialiser les filtres EMA avec une première lecture
  for (int i = 0; i < 5; i++) flexFiltre[i] = lireCanalMux(i);
  pressionFiltree[0] = analogRead(PIN_PRESSION_POUCE);
  pressionFiltree[1] = analogRead(PIN_PRESSION_INDEX);
  pressionFiltree[2] = analogRead(PIN_PRESSION_MAJEUR);

  // OLED
  Wire.begin(21, 22);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("ERREUR OLED non detecte !");
  } else {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("Gant AVC");
    display.println("Demarrage...");
    display.display();
  }

  connecterWifi();

  // Signal de démarrage
  digitalWrite(PIN_LED, HIGH);
  digitalWrite(PIN_BUZZER, HIGH); delay(100);
  digitalWrite(PIN_BUZZER, LOW);  delay(200);
  digitalWrite(PIN_LED, LOW);

  Serial.println("=== PRET - BOUCLE EN COURS ===");
}

// ============================================================================
// LOOP
// ============================================================================
void loop() {
  // Mise à jour filtres EMA (couche 1 pipeline IA)
  mettreAJourFiltres();

  // Bouton : LED allumée tant qu'appuyé
  bool bouton = (digitalRead(PIN_BOUTON) == LOW);
  digitalWrite(PIN_LED, bouton ? HIGH : LOW);

  // Affichage OLED toutes les 300ms
  if (millis() - dernierAffichage >= INTERVALLE_AFFICHAGE) {
    afficherOLED();
    afficherSerial();
    dernierAffichage = millis();
  }

  // Envoi WiFi toutes les 500ms
  if (millis() - dernierEnvoi >= INTERVALLE_ENVOI) {
    envoyerDonnees();
    dernierEnvoi = millis();
  }

  delay(10);
}
