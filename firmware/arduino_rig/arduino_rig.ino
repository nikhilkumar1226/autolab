/*
  =============================================================================
  AutoLab rig firmware v2.0.0  (Arduino Uno)
  -----------------------------------------------------------------------------
  Actuator: SG90 servo on pin 9 (rotates the second polariser, 0-180 deg).
  Sensor, chosen at compile time (exactly one):
    SENSOR_EMULATED  (default) intensity computed on-chip from the servo angle
                     with Malus's law plus Gaussian noise: for firmware-in-the-
                     loop runs in the Wokwi simulator, where nothing is optical.
    SENSOR_LDR       light-dependent resistor in a voltage divider on A0
                     (nominal GL5528 calibration; calibrate against a lux meter).
    SENSOR_TSL2591   Adafruit TSL2591 lux sensor on I2C (needs the
                     "Adafruit TSL2591 Library").
  Select one with a build flag, e.g.
    scripts/build_firmware.sh LDR     (or EMULATED, TSL2591)

  Serial protocol, 115200 baud, one command per line, one reply line each:
    PING            -> PONG
    VERSION         -> VERSION: 2.0.0, SENSOR: EMULATED, MIN: 0.00, MAX: 180.00, MAX_STEP: 180.00, WATCHDOG_S: 300
    STATUS          -> STATUS: OK, POS: <deg>, HOMED: <0|1>, ESTOP: <0|1>, REASON: <none|operator|watchdog>
    HOME            -> STATUS: OK, POS: 0.00
    MOVE <deg>      -> STATUS: OK, POS: <deg>
    READ            -> STATUS: OK, DATA: <lux>
    ESTOP           -> STATUS: OK, ESTOP: 1        (servo released at once; latched)
    RESET           -> STATUS: OK, ESTOP: 0, POS: <deg>
  Errors:           -> STATUS: ERROR, CODE: <code>, MSG: <text>
    E_SYNTAX   not a plain decimal number, missing argument, or line too long
    E_BOUNDS   target outside [MIN_ANGLE, MAX_ANGLE]
    E_STEP     move larger than MAX_STEP_DEG
    E_ESTOP    E-stop latched (by ESTOP or the watchdog): send RESET
    E_SENSOR   the sensor did not give a valid reading
    E_UNKNOWN  unknown command
  Unsolicited lines start with "EVENT:" (the boot banner, a watchdog trip);
  the host skips them.

  Firmware-side safety, independent of the host's safety guard:
    - bounds and step limit on every MOVE; arguments must be plain decimals
      ("abc" is an error, not 0 as String.toFloat would make it);
    - command watchdog: no command for WATCHDOG_S seconds -> servo released,
      E-stop latched until RESET;
    - ESTOP releases the servo (no holding torque) and refuses every MOVE and
      HOME until RESET.
  =============================================================================
*/

#include <Arduino.h>
#include <Servo.h>

#if !defined(SENSOR_LDR) && !defined(SENSOR_TSL2591) && !defined(SENSOR_EMULATED)
#define SENSOR_EMULATED
#endif
#if (defined(SENSOR_LDR) + defined(SENSOR_TSL2591) + defined(SENSOR_EMULATED)) != 1
#error "Select exactly one sensor: SENSOR_EMULATED, SENSOR_LDR or SENSOR_TSL2591"
#endif

#ifdef SENSOR_TSL2591
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include "Adafruit_TSL2591.h"
#endif

const char FIRMWARE_VERSION[] = "2.0.0";

// ---- Pins ----------------------------------------------------------------
const int SERVO_PIN = 9;
const int LDR_PIN = A0;

// ---- Safety limits (the host guard has its own; these hold even if it fails)
const float MIN_ANGLE = 0.0;
const float MAX_ANGLE = 180.0;
const float MAX_STEP_DEG = 180.0;
const unsigned long WATCHDOG_S = 300;          // 5 min: generous, a real LLM may think between moves
const unsigned long WATCHDOG_MS = WATCHDOG_S * 1000UL;
const int SERVO_MIN_US = 544;                  // Servo library defaults for 0 and 180 deg
const int SERVO_MAX_US = 2400;

// ---- Emulated optics (Malus's law; same values as the digital twin) ----------
const float EMU_AMPLITUDE = 1000.0;            // lux through aligned polarisers
const float EMU_PHASE_DEG = 20.0;              // angle of maximum transmission
const float EMU_AMBIENT = 15.0;                // stray light, lux
const float EMU_NOISE_SD = 10.0;               // sensor noise, lux
const long EMU_SEED = 12345;                   // fixed, so a simulated session is repeatable

// ---- LDR divider (SENSOR_LDR) --------------------------------------------------
const float LDR_FIXED_OHM = 10000.0;           // fixed resistor from A0 to GND; LDR from 5 V to A0
const float LDR_R10_OHM = 50000.0;             // GL5528: about 50 kOhm at 10 lux (nominal)
const float LDR_GAMMA = 0.7;

const int LINE_MAX = 32;

Servo polariserServo;
float currentAngle = 0.0;
bool isHomed = false;
bool estopLatched = false;
const char* estopReason = "none";          // "operator" or "watchdog" while latched
unsigned long lastCommandMs = 0;
char line[LINE_MAX + 1];
int lineLength = 0;
bool lineOverflow = false;

#ifdef SENSOR_TSL2591
Adafruit_TSL2591 tsl = Adafruit_TSL2591(2591);
bool tslReady = false;
#endif

// ---- Replies -----------------------------------------------------------------
void replyError(const char* code, const __FlashStringHelper* msg) {
  Serial.print(F("STATUS: ERROR, CODE: "));
  Serial.print(code);
  Serial.print(F(", MSG: "));
  Serial.println(msg);
}

void replyPosition() {
  Serial.print(F("STATUS: OK, POS: "));
  Serial.println(currentAngle, 2);
}

// ---- Actuator --------------------------------------------------------------------
void driveServo(float angle) {
  int us = SERVO_MIN_US + (int)((angle - MIN_ANGLE) * (SERVO_MAX_US - SERVO_MIN_US) / (MAX_ANGLE - MIN_ANGLE) + 0.5);
  polariserServo.writeMicroseconds(us);
}

void settle(float degrees) {
  // SG90: about 0.1 s per 60 deg, plus time to stop ringing.
  delay(100 + (unsigned long)(degrees * 2.0));
}

void releaseServo() {
  polariserServo.detach();                     // no holding torque: the safe state
}

void latchEstop(const char* reason) {
  estopLatched = true;
  estopReason = reason;
  releaseServo();
}

// A plain decimal: optional sign, digits, at most one point, at least one digit.
bool parseNumber(const char* text, float* out) {
  const char* p = text;
  while (*p == ' ') p++;
  if (*p == '+' || *p == '-') p++;
  int digits = 0, points = 0;
  for (const char* q = p; *q; q++) {
    if (*q >= '0' && *q <= '9') digits++;
    else if (*q == '.') points++;
    else if (*q == ' ' && *(q + 1) == '\0') break;
    else return false;
  }
  if (digits == 0 || points > 1 || digits > 8) return false;
  *out = (float)atof(text);
  return true;
}

// ---- Sensor --------------------------------------------------------------------------
#ifdef SENSOR_EMULATED
float gaussian() {
  float u1 = (random(1, 10001)) / 10001.0;
  float u2 = random(0, 10000) / 10000.0;
  return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}
#endif

bool readLux(float* lux) {
#if defined(SENSOR_EMULATED)
  float c = cos((currentAngle - EMU_PHASE_DEG) * DEG_TO_RAD);
  float value = EMU_AMPLITUDE * c * c + EMU_AMBIENT + EMU_NOISE_SD * gaussian();
  *lux = value < 0.0 ? 0.0 : value;
  delay(20);                                   // stands in for the integration time
  return true;
#elif defined(SENSOR_LDR)
  long sum = 0;
  for (int i = 0; i < 16; i++) {
    sum += analogRead(LDR_PIN);
    delay(2);
  }
  float volts = (sum / 16.0) * 5.0 / 1023.0;
  if (volts <= 0.01 || volts >= 4.99) return false;   // open or shorted divider: no valid reading
  float rLdr = LDR_FIXED_OHM * (5.0 - volts) / volts;
  *lux = 10.0 * pow(LDR_R10_OHM / rLdr, 1.0 / LDR_GAMMA);
  return true;
#elif defined(SENSOR_TSL2591)
  if (!tslReady) return false;
  uint32_t lum = tsl.getFullLuminosity();
  uint16_t ir = lum >> 16;
  uint16_t full = lum & 0xFFFF;
  float value = tsl.calculateLux(full, ir);
  if (isnan(value) || value < 0.0) return false;      // saturated or failed conversion
  *lux = value;
  return true;
#endif
}

const char* sensorName() {
#if defined(SENSOR_EMULATED)
  return "EMULATED";
#elif defined(SENSOR_LDR)
  return "LDR";
#else
  return "TSL2591";
#endif
}

// ---- Commands ------------------------------------------------------------------------
void handleMove(const char* arg) {
  float target;
  if (arg == NULL || !parseNumber(arg, &target)) {
    replyError("E_SYNTAX", F("MOVE needs a plain decimal angle"));
    return;
  }
  if (estopLatched) {
    replyError("E_ESTOP", F("E-stop latched; send RESET"));
    return;
  }
  if (target < MIN_ANGLE || target > MAX_ANGLE) {
    replyError("E_BOUNDS", F("angle outside [0, 180]"));
    return;
  }
  float step = fabs(target - currentAngle);
  if (step > MAX_STEP_DEG) {
    replyError("E_STEP", F("step larger than MAX_STEP"));
    return;
  }
  if (!polariserServo.attached()) polariserServo.attach(SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
  driveServo(target);
  settle(step);
  currentAngle = target;
  replyPosition();
}

void handleCommand(char* cmd) {
  // Upper-case the verb; split off the argument.
  char* arg = strchr(cmd, ' ');
  if (arg != NULL) {
    *arg = '\0';
    arg++;
    while (*arg == ' ') arg++;
    if (*arg == '\0') arg = NULL;
  }
  for (char* p = cmd; *p; p++) *p = toupper(*p);

  if (strcmp(cmd, "PING") == 0) {
    Serial.println(F("PONG"));
  } else if (strcmp(cmd, "VERSION") == 0) {
    Serial.print(F("VERSION: "));
    Serial.print(FIRMWARE_VERSION);
    Serial.print(F(", SENSOR: "));
    Serial.print(sensorName());
    Serial.print(F(", MIN: "));
    Serial.print(MIN_ANGLE, 2);
    Serial.print(F(", MAX: "));
    Serial.print(MAX_ANGLE, 2);
    Serial.print(F(", MAX_STEP: "));
    Serial.print(MAX_STEP_DEG, 2);
    Serial.print(F(", WATCHDOG_S: "));
    Serial.println(WATCHDOG_S);
  } else if (strcmp(cmd, "STATUS") == 0) {
    Serial.print(F("STATUS: OK, POS: "));
    Serial.print(currentAngle, 2);
    Serial.print(F(", HOMED: "));
    Serial.print(isHomed ? 1 : 0);
    Serial.print(F(", ESTOP: "));
    Serial.print(estopLatched ? 1 : 0);
    Serial.print(F(", REASON: "));
    Serial.println(estopReason);
  } else if (strcmp(cmd, "HOME") == 0) {
    if (estopLatched) {
      replyError("E_ESTOP", F("E-stop latched; send RESET"));
      return;
    }
    if (!polariserServo.attached()) polariserServo.attach(SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
    driveServo(MIN_ANGLE);
    settle(fabs(currentAngle - MIN_ANGLE));
    currentAngle = MIN_ANGLE;
    isHomed = true;
    replyPosition();
  } else if (strcmp(cmd, "MOVE") == 0) {
    handleMove(arg);
  } else if (strcmp(cmd, "READ") == 0) {
    float lux;
    if (readLux(&lux)) {
      Serial.print(F("STATUS: OK, DATA: "));
      Serial.println(lux, 2);
    } else {
      replyError("E_SENSOR", F("no valid sensor reading"));
    }
  } else if (strcmp(cmd, "ESTOP") == 0) {
    latchEstop("operator");
    Serial.println(F("STATUS: OK, ESTOP: 1"));
  } else if (strcmp(cmd, "RESET") == 0) {
    estopLatched = false;
    estopReason = "none";
    polariserServo.attach(SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
    driveServo(currentAngle);                  // hold where it is; no sudden move
    Serial.print(F("STATUS: OK, ESTOP: 0, POS: "));
    Serial.println(currentAngle, 2);
  } else {
    replyError("E_UNKNOWN", F("unknown command"));
  }
}

// ---- Main loop --------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
  }
#ifdef SENSOR_EMULATED
  randomSeed(EMU_SEED);
#endif
#ifdef SENSOR_TSL2591
  Wire.begin();
  tslReady = tsl.begin();
  if (tslReady) {
    tsl.setGain(TSL2591_GAIN_MED);
    tsl.setTiming(TSL2591_INTEGRATIONTIME_100MS);
  }
#endif
  // The servo stays released until the first HOME or MOVE.
  lastCommandMs = millis();
  Serial.print(F("EVENT: READY, VERSION: "));
  Serial.print(FIRMWARE_VERSION);
  Serial.print(F(", SENSOR: "));
  Serial.println(sensorName());
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line[lineLength] = '\0';
      lastCommandMs = millis();
      if (lineOverflow) {
        replyError("E_SYNTAX", F("line too long"));
      } else if (lineLength > 0) {
        handleCommand(line);
      }
      lineLength = 0;
      lineOverflow = false;
    } else if (lineLength < LINE_MAX) {
      line[lineLength++] = c;
    } else {
      lineOverflow = true;
    }
  }

  // Command watchdog: a silent host must not leave the servo driven indefinitely.
  if (!estopLatched && isHomed && millis() - lastCommandMs > WATCHDOG_MS) {
    latchEstop("watchdog");
    Serial.print(F("EVENT: WATCHDOG, MSG: no command for "));
    Serial.print(WATCHDOG_S);
    Serial.println(F(" s; servo released; send RESET"));
  }
}
