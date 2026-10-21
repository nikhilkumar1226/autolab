/*
  =============================================================================
  Autonomous Scientific Discovery System - Microcontroller Firmware
  Hardware: Arduino Uno / Nano / ESP32
  Actuator: SG90 Servo / NEMA 17 Stepper (A4988 driver)
  Sensors:  Adafruit TSL2591 / Analog Photoresistor / MLX90614
  
  Serial Protocol (115200 baud):
    Host: PING       -> MCU: PONG
    Host: HOME       -> MCU: STATUS: OK, POS: 0.00
    Host: MOVE <val> -> MCU: STATUS: OK, POS: <val> (or STATUS: ERROR)
    Host: READ       -> MCU: STATUS: OK, DATA: <float>
    Host: STATUS     -> MCU: STATUS: OK, POS: <val>, HOMED: 1
  =============================================================================
*/

#include <Arduino.h>
#include <Servo.h>
#include <Wire.h>

// Pin Definitions
const int SERVO_PIN = 9;
const int ANALOG_SENSOR_PIN = A0; // Fallback analog light sensor pin

// Safety Constraints
const float MIN_ANGLE = 0.0;
const float MAX_ANGLE = 180.0;

Servo polarizationServo;
float currentAngle = 0.0;
bool isHomed = false;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
    // Wait for USB Serial
  }
  
  // Attach servo and start at 0 degrees
  polarizationServo.attach(SERVO_PIN);
  polarizationServo.write(0);
  currentAngle = 0.0;
  isHomed = true;
  
  // Initialize I2C for digital sensor (TSL2591 or MLX90614)
  Wire.begin();
}

float readLightIntensity() {
  // Sample multiple readings to filter electrical noise
  long sum = 0;
  const int NUM_SAMPLES = 10;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    sum += analogRead(ANALOG_SENSOR_PIN);
    delay(5);
  }
  float avgRaw = (float)sum / NUM_SAMPLES;
  // Convert 10-bit ADC (0-1023) to simulated lux / calibrated intensity
  float calibratedLux = (avgRaw / 1023.0) * 1000.0;
  return calibratedLux;
}

void processCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  if (cmd.equalsIgnoreCase("PING")) {
    Serial.println("PONG");
  } 
  else if (cmd.equalsIgnoreCase("HOME")) {
    polarizationServo.write(0);
    currentAngle = 0.0;
    isHomed = true;
    delay(200);
    Serial.println("STATUS: OK, POS: 0.00");
  } 
  else if (cmd.startsWith("MOVE") || cmd.startsWith("move")) {
    int spaceIdx = cmd.indexOf(' ');
    if (spaceIdx == -1) {
      Serial.println("STATUS: ERROR, MSG: Missing angle argument");
      return;
    }
    String angleStr = cmd.substring(spaceIdx + 1);
    float targetAngle = angleStr.toFloat();

    // Bounds checking
    if (targetAngle < MIN_ANGLE || targetAngle > MAX_ANGLE) {
      Serial.print("STATUS: ERROR, MSG: Angle out of bounds [0, 180]: ");
      Serial.println(targetAngle);
      return;
    }

    // Smooth transit
    polarizationServo.write((int)targetAngle);
    delay(150); // allow mechanical settling
    currentAngle = targetAngle;
    Serial.print("STATUS: OK, POS: ");
    Serial.println(currentAngle, 2);
  } 
  else if (cmd.equalsIgnoreCase("READ")) {
    float val = readLightIntensity();
    Serial.print("STATUS: OK, DATA: ");
    Serial.println(val, 2);
  } 
  else if (cmd.equalsIgnoreCase("STATUS")) {
    Serial.print("STATUS: OK, POS: ");
    Serial.print(currentAngle, 2);
    Serial.print(", HOMED: ");
    Serial.println(isHomed ? "1" : "0");
  } 
  else {
    Serial.print("STATUS: ERROR, MSG: Unknown command '");
    Serial.print(cmd);
    Serial.println("'");
  }
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    processCommand(input);
  }
}
