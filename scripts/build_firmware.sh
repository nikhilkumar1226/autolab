#!/usr/bin/env bash
# Build the rig firmware with arduino-cli.
#   scripts/build_firmware.sh            # SENSOR_EMULATED -> firmware/build/ (what wokwi.toml loads)
#   scripts/build_firmware.sh LDR        # SENSOR_LDR      -> firmware/build-ldr/
#   scripts/build_firmware.sh TSL2591    # SENSOR_TSL2591  -> firmware/build-tsl2591/ (needs "Adafruit TSL2591 Library")
# One-time setup: arduino-cli core install arduino:avr && arduino-cli lib install Servo "Adafruit TSL2591 Library"
# Upload to a board: arduino-cli upload -p /dev/ttyUSB0 --fqbn arduino:avr:uno --input-dir firmware/build-ldr
set -euo pipefail
SENSOR="${1:-EMULATED}"
case "$SENSOR" in
  EMULATED) OUT="firmware/build" ;;
  LDR) OUT="firmware/build-ldr" ;;
  TSL2591) OUT="firmware/build-tsl2591" ;;
  *) echo "usage: $0 [EMULATED|LDR|TSL2591]" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/.."
arduino-cli compile --fqbn arduino:avr:uno --warnings all \
  --build-property "build.extra_flags=-DSENSOR_${SENSOR}" \
  --output-dir "$OUT" firmware/arduino_rig
echo "Built SENSOR_${SENSOR} into $OUT"
