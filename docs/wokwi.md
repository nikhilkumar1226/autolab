# Firmware-in-the-loop with Wokwi

The Wokwi simulator runs the real rig firmware (`firmware/arduino_rig/arduino_rig.ino`) on a
simulated Arduino Uno with an SG90 servo on pin 9 and an LDR module on A0. Wokwi forwards the
board's serial port over RFC2217 on port 4000, and the unchanged Python driver talks to it as it
would to a USB board. Runs are labelled **FIRMWARE-IN-THE-LOOP (Wokwi)**.

The firmware for Wokwi is the `SENSOR_EMULATED` build. Nothing optical is simulated, so the
firmware computes the light intensity on the chip from the servo angle. It uses Malus's law with
the same values as the digital twin: 1000 lux amplitude, maximum at 20°, 15 lux ambient, and
10 lux Gaussian noise. What this tests is the real firmware, its serial protocol and its safety
checks (bounds, step limit, watchdog, ESTOP and RESET), with the Python driver and the whole
AutoLab loop on top. It does not test real optics.

## One-time setup

1. Install the **Wokwi Simulator** extension in VS Code.
2. Press F1 and choose **Wokwi: Request a New License**. Sign in; the free licence is enough.
3. **Nothing to build.** The built firmware is already in the repository (`firmware/build/arduino_rig.ino.hex` and `.elf`), and `wokwi.toml` points at it. Skip the build unless you changed the sketch. (Without arduino-cli, `scripts/build_firmware.sh` fails with "arduino-cli: command not found"; that is expected and does not matter.)

Only to rebuild it after changing the sketch, install [arduino-cli](https://arduino.github.io/arduino-cli/) and run:

```bash
arduino-cli core install arduino:avr
arduino-cli lib install Servo "Adafruit TSL2591 Library"
scripts/build_firmware.sh            # SENSOR_EMULATED -> firmware/build/ (what wokwi.toml loads)
```

## Run AutoLab against the simulated board

1. Open the `autolab` folder in VS Code. `wokwi.toml` and `diagram.json` are in the project root.
2. Press F1 and choose **Wokwi: Start Simulator**. The diagram shows the Uno, the servo and the LDR module. Keep the simulator tab visible: Wokwi pauses when its tab is hidden.
3. In a terminal, with the virtual environment active:

   ```bash
   python main.py --experiment malus_law --hardware --port rfc2217://localhost:4000 --provider mock
   ```

   The console banner shows `DATA SOURCE: FIRMWARE-IN-THE-LOOP (Wokwi)`. The servo turns for every measurement. The run record, figure and report carry the same label.
4. For the black book, take a screenshot of the Wokwi circuit while the run is going.

`--hardware` supports `malus_law` only, because the rig is the polariser experiment.

## Checking the link by hand

Any serial terminal that speaks RFC2217 can talk to the board, for example:

```bash
python -m serial.tools.miniterm rfc2217://localhost:4000 115200
```

Type `PING`, `VERSION`, `HOME`, `MOVE 45`, `READ`, `STATUS`, `ESTOP` and `RESET`. These
commands should be refused with an error code: `MOVE 200` (E_BOUNDS), `MOVE abc` (E_SYNTAX),
and any `MOVE` after `ESTOP` (E_ESTOP until `RESET`).

## The real rig

The same driver works over USB:

1. Build the LDR or TSL2591 firmware: `scripts/build_firmware.sh LDR`.
2. Upload it: `arduino-cli upload -p /dev/ttyUSB0 --fqbn arduino:avr:uno --input-dir firmware/build-ldr`.
3. Run with `--hardware --port /dev/ttyUSB0` (Windows: `COM3`).

The run is labelled **PHYSICAL RIG**. The LDR conversion to lux is a nominal GL5528 calibration; check it against a lux meter before quoting lux values.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Run stopped before anything moved: could not connect to the rig on rfc2217://localhost:4000 (… Connection refused)` | The simulator is not running (start it first and wait until the board is drawn), or another program holds port 4000. |
| `arduino-cli: command not found` | You ran the build script. It is not needed: the built firmware is committed. |
| Wokwi says it cannot find `wokwi.toml` or the firmware | VS Code has a different folder open. Open the `autolab` folder itself (File → Open Folder). |
| Wokwi asks for a licence | F1 → **Wokwi: Request a New License**, finish the sign-in in the browser, and let the browser return to VS Code. |
| Every command times out | The Wokwi tab is in the background (the simulation pauses), or the firmware did not load. Check the `firmware` path in `wokwi.toml`. |
| `firmware E_ESTOP` | The firmware's E-stop is latched (an `ESTOP` command, or 300 s without a command). The run aborts; start a new run, or send `RESET`. |
