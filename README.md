# Autonomous Scientific Discovery System
> **Capstone Project:** LLM Agents for Hardware-Grounded Experimental Design & Closed-Loop Science

An autonomous "Self-Driving Laboratory" (SDL) pairing physical hardware (or Digital Twin physics simulations) with multi-agent intelligence and active learning (Bayesian Optimization). The system formulates hypotheses, executes experiments, statistically confirms mathematical laws, and synthesizes publication-grade research papers.

---

## 🌟 Key Features

1. **Digital Twin Simulation Layer**: Full physics and noise modeling (Malus's Law, Inverse Square Law, Beer-Lambert Law) allowing **100% development and verification without physical hardware**.
2. **Seamless Hardware Transition**: Arduino Uno firmware (`firmware/arduino_rig/arduino_rig.ino`: SG90 servo on pin 9; LDR, TSL2591 or emulated sensor). Run on the board or in the Wokwi simulator with `--hardware` (see `docs/wokwi.md`).
3. **Nervous System (Safety Layer)**: Hardcoded boundary checking, angular velocity clamping, and Emergency Stop (`ESTOP`) interceptors preventing hardware damage from AI hallucinations.
4. **Bayesian Optimization & Active Learning**: Gaussian Process regression with epistemic uncertainty sampling ($\arg\max \sigma(x)$) and Expected Improvement (EI) for sample efficiency.
5. **Multi-Agent Architecture**:
   - **Lead Scientist Agent**: Formulates hypotheses and exploration boundaries from natural language research prompts.
   - **Hardware Executor Agent**: Validates safety and triggers physical actuation and sensor reading.
   - **Data Analyst Agent**: Performs multi-model non-linear curve fitting (Trigonometric, Inverse Square, Polynomial, Linear), ranks models via AIC/BIC, and evaluates convergence criteria.
   - **Scientific Writer Agent**: Automatically generates publication-grade 3-panel figures and synthesized research papers in **Markdown** and **LaTeX**.

---

## 📁 Repository Structure

```text
AI-Laboratory/
├── config.py                  # Global system configuration (Simulated vs Hardware mode)
├── hardware/
│   ├── base.py                # Abstract Base Class and telemetry schemas
│   ├── simulator.py           # Digital Twin physics engine (Malus's Law, Inverse Square Law)
│   ├── serial_driver.py       # PySerial driver for Arduino/ESP32
│   └── safety_guard.py        # Hardware safety boundaries and interceptors
├── optimization/
│   └── bayesian_opt.py        # Gaussian Process Active Learning engine
├── agents/
│   ├── models.py              # Pydantic data schemas (Hypothesis, AnalysisResult, etc.)
│   ├── tools.py               # Executable tool wrappers for agents
│   ├── lead_scientist.py      # Hypothesis formulation and strategy
│   ├── executor.py            # Hardware actuation agent
│   ├── analyst.py             # Multi-model fitting and statistical convergence
│   └── orchestrator.py        # Closed-loop autonomous controller
├── reporting/
│   ├── visualizer.py          # Publication-grade 3-panel matplotlib figures
│   └── paper_generator.py     # Markdown and LaTeX research paper synthesis
├── firmware/
│   └── arduino_rig/arduino_rig.ino  # Arduino Uno firmware (servo + LDR / TSL2591 / emulated sensor)
├── tests/                     # Unit and integration test suite
├── main.py                    # Main discovery CLI entrypoint
├── requirements.txt           # Python dependencies
└── pytest.ini                 # Pytest configuration
```

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Autonomous Discovery (Simulation Mode)

Run the autonomous discovery engine on **Malus's Law**:
```bash
python3 main.py --experiment malus_law
```

Run on the **Inverse-Square Law**:
```bash
python3 main.py --experiment inverse_square_law
```

Try an experiment that is not in the registry (dashboard, `streamlit run app/Home.py` → **Live Lab** → Experiment: *✎ Your own law*):
- **By name:** type e.g. "simple pendulum: period against length" and press *Draft the experiment*. The LLM drafts the variables, units, range and law; you check them, tick the confirmation box and press Run. The agents never see the name or the law. Without a real LLM unlocked, an offline table of eight textbook experiments is used.
- **Type the law yourself:** enter any y = f(x), a range and units.

Both run on the digital twin, and the result says whether the law was recovered.

### 3. Output Artifacts
Every run automatically produces:
- **Publication Plot:** High-resolution 3-panel chart saved in `output/plots/discovery_plot_*.png`
- **Markdown Paper:** Formal research paper saved in `output/reports/scientific_paper_*.md`
- **LaTeX Source:** LaTeX publication paper saved in `output/reports/scientific_paper_*.tex`

---

## 🔌 Transitioning to Physical Hardware

When your Arduino / ESP32 and motor/sensor hardware arrive:
1. Build and upload the firmware (`scripts/build_firmware.sh LDR`, then `arduino-cli upload`; see `docs/wokwi.md`).
2. Connect the microcontroller via USB.
3. Run the discovery loop with the `--hardware` flag:
   ```bash
   python3 main.py --hardware --port /dev/ttyUSB0 --baud 115200
   ```

No code modifications to the agents, optimizer, or reporting layers are needed!
