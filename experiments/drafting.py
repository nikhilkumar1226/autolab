"""
Experiments drafted from a name: the operator types the name of an experiment ("simple pendulum: period
against length"), a language model drafts it (variables, units, range and the textbook law), the operator
checks and confirms the draft on the dashboard, and the run then proceeds exactly like any custom experiment.

Rules this module keeps:
- The draft is a proposal. It goes through the same gates as an operator-typed law: the law is compiled by the
  whitelist gate (never eval, exec or sympy) and build_custom_experiment checks the range, the finiteness and
  the span. Variable names that would hint at the law are refused, because the agents see them.
- The operator, not the model, sets the ground truth: nothing runs until the operator has confirmed the draft
  (the dashboard enforces this) and may edit every field first. The run record says the law was drafted by a
  model, from which name, by which model, and whether the operator edited it.
- The Drafter takes no part in the run. The discovery agents never see the experiment name, the draft's
  apparatus text or the law: they get the generic custom-experiment description, the variable names, units and
  the range, as for any custom experiment.
- Digital twin only. A drafted experiment is simulated; it can never drive hardware.

With the mock provider a small offline table of textbook experiments stands in for the model, so tests and the
offline demo work; an unknown name is reported as not available offline. With provider "none" there is no
drafting (DraftingUnavailable).
"""
import re
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from agents.prompts import DRAFT_PROMPT_VERSION, DRAFTER_SYSTEM, to_payload
from experiments.custom import CustomExperimentError, CustomLaw, build_custom_experiment
from experiments.registry import ExperimentSpec
from llm.client import LLMClient, LLMUnavailableError, NotAvailable
from llm.types import LLMCallRecord

MAX_NAME_CHARS = 120
MAX_REASKS = 2            # re-asks with the problem fed back when a draft fails the gates

# Words in a variable name that would hint at the law (the agents see the names).
HINT_WORDS = ("law", "malus", "hooke", "ohm", "newton", "boyle", "charles", "coulomb", "stefan", "boltzmann",
              "shockley", "beer", "lambert", "kepler", "inverse", "square", "exponential", "logarithm", "linear",
              "proportional", "quadratic", "cos", "sin", "exp", "sqrt", "power law", "decay constant", "time constant")


class ExperimentDraft(BaseModel):
    """What the Drafter returns (validated like every agent answer)."""
    model_config = ConfigDict(extra="ignore")

    feasible: bool = True
    reason: str = Field(default="", max_length=400)
    title: str = Field(default="", max_length=120)
    apparatus: str = Field(default="", max_length=600)
    x_name: str = Field(default="input", max_length=60)
    x_units: str = Field(default="a.u.", max_length=24)
    x_min: float = 0.0
    x_max: float = 1.0
    y_name: str = Field(default="response", max_length=60)
    y_units: str = Field(default="a.u.", max_length=24)
    law: str = Field(default="x", max_length=300)
    constants: str = Field(default="", max_length=600)
    noise_pct: float = 1.0

    def to_custom_law(self) -> CustomLaw:
        return CustomLaw(expression=self.law, x_name=self.x_name.strip() or "input", x_units=self.x_units.strip(),
                         x_min=self.x_min, x_max=self.x_max, y_name=self.y_name.strip() or "response",
                         y_units=self.y_units.strip(), noise_pct=min(10.0, max(0.2, float(self.noise_pct))))


class DraftProvenance(BaseModel):
    """Saved in the run record of a drafted experiment."""
    drafted_from: str = Field(description="The experiment name the operator typed")
    provider: str = Field(description="litellm (a real model) or mock (offline table)")
    model: Optional[str] = Field(default=None, description="The model whose draft was used")
    title: str = ""
    apparatus: str = ""
    constants: str = ""
    drafted_law: str = Field(description="The law as the model drafted it")
    edited_by_operator: bool = Field(default=False, description="The operator changed the law, range, names or units")
    confirmed_by_operator: bool = Field(default=False, description="The operator confirmed the draft before the run")
    reasks: int = Field(default=0, description="Drafts rejected by the gates and re-asked")
    calls: List[LLMCallRecord] = Field(default_factory=list, description="Every drafting call, prompt and answer")


class DraftResult(BaseModel):
    ok: bool
    error: Optional[str] = None
    draft: Optional[ExperimentDraft] = None
    provenance: Optional[DraftProvenance] = None


class DraftingUnavailable(RuntimeError):
    """Provider 'none': there is no model to draft with."""


def clean_name(name: str) -> str:
    text = " ".join(str(name).split())
    if len(text) < 3:
        raise ValueError("type the name of an experiment (at least 3 characters)")
    if len(text) > MAX_NAME_CHARS:
        raise ValueError(f"the name is longer than {MAX_NAME_CHARS} characters")
    return text


def hinting_names(draft: ExperimentDraft) -> List[str]:
    """Variable names that contain a word hinting at the law."""
    bad = []
    for label, value in (("x_name", draft.x_name), ("y_name", draft.y_name)):
        low = value.lower()
        for w in HINT_WORDS:
            if re.search(rf"\b{re.escape(w)}\b", low):
                bad.append(f"{label} '{value}' contains '{w}'")
                break
    return bad


def with_provenance(spec: ExperimentSpec, provenance: DraftProvenance) -> ExperimentSpec:
    """The spec with its draft provenance attached, so the run record says where the law came from."""
    return spec.model_copy(update={"draft": provenance.model_dump(mode="json")})


def check_draft(draft: ExperimentDraft) -> ExperimentSpec:
    """The spec of a draft, through the same gates as an operator-typed law. Raises CustomExperimentError."""
    if not draft.feasible:
        raise CustomExperimentError(draft.reason or "the model says this is not a single-variable experiment")
    hints = hinting_names(draft)
    if hints:
        raise CustomExperimentError("variable names must not hint at the law: " + "; ".join(hints))
    try:
        law = draft.to_custom_law()
    except ValueError as exc:
        raise CustomExperimentError(str(exc).splitlines()[-1]) from None
    return build_custom_experiment(law)


# ----------------------------------------------------------------------------- offline stand-in
# The mock is not a language model: a fixed table of textbook experiments, matched by keyword.
MOCK_TABLE: List[Dict[str, Any]] = [
    dict(keys=("pendulum",), title="Simple pendulum: period against length",
         apparatus="A small bob on a light string swings with a small amplitude; a light gate times the period.",
         x_name="pendulum length", x_units="m", x_min=0.1, x_max=2.0, y_name="period", y_units="s",
         law="2*pi*sqrt(x/9.81)", constants="g = 9.81 m/s²", noise_pct=1.0),
    dict(keys=("ohm", "resistor", "current-voltage", "current voltage"), title="Resistor: current against voltage",
         apparatus="A bench supply drives a 470 Ω resistor; a multimeter reads the current.",
         x_name="applied voltage", x_units="V", x_min=0.0, x_max=12.0, y_name="current", y_units="mA",
         law="1000*x/470", constants="R = 470 Ω", noise_pct=1.0),
    dict(keys=("cooling", "cools", "coffee", "hot water"), title="Cooling of hot water",
         apparatus="A beaker of hot water cools in a room at 25 °C; a thermocouple logs its temperature.",
         x_name="elapsed time", x_units="min", x_min=0.0, x_max=60.0, y_name="water temperature", y_units="°C",
         law="25 + 65*exp(-x/15)", constants="room 25 °C, start 90 °C, 15 min characteristic time", noise_pct=1.0),
    dict(keys=("free fall", "falling", "dropped", "drop"), title="Free fall: distance against time",
         apparatus="A steel ball is released from rest; a high-speed camera measures the distance fallen.",
         x_name="time after release", x_units="s", x_min=0.0, x_max=1.5, y_name="distance fallen", y_units="m",
         law="0.5*9.81*x**2", constants="g = 9.81 m/s²", noise_pct=1.0),
    dict(keys=("discharg",), title="Capacitor discharge through a resistor",
         apparatus="A 4.7 µF capacitor charged to 5 V discharges through 10 kΩ; an oscilloscope reads its voltage.",
         x_name="time after switching", x_units="ms", x_min=0.0, x_max=250.0, y_name="capacitor voltage", y_units="V",
         law="5*exp(-x/47)", constants="V0 = 5 V, R = 10 kΩ, C = 4.7 µF (RC = 47 ms)", noise_pct=1.0),
    dict(keys=("gas", "boyle", "syringe"), title="Gas in a syringe: pressure against volume",
         apparatus="A sealed syringe of air at room temperature is compressed; a gauge reads the pressure.",
         x_name="gas volume", x_units="mL", x_min=20.0, x_max=100.0, y_name="gas pressure", y_units="kPa",
         law="5066/x", constants="pV = 5066 kPa·mL (50 mL at 101.3 kPa), constant temperature", noise_pct=1.0),
    dict(keys=("coulomb", "charges", "electrostatic"), title="Force between two charged spheres",
         apparatus="Two small charged spheres on a torsion balance; the separation is varied.",
         x_name="sphere separation", x_units="cm", x_min=2.0, x_max=20.0, y_name="force", y_units="mN",
         law="899/x**2", constants="k q1 q2 = 8.99e-5 N·m² (q = 100 nC each)", noise_pct=1.0),
    dict(keys=("spring", "elastic", "extension"), title="Spring: extension against load",
         apparatus="Masses hang from a steel spring; a ruler reads the extension.",
         x_name="hanging mass", x_units="g", x_min=0.0, x_max=500.0, y_name="extension", y_units="mm",
         law="0.0981*x/0.25", constants="g = 9.81 m/s², spring constant 25 N/m", noise_pct=1.0),
]


def mock_draft(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Offline stand-in for the Drafter: a fixed table matched by keyword."""
    name = str(payload.get("experiment_name", "")).lower()
    for row in MOCK_TABLE:
        if any(k in name for k in row["keys"]):
            out = {k: v for k, v in row.items() if k != "keys"}
            return {"feasible": True, "reason": "found in the offline table of textbook experiments", **out}
    known = ", ".join(r["title"].split(":")[0].lower() for r in MOCK_TABLE)
    return {"feasible": False, "reason": f"the offline mock only knows a few textbook experiments ({known}); "
                                         f"use the real LLM to draft '{payload.get('experiment_name', '')}'"}


# ----------------------------------------------------------------------------- the drafting call
def draft_experiment(name: str, client: LLMClient, mock=None, max_reasks: int = MAX_REASKS,
                     on_progress: Optional[Callable[[str], None]] = None) -> DraftResult:
    """Ask the model for a draft and check it; re-ask with the problem when the draft fails the gates.

    on_progress, if given, receives one short message per step (the dashboard shows them as they happen).
    """
    say = on_progress or (lambda message: None)
    try:
        name = clean_name(name)
    except ValueError as exc:
        return DraftResult(ok=False, error=str(exc))
    if client.provider == "none":
        raise DraftingUnavailable("provider 'none': choose the real LLM or the mock to draft an experiment")
    payload: Dict[str, Any] = {"experiment_name": name}
    first_call = len(client.calls)
    problem, draft = None, None
    who = "the offline mock table" if client.provider == "mock" else " → ".join(client.cfg.chain)
    for attempt in range(max_reasks + 1):
        say(f"Asking {who} to draft “{name}”" + (f" again (attempt {attempt + 1})" if attempt else "") + "…")
        try:
            draft = client.complete_json(DRAFTER_SYSTEM, to_payload(payload), ExperimentDraft, agent="drafter",
                                         purpose="draft_experiment", mock=mock or mock_draft,
                                         prompt_version=DRAFT_PROMPT_VERSION)
        except (LLMUnavailableError, NotAvailable) as exc:
            return DraftResult(ok=False, error=f"no model could draft the experiment: {exc}")
        model = client.calls[-1].model if client.calls else None
        if not draft.feasible:
            say(f"{model or 'The model'} says it cannot be drafted: {draft.reason}")
            return DraftResult(ok=False, draft=draft,
                               error=f"cannot draft '{name}' as a single-variable experiment: {draft.reason}")
        say(f"Draft from {model}: {draft.x_name} ({draft.x_units}) → {draft.y_name} ({draft.y_units}), "
            f"y = {draft.law}")
        say("Checking the law with the whitelist compiler, the range and the variable names…")
        try:
            check_draft(draft)
        except CustomExperimentError as exc:
            problem = str(exc)
            say(f"Rejected: {problem}" + (". Sending the problem back to the model." if attempt < max_reasks else "."))
            payload = {"experiment_name": name, "previous_draft": draft.model_dump(), "previous_problem": problem}
            continue
        say("All checks passed. Check the draft below and confirm it to run.")
        calls = client.calls[first_call:]
        return DraftResult(ok=True, draft=draft, provenance=DraftProvenance(
            drafted_from=name, provider=client.provider, model=calls[-1].model if calls else None,
            title=draft.title, apparatus=draft.apparatus, constants=draft.constants, drafted_law=draft.law,
            reasks=attempt, calls=calls))
    return DraftResult(ok=False, draft=draft,
                       error=f"the drafts failed the checks {max_reasks + 1} times; the last problem: {problem}")
