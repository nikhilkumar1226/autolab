"""
Hardware Executor Agent.
Executes physical or simulated actuation through the safety guard
and reports the outcome of every attempt.
"""
import logging
from typing import Optional
from agents.models import ExecutionOutcome
from agents.tools import LabTools

logger = logging.getLogger(__name__)

class HardwareExecutorAgent:
    """
    Agent responsible for translating experimental directives into validated hardware actions.
    """

    def __init__(self, lab_tools: LabTools):
        self.tools = lab_tools

    def execute_step(self, target_position: float, rationale: str = "", confirm: Optional[float] = None) -> ExecutionOutcome:
        """
        Executes a safe actuator movement and takes a sensor reading. `confirm` is an earlier
        reading at the same position that was flagged as an outlier (see LabTools).
        """
        logger.info("Executor: actuating to %s (%s)", target_position, rationale)
        outcome = self.tools.tool_take_measurement(target_position, confirm=confirm)

        if outcome.status == "rejected":
            logger.warning("Executor: safety guard rejected %s: %s", target_position, outcome.message)
        elif outcome.status == "anomaly":
            logger.warning("Executor: reading at %s flagged: %s", target_position, ", ".join(outcome.flags))
        elif outcome.status == "error":
            logger.error("Executor: measurement at %s failed: %s", target_position, outcome.message)
        else:
            logger.info("Executor: recorded %s at position %s", outcome.value, outcome.position)
        return outcome
