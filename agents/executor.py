"""
Hardware Executor Agent.
Executes physical or simulated actuation, handles settling times,
and verifies sensor response telemetry.
"""
import logging
from typing import Dict, Any
from agents.tools import LabTools

logger = logging.getLogger(__name__)

class HardwareExecutorAgent:
    """
    Agent responsible for translating experimental directives into validated hardware actions.
    """

    def __init__(self, lab_tools: LabTools):
        self.tools = lab_tools

    def execute_step(self, target_position: float, rationale: str = "") -> Dict[str, Any]:
        """
        Executes a safe actuator movement and takes calibrated sensor readings.
        """
        logger.info(f"Executor: Actuating to {target_position} ({rationale})...")
        res = self.tools.tool_take_measurement(target_position)

        if res["status"] == "ERROR":
            logger.error(f"Executor failed during actuation to {target_position}: {res.get('message')}")
            return {
                "success": False,
                "position": target_position,
                "error": res.get("message")
            }

        logger.info(f"Executor: Recorded {res['sensor_value']} at position {res['position']}.")
        return {
            "success": True,
            "position": res["position"],
            "sensor_value": res["sensor_value"],
            "total_samples": res["total_samples"]
        }
