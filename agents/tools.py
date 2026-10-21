"""
Tool interface for agentic execution.
Binds the hardware safety layer, active learning engine, and curve fitters into callable tools.
"""
from typing import Dict, Any, List
import numpy as np
from hardware.safety_guard import HardwareSafetyGuard
from optimization.bayesian_opt import ActiveLearningEngine, AcquisitionStrategy

class LabTools:
    """
    Executable toolbox accessible to Planner, Executor, and Analyst agents.
    """

    def __init__(self, safety_guard: HardwareSafetyGuard, optimizer: ActiveLearningEngine):
        self.safety_guard = safety_guard
        self.optimizer = optimizer
        self.history_x: List[float] = []
        self.history_y: List[float] = []

    def tool_move_actuator(self, position: float) -> Dict[str, Any]:
        """
        Moves the physical or simulated actuator to a designated target with safety checks.
        """
        try:
            res = self.safety_guard.safe_move(position)
            return {"status": "SUCCESS", "position": res["new_position"]}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    def tool_read_sensor(self) -> Dict[str, Any]:
        """
        Reads the sensor telemetry at the current position.
        """
        try:
            measurement = self.safety_guard.safe_read()
            return {
                "status": "SUCCESS",
                "position": measurement.actuator_position,
                "sensor_value": round(measurement.sensor_value, 4),
                "units": measurement.units_sensor,
                "valid": measurement.is_valid
            }
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    def tool_take_measurement(self, position: float) -> Dict[str, Any]:
        """
        Compound tool: Moves to the target position, lets the system settle, and records sensor data.
        """
        move_res = self.tool_move_actuator(position)
        if move_res["status"] == "ERROR":
            return move_res

        read_res = self.tool_read_sensor()
        if read_res["status"] == "ERROR":
            return read_res

        pos = read_res["position"]
        val = read_res["sensor_value"]

        self.history_x.append(pos)
        self.history_y.append(val)

        return {
            "status": "SUCCESS",
            "position": pos,
            "sensor_value": val,
            "total_samples": len(self.history_x)
        }

    def tool_suggest_next_point(self, strategy: str = "uncertainty") -> Dict[str, Any]:
        """
        Invokes Bayesian Optimization / Active Learning to calculate the most statistically
        informative next parameter coordinate.
        """
        if len(self.history_x) < 2:
            # Not enough points for GP, propose equidistant or random
            span = self.safety_guard.config.max_actuator_value - self.safety_guard.config.min_actuator_value
            mid = self.safety_guard.config.min_actuator_value + span / 2.0
            return {
                "status": "SUCCESS",
                "suggested_position": round(mid, 2),
                "strategy": "initial_midpoint",
                "mean_sigma": 1.0
            }

        strat = AcquisitionStrategy(strategy) if strategy in [s.value for s in AcquisitionStrategy] else AcquisitionStrategy.UNCERTAINTY
        X_arr = np.array(self.history_x)
        y_arr = np.array(self.history_y)

        suggestion = self.optimizer.suggest_next_sample(X_arr, y_arr, strategy=strat)
        return {"status": "SUCCESS", **suggestion}
