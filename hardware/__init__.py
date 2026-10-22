"""
Hardware abstraction and digital twin package for AI Laboratory.
"""
from hardware.base import BaseHardwareRig, DataSource, MeasurementResult, RigStatus
from hardware.simulator import PhysicsSimulationRig
from hardware.serial_driver import SerialHardwareRig
from hardware.safety_guard import EmergencyStopActive, HardwareSafetyGuard, SafetyLimits, SafetyViolationError
from hardware.audit import AuditEntry, AuditLog, verify_chain

__all__ = [
    "BaseHardwareRig",
    "DataSource",
    "MeasurementResult",
    "RigStatus",
    "PhysicsSimulationRig",
    "SerialHardwareRig",
    "HardwareSafetyGuard",
    "SafetyLimits",
    "SafetyViolationError",
    "EmergencyStopActive",
    "AuditEntry",
    "AuditLog",
    "verify_chain",
]
