"""
PySerial Hardware Driver for Arduino Uno and ESP32 Microcontrollers.
Handles serial communication, ASCII protocol handshakes, and hardware command parsing.
"""
import time
import logging
from typing import Optional
from datetime import datetime
from hardware.base import BaseHardwareRig, MeasurementResult, RigStatus
from config import HardwareConfig

logger = logging.getLogger(__name__)

class SerialHardwareRig(BaseHardwareRig):
    """
    Physical hardware interface connecting to Arduino/ESP32 firmware via USB serial.
    Protocol:
      Host -> MCU: 'PING\n'        MCU -> Host: 'PONG\n'
      Host -> MCU: 'HOME\n'        MCU -> Host: 'STATUS: OK, POS: 0.0\n'
      Host -> MCU: 'MOVE 45.0\n'   MCU -> Host: 'STATUS: OK, POS: 45.0\n'
      Host -> MCU: 'READ\n'        MCU -> Host: 'STATUS: OK, DATA: 450.2\n'
    """

    def __init__(self, hw_config: Optional[HardwareConfig] = None):
        self.config = hw_config or HardwareConfig()
        self._serial = None
        self._connected = False
        self._is_homed = False
        self._current_position = 0.0
        self._is_busy = False

    def connect(self) -> bool:
        try:
            import serial
        except ImportError:
            raise ImportError("pyserial is required for physical hardware. Run `pip install pyserial`.")

        try:
            logger.info(f"Connecting to serial port {self.config.port} at {self.config.baud_rate} baud...")
            self._serial = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baud_rate,
                timeout=self.config.timeout_seconds,
                write_timeout=self.config.timeout_seconds
            )
            # Give Arduino bootloader time to reset after DTR toggle
            time.sleep(2.0)
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()

            # Handshake
            if self._send_command("PING") == "PONG":
                self._connected = True
                logger.info("Hardware handshake successful.")
                return True
            else:
                logger.warning("No PONG response from microcontroller.")
                self._connected = True # Port opened, continue with caution
                return True
        except Exception as e:
            logger.error(f"Failed to connect to hardware on {self.config.port}: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except Exception as e:
                logger.error(f"Error closing serial port: {e}")
        self._connected = False

    def is_connected(self) -> bool:
        return bool(self._serial and self._serial.is_open and self._connected)

    def _send_command(self, cmd: str) -> str:
        if not self._serial or not self._serial.is_open:
            raise ConnectionError("Hardware serial port is not open.")

        self._is_busy = True
        try:
            payload = (cmd.strip() + "\n").encode("utf-8")
            self._serial.write(payload)
            self._serial.flush()
            response = self._serial.readline().decode("utf-8", errors="replace").strip()
            return response
        finally:
            self._is_busy = False

    def home(self) -> bool:
        res = self._send_command("HOME")
        if "STATUS: OK" in res:
            self._current_position = 0.0
            self._is_homed = True
            return True
        return False

    def move_actuator(self, position: float) -> bool:
        cmd = f"MOVE {position:.2f}"
        res = self._send_command(cmd)
        if "STATUS: OK" in res:
            self._current_position = float(position)
            return True
        elif "ERROR" in res:
            raise RuntimeError(f"Microcontroller error during MOVE: {res}")
        return False

    def read_sensor(self) -> MeasurementResult:
        res = self._send_command("READ")
        # Expected format: "STATUS: OK, DATA: 450.2"
        if "STATUS: OK" in res and "DATA:" in res:
            try:
                data_part = res.split("DATA:")[1].strip()
                val = float(data_part.split()[0])
                return MeasurementResult(
                    timestamp=datetime.utcnow(),
                    actuator_position=self._current_position,
                    sensor_value=val,
                    units_actuator="degrees",
                    units_sensor="lux",
                    is_valid=True,
                    metadata={"raw_response": res}
                )
            except Exception as e:
                raise ValueError(f"Failed to parse sensor reading '{res}': {e}")
        raise RuntimeError(f"Failed to obtain valid sensor reading. Response: {res}")

    def get_status(self) -> RigStatus:
        return RigStatus(
            is_connected=self.is_connected(),
            is_homed=self._is_homed,
            current_position=self._current_position,
            is_busy=self._is_busy,
            telemetry={"port": self.config.port, "baud": self.config.baud_rate}
        )
