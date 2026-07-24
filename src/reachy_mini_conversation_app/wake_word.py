"""On-device "hey jarvis" wake word gate for the microphone stream."""

import time
import logging

import numpy as np
import openwakeword
from numpy.typing import NDArray
from openwakeword.model import Model

from reachy_mini_conversation_app.streaming import audio_to_int16


logger = logging.getLogger(__name__)

CHUNK_SAMPLES = 1280  # 80 ms at openwakeword's required 16 kHz rate
DETECTION_THRESHOLD = 0.5
REARM_SECONDS = 60.0


class WakeWordGate:
    """Blocks mic audio until "hey jarvis" is heard; re-arms after conversation inactivity."""

    def __init__(self, rearm_seconds: float = REARM_SECONDS) -> None:
        """Load the bundled hey_jarvis model and start in the armed state."""
        self.rearm_seconds = rearm_seconds
        self._model = self._new_model()
        self._pending = np.empty(0, dtype=np.int16)
        self._awake_at: float | None = None

    @staticmethod
    def _new_model() -> Model:
        return Model(wakeword_model_paths=[openwakeword.models["hey_jarvis"]["model_path"]])

    @property
    def is_awake(self) -> bool:
        """Return True while mic audio is being forwarded to the backend."""
        return self._awake_at is not None

    def arm(self) -> None:
        """Return to the armed state immediately, discarding any awake window."""
        if self._awake_at is None:
            return
        self._awake_at = None
        # Model.reset() keeps ~10 s of feature history, so rebuild for a clean armed state.
        self._model = self._new_model()
        self._pending = np.empty(0, dtype=np.int16)

    def allows(self, audio_frame: NDArray[np.float32], idle_seconds: float, conversation_idle: bool) -> bool:
        """Return True when mic audio may reach the backend; run detection while gated."""
        if self._awake_at is not None:
            if min(idle_seconds, time.monotonic() - self._awake_at) <= self.rearm_seconds or not conversation_idle:
                return True
            self.arm()
            logger.info("Wake word gate re-armed after %.0f s of inactivity", self.rearm_seconds)
        first_channel = audio_frame[:, 0] if audio_frame.ndim == 2 else audio_frame
        self._pending = np.concatenate([self._pending, audio_to_int16(first_channel)])
        detected = False
        while self._pending.size >= CHUNK_SAMPLES:
            chunk, self._pending = self._pending[:CHUNK_SAMPLES], self._pending[CHUNK_SAMPLES:]
            if max(self._model.predict(chunk).values()) >= DETECTION_THRESHOLD:
                detected = True
        if detected:
            self._awake_at = time.monotonic()
            logger.info("Wake word detected; forwarding mic audio")
        return False
