"""Tests for the wake word gate."""

from types import SimpleNamespace

import numpy as np
import pytest

from reachy_mini_conversation_app import wake_word
from reachy_mini_conversation_app.wake_word import CHUNK_SAMPLES, REARM_SECONDS, WakeWordGate


class _StubModel:
    """Stands in for openwakeword.Model with a controllable score."""

    instances: list["_StubModel"] = []
    score = 0.0

    def __init__(self, wakeword_model_paths: list[str] | None = None) -> None:
        self.predict_calls: list[np.ndarray] = []
        _StubModel.instances.append(self)

    def predict(self, chunk: np.ndarray) -> dict[str, float]:
        self.predict_calls.append(chunk)
        return {"hey_jarvis_v0.1": _StubModel.score}


@pytest.fixture
def stub_model(monkeypatch: pytest.MonkeyPatch) -> type[_StubModel]:
    """Replace the openwakeword model with a controllable stub."""
    _StubModel.instances = []
    _StubModel.score = 0.0
    monkeypatch.setattr(wake_word, "Model", _StubModel)
    return _StubModel


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Give the wake word module a controllable monotonic clock."""
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(wake_word, "time", SimpleNamespace(monotonic=lambda: clock.now))
    return clock


def _frame(samples: int, value: float = 0.0) -> np.ndarray:
    return np.full(samples, value, dtype=np.float32)


def test_frames_accumulate_into_full_chunks(stub_model: type[_StubModel]) -> None:
    """Predict runs once per full 1280-sample chunk, not per mic frame."""
    gate = WakeWordGate()
    model = stub_model.instances[0]

    assert gate.allows(_frame(CHUNK_SAMPLES // 2), 0.0, True) is False
    assert model.predict_calls == []

    assert gate.allows(_frame(CHUNK_SAMPLES // 2), 0.0, True) is False
    assert len(model.predict_calls) == 1
    assert model.predict_calls[0].size == CHUNK_SAMPLES


def test_stereo_frames_use_first_channel_as_int16(stub_model: type[_StubModel]) -> None:
    """Stereo input is reduced to the first channel and converted to int16."""
    gate = WakeWordGate()
    stereo = np.stack([_frame(CHUNK_SAMPLES, 0.5), _frame(CHUNK_SAMPLES, -0.5)], axis=1)

    gate.allows(stereo, 0.0, True)

    chunk = stub_model.instances[0].predict_calls[0]
    assert chunk.dtype == np.int16
    assert np.all(chunk == np.int16(0.5 * 32767))


def test_detection_opens_gate_from_next_frame(stub_model: type[_StubModel], fake_clock: SimpleNamespace) -> None:
    """The detecting frame is withheld; subsequent frames flow without detection."""
    gate = WakeWordGate()
    stub_model.score = 0.9

    assert gate.allows(_frame(CHUNK_SAMPLES), 0.0, True) is False

    stub_model.score = 0.0
    assert gate.allows(_frame(CHUNK_SAMPLES), 0.0, True) is True
    assert len(stub_model.instances[0].predict_calls) == 1  # no detection while awake


def test_awake_window_survives_stale_handler_clock(stub_model: type[_StubModel], fake_clock: SimpleNamespace) -> None:
    """Right after wake, a large handler idle time must not re-arm the gate."""
    gate = WakeWordGate()
    stub_model.score = 0.9
    gate.allows(_frame(CHUNK_SAMPLES), 0.0, True)

    assert gate.allows(_frame(CHUNK_SAMPLES), 999.0, True) is True


def test_rearm_after_inactivity_rebuilds_model(stub_model: type[_StubModel], fake_clock: SimpleNamespace) -> None:
    """After the inactivity window the gate closes and gets a fresh model."""
    gate = WakeWordGate()
    stub_model.score = 0.9
    gate.allows(_frame(CHUNK_SAMPLES), 0.0, True)
    stub_model.score = 0.0

    fake_clock.now += REARM_SECONDS + 1
    assert gate.allows(_frame(CHUNK_SAMPLES), REARM_SECONDS + 1, True) is False
    assert len(stub_model.instances) == 2  # fresh model: reset() keeps stale feature history

    stub_model.score = 0.9
    gate.allows(_frame(CHUNK_SAMPLES), REARM_SECONDS + 2, True)
    stub_model.score = 0.0
    assert gate.allows(_frame(CHUNK_SAMPLES), 0.0, True) is True


def test_no_rearm_while_conversation_active(stub_model: type[_StubModel], fake_clock: SimpleNamespace) -> None:
    """An active conversation (listening/moving) blocks re-arm past the deadline."""
    gate = WakeWordGate()
    stub_model.score = 0.9
    gate.allows(_frame(CHUNK_SAMPLES), 0.0, True)
    stub_model.score = 0.0

    fake_clock.now += REARM_SECONDS + 1
    assert gate.allows(_frame(CHUNK_SAMPLES), REARM_SECONDS + 1, False) is True
    assert len(stub_model.instances) == 1


def test_manual_arm_regates_and_rebuilds_model(stub_model: type[_StubModel]) -> None:
    """arm() closes an awake gate immediately and swaps in a fresh model."""
    gate = WakeWordGate()
    stub_model.score = 0.9
    gate.allows(_frame(CHUNK_SAMPLES), 0.0, True)
    stub_model.score = 0.0
    assert gate.is_awake is True

    gate.arm()

    assert gate.is_awake is False
    assert len(stub_model.instances) == 2
    assert gate.allows(_frame(CHUNK_SAMPLES), 0.0, True) is False


def test_manual_arm_is_noop_while_already_armed(stub_model: type[_StubModel]) -> None:
    """arm() on an armed gate must not rebuild the model."""
    gate = WakeWordGate()

    gate.arm()

    assert gate.is_awake is False
    assert len(stub_model.instances) == 1


def test_rearm_timeout_change_applies_immediately(stub_model: type[_StubModel], fake_clock: SimpleNamespace) -> None:
    """Shrinking rearm_seconds at runtime re-arms an already-awake gate sooner."""
    gate = WakeWordGate(rearm_seconds=REARM_SECONDS)
    stub_model.score = 0.9
    gate.allows(_frame(CHUNK_SAMPLES), 0.0, True)
    stub_model.score = 0.0

    gate.rearm_seconds = 10.0
    fake_clock.now += 11.0
    assert gate.allows(_frame(CHUNK_SAMPLES), 11.0, True) is False
    assert gate.is_awake is False


def test_real_model_scores_silence_below_threshold() -> None:
    """Smoke test against the real bundled model: silence must not wake the gate."""
    gate = WakeWordGate()

    assert gate.allows(_frame(CHUNK_SAMPLES), 0.0, True) is False
    assert gate._awake_at is None
