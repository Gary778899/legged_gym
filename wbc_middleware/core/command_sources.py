from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np


class CommandSource(ABC):
    @abstractmethod
    def get_command(self) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        return None


class ZeroCommandSource(CommandSource):
    def __init__(self):
        self._zero_command = np.zeros(3, dtype=np.float32)

    def get_command(self) -> np.ndarray:
        return self._zero_command.copy()


@dataclass(frozen=True)
class ScriptedCommandSegment:
    duration_s: float
    command: np.ndarray


class ScriptedCommandSource(CommandSource):
    def __init__(
        self,
        segments: list[ScriptedCommandSegment],
        *,
        time_fn: Callable[[], float] | None = None,
    ):
        if not segments:
            raise ValueError("ScriptedCommandSource requires at least one segment")
        self._segments = segments
        self._time_fn = time_fn or time.monotonic
        self._segment_end_times_s = self._build_segment_end_times(segments)
        self._start_time_s = self._time_fn()

    def get_command(self) -> np.ndarray:
        elapsed_s = max(0.0, float(self._time_fn()) - self._start_time_s)
        segment_index = self._find_segment_index(elapsed_s)
        return self._segments[segment_index].command.copy()

    def reset(self) -> None:
        self._start_time_s = self._time_fn()

    @staticmethod
    def _build_segment_end_times(segments: list[ScriptedCommandSegment]) -> np.ndarray:
        durations_s = np.asarray([segment.duration_s for segment in segments], dtype=np.float64)
        return np.cumsum(durations_s, dtype=np.float64)

    def _find_segment_index(self, elapsed_s: float) -> int:
        segment_index = int(np.searchsorted(self._segment_end_times_s, elapsed_s, side="right"))
        return min(segment_index, len(self._segments) - 1)


def _coerce_command_vector(command_value: Any) -> np.ndarray:
    command = np.asarray(command_value, dtype=np.float32)
    if command.shape != (3,):
        raise ValueError(f"command must have shape (3,), got {command.shape}")
    return command


def _parse_scripted_segments(config: dict[str, Any]) -> list[ScriptedCommandSegment]:
    segment_values = config.get("segments", config.get("script", config.get("commands")))
    if not isinstance(segment_values, list) or not segment_values:
        raise ValueError("scripted command source requires a non-empty segment list")

    segments: list[ScriptedCommandSegment] = []
    for index, segment_value in enumerate(segment_values):
        if not isinstance(segment_value, dict):
            raise ValueError(f"scripted segment {index} must be a mapping")

        duration_s = float(segment_value["duration_s"])
        if duration_s <= 0.0:
            raise ValueError(f"scripted segment {index} duration_s must be > 0")

        command = _coerce_command_vector(segment_value["command"])
        segments.append(ScriptedCommandSegment(duration_s=duration_s, command=command))

    return segments


def build_command_source(config: dict[str, Any] | None) -> CommandSource:
    config = config or {}
    source_type = str(config.get("type", "zero")).lower()
    if source_type == "zero":
        return ZeroCommandSource()
    if source_type == "scripted":
        try:
            return ScriptedCommandSource(_parse_scripted_segments(config))
        except (KeyError, TypeError, ValueError):
            return ZeroCommandSource()
    if source_type != "zero":
        return ZeroCommandSource()
    return ZeroCommandSource()
