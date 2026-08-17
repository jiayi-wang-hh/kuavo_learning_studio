from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Condition
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ActionEntry:
    """An action tagged with the physical timestep it is intended to control."""

    global_step: int
    chunk_id: int
    chunk_offset: int
    action: np.ndarray


@dataclass(frozen=True)
class InferenceTrigger:
    trigger_global_step: int
    previous_chunk_id: int | None
    previous_chunk_start_global_step: int | None
    executed_offset_at_trigger: int | None

    def as_request_context(self) -> dict[str, int | None]:
        return {
            "trigger_global_step": self.trigger_global_step,
            "previous_chunk_id": self.previous_chunk_id,
            "previous_chunk_start_global_step": self.previous_chunk_start_global_step,
            "executed_offset_at_trigger": self.executed_offset_at_trigger,
        }


@dataclass(frozen=True)
class MergeResult:
    inserted: int
    stale: int
    ready_global_step: int
    first_chunk_offset: int | None


class ActionTimelineBuffer:
    """Thread-safe action queue with an explicit global execution timeline.

    Calls that combine environment access with this buffer still need an external
    timeline lock. The lock makes observation capture, action execution, and
    queue replacement occur between control steps rather than during one.
    """

    def __init__(self, maxlen: int):
        self.maxlen = max(1, int(maxlen))
        self._entries: deque[ActionEntry] = deque()
        self._next_global_step = 0
        self._cond = Condition()

    def clear(self, *, reset_global_step: bool = True) -> None:
        with self._cond:
            self._entries.clear()
            if reset_global_step:
                self._next_global_step = 0
            self._cond.notify_all()

    def qsize(self) -> int:
        with self._cond:
            return len(self._entries)

    def current_global_step(self) -> int:
        with self._cond:
            return self._next_global_step

    def snapshot_trigger(self) -> InferenceTrigger:
        with self._cond:
            trigger_step = self._next_global_step
            if not self._entries:
                return InferenceTrigger(trigger_step, None, None, None)

            first = self._entries[0]
            if first.global_step != trigger_step:
                raise RuntimeError(
                    "Action timeline is not contiguous at trigger: "
                    f"next={trigger_step}, first={first.global_step}"
                )
            return InferenceTrigger(
                trigger_global_step=trigger_step,
                previous_chunk_id=first.chunk_id,
                previous_chunk_start_global_step=first.global_step - first.chunk_offset,
                executed_offset_at_trigger=first.chunk_offset,
            )

    def wait_for_size(self, size: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._cond:
            while len(self._entries) < size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=remaining)
            return True

    def wait_for_action(self, timeout: float) -> bool:
        return self.wait_for_size(1, timeout)

    def pop_next(self) -> ActionEntry | None:
        """Pop without advancing time; call mark_step_executed after env.step."""
        with self._cond:
            if not self._entries:
                return None
            entry = self._entries.popleft()
            if entry.global_step != self._next_global_step:
                raise RuntimeError(
                    "Action timeline is not contiguous at pop: "
                    f"next={self._next_global_step}, entry={entry.global_step}"
                )
            return entry

    def mark_step_executed(self) -> int:
        with self._cond:
            self._next_global_step += 1
            self._cond.notify_all()
            return self._next_global_step

    def replace_with_chunk(
        self,
        actions: Sequence[Any],
        *,
        chunk_id: int,
        chunk_start_global_step: int,
        execution_horizon: int,
    ) -> MergeResult:
        """Atomically replace queued future actions and remove the stale prefix.

        ``actions[k]`` is defined at ``chunk_start_global_step + k``. At merge
        time, all actions before ``_next_global_step`` are stale. The replacement
        therefore begins at the first action that refers to the current physical
        timestep, and is capped to one configured execution horizon.
        """
        if int(execution_horizon) <= 0:
            raise ValueError(f"execution_horizon must be positive, got {execution_horizon}")
        with self._cond:
            ready_step = self._next_global_step
            stale = ready_step - int(chunk_start_global_step)
            if stale < 0:
                raise ValueError(
                    f"Chunk starts in the future: start={chunk_start_global_step}, ready={ready_step}"
                )
            if stale >= len(actions):
                return MergeResult(0, stale, ready_step, None)

            count = min(
                int(execution_horizon),
                len(actions) - stale,
                self.maxlen,
            )
            replacement = deque(
                ActionEntry(
                    global_step=ready_step + index,
                    chunk_id=int(chunk_id),
                    chunk_offset=stale + index,
                    action=np.asarray(actions[stale + index]),
                )
                for index in range(count)
            )
            self._entries = replacement
            self._cond.notify_all()
            return MergeResult(
                inserted=count,
                stale=stale,
                ready_global_step=ready_step,
                first_chunk_offset=stale if count else None,
            )

    def entries_snapshot(self) -> list[ActionEntry]:
        with self._cond:
            return list(self._entries)
