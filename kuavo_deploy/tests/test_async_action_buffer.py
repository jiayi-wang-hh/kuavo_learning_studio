from __future__ import annotations

import numpy as np

from kuavo_deploy.src.eval.async_action_buffer import ActionTimelineBuffer


def _chunk(start: int, size: int = 32) -> list[np.ndarray]:
    return [np.asarray([start + index], dtype=np.float32) for index in range(size)]


def _execute(buffer: ActionTimelineBuffer, count: int) -> None:
    for _ in range(count):
        entry = buffer.pop_next()
        assert entry is not None
        buffer.mark_step_executed()


def test_trigger_tracks_active_chunk_offset() -> None:
    buffer = ActionTimelineBuffer(maxlen=32)
    buffer.replace_with_chunk(
        _chunk(0), chunk_id=7, chunk_start_global_step=0, execution_horizon=16
    )
    _execute(buffer, 13)

    trigger = buffer.snapshot_trigger()

    assert trigger.trigger_global_step == 13
    assert trigger.previous_chunk_id == 7
    assert trigger.previous_chunk_start_global_step == 0
    assert trigger.executed_offset_at_trigger == 13


def test_ready_merge_drops_stale_prefix_and_replaces_future_atomically() -> None:
    buffer = ActionTimelineBuffer(maxlen=32)
    buffer.replace_with_chunk(
        _chunk(0), chunk_id=0, chunk_start_global_step=0, execution_horizon=16
    )
    _execute(buffer, 13)
    trigger = buffer.snapshot_trigger()

    # Two old actions execute while inference is in flight. A[15] remains queued.
    _execute(buffer, 2)
    result = buffer.replace_with_chunk(
        _chunk(100),
        chunk_id=1,
        chunk_start_global_step=trigger.trigger_global_step,
        execution_horizon=16,
    )

    entries = buffer.entries_snapshot()
    assert result.stale == 2
    assert result.first_chunk_offset == 2
    assert result.inserted == 16
    assert entries[0].global_step == 15
    assert entries[0].chunk_id == 1
    assert entries[0].chunk_offset == 2
    assert entries[0].action.item() == 102
    assert [entry.global_step for entry in entries] == list(range(15, 31))


def test_three_step_latency_maps_b3_to_global_boundary() -> None:
    buffer = ActionTimelineBuffer(maxlen=32)
    buffer.replace_with_chunk(
        _chunk(0), chunk_id=0, chunk_start_global_step=0, execution_horizon=16
    )
    _execute(buffer, 13)
    trigger = buffer.snapshot_trigger()
    _execute(buffer, 3)

    result = buffer.replace_with_chunk(
        _chunk(100),
        chunk_id=1,
        chunk_start_global_step=trigger.trigger_global_step,
        execution_horizon=16,
    )

    first = buffer.entries_snapshot()[0]
    assert result.stale == 3
    assert first.global_step == 16
    assert first.chunk_offset == 3
    assert first.action.item() == 103


def test_fully_stale_response_is_rejected_without_overwriting_queue() -> None:
    buffer = ActionTimelineBuffer(maxlen=64)
    buffer.replace_with_chunk(
        _chunk(0, 40), chunk_id=0, chunk_start_global_step=0, execution_horizon=40
    )
    _execute(buffer, 32)
    before = buffer.entries_snapshot()

    result = buffer.replace_with_chunk(
        _chunk(100, 16), chunk_id=1, chunk_start_global_step=0, execution_horizon=16
    )

    assert result.inserted == 0
    after = buffer.entries_snapshot()
    assert len(after) == len(before)
    assert all(current is original for current, original in zip(after, before))
