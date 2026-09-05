# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.engine.core import DPEngineCoreProc


def _core(interval: int, step: int) -> DPEngineCoreProc:
    core = object.__new__(DPEngineCoreProc)
    core.prefill_schedule_interval = interval
    core.step_counter = step
    return core


def test_dp_prefill_cadence_waits_before_first_release():
    assert _core(2, 0)._should_throttle_prefills()
    assert not _core(2, 1)._should_throttle_prefills()
    assert _core(2, 2)._should_throttle_prefills()
    assert not _core(2, 3)._should_throttle_prefills()


def test_dp_prefill_cadence_one_never_throttles():
    assert not _core(1, 0)._should_throttle_prefills()
    assert not _core(1, 9)._should_throttle_prefills()
