# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import threading
from collections import namedtuple, deque, Counter
from unittest.mock import patch, MagicMock, call
import pytest
import ms_service_profiler

from .fake_ms_service_profiler import Profiler, Level

# Setup environment
os.environ["VLLM_USE_V1"] = "-1"
sys.modules["ms_service_profiler"].Profiler = Profiler
sys.modules["ms_service_profiler"].Level = Level

from msserviceprofiler.vllm_profiler.vllm_v1 import batch_hookers

# Test helpers
SequenceGroup = namedtuple("SequenceGroup", ["request_id", "prompt_token_ids"])
SchedulerOutput = namedtuple(
    "SchedulerOutput", ["scheduled_new_reqs", "scheduled_cached_reqs", "num_scheduled_tokens", "finished_req_ids"]
)


# Reset profiler before each test
@pytest.fixture(autouse=True, scope="function")
def reset_profiler():
    Profiler.reset()
    yield
    Profiler.reset()


@pytest.fixture
def hook_state():
    state = batch_hookers.HookState()
    state.request_id_to_prompt_token_len = {}
    state.request_id_to_iter_size = {}
    return state


@pytest.fixture
def mock_scheduler(hook_state):
    sched = MagicMock()
    sched.running = deque()
    sched.waiting = deque()

    with patch.object(batch_hookers, "_get_state", return_value=hook_state):
        yield sched


def create_request(request_id, token_count=10):
    return MagicMock(request_id=request_id, req_id=request_id, prompt_token_ids=[0] * token_count)


# compare_deques tests
@pytest.mark.parametrize(
    "q1, q2, expected",
    [
        # Normal cases
        ([1, 2, 3], [2, 3, 4], Counter({1: 1})),
        ([1, 1, 2], [1, 2], Counter({1: 1})),
        # Empty cases
        ([], [1, 2], Counter()),
        ([1, 2], [], Counter({1: 1, 2: 1})),
        # Identical cases
        ([1, 2], [1, 2], Counter()),
        # Duplicates
        ([1, 1, 1], [1], Counter({1: 2})),
    ],
)
def test_compare_deques_given_two_deques_when_difference_exists_then_return_counter_diff(q1, q2, expected):
    result = batch_hookers.compare_deques(deque(q1), deque(q2))
    assert result == expected


def test_queue_profiler_given_enqueue_when_queue_changes_then_log_enqueue_event():
    before = deque([SequenceGroup("req1", (1, 2, 3))])
    after = deque([SequenceGroup("req1", (1, 2, 3)), SequenceGroup("req2", (4, 5))])

    batch_hookers.queue_profiler(before, after, "test_queue")

    assert len(Profiler.instance_calls) == 1
    calls = Profiler.instance_calls[0]
    assert ("res", ["req2"]) in [call[:2] for call in calls]
    assert any(call[0] == "event" and call[1] == "Enqueue" for call in calls)
    assert any(call[0] == "metric_scope" and call[2] == "test_queue" for call in calls)


def test_queue_profiler_given_dequeue_when_queue_changes_then_log_dequeue_event():
    before = deque([SequenceGroup("req1", (1,)), SequenceGroup("req2", (2,))])
    after = deque([SequenceGroup("req2", (2,))])

    batch_hookers.queue_profiler(before, after, "test_queue")

    assert len(Profiler.instance_calls) == 1
    calls = Profiler.instance_calls[0]
    assert ("res", ["req1"]) in [call[:2] for call in calls]
    assert any(call[0] == "event" and call[1] == "Dequeue" for call in calls)


def test_queue_profiler_given_no_changes_when_queues_identical_then_no_events_logged():
    before = deque([SequenceGroup("req1", (1,))])
    after = deque([SequenceGroup("req1", (1,))])

    batch_hookers.queue_profiler(before, after, "test_queue")
    assert len(Profiler.instance_calls) == 0


def test_get_state_given_first_call_when_no_existing_state_then_create_new_state():
    if hasattr(batch_hookers._thread_local, "hook_state"):
        del batch_hookers._thread_local.hook_state

    state = batch_hookers._get_state()
    assert isinstance(state, batch_hookers.HookState)
    assert state == batch_hookers._thread_local.hook_state


def test_get_state_given_existing_state_when_called_then_return_same_instance():
    original_state = batch_hookers.HookState()
    batch_hookers._thread_local.hook_state = original_state
    assert batch_hookers._get_state() is original_state


def test_process_inputs_given_valid_request_when_called_then_log_event():
    mock_original = MagicMock(return_value="result")
    mock_this = MagicMock()

    result = batch_hookers.process_inputs(mock_original, mock_this, "req1")

    mock_original.assert_called_with(mock_this, "req1")
    assert result == "result"
    assert len(Profiler.instance_calls) == 1
    assert Profiler.instance_calls[0][-1] == ("event", "ReqState")


def test_schedule_given_new_requests_when_processing_then_update_state_and_log(hook_state, mock_scheduler):
    req1, req2 = create_request("req1", 5), create_request("req2", 3)
    mock_scheduler.running = deque([SequenceGroup("req1", (1, 2, 3)), SequenceGroup("req2", (4, 5))])

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[req1, req2],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={"req1": 5, "req2": 3},
        finished_req_ids=[],
    )

    mock_original = MagicMock(return_value=scheduler_output)
    with patch.object(batch_hookers, "_get_state", return_value=hook_state):
        result = batch_hookers.schedule(mock_original, mock_scheduler)

    assert result == scheduler_output
    assert hook_state.request_id_to_prompt_token_len == {
        "req1": len(req1.prompt_token_ids),
        "req2": len(req2.prompt_token_ids),
    }
    assert hook_state.request_id_to_iter_size == {"req1": 0, "req2": 0}


def test_schedule_given_finished_requests_when_processing_then_clean_state(hook_state, mock_scheduler):
    hook_state.request_id_to_prompt_token_len = {"req1": 10, "req2": 20}
    hook_state.request_id_to_iter_size = {"req1": 3, "req2": 5}

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={"req1": 5, "req2": 0},
        finished_req_ids=["req2"],
    )

    mock_original = MagicMock(return_value=scheduler_output)
    with patch.object(batch_hookers, "_get_state", return_value=hook_state):
        batch_hookers.schedule(mock_original, mock_scheduler)

    assert "req2" not in hook_state.request_id_to_prompt_token_len
    assert "req2" not in hook_state.request_id_to_iter_size


def test_schedule_given_prefill_batch_when_iter_size_zero_then_set_batch_type(hook_state, mock_scheduler):
    hook_state.request_id_to_iter_size = {"req1": 0, "req2": 1}
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={"req1": 5, "req2": 3, "req3": 2},
        finished_req_ids=[],
    )

    mock_original = MagicMock(return_value=scheduler_output)
    with patch.object(batch_hookers, "_get_state", return_value=hook_state):
        batch_hookers.schedule(mock_original, mock_scheduler)

    # Verify batch type attribute
    span_calls = None
    for calls in Profiler.instance_calls:
        if any(call[0] == "span_start" for call in calls):
            span_calls = calls
            break

    assert span_calls is not None
    assert ("attr", "batch_type", "Prefill") in span_calls


def test_schedule_given_no_requests_when_processing_then_early_return(hook_state, mock_scheduler):
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[], scheduled_cached_reqs=[], num_scheduled_tokens={}, finished_req_ids=[]
    )

    mock_original = MagicMock(return_value=scheduler_output)
    with patch.object(batch_hookers, "_get_state", return_value=hook_state):
        result = batch_hookers.schedule(mock_original, mock_scheduler)

    assert result == scheduler_output
    # Should still profile queues
    assert any(["BatchSchedule" in call for calls in Profiler.instance_calls for call in calls])
    assert not any(["QueueSize" in call for calls in Profiler.instance_calls for call in calls])


def test_free_request_given_running_request_when_freed_then_transition_state(hook_state):
    request = create_request("req1")

    with patch.object(batch_hookers, "_get_state", return_value=hook_state):
        batch_hookers.free_request(MagicMock(), MagicMock(), request)

    assert any(
        [
            call[0] == "metric_inc" and call[1] == "FINISHED" and call[2] == 1
            for calls in Profiler.instance_calls
            for call in calls
        ]
    )


def test_free_request_given_waiting_request_when_freed_then_transition_state(hook_state):
    request = create_request("req1")

    with patch.object(batch_hookers, "_get_state", return_value=hook_state):
        batch_hookers.free_request(MagicMock(), MagicMock(), request)

    assert any(
        [
            call[0] == "metric_inc" and call[1] == "FINISHED" and call[2] == 1
            for calls in Profiler.instance_calls
            for call in calls
        ]
    )


def test_free_request_given_unknown_request_when_freed_then_no_state_change(hook_state):
    request = create_request("req1")

    with patch.object(batch_hookers, "_get_state", return_value=hook_state):
        batch_hookers.free_request(MagicMock(), MagicMock(), request)

    # Should still create Profiler event
    assert len(Profiler.instance_calls) == 1
    calls = Profiler.instance_calls[0]
    # Verify that res() was called with the request_id
    assert ("res", "req1") in calls
    # Verify that event("ReqState") was called
    assert ("event", "ReqState") in calls
    # Verify that metric_inc("FINISHED", 1) was called
    assert ("metric_inc", "FINISHED", 1) in calls


def test_add_request_given_new_request_when_added_then_update_state_and_log(hook_state):
    scheduler = MagicMock()
    scheduler.waiting = deque([SequenceGroup("req1", [1, 2, 3])])
    request = create_request("req1")

    with patch.object(batch_hookers, "_get_state", return_value=hook_state):
        batch_hookers.add_request(MagicMock(), scheduler, request)

    assert any(
        call[0] == "metric_inc" and call[1] == "WAITING" and call[2] == 1
        for calls in Profiler.instance_calls
        for call in calls
    )
    # Verify queue enqueue event
    assert any(call[0] == "event" and call[1] == "Enqueue" for calls in Profiler.instance_calls for call in calls)
