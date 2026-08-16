import importlib
import json

import pytest


@pytest.mark.parametrize("version", ("v1", "v2", "v3", "v4"))
def test_running_completed_and_failed_status_updates(tmp_path, version: str) -> None:
    update_status = importlib.import_module(f"src.{version}.experiment").update_status
    state = {"status": "running", "last_epoch": 0}

    state = update_status(tmp_path, state, status="running", last_epoch=1)
    assert state["status"] == "running"
    assert json.loads((tmp_path / "status.json").read_text())["last_epoch"] == 1

    state = update_status(tmp_path, state, status="completed")
    assert state["status"] == "completed"
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "completed"

    state = update_status(
        tmp_path,
        state,
        status="failed",
        exception_type="SyntheticError",
        exception_message="synthetic failure",
    )
    persisted = json.loads((tmp_path / "status.json").read_text())
    assert state["status"] == persisted["status"] == "failed"
    assert persisted["exception_type"] == "SyntheticError"
