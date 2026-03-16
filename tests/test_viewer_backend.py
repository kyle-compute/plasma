from __future__ import annotations

from plasma.live.viewer_backend import (
    backend_candidates,
    backend_failure_message,
    is_uv_managed_python,
)


def test_is_uv_managed_python_detects_uv_runtime_path():
    assert is_uv_managed_python("/home/tax/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12")
    assert not is_uv_managed_python("/usr/bin/python3")


def test_backend_candidates_prefer_webagg_for_uv_python():
    assert backend_candidates("/home/tax/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12") == (
        "WebAgg",
        "TkAgg",
    )
    assert backend_candidates("/usr/bin/python3") == ("TkAgg", "WebAgg")


def test_backend_failure_message_mentions_tornado_for_uv_python():
    message = backend_failure_message(
        "/home/tax/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12"
    )
    assert "uv pip install --python .venv/bin/python tornado" in message
