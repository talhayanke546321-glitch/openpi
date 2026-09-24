# ruff: noqa: SLF001

import pytest

from scripts import serve_policy


def test_loopback_listener_can_be_unauthenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENPI_API_KEY", raising=False)
    args = serve_policy.Args(host="127.0.0.1")
    assert serve_policy._load_api_key(args) is None


def test_non_loopback_listener_requires_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENPI_API_KEY", raising=False)
    args = serve_policy.Args(host="0.0.0.0")
    with pytest.raises(ValueError, match="Refusing unauthenticated"):
        serve_policy._load_api_key(args)


def test_api_key_is_loaded_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENPI_API_KEY", "secret")
    args = serve_policy.Args(host="0.0.0.0")
    assert serve_policy._load_api_key(args) == "secret"
