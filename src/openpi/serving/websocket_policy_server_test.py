# ruff: noqa: SLF001

import dataclasses
import http

from openpi.serving import websocket_policy_server


@dataclasses.dataclass
class _Response:
    status_code: http.HTTPStatus
    text: str
    headers: dict[str, str] = dataclasses.field(default_factory=dict)


class _Connection:
    def respond(self, status: http.HTTPStatus, text: str) -> _Response:
        return _Response(status, text)


@dataclasses.dataclass
class _Request:
    path: str
    headers: dict[str, str] = dataclasses.field(default_factory=dict)


def test_health_check_does_not_require_authentication() -> None:
    response = websocket_policy_server._process_request(
        _Connection(),
        _Request("/healthz"),
        api_key="secret",
    )
    assert response is not None
    assert response.status_code == http.HTTPStatus.OK


def test_policy_connection_requires_matching_api_key() -> None:
    response = websocket_policy_server._process_request(
        _Connection(),
        _Request("/", {"Authorization": "Api-Key wrong"}),
        api_key="secret",
    )
    assert response is not None
    assert response.status_code == http.HTTPStatus.UNAUTHORIZED
    assert response.headers["WWW-Authenticate"] == "Api-Key"

    accepted = websocket_policy_server._process_request(
        _Connection(),
        _Request("/", {"Authorization": "Api-Key secret"}),
        api_key="secret",
    )
    assert accepted is None


def test_policy_connection_can_run_without_auth_when_explicitly_unconfigured() -> None:
    accepted = websocket_policy_server._process_request(
        _Connection(),
        _Request("/"),
        api_key=None,
    )
    assert accepted is None


def test_serve_forever_treats_keyboard_interrupt_as_clean_shutdown(monkeypatch) -> None:
    """收到 Ctrl-C 或 systemd SIGINT 时应安静退出，不抛出异常栈。"""
    server = websocket_policy_server.WebsocketPolicyServer(object())

    def interrupt(coroutine) -> None:
        coroutine.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(websocket_policy_server.asyncio, "run", interrupt)

    server.serve_forever()
