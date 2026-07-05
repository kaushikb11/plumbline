"""Tiny prereq-UX helpers shared by the runnable examples (examples/ only).

These turn the two common first-run failures into a single actionable line plus a
clean ``exit(1)`` — instead of a ``KeyError`` or a ~25-line ``httpx.ConnectError``
traceback:

* a missing required environment variable, via :func:`require_env`;
* an unreachable Ollama / Modal / WebSocket endpoint, via :func:`friendly_endpoint`.

Pure standard library at import time (``httpx`` / ``websockets`` are imported lazily
and optionally), so importing this module never pulls in an optional extra and never
changes what an example DOES once its prerequisites are present.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator


def require_env(name: str, desc: str) -> str:
    """Return ``os.environ[name]``, or print one actionable line and ``exit(1)``.

    ``desc`` is a short human description of the value (e.g. ``"the LLM endpoint URL"``)
    used to build the message ``This example needs NAME=<desc>. See examples/README.md.``
    """
    value = os.environ.get(name)
    if not value:
        print(
            f"This example needs {name}={desc}. See examples/README.md.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return value


def _connection_error_types() -> tuple[type[BaseException], ...]:
    """The exception types that mean "could not reach the endpoint".

    ``OSError`` covers refused sockets and DNS failures; ``httpx.TransportError`` and
    ``websockets``' base exception are added only if those libraries are importable, so
    this stays dependency-free for callers that use neither.
    """
    types: list[type[BaseException]] = [OSError, TimeoutError]
    with contextlib.suppress(ImportError):
        import httpx

        types.append(httpx.TransportError)
    with contextlib.suppress(ImportError):
        from websockets.exceptions import WebSocketException

        types.append(WebSocketException)
    return tuple(types)


@contextlib.contextmanager
def friendly_endpoint(service: str, url: str, *, hint: str) -> Iterator[None]:
    """Turn a connection failure reaching ``service`` at ``url`` into one line + ``exit(1)``.

    Wrap the network entrypoint (the first connecting call, or ``main()``) in this. A
    transport/socket error prints ``Could not reach <service> at <url>. <hint>`` and
    exits cleanly; any other error propagates unchanged.
    """
    try:
        yield
    except _connection_error_types() as exc:
        print(
            f"Could not reach {service} at {url}. {hint} ({exc.__class__.__name__})",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
