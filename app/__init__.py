"""Alvin real-time WebSocket Speech-to-Text service.

The FastAPI application object lives in :mod:`app.main`, which in turn pulls in
:class:`app.connection.Connection` and the rest of the pipeline -- imports that
are somewhat heavy (edge-tts, numpy, miniaudio, etc). ``__all__`` previously
eagerly exported ``"app"`` as a string literal, which is useless to tools that
import it and still forces the whole tree to load whenever the package is
imported. Bug 9: lazy-import the application on attribute access instead.
"""

from __future__ import annotations

__all__ = ["app"]


def _load_app():
    """Deferred import so merely importing ``app`` stays cheap."""
    from .main import app as _app  # noqa: WPS433 -- intentional lazy load
    return _app


def __getattr__(name: str):
    if name == "app":
        app = _load_app()
        globals()["app"] = app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
