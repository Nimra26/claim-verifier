"""Progress reports (how far a long step is) are a nicety: they must never make the work itself fail.

The Add documents page passes on_progress(done, total) down to the ingestion service and the embedding provider.
Other implementations (test stand-ins, older callers) don't take that argument, so it is only passed to those that do.
"""

from __future__ import annotations

import inspect


def takes(fn, name: str) -> bool:
    """Whether fn accepts the keyword argument `name` (by name, or through **kwargs)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
