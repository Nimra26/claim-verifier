"""Timeouts, fast retries and circuit breakers for calls to the model server.

The rules (29 Oct, from what went wrong on this laptop):
  - Retry only fast failures — the connection was refused or dropped, e.g. Ollama restarting — once or twice with a
    short backoff and jitter. Never retry a timeout: a 180 s generation retried twice kept people waiting 15 minutes.
  - After repeated failures to reach the server (refused, dropped, 5xx), stop calling for a while (circuit breaker) and
    say so at once, instead of making every person wait for their own failure. One trial call goes through when the
    pause ends. A timeout means the server is up but slow, so it doesn't open the breaker; a fallback model handles it.
Settings: WORKSPACE_BREAKER_FAILURES (default 3), WORKSPACE_BREAKER_PAUSE_S (default 60).
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Awaitable, Callable, TypeVar

import openai

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Fast failures worth one more try. APITimeoutError is a subclass of APIConnectionError, so it is excluded by name.
FAST_FAILURES = (openai.APIConnectionError, openai.InternalServerError)  # refused/dropped, or a 5xx while loading


def _env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class CircuitOpenError(RuntimeError):
    """Raised instead of calling a service that failed several times in a row."""

    def __init__(self, name: str, retry_in_s: float) -> None:
        self.name, self.retry_in_s = name, retry_in_s
        super().__init__(f"{name} failed several times in a row; calls are paused for {retry_in_s:.0f} s")


class CircuitBreaker:
    """closed → (N failures in a row) → open for a pause → half-open (one trial call) → closed or open again."""

    def __init__(self, name: str, *, failures: int | None = None, pause_s: float | None = None) -> None:
        self.name = name
        self.failures_to_open = int(failures if failures is not None else _env("WORKSPACE_BREAKER_FAILURES", 3))
        self.pause_s = pause_s if pause_s is not None else _env("WORKSPACE_BREAKER_PAUSE_S", 60)
        self.failures = 0
        self.opened_at: float | None = None
        self.trial_running = False

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half-open" if time.monotonic() - self.opened_at >= self.pause_s else "open"

    def retry_in(self) -> float:
        return 0.0 if self.opened_at is None else max(0.0, self.pause_s - (time.monotonic() - self.opened_at))

    def before_call(self) -> None:
        state = self.state
        if state == "open" or (state == "half-open" and self.trial_running):
            raise CircuitOpenError(self.name, self.retry_in() or self.pause_s)
        if state == "half-open":
            self.trial_running = True

    def success(self) -> None:
        if self.opened_at is not None:
            logger.info("Circuit %s closed again", self.name)
        self.failures, self.opened_at, self.trial_running = 0, None, False

    def failure(self) -> None:
        self.trial_running = False
        self.failures += 1
        if self.opened_at is not None or self.failures >= self.failures_to_open:
            self.opened_at = time.monotonic()  # (re)open: a failed trial starts a new pause
            logger.warning("Circuit %s open for %.0f s after %d failures", self.name, self.pause_s, self.failures)


BREAKERS: dict[str, CircuitBreaker] = {}


def breaker(name: str) -> CircuitBreaker:
    if name not in BREAKERS:
        BREAKERS[name] = CircuitBreaker(name)
    return BREAKERS[name]


def open_breakers() -> list[dict]:
    """For the health bar and Monitor: services whose calls are paused right now."""
    return [{"name": b.name, "state": b.state, "retry_in_s": round(b.retry_in())} for b in BREAKERS.values() if b.state != "closed"]


async def call(name: str, fn: Callable[[], Awaitable[T]], *, retries: int = 2, delays: tuple[float, ...] = (0.5, 1.5),
               sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> T:
    """Run fn under the named breaker, retrying fast failures only. Timeouts and other errors go straight up."""
    b = breaker(name)
    attempt = 0
    while True:
        b.before_call()
        try:
            result = await fn()
        except openai.APITimeoutError:
            # The server is up but slow: not a breaker failure, and never retried (the fallback path handles it).
            b.trial_running = False  # a half-open trial ended without a verdict: the next call may try again
            raise
        except FAST_FAILURES as exc:
            b.failure()
            if attempt >= retries or b.state != "closed":
                raise
            wait = delays[min(attempt, len(delays) - 1)] * (0.8 + 0.4 * random.random())  # jitter: don't retry in step
            logger.info("%s: %s; retrying in %.1f s", name, type(exc).__name__, wait)
            attempt += 1
            await sleep(wait)
            continue
        except openai.APIStatusError:
            # A 4xx (bad request, model missing): the server is up and answering, so this isn't a breaker failure.
            b.success()
            raise
        except BaseException:
            # Cancelled (Cancel on a task, a closed page) or an unexpected error during a half-open trial: without
            # this, trial_running stayed True and every later call got CircuitOpenError until a restart.
            b.trial_running = False
            raise
        b.success()
        return result
