"""
Trace budgets for FORECAST event collection.

Instrumenting a program records every line, branch, condition, definition and
use it executes.  For small subjects that is cheap, but for realistic projects
a single test can emit an unbounded event stream: hot loops contribute millions
of events, and every definition of a large ``str``/``bytes`` value serialises
that value in full, so one assignment of a multi-megabyte buffer writes
megabytes into the trace.  Left unbounded, individual tests have produced
hundreds of gigabytes.

A :class:`TraceBudget` caps what one run may emit.  It is handed to the runtime
tracer through environment variables, so the limits are enforced *while* the
program runs rather than by cleaning up afterwards.

Two of the limits are information-preserving for the analyses FORECAST
performs:

* **Loop hits** — SFLKit's loop spectra distinguish only "never", "once" and
  "more than once", so recording at most two iterations per loop frame is
  lossless while removing what is usually the dominant event class.
* **Value size** — length events carry the true length separately, so length
  analyses stay exact.  Only string/bytes predicates (empty, ASCII-only,
  contains-digit, contains-special) see the retained prefix instead of the full
  value.

The remaining limits are hard stops.  Because events are written whole, a trace
cut off by a budget stays readable up to the cut.

Every field can be overridden from the host environment, so limits can be tuned
for a campaign without editing code.
"""

import os
from dataclasses import dataclass

#: Longest str/bytes value serialised into a single event.
DEFAULT_MAX_VALUE_BYTES = 1024
#: Loop-hit events recorded per loop iteration frame.
DEFAULT_MAX_LOOP_HITS = 2
#: Hard cap on the size of one trace, before compression.
DEFAULT_MAX_BYTES = 2 * 1024**3
#: Hard cap on the number of events in one trace (0 disables).
DEFAULT_MAX_EVENTS = 0
#: Compression codec for the event stream ("gzip", "zstd" or "" for none).
DEFAULT_COMPRESS = "gzip"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _extra_run_env() -> dict[str, str]:
    """
    Extra environment variables to hand the traced run, from ``FORECAST_RUN_ENV``.

    Some subjects cannot execute their own test suite under a bare
    ``pytest <node-id>`` invocation because the framework needs configuring
    first.  Django is the motivating case: its suite is normally driven by
    ``tests/runtests.py``, and without ``DJANGO_SETTINGS_MODULE`` the first
    access to ``settings`` raises ``ImproperlyConfigured`` while the package is
    still being imported.  The run then dies before reaching the test body, so
    every trace is the same import prefix and nothing discriminates.

    Rather than special-casing individual projects in code, the campaign
    supplies whatever the subject needs:

        FORECAST_RUN_ENV="DJANGO_SETTINGS_MODULE=test_sqlite,PYTHONPATH=/testbed/tests"

    Pairs are separated by commas (or newlines) and split on the first ``=``,
    so values may themselves contain ``=``.  Malformed entries are ignored:
    a bad campaign variable must never break collection.

    :returns: Parsed environment variables, empty when the variable is unset.
    """
    raw = os.getenv("FORECAST_RUN_ENV")
    if not raw:
        return {}
    extra: dict[str, str] = {}
    for item in raw.replace("\n", ",").split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, _, value = item.partition("=")
        key = key.strip()
        if key:
            extra[key] = value.strip()
    return extra


@dataclass(frozen=True)
class TraceBudget:
    """
    Limits applied to a single instrumented run.

    :param max_value_bytes: Longest ``str``/``bytes`` value serialised into a
        definition or return event; longer values are truncated to this prefix.
        ``0`` disables truncation.
    :param max_loop_hits: Loop-hit events recorded per loop iteration frame.
        ``0`` records every iteration.
    :param max_bytes: Size at which the trace is cut off, before compression.
        ``0`` means unbounded.
    :param max_events: Number of events at which the trace is cut off.  ``0``
        means unbounded.
    :param compress: Codec used to write the event stream — ``"gzip"``,
        ``"zstd"``, or ``""`` to store it uncompressed.  Traces are highly
        repetitive, so compression usually shrinks them by an order of
        magnitude.  The reader detects the codec from the file's magic bytes,
        so compressed and uncompressed traces can be mixed freely.
    """

    max_value_bytes: int = DEFAULT_MAX_VALUE_BYTES
    max_loop_hits: int = DEFAULT_MAX_LOOP_HITS
    max_bytes: int = DEFAULT_MAX_BYTES
    max_events: int = DEFAULT_MAX_EVENTS
    compress: str = DEFAULT_COMPRESS

    @classmethod
    def from_environment(cls) -> "TraceBudget":
        """
        Build a budget from ``FORECAST_TRACE_*`` host environment variables,
        falling back to the module defaults.

        :returns: The configured budget.
        """
        return cls(
            max_value_bytes=_env_int(
                "FORECAST_TRACE_MAX_VALUE_BYTES", DEFAULT_MAX_VALUE_BYTES
            ),
            max_loop_hits=_env_int(
                "FORECAST_TRACE_MAX_LOOP_HITS", DEFAULT_MAX_LOOP_HITS
            ),
            max_bytes=_env_int("FORECAST_TRACE_MAX_BYTES", DEFAULT_MAX_BYTES),
            max_events=_env_int("FORECAST_TRACE_MAX_EVENTS", DEFAULT_MAX_EVENTS),
            compress=os.getenv("FORECAST_TRACE_COMPRESS", DEFAULT_COMPRESS).strip(),
        )

    def environment(self, events_path: str) -> dict[str, str]:
        """
        Render this budget as the environment the runtime tracer reads.

        :param events_path: Path the tracer should write the event stream to.
        :returns: Environment variables to pass to the instrumented run.
        """
        env = {
            "EVENTS_PATH": str(events_path),
            "EVENTS_MAX_VALUE_BYTES": str(self.max_value_bytes),
            "EVENTS_MAX_LOOP_HITS": str(self.max_loop_hits),
            "EVENTS_MAX_BYTES": str(self.max_bytes),
            "EVENTS_MAX_EVENTS": str(self.max_events),
            # atexit handlers do not run when a run is killed for overrunning
            # its timeout, so ask the tracer to flush on the signal instead.
            "EVENTS_FLUSH_ON_SIGNAL": "1",
        }
        if self.compress:
            env["EVENTS_COMPRESS"] = self.compress
        # Subject-specific setup the traced run needs before its tests can run
        # at all (see _extra_run_env). Applied last, but never allowed to
        # override the tracer's own variables: a campaign typo in
        # FORECAST_RUN_ENV must not silently redirect or uncap the trace.
        for key, value in _extra_run_env().items():
            if key not in env:
                env[key] = value
        return env


#: Budget used when a caller does not supply one.
DEFAULT_BUDGET = TraceBudget()


def resolve_budget(budget: "TraceBudget | None") -> TraceBudget:
    """
    Return *budget*, or one built from the environment when it is ``None``.

    :param budget: Explicit budget, or ``None``.
    :returns: The budget to apply.
    """
    if budget is not None:
        return budget
    return TraceBudget.from_environment()
