"""
Event-file access for FORECAST.

Traces written by the instrumented program are not always plain, complete
binary streams:

* they may be **compressed** (gzip or zstd), because raw traces of realistic
  programs are large and highly repetitive; and
* they may be **truncated**, because the run hit its timeout or exhausted the
  trace budget enforced by the runtime tracer.

:class:`ForecastEventFile` handles both.  The codec is detected from the
leading magic bytes rather than the file name, and any decoding error ends the
stream instead of failing the analysis — events are written whole, so
everything read before the cut is valid.

Use :func:`make_event_file` instead of constructing
:class:`~sflkit.events.event_file.EventFile` directly.
"""

import io
import os
import zlib
from pathlib import Path
from pickle import PickleError

from sflkit.events.event_file import EventFile
from sflkit.events.mapping import EventMapping
from sflkitlib.events import event

GZIP_MAGIC = b"\x1f\x8b"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

#: Suffixes appended by the collector when a trace is stored compressed.
COMPRESSED_SUFFIXES = (".gz", ".zst")


def open_event_stream(path: os.PathLike | str) -> io.BufferedReader:
    """
    Open an event file for reading, transparently decompressing it.

    :param path: Path of the event file.
    :returns: A buffered binary reader over the decoded event stream.  The
        result always supports ``peek``, which the event decoder relies on to
        detect the end of the stream.
    :raises ImportError: If the file is zstd-compressed but ``zstandard`` is
        not installed.
    """
    raw = open(path, "rb")
    try:
        header = raw.read(4)
        raw.seek(0)
    except OSError:
        return raw

    if header.startswith(GZIP_MAGIC):
        import gzip

        return io.BufferedReader(gzip.GzipFile(fileobj=raw, mode="rb"))
    if header.startswith(ZSTD_MAGIC):
        try:
            import zstandard
        except ImportError as e:
            raw.close()
            raise ImportError(
                f"{path} is zstd-compressed but the 'zstandard' package is not "
                "installed. Run: pip install zstandard"
            ) from e
        return io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(raw))
    return raw


def resolve_event_path(path: os.PathLike | str) -> Path:
    """
    Return the on-disk path of a trace, allowing for a compression suffix.

    The collector may store ``<name>`` as ``<name>.gz`` or ``<name>.zst``, so
    callers that only know the logical name can resolve the real file here.

    :param path: Logical event-file path.
    :returns: The existing path, preferring the exact name and falling back to
        the compressed variants.  Returns *path* unchanged when nothing exists.
    """
    candidate = Path(path)
    if candidate.exists():
        return candidate
    for suffix in COMPRESSED_SUFFIXES:
        compressed = candidate.with_name(candidate.name + suffix)
        if compressed.exists():
            return compressed
    return candidate


class ForecastEventFile(EventFile):
    """
    An :class:`~sflkit.events.event_file.EventFile` that reads compressed and
    truncated traces.
    """

    def __enter__(self) -> "ForecastEventFile":
        self._file_pointer = open_event_stream(self.path)
        return self

    def __getstate__(self) -> dict:
        # The relevance tree keeps event files as node run-ids, and the tree is
        # pickled to disk. An open ``_file_pointer`` (a BufferedReader over the
        # possibly-compressed stream) cannot be pickled, so drop it; it is only
        # needed while actively reading events and is reopened by ``__enter__``.
        state = dict(self.__dict__)
        state["_file_pointer"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._file_pointer = None

    def load(self):
        """
        Yield the events in this file, stopping cleanly at the end of a
        truncated *or corrupted* stream.

        A trace can be cut mid-record (timeout / trace-budget) or its
        compressed body can be damaged; decompressing past the damage raises
        :class:`zlib.error`, which — like the truncation errors — ends the
        stream instead of failing the whole analysis, since every event read
        before the damage is intact.
        """
        while True:
            try:
                if not self._file_pointer.peek(1):
                    break
                decoded = event.load_next_event(
                    self._file_pointer,
                    self.mapping.mapping,
                    with_thread_id=self.thread_support,
                )
            except (
                IndexError,
                ValueError,
                PickleError,
                KeyError,
                EOFError,
                OSError,
                zlib.error,
            ):
                break
            if self.mapping.is_valid(decoded):
                yield decoded


def make_event_file(
    path: os.PathLike | str,
    run_id: int,
    mapping: EventMapping,
    failing: bool = False,
    thread_support: bool = False,
) -> ForecastEventFile:
    """
    Build a :class:`ForecastEventFile`, resolving any compression suffix.

    :param path: Logical path of the event file.
    :param run_id: Unique identifier of the run that produced the trace.
    :param mapping: Event mapping used to decode event IDs.
    :param failing: Whether the trace comes from a failing test.
    :param thread_support: Whether the trace carries thread IDs.
    :returns: The event file, ready to be used as a context manager.
    """
    return ForecastEventFile(
        resolve_event_path(path),
        run_id,
        mapping,
        failing=failing,
        thread_support=thread_support,
    )
