"""
Event-handling and tree-building pipeline for FORECAST.

This module bridges the raw ``sflkit`` event stream and the
:class:`~forecast.tree.TracedRelevanceTree` data structure used by the
diagnosis engine.  Events are processed file-by-file; each
:class:`FunctionEnterEvent` pushes a new tree node onto a per-thread stack
and each :class:`FunctionExitEvent` / :class:`FunctionErrorEvent` pops it,
recording the feature vectors observed at entry and exit.

Main entry point: :class:`ForecastHandler`.
"""

import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from sflkit.analysis.factory import ScalarPairFactory
from sflkit.analysis.predicate import Predicate
from sflkit.analysis.spectra import Spectrum
from sflkit.events.event_file import EventFile
from sflkit.features.handler import FeatureBuilder, EventHandler
from sflkit.features.value import BinaryFeature, TertiaryFeature
from sflkit.features.vector import FeatureVector
from sflkit.language.meta import IDGenerator
from sflkit.model.parallel import ParallelModel
from sflkit.runners.run import TestResult
from sflkitlib.events import EventType
from sflkitlib.events.event import (
    FunctionEnterEvent,
    FunctionExitEvent,
    FunctionErrorEvent,
    Event,
)

from forecast.logger import LOGGER
from forecast.tree import (
    TracedRelevanceTree,
    merge_into,
    structure_only_copy,
)

# A cause chain explains a failure in terms of what a function had in hand, so
# the pairs worth ranking are the ones relating a definition to the rest of its
# invocation. Pairing it with the imported module's namespace instead -- which
# is where the great majority of in-scope variables live once a library is
# loaded, and which no execution of the program changes -- yields features that
# hold equally in passing and failing runs, so they discriminate nothing while
# still being evaluated at every definition. On the heaviest subjects those
# pairs were roughly two thirds of all analysis work.
ScalarPairFactory.PAIR_WITH_GLOBALS = False


class ForecastFeatureVector(FeatureVector):
    """
    Feature vector extended with a per-thread context tag.

    Extends :class:`~sflkit.features.vector.FeatureVector` to track which
    thread produced the vector and what execution context depth it belongs to.

    :ivar context: Execution context / call-depth counter.
    :type context: int
    :ivar thread_id: OS thread identifier for the test run that produced this
        vector.
    :type thread_id: int | None
    """

    def __init__(
        self,
        run_id: int | EventFile,
        result: TestResult,
        context: int = 0,
        thread_id: int | None = None,
    ) -> None:
        """
        Initialize the feature vector.

        :param run_id: Unique run identifier (index into the event-file list).
        :type run_id: int | EventFile
        :param result: Whether the test that produced this vector was passing
            or failing.
        :type result: TestResult
        :param context: Execution context / call-depth counter.  Defaults to 0.
        :type context: int
        :param thread_id: Thread that produced this vector.  Defaults to ``None``.
        :type thread_id: int | None
        """
        super().__init__(run_id, result)
        self.context = context
        self.thread_id = thread_id
        assert isinstance(result, TestResult)

    def __eq__(self, other: object) -> bool:
        """
        Two vectors are equal when their base equality holds *and* their thread
        IDs match.

        :param other: Object to compare against.
        :type other: object
        :returns: ``True`` if equal.
        :rtype: bool
        """
        if not isinstance(other, ForecastFeatureVector):
            return False
        return super().__eq__(other) and self.thread_id == other.thread_id


class FeatureGuard:
    """
    Per-event-file registry of :class:`ForecastFeatureVector` instances keyed
    by thread ID.

    Ensures that each thread gets its own feature vector so that concurrent
    test execution does not mix observations from different threads.

    :ivar features: Thread-ID → vector map.
    :type features: dict[int, ForecastFeatureVector]
    :ivar test_result: Result label for all vectors in this guard.
    :type test_result: TestResult
    :ivar event_file: The event file these vectors belong to.
    :type event_file: EventFile
    :ivar context: Execution context depth passed to new vectors.
    :type context: int
    :ivar main_thread: Thread ID of the main test thread.
    :type main_thread: int | None
    """

    def __init__(
        self, test_result: TestResult, event_file: EventFile, context: int = 0
    ):
        """
        Initialize the guard.

        :param test_result: Result label (``PASSING`` or ``FAILING``).
        :type test_result: TestResult
        :param event_file: Owning event file.
        :type event_file: EventFile
        :param context: Execution context depth.  Defaults to 0.
        :type context: int
        :returns: None
        :rtype: None
        """
        self.features: dict[int, ForecastFeatureVector] = dict()
        self.test_result = test_result
        assert isinstance(test_result, TestResult), str(test_result)
        self.event_file = event_file
        self.context: int = context
        self.main_thread: Optional[int] = None

    def __getitem__(self, item: int) -> ForecastFeatureVector:
        """
        Return the vector for thread *item*, creating one on first access.

        :param item: Thread ID.
        :type item: int
        :returns: The :class:`ForecastFeatureVector` for that thread.
        :rtype: ForecastFeatureVector
        """
        if item not in self.features:
            self.features[item] = ForecastFeatureVector(
                result=self.test_result,
                context=self.context,
                run_id=self.event_file,
                thread_id=item,
            )
        return self.features[item]

    def __setitem__(self, key: int, value: ForecastFeatureVector) -> None:
        """Set the vector for thread *key*."""
        self.features[key] = value

    def __contains__(self, item: int) -> bool:
        """Return ``True`` if a vector exists for thread *item*."""
        return item in self.features


class TreeGuard:
    """
    Per-event-file registry of :class:`~forecast.tree.TracedRelevanceTree`
    nodes keyed by thread ID.

    Ensures each thread navigates its own position in the tree.  When a new
    thread is seen it inherits the current node of the main thread.

    :ivar trees: Thread-ID → current-node map.
    :type trees: dict[int, TracedRelevanceTree]
    :ivar default: The shared root node used as fallback.
    :type default: TracedRelevanceTree
    :ivar main_thread: Thread ID of the main test thread.
    :type main_thread: int | None
    """

    def __init__(self, default: TracedRelevanceTree) -> None:
        """
        Initialize the guard with the shared root node.

        :param default: Shared root to fall back to for newly-seen threads.
        :type default: TracedRelevanceTree
        """
        self.trees: dict[int, TracedRelevanceTree] = dict()
        self.default = default
        self.main_thread = None

    def __getitem__(self, item: int) -> TracedRelevanceTree:
        """
        Return the current tree node for thread *item*.

        If the thread has not been seen before, it inherits the main thread's
        node (if available) or the root default.

        :param item: Thread ID.
        :type item: int
        :returns: The current :class:`~forecast.tree.TracedRelevanceTree` node
            for that thread.
        :rtype: TracedRelevanceTree
        """
        if item not in self.trees:
            if self.main_thread in self.trees:
                self.trees[item] = self.trees[self.main_thread]
            else:
                self.trees[item] = self.default
        return self.trees[item]

    def __setitem__(self, key: int, value: TracedRelevanceTree) -> None:
        """Set the current node for thread *key*."""
        self.trees[key] = value

    def __contains__(self, item: int) -> bool:
        """Return ``True`` if a node is registered for thread *item*."""
        return item in self.trees


class FunctionGuard:
    """
    Per-event-file call stack keyed by thread ID.

    Maintains a separate list of active function names for each thread so that
    recursive or multithreaded call stacks do not interfere with each other.

    :ivar functions: Thread-ID → call stack (list of function representations).
    :type functions: dict[int, list[str]]
    """

    def __init__(self) -> None:
        """Initialize with an empty call-stack registry."""
        self.functions: dict[int, list[str]] = dict()

    def __getitem__(self, item) -> list[str]:
        """
        Return the call stack for thread *item*, creating an empty one on first
        access.

        :param item: Thread ID.
        :returns: The call stack (list of function representation strings) for
            that thread.
        """
        if item not in self.functions:
            self.functions[item] = list()
        return self.functions[item]

    def __setitem__(self, key, value):
        """Set the call stack for thread *key*."""
        self.functions[key] = value

    def __contains__(self, item):
        """Return ``True`` if a call stack exists for thread *item*."""
        return item in self.functions


class OffsideGuard:
    """
    Per-event-file "off-footprint" depth counter keyed by thread ID.

    Used by the failing-first pruning optimization: while a passing run is
    executing inside a function that no failing run ever reached, the counter
    for that thread is positive and grows/shrinks with nested enters/exits, so
    the whole off-footprint subtree can be skipped without recording anything.

    :ivar depth: Thread-ID → current off-footprint nesting depth (0 = on the
        failing footprint).
    :type depth: dict[int, int]
    """

    def __init__(self) -> None:
        """Initialize with every thread on the footprint (depth 0)."""
        self.depth: dict[int, int] = dict()

    def __getitem__(self, item: int) -> int:
        """Return the off-footprint depth for thread *item* (0 if unseen)."""
        return self.depth.get(item, 0)

    def __setitem__(self, key: int, value: int) -> None:
        """Set the off-footprint depth for thread *key*."""
        self.depth[key] = value


class TreeBuilder(FeatureBuilder):
    """
    Builds a :class:`~forecast.tree.TracedRelevanceTree` from an event stream.

    Receives ``FUNCTION_ENTER``, ``FUNCTION_EXIT``, and ``FUNCTION_ERROR``
    events and updates the shared relevance tree accordingly.  Spectra and
    predicates observed at each probe point are recorded as features on the
    current tree node.

    :ivar context: Execution context depth passed to feature vectors.
    :type context: int
    :ivar root: The shared root node of the call tree.
    :type root: TracedRelevanceTree
    :ivar current_trees: Per-file current-node map.
    :type current_trees: dict[EventFile, TreeGuard]
    :ivar main_threads: Per-file main-thread ID.
    :type main_threads: dict[EventFile, int]
    :ivar id_gen: Monotonically-increasing feature-vector ID source.
    :type id_gen: IDGenerator
    :ivar current_features: Per-file feature vectors.
    :type current_features: dict[EventFile, FeatureGuard]
    :ivar stack: Per-file per-thread call stacks.
    :type stack: dict[EventFile, FunctionGuard]
    """

    def __init__(
        self,
        context: int = 0,
        prune: bool = True,
        footprint=None,
        carry_forward: bool = False,
        collapse_cycles: bool = None,
    ):
        """
        Initialize the tree builder.

        :param context: Execution context depth to stamp on feature vectors.
            Defaults to 0.
        :param prune: When ``True`` (default) passing runs are pruned to the
            failing footprint: a function a passing run enters that no failing
            run ever reached (and its whole subtree) is not recorded, because
            such nodes are passing-only and always discarded by diagnosis.
            This assumes all failing event files are processed before any
            passing one (which :meth:`_load_event_files` guarantees). Set to
            ``False`` to build the full, unpruned tree.
        :param footprint: Optional pre-built root to build against instead of a
            fresh one. The parallel builder passes a structure-only skeleton of
            the failing footprint here so that passing-file workers prune
            against it (``create=False``); their observations are merged back
            with :func:`~forecast.tree.merge_into`.
        :param carry_forward: When ``False`` (default) each node records only its
            LOCAL features: on a nested call the current vector is closed and the
            new one starts empty, so a node's vectors hold only the features
            observed at that scope. The call tree already encodes the path, so a
            caller's features are discriminated at the caller's own node and need
            not be duplicated into every descendant. Set to ``True`` for the
            legacy accumulating behavior, where each vector carries forward all
            features seen so far along the path (O(runs x path-length) memory).
        :param collapse_cycles: When ``True`` a function that is already on the
            call stack does not open a new subtree: the tree pointer navigates
            back to the node of its outermost active occurrence, so a recursive
            cycle folds onto itself. Direct recursion (``f -> f``) is collapsed
            in either mode; this extends the same treatment to *indirect*
            cycles (``f -> g -> f``), which otherwise mint a fresh node chain
            per turn of the cycle and dominate deeply recursive traces (sympy's
            symbolic evaluation). Evidence then aggregates at one node per
            function instead of being smeared over thousands of thin frames.
            Defaults to the ``FORECAST_COLLAPSE_CYCLES`` environment variable
            (unset = ``False``, i.e. legacy behavior).
        """
        super().__init__()
        self.context = context
        self._prune = prune
        self._carry_forward = carry_forward
        if collapse_cycles is None:
            collapse_cycles = os.environ.get("FORECAST_COLLAPSE_CYCLES", "") not in (
                "",
                "0",
                "false",
                "False",
            )
        self._collapse_cycles = collapse_cycles
        self.root: TracedRelevanceTree = (
            footprint if footprint is not None else TracedRelevanceTree("FORECAST_ROOT")
        )
        self.current_trees: dict[EventFile, TreeGuard] = dict()
        self.main_threads: dict[EventFile, int] = dict()
        self.id_gen = IDGenerator()
        self.current_features: dict[EventFile, FeatureGuard] = dict()
        self.stack: dict[EventFile, FunctionGuard] = dict()
        self.offside: dict[EventFile, OffsideGuard] = dict()
        # Cycle-collapse bookkeeping (only maintained when collapsing). For each
        # thread: the node to return to on each exit, the node of the outermost
        # active occurrence of every function on the path, and how many times
        # each function is currently on it.
        self.node_stack: dict[EventFile, dict[int, list]] = dict()
        self.path_node: dict[EventFile, dict[int, dict[str, TracedRelevanceTree]]] = (
            dict()
        )
        self.path_count: dict[EventFile, dict[int, dict[str, int]]] = dict()

    def prepare(self, event_file: EventFile, test_result: TestResult):
        """
        Initialize per-file state before processing an event file.

        Called by the event-handling infrastructure before the first event of
        *event_file* is dispatched.

        :param event_file: The event file about to be processed.
        :param test_result: Result label for this test run.
        """
        self.current_features[event_file] = FeatureGuard(
            test_result, event_file, context=self.context
        )
        self.current_trees[event_file] = TreeGuard(self.root)
        self.stack[event_file] = FunctionGuard()
        self.offside[event_file] = OffsideGuard()
        if self._collapse_cycles:
            self.node_stack[event_file] = defaultdict(list)
            self.path_node[event_file] = defaultdict(dict)
            self.path_count[event_file] = defaultdict(Counter)

    @staticmethod
    def get_function_repr(
        event: FunctionEnterEvent | FunctionExitEvent | FunctionErrorEvent,
    ):
        """
        Return a canonical string representation for the function in *event*.

        The format is ``<file>:<function_name>:<function_id>`` which is unique
        within a project even for overloaded names.

        :param event: Any function-boundary event.
        :returns: Canonical function identifier string.
        """
        return f"{event.file}:{event.function}:{event.function_id}"

    def hit(self, id_, *args, **kwargs):
        """
        Process a single probe hit from the event stream.

        Records spectrum and predicate features onto the current feature vector
        for the event's thread, then dispatches function-boundary events to the
        appropriate handler.

        :param id_: The event file object used as the per-file key.
        :param args: Positional arguments; the first element is the
            :class:`~sflkitlib.events.event.Event`.
        :param kwargs: Keyword arguments; ``"event"`` key is used if present.
        """
        if "event" in kwargs:
            event: Event = kwargs["event"]
        else:
            event: Event = args[0]
        # Record the first event's thread as the main thread for this file
        if id_ not in self.main_threads:
            self.main_threads[id_] = event.thread_id
            self.current_features[id_].main_thread = event.thread_id
            self.current_trees[id_].main_thread = event.thread_id
        # Record spectrum and predicate features observed at this probe point.
        #
        # Mirrors FeatureBuilder.hit: reuse the cached Feature for an analysis
        # object via _feature(), and skip anything already recorded on the
        # current vector. FeatureVector.set_feature keeps the FIRST value given
        # for a vector, so a repeat hit cannot change it -- checking that up
        # front avoids reading the evaluation, taking the vector's lock and
        # touching the feature set, which is most of the work on every hit after
        # the first. FORECAST forks a fresh vector at each function boundary
        # (LOCAL features), so "already recorded" is scoped to the current
        # function, not the whole run.
        vector = self.current_features[id_][event.thread_id]
        recorded = vector.features
        thread_id = event.thread_id
        all_features = self.root.all_features
        for a in self.analysis[id_]:
            feature = self._feature(a)
            if feature is None or feature in recorded:
                continue
            vector.set_feature(feature, self.map_evaluation(a, id_, thread_id))
            all_features.add(feature)
        # Dispatch function-boundary events to tree navigation handlers
        if event.event_type is EventType.FUNCTION_ENTER:
            # noinspection PyTypeChecker
            self._handle_function_enter(
                event, event_file=id_, id_=0, thread_id=event.thread_id
            )
        elif event.event_type is EventType.FUNCTION_EXIT:
            # noinspection PyTypeChecker
            self._handle_function_exit(
                event, event_file=id_, id_=0, thread_id=event.thread_id
            )
        elif event.event_type is EventType.FUNCTION_ERROR:
            # noinspection PyTypeChecker
            self._handle_function_error(
                event, event_file=id_, id_=0, thread_id=event.thread_id
            )

    def _fork_features(self, event_file: EventFile, thread_id: int):
        """
        Snapshot the current feature vector and continue with a fresh copy.

        Records nothing on the tree; it only forks the per-thread feature
        vector (new ID, same accumulated features) so that observations already
        recorded on tree nodes are not retroactively mutated by later probes.
        Runs on every enter/exit — including off-footprint ones — so that the
        feature state at footprint nodes is identical whether or not pruning is
        active.

        :param event_file: Owning event file.
        :param thread_id: Thread whose feature vector is forked.
        """
        # Start a fresh feature vector for the new scope. The just-recorded
        # vector keeps its own features regardless of mode.
        old = self.current_features[event_file][thread_id]
        new = ForecastFeatureVector(
            self.id_gen.get_next_id(),
            old.result,
            context=self.context,
        )
        if self._carry_forward:
            # Legacy accumulating behavior: the new vector inherits every feature
            # seen so far along the path (O(runs x path-length) memory).
            new.features = old.get_features()
        # Otherwise the new vector begins empty and accumulates only what happens
        # until the next function boundary — LOCAL features. A caller's features
        # are already recorded (and discriminated) at the caller's own node, so
        # carrying them onto every descendant only duplicates them (the call tree
        # already encodes the path) and blows up memory.
        self.current_features[event_file][thread_id] = new

    def _handle_function_enter(
        self,
        event: FunctionEnterEvent,
        event_file: EventFile,
        id_: int,
        thread_id: int = None,
    ):
        """
        Navigate into the child node for the entered function and record entry
        features.

        Pushes the function onto the per-thread call stack, moves the current
        tree pointer to the corresponding child node (creating it if necessary),
        records the current feature vector at the node's entry, then resets the
        vector for subsequent probes within the function body.

        When pruning is active and this is a passing run, a function that no
        failing run reached (so the child node does not exist) opens an
        off-footprint region: nothing is recorded and the entire subtree is
        skipped until the matching exit, because such nodes are passing-only
        and always discarded by diagnosis.

        :param event: The enter event.
        :param event_file: Owning event file.
        :param id_: Probe ID (0 for function-boundary events).
        :param thread_id: Thread that fired the event.
        """
        function = self.get_function_repr(event)
        self.stack[event_file][thread_id].append(function)
        pruning = self._prune and not event_file.failing
        if pruning and self.offside[event_file][thread_id] > 0:
            # Already off the failing footprint: descend deeper without recording.
            self.offside[event_file][thread_id] += 1
            self._fork_features(event_file, thread_id)
            return
        current = self.current_trees[event_file][thread_id]
        if self._collapse_cycles:
            # Remember where to return on the matching exit, so navigation is
            # exact no matter how far back a cycle jumps.
            self.node_stack[event_file][thread_id].append(current)
            counts = self.path_count[event_file][thread_id]
            if counts[function]:
                # Already on the call stack: fold this turn of the cycle onto the
                # node of its outermost active occurrence instead of opening a
                # fresh subtree. Subsumes direct recursion (f -> f).
                self.current_trees[event_file][thread_id] = self.path_node[event_file][
                    thread_id
                ][function]
            else:
                child = current.get_child(function, create=not pruning)
                if child is None:
                    # Passing run entered a function no failing run reached.
                    self.offside[event_file][thread_id] = 1
                    self.node_stack[event_file][thread_id].pop()
                    self._fork_features(event_file, thread_id)
                    return
                self.current_trees[event_file][thread_id] = child
                self.path_node[event_file][thread_id][function] = child
            counts[function] += 1
        elif current.name != function:
            child = current.get_child(function, create=not pruning)
            if child is None:
                # Passing run entered a function no failing run reached: record
                # nothing here and skip its whole subtree.
                self.offside[event_file][thread_id] = 1
                self._fork_features(event_file, thread_id)
                return
            self.current_trees[event_file][thread_id] = child
        # The sliding-window cap is PER RUN, so key it on the run id (the event
        # file), not the probe id (which is 0 for function boundaries). Keying
        # on 0 would cap 4 observations *total* per node and evict across runs,
        # silently dropping other runs' observations at hot nodes.
        self.current_trees[event_file][thread_id].enter(
            self.current_features[event_file][thread_id], id_=event_file.run_id
        )
        # Snapshot current features and start a fresh vector inside the function
        self._fork_features(event_file, thread_id)

    def _handle_function_exit(
        self,
        event: FunctionExitEvent,
        event_file: EventFile,
        id_: int,
        thread_id: int = None,
    ):
        """
        Delegate a normal function exit to :meth:`_handle_function_leave`.

        :param event: The exit event.
        :param event_file: Owning event file.
        :param id_: Probe ID.
        :param thread_id: Thread that fired the event.
        """
        self._handle_function_leave(event, event_file, id_, thread_id)

    def _handle_function_error(
        self,
        event: FunctionErrorEvent,
        event_file: EventFile,
        id_: int,
        thread_id: int = None,
    ):
        """
        Delegate an error-propagating function exit to
        :meth:`_handle_function_leave`.

        :param event: The error event.
        :param event_file: Owning event file.
        :param id_: Probe ID.
        :param thread_id: Thread that fired the event.
        """
        self._handle_function_leave(event, event_file, id_, thread_id)

    def _handle_function_leave(
        self,
        event: FunctionExitEvent | FunctionErrorEvent,
        event_file: EventFile,
        id_: int,
        thread_id: int = None,
    ):
        """
        Record exit features and navigate back to the parent tree node.

        Pops the function from the call stack (with a mismatch warning if
        necessary), records the current feature vector at the node's exit, then
        moves the current tree pointer back to the parent node when the stack
        confirms we have left the function entirely.

        :param event: The leave event (exit or error).
        :param event_file: Owning event file.
        :param id_: Probe ID.
        :param thread_id: Thread that fired the event.
        """
        function = self.get_function_repr(event)
        stack = self.stack[event_file][thread_id]
        if stack:
            expected = stack.pop()
            if function != expected:
                # Expected and benign for interleaved generators/coroutines
                # (e.g. sympy's _preorder_traversal yielding into __next__),
                # where enter/exit legitimately do not nest. Kept at debug so it
                # does not flood (and I/O-throttle) generator-heavy builds.
                LOGGER.debug(
                    f"Function exit mismatch: expected {expected}, got {function}"
                )
                # Push it back so the stack stays consistent
                stack.append(expected)
        else:
            # A trace can start inside a call: the run may have been killed by a
            # timeout or have exhausted its trace budget, and the events before
            # the recorded ones are simply absent. Record the exit and carry on.
            LOGGER.debug(
                f"Function exit without a recorded entry: {function}. "
                "The trace is most likely truncated."
            )
        pruning = self._prune and not event_file.failing
        if pruning and self.offside[event_file][thread_id] > 0:
            # Leaving an off-footprint region: unwind depth, record nothing, and
            # stay on the footprint node we suspended at (no parent navigation).
            self.offside[event_file][thread_id] -= 1
            self._fork_features(event_file, thread_id)
            return
        # Per-run cap: key on the run id (event file), not the probe id. See
        # :meth:`_handle_function_enter`.
        self.current_trees[event_file][thread_id].exit(
            self.current_features[event_file][thread_id], id_=event_file.run_id
        )
        if self._collapse_cycles:
            # Restore the caller's node recorded on entry. This reproduces the
            # legacy "stay put on recursion, else go to parent" rule and also
            # unwinds a collapsed cycle correctly (exiting the inner f of
            # f -> g -> f returns to g, which is not f's tree parent).
            counts = self.path_count[event_file][thread_id]
            if counts[function]:
                counts[function] -= 1
                if not counts[function]:
                    del counts[function]
                    self.path_node[event_file][thread_id].pop(function, None)
            if self.node_stack[event_file][thread_id]:
                self.current_trees[event_file][thread_id] = self.node_stack[event_file][
                    thread_id
                ].pop()
        # Navigate back to the caller whenever we are no longer inside this
        # function. Staying put is correct only for direct recursion, where the
        # same function is still on the stack (mirroring the enter side, which
        # does not descend in that case).
        #
        # The stack being EMPTY also means we left the function -- we returned
        # to top level -- so it must navigate too. Requiring a non-empty stack
        # here left the pointer on the function that had just returned, and the
        # next top-level call was then created as its CHILD. Sequential calls
        # were chained instead of becoming siblings under the root: django's
        # import alone produced a 776-deep chain from a call tree that never
        # nests deeper than 22, with twice the nodes it should have.
        elif (
            not self.stack[event_file][thread_id]
            or self.stack[event_file][thread_id][-1] != function
        ):
            self.current_trees[event_file][thread_id] = self.current_trees[event_file][
                thread_id
            ].get_parent()
        # Snapshot features and reset for the caller's context
        self._fork_features(event_file, thread_id)

    def post_process(self, event_file: EventFile):
        """
        Finalize processing of *event_file* by recording the root node exit.

        Must be called after all events in *event_file* have been dispatched.
        Records the main-thread feature vector at the root's exit point so that
        the root node participates in discriminability analysis.

        :param event_file: The event file that has just been fully processed.
        """
        if event_file not in self.main_threads:
            # The file carried no events at all, so there is no run to record.
            LOGGER.debug(f"No events found in {event_file}; skipping.")
            return
        self.root.exit(
            self.current_features[event_file][self.main_threads[event_file]],
            id_=event_file.run_id,
        )


def build_share(
    config: dict, footprint: TracedRelevanceTree, event_files: list
) -> TracedRelevanceTree:
    """
    Build a partial tree from a share of the passing runs, in one process.

    At module level so it can be shipped to a worker.

    :param config: Keyword arguments describing the builder, so the worker
        constructs one configured exactly like the parent's.
    :param footprint: Structure-only skeleton of the failing footprint. The
        worker prunes against its own copy, so the tree it returns carries only
        its own observations and nothing else.
    :param event_files: The passing runs this worker is responsible for.
    :returns: The worker's partial tree, ready for :func:`merge_into`.
    """
    handler = ForecastHandler(footprint=footprint, **config)
    for event_file in event_files:
        handler.handle(event_file)
    return handler.tree


class ForecastHandler(EventHandler):
    """
    Top-level event handler that drives tree construction for FORECAST.

    Wraps a :class:`TreeBuilder` inside a
    :class:`~sflkit.model.parallel.ParallelModel` for concurrent event-file
    processing, then exposes the finished
    :class:`~forecast.tree.TracedRelevanceTree` via :attr:`tree`.

    :ivar context: Execution context depth forwarded to the builder.
    :ivar builder: The underlying tree-construction logic.
    :ivar model: Parallel dispatcher for event files.
    """

    def __init__(
        self,
        context: int = 0,
        workers: int = 4,
        prune: bool = True,
        footprint=None,
        carry_forward: bool = False,
        collapse_cycles: bool = None,
    ):
        """
        Initialize the handler.

        :param context: Execution context depth.  Defaults to 0.
        :param workers: Number of parallel worker threads for event processing.
            Defaults to 4.
        :param prune: When ``True`` (default) passing runs are pruned to the
            failing footprint. Requires all failing event files to be processed
            before any passing one. See :class:`TreeBuilder`.
        :param footprint: Optional pre-built skeleton root (parallel builder).
            See :class:`TreeBuilder`.
        :param carry_forward: When ``False`` (default) nodes record only LOCAL
            features; ``True`` restores the legacy accumulating behavior. See
            :class:`TreeBuilder`.
        :param collapse_cycles: Fold indirect recursion onto the node of the
            outermost active occurrence. Defaults to the
            ``FORECAST_COLLAPSE_CYCLES`` environment variable. See
            :class:`TreeBuilder`.
        """
        super().__init__(workers=workers, thread_support=True)
        self.context = context
        self.builder = TreeBuilder(
            context=context,
            prune=prune,
            footprint=footprint,
            carry_forward=carry_forward,
            collapse_cycles=collapse_cycles,
        )
        self.model = ParallelModel(self.builder, workers=workers)

    def handle(self, event_file: EventFile):
        """
        Process all events in *event_file* and finalize the tree node.

        Calls the parent :meth:`~sflkit.features.handler.EventHandler.handle`
        to dispatch individual events, then calls
        :meth:`TreeBuilder.post_process` to record the root exit vector.

        :param event_file: The event file to process.
        """
        super().handle(event_file)
        self.builder.post_process(event_file)

    def build(self, failing: list, passing: list, processes: int = 1):
        """
        Build the tree from *failing* runs first, then *passing* ones.

        :param failing: The failing runs.
        :param passing: The passing runs.
        :param processes: Worker processes to spread the passing runs over.
        :returns: The finished tree.

        The order is not a convenience, it is what pruning requires: a passing
        run is only recorded where a failing run has already been, so the
        failing footprint has to exist before any passing run is handled. That
        also makes the two phases fundamentally different. Failing runs grow
        the tree and so must agree on one structure, which is why they are
        handled here, in one process. Passing runs only add observations to
        nodes that already exist, so they are independent of one another: each
        worker prunes against its own copy of the footprint, and what it
        produces folds back with :func:`~forecast.tree.merge_into`.

        With ``processes`` above one, workers are started with
        :class:`~concurrent.futures.ProcessPoolExecutor`, so a caller on macOS
        or Windows must guard its entry point with
        ``if __name__ == "__main__":``.
        """
        for event_file in failing:
            self.handle(event_file)
        if not passing:
            return self.tree
        if processes <= 1 or len(passing) < 2:
            for event_file in passing:
                self.handle(event_file)
            return self.tree
        self._build_passing_in_parallel(passing, processes)
        return self.tree

    def _build_passing_in_parallel(self, passing: list, processes: int) -> None:
        """
        Spread the passing runs over worker processes and merge them back.

        :param passing: The passing runs.
        :param processes: Number of worker processes.
        """
        chunks: list[list] = [[] for _ in range(processes)]
        # Round-robin: runs differ wildly in length, so dealing them out keeps
        # one worker from drawing all the long ones.
        for index, event_file in enumerate(passing):
            chunks[index % processes].append(event_file)
        chunks = [chunk for chunk in chunks if chunk]
        # Structure only: a worker must contribute its own observations and
        # nothing else, or the failing runs' observations would be counted once
        # per worker when the shares are merged back.
        skeleton = structure_only_copy(self.tree)
        config = {
            "context": self.context,
            "workers": 1,
            "prune": self.builder._prune,
            "carry_forward": self.builder._carry_forward,
            "collapse_cycles": self.builder._collapse_cycles,
        }
        with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(build_share, config, skeleton, chunk)
                for chunk in chunks
            ]
            for future in futures:
                merge_into(self.root_tree, future.result())

    @property
    def root_tree(self):
        """The shared root, under its own name so :attr:`tree` stays the API."""
        return self.builder.root

    @property
    def tree(self):
        """
        The root :class:`~forecast.tree.TracedRelevanceTree` built so far.

        :returns: The shared root node populated during :meth:`handle` calls.
        """
        return self.builder.root
