"""
Traced relevance tree for FORECAST.

Defines :class:`TracedRelevanceTree`, a call-tree data structure that records
:class:`~sflkit.features.vector.FeatureVector` observations at each function's
entry and exit points across all test runs.  After event collection the tree is
analyzed by :class:`~forecast.diagnosis.Forecast` to identify discriminative
nodes and derive a cause-effect chain.

Supporting utilities
--------------------
* :class:`IDCounter` — per-run entry/exit counters with sliding-window capping.
* :class:`TreeVisitor` / :class:`PrintTreeVisitor` / :class:`ExhaustivePrintTreeVisitor`
  — visitor pattern for traversal and debugging.
"""

import gzip
import os
import pickle
from threading import Lock
from typing import List, Dict, Any, Optional, Set

from sflkit.features.value import Feature
from sflkit.features.vector import FeatureVector

#: Leading bytes of a gzip stream, used to detect compressed tree pickles.
_GZIP_MAGIC = b"\x1f\x8b"


def dump_tree(tree: Any, path: os.PathLike | str, compress: bool = True) -> None:
    """
    Pickle a relevance *tree* to *path*, gzip-compressed by default.

    Relevance trees are large and highly repetitive, so gzip typically shrinks
    the pickle by an order of magnitude. The on-disk name is unchanged
    (``tree.pkl``); :func:`load_tree` detects the codec from the file's magic
    bytes, so compressed and uncompressed trees are read transparently.

    :param tree: The tree (root node) to persist.
    :param path: Destination path.
    :param compress: When ``True`` (default) write a gzip-compressed pickle;
        when ``False`` write a plain pickle.

    The write is atomic: the tree is serialized to a temporary file in the same
    directory and only then renamed into place. Large trees take minutes to
    gzip, so a process killed mid-write (memory guard, OOM, operator) would
    otherwise leave a TRUNCATED file that still carries valid gzip magic bytes
    — it passes a cheap header check but raises ``EOFError`` when finally read.
    With the rename, a killed build leaves either the previous tree or nothing
    at all, never a plausible-looking broken one.
    """
    opener = gzip.open if compress else open
    target = os.fspath(path)
    tmp = f"{target}.tmp"
    try:
        with opener(tmp, "wb") as f:
            pickle.dump(tree, f)
        os.replace(tmp, target)
    except BaseException:
        # Do not leave a partial temp file behind on failure/interrupt.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_tree(path: os.PathLike | str) -> Any:
    """
    Load a relevance tree persisted by :func:`dump_tree`.

    The codec is detected from the leading magic bytes rather than the file
    name, so both gzip-compressed and plain pickles are read transparently.

    :param path: Path of the persisted tree.
    :returns: The unpickled tree (root node).
    """
    with open(path, "rb") as f:
        header = f.read(2)
    opener = gzip.open if header.startswith(_GZIP_MAGIC) else open
    with opener(path, "rb") as f:
        return pickle.load(f)


class IDCounter:
    """
    Per-run entry/exit counter for a :class:`TracedRelevanceTree` node.

    Tracks how many times a given run ID has entered and exited the node, and
    remembers the most recent feature vectors so that the per-run sliding-window
    cap in :meth:`TracedRelevanceTree.enter` / :meth:`TracedRelevanceTree.exit`
    can evict stale observations for hot loops.

    :ivar enter: Number of times this run has entered the node.
    :ivar exit: Number of times this run has exited the node.
    :ivar last_enter: The most recently appended entry vector (used for
        eviction when the cap is exceeded).
    :ivar last_exit: The most recently appended exit vector (used for
        eviction when the cap is exceeded).
    """

    def __init__(self):
        """Initialize all counters and last-vector references to zero / None."""
        self.enter = 0
        self.exit = 0
        self.last_enter: Optional[FeatureVector] = None
        self.last_exit: Optional[FeatureVector] = None


class TracedRelevanceTree:
    """
    A node in the function call tree annotated with execution-trace features.

    Each node corresponds to one function (identified by its canonical
    ``file:function:id`` string) and stores the feature vectors observed when
    tests entered and exited that function.  The tree is built incrementally by
    :class:`~forecast.handler.TreeBuilder` as event files are processed.

    Thread safety: :meth:`enter` and :meth:`exit` acquire an internal
    :class:`~threading.Lock` so concurrent event-file processing is safe.

    :ivar name: Canonical function identifier (``file:function:id``) or
        ``"FORECAST_ROOT"`` for the virtual root.
    :ivar parent: Parent node, or ``None`` for the root.
    :ivar enter_features: Feature vectors recorded at function entry across all
        runs (capped per run at 4).
    :ivar exit_features: Feature vectors recorded at function exit across all
        runs (capped per run at 4).
    :ivar children: Child nodes keyed by their canonical function name.
    :ivar all_features: Union of all :class:`~sflkit.features.value.Feature`
        objects observed anywhere in the subtree (only populated at the root).
    :ivar id_counter: Per-run-ID entry/exit counters.
    :ivar is_discriminative: ``True`` after
        :meth:`~forecast.diagnosis.Forecast._mark_discriminative_nodes` marks
        this node as separating passing from failing runs.
    """

    #: How many observations to keep per run at each node (sliding window of the
    #: first hits plus the last). This is PER RUN, so cross-run observations are
    #: never evicted. 2 = first + last hit per run. Even with LOCAL vectors, the
    #: high-run-count instances (matplotlib P2P=817, big sympy) blow up at cap 4
    #: because a node holds up to 4 x R vectors; cap 2 halves that so the biggest
    #: trees (was 60-121GB at cap 4) become buildable in parallel and diagnosable.
    _PER_RUN_CAP = 2

    def __init__(self, name: str, parent: Optional[Any] = None):
        """
        Initialize a new tree node.

        :param name: Canonical function identifier or ``"FORECAST_ROOT"``.
        :param parent: Parent node.  ``None`` for the root.
        """
        self.name: str = name
        self.parent: Optional[TracedRelevanceTree] = parent
        self.enter_features: List[FeatureVector] = list()
        self.exit_features: List[FeatureVector] = list()
        self.children: Dict[str, TracedRelevanceTree] = dict()
        self.all_features: Set[Feature] = set()
        self.id_counter: Dict[int, IDCounter] = dict()
        self.is_discriminative: bool = False
        self._lock: Lock = Lock()

    def __getstate__(self) -> dict:
        # The per-node lock guards concurrent enter/exit recording during a
        # parallel build; a ``threading.Lock`` is not picklable, so drop it.
        # It carries no state worth persisting and is recreated on load.
        state = dict(self.__dict__)
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = Lock()

    def __repr__(self):
        return f"TracedRelevanceTree(name={self.name}, children={list(self.children.keys())})"

    def __str__(self):
        return (
            f"{self.name}({', '.join(str(child) for child in self.children.values())})"
        )

    def get_child(self, name: str, create: bool = True) -> Optional[Any]:
        """
        Return the child node with *name*, optionally creating it.

        :param name: Canonical function identifier of the desired child.
        :param create: When ``True`` (default) a new child is created when
            *name* is not found.
        :returns: The child node, or ``None`` if it does not exist and
            *create* is ``False``.
        """
        if name in self.children:
            return self.children[name]
        elif create:
            child = TracedRelevanceTree(name, parent=self)
            self.children[name] = child
            return child
        return None

    def get_parent(self) -> Any:
        """
        Return the parent node, or *self* if this is the root.

        :returns: The parent :class:`TracedRelevanceTree` node (never ``None``).
        """
        if self.parent:
            return self.parent
        else:
            return self

    def _get_counter(self, id_: int):
        """
        Return the :class:`IDCounter` for run *id_*, creating it on first access.

        :param id_: Run identifier.
        :returns: The :class:`IDCounter` for that run.
        """
        if id_ not in self.id_counter:
            self.id_counter[id_] = IDCounter()
        return self.id_counter[id_]

    def enter(self, features: FeatureVector, id_: int):
        """
        Record *features* as an entry observation for run *id_*.

        Applies a sliding-window cap of :attr:`_PER_RUN_CAP` per run (first hit
        plus the sliding last hit): once that many entry observations have been
        recorded for *id_*, this run's previous last observation is evicted
        before the new one is appended. This prevents hot loops from dominating
        the feature set while keeping every run's observations.

        Thread-safe.

        :param features: Feature vector observed at function entry.
        :param id_: Run identifier.
        """
        with self._lock:
            counter = self._get_counter(id_)
            if counter.enter >= self._PER_RUN_CAP and counter.last_enter is not None:
                # Evict this run's previous observation to keep its count bounded.
                # Remove by IDENTITY (not value): equal-valued vectors from other
                # runs must not be evicted, or the kept set would depend on how
                # runs are grouped (which breaks order-invariance and the
                # parallel merge).
                self._evict_by_identity(self.enter_features, counter.last_enter)
            else:
                counter.enter += 1
            counter.last_enter = features
            self.enter_features.append(features)

    def exit(self, features: FeatureVector, id_: int):
        """
        Record *features* as an exit observation for run *id_*.

        Applies the same sliding-window cap as :meth:`enter`.

        Thread-safe.

        :param features: Feature vector observed at function exit.
        :param id_: Run identifier.
        """
        with self._lock:
            counter = self._get_counter(id_)
            if counter.exit >= self._PER_RUN_CAP and counter.last_exit is not None:
                # Evict by identity (see :meth:`enter`).
                self._evict_by_identity(self.exit_features, counter.last_exit)
            else:
                counter.exit += 1
            counter.last_exit = features
            self.exit_features.append(features)

    @staticmethod
    def _evict_by_identity(vectors: list, obj: FeatureVector) -> None:
        """
        Remove *obj* from *vectors* by identity (``is``), scanning from the end
        (the most recent append is near the tail). No-op if not present.
        """
        for i in range(len(vectors) - 1, -1, -1):
            if vectors[i] is obj:
                del vectors[i]
                return

    def get_all_features(self):
        """
        Return the global feature set, delegating to the root node.

        :returns: All :class:`~sflkit.features.value.Feature` objects observed
            anywhere in the tree.
        """
        if self.parent:
            return self.parent.get_all_features()
        else:
            return self.all_features


def structure_only_copy(root: "TracedRelevanceTree") -> "TracedRelevanceTree":
    """
    Copy the node *structure* of *root* (names + parent/child links) with EMPTY
    feature state (no enter/exit vectors, no id counters, no all_features).

    Used by the parallel builder as a pruning *skeleton*: passing-file workers
    build against their own copy of this skeleton (``create=False``), so each
    worker's tree carries only that worker's passing observations, which
    :func:`merge_into` then folds back onto the real (failing) footprint.

    Iterative to tolerate very deep call trees.
    """
    new_root = TracedRelevanceTree(root.name)
    stack = [(root, new_root)]
    while stack:
        src, dst = stack.pop()
        for name, child in src.children.items():
            new_child = TracedRelevanceTree(name, parent=dst)
            dst.children[name] = new_child
            stack.append((child, new_child))
    return new_root


def merge_into(base: "TracedRelevanceTree", partial: "TracedRelevanceTree") -> None:
    """
    Fold the observations of *partial* into *base* (same node structure).

    At each matching-path node, appends *partial*'s enter/exit feature vectors
    and merges its per-run id counters into *base*. Run IDs are globally unique
    per event file, so counters never collide and the per-run 4-vector cap
    (already applied during each partial build) is preserved by concatenation.
    Nodes present only in *partial* are created (defensive; with a shared
    skeleton this does not happen). Unions ``all_features`` at the root.

    Iterative to tolerate very deep call trees.
    """
    base.all_features |= partial.all_features
    stack = [(base, partial)]
    while stack:
        b, p = stack.pop()
        if p.enter_features:
            b.enter_features.extend(p.enter_features)
        if p.exit_features:
            b.exit_features.extend(p.exit_features)
        if p.id_counter:
            b.id_counter.update(p.id_counter)
        for name, p_child in p.children.items():
            b_child = b.children.get(name)
            if b_child is None:
                b_child = TracedRelevanceTree(name, parent=b)
                b.children[name] = b_child
            stack.append((b_child, p_child))


class TreeVisitor:
    """
    Abstract visitor for :class:`TracedRelevanceTree` traversal.

    Subclasses override :meth:`visit` to perform work at each node.
    Call :meth:`traverse` to start a full DFS traversal from a root node,
    or :meth:`dfs` to traverse from an arbitrary node.
    """

    def visit(self, node: TracedRelevanceTree, level: int = 0):
        """
        Visit a single tree node.

        :param node: The node being visited.
        :param level: Depth of *node* in the tree (root = 0).
        :raises NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("visit not implemented")

    def dfs(self, node: TracedRelevanceTree, level: int = 0):
        """
        Depth-first traversal starting at *node*.

        Calls :meth:`visit` on *node* before recursing into its children.

        :param node: Starting node.
        :param level: Depth of *node* (used to compute child depths).
        """
        if node is not None:
            self.visit(node, level)
            for child in node.children.keys():
                self.dfs(node.children[child], level + 1)

    def traverse(self, node: TracedRelevanceTree):
        """
        Prepare and then DFS-traverse the tree rooted at *node*.

        Calls :meth:`prepare` first to allow subclasses to initialize state,
        then calls :meth:`dfs`.

        :param node: Root of the subtree to traverse.
        """
        self.prepare(node)
        self.dfs(node)

    def prepare(self, node: TracedRelevanceTree):
        """
        Called once before traversal begins.  No-op by default.

        :param node: The root node passed to :meth:`traverse`.
        """
        pass


class PrintTreeVisitor(TreeVisitor):
    """
    Visitor that prints each node's name indented by its depth.

    Produces a compact ASCII-art representation of the call tree, e.g.::

        FORECAST_ROOT
           module.py:foo:0
              module.py:bar:1
    """

    def visit(self, node: TracedRelevanceTree, level: int = 0):
        """
        Print *node*'s name indented proportionally to *level*.

        :param node: The node to print.
        :param level: Depth of *node* in the tree.
        """
        if level == 0:
            print(node.name)
        else:
            print("   " * level + node.name)


class ExhaustivePrintTreeVisitor(TreeVisitor):
    """
    Visitor that prints each node's name together with its entry and exit
    feature vectors.

    More verbose than :class:`PrintTreeVisitor`; useful for debugging the
    feature-vector contents of individual nodes.
    """

    def visit(self, node: TracedRelevanceTree, level: int = 0):
        """
        Print *node*'s name, entry vectors, and exit vectors.

        :param node: The node to print.
        :param level: Depth of *node* in the tree.
        """
        if level == 0:
            print(node.name)
            print(f"  Enter: {node.enter_features}")
            print(f"  Exit: {node.exit_features}")
        else:
            print("    " * level + node.name)
            print("    " * level + f"  Enter: {node.enter_features}")
            print("    " * level + f"  Exit: {node.exit_features}")
