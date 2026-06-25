from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Protocol
import numpy as np
import math
import random


# ------------------------------- Geometry ------------------------------------

@dataclass
class EnvelopeNDLite:
    """Lightweight N-D axis-aligned rectangle (min/max per dimension)."""
    mins: np.ndarray  # shape (D,)
    maxs: np.ndarray  # shape (D,)

    def __post_init__(self):
        self.mins = np.asarray(self.mins, dtype=float)
        self.maxs = np.asarray(self.maxs, dtype=float)
        assert self.mins.shape == self.maxs.shape
        assert self.mins.ndim == 1
        # normalize
        bad = self.mins > self.maxs
        if np.any(bad):
            a = self.mins.copy(); b = self.maxs.copy()
            self.mins = np.minimum(a, b); self.maxs = np.maximum(a, b)

    def getCoordinateDimension(self) -> int:
        return int(self.mins.size)

    def len(self) -> int:
        return self.mins.size()
    
    def isEmpty(self) -> bool:
        return bool(np.any(self.maxs <= self.mins))

    def getMinCoord(self, d: int) -> float:
        return float(self.mins[d])

    def getMaxCoord(self, d: int) -> float:
        return float(self.maxs[d])

    def setCoordinateDimension(self, d: int):
        if self.mins.size != d:
            self.mins = np.full(d, np.inf)
            self.maxs = np.full(d, -np.inf)

    def setEmpty(self):
        self.mins[:] = np.inf
        self.maxs[:] = -np.inf

    def merge_point(self, coord: np.ndarray):
        self.mins = np.minimum(self.mins, coord)
        self.maxs = np.maximum(self.maxs, coord)

    def merge_box(self, other: "EnvelopeNDLite"):
        self.mins = np.minimum(self.mins, other.mins)
        self.maxs = np.maximum(self.maxs, other.maxs)

    def copy(self) -> "EnvelopeNDLite":
        return EnvelopeNDLite(self.mins.copy(), self.maxs.copy())

    def area(self) -> float:
        side = np.maximum(0.0, self.maxs - self.mins)
        vol = float(np.prod(side))
        return vol

    def margin(self) -> float:
        # R*-tree uses sum of edge lengths (perimeter in 2D, "surface area" proxy in N-D)
        side = np.maximum(0.0, self.maxs - self.mins)
        if side.size == 0:
            return 0.0
        # generalized "Manhattan perimeter": 2 * sum(side) in 2D; here use sum(side) as a proxy
        return float(np.sum(side))

    @staticmethod
    def from_points(coords: np.ndarray) -> "EnvelopeNDLite":
        # coords: (D, N) or (N, D)
        if coords.ndim != 2:
            raise ValueError("coords must be 2-D")
        if coords.shape[0] < coords.shape[1]:  # assume (D,N)
            mins = np.min(coords, axis=1)
            maxs = np.max(coords, axis=1)
        else:  # (N,D)
            mins = np.min(coords, axis=0)
            maxs = np.max(coords, axis=0)
        return EnvelopeNDLite(mins, maxs)


class KDPartitionTree:
    """
    kd-tree-like split structure for point partition lookup.

    While building, every row is either an internal split or an explicit leaf.
    Leaves are marked with split axis ``-1`` and have stable row IDs, which lets
    callers split a specific leaf later. The tree starts with one leaf at row 0
    that represents the entire space.

    After ``assign_partition_ids()``, the table is compacted to contain only
    internal split nodes. Child references to leaves are replaced with negative
    partition references encoded as ``-partition_id - 1``.
    """

    LEAF_AXIS = -1

    def __init__(self):
        self._rows = self._leaf_row()
        self._finalized = False
        self._num_partitions = 1

    @property
    def rows(self) -> np.ndarray:
        """Return the raw split table."""
        return self._rows

    def __len__(self) -> int:
        return int(self._rows.shape[0])

    def is_empty(self) -> bool:
        return len(self) == 0

    def numPartitions(self) -> int:
        return self._num_partitions

    def add_split(self, node_id: int, split_axis: int, split_value: float) -> Tuple[int, int]:
        """
        Split a leaf node and return the lower and higher leaf node IDs.

        ``node_id`` must point to an existing leaf row, identified by split
        axis ``-1``. The row is updated in place with the split information,
        and two new leaf rows are appended as its lower and higher children.
        """
        if self._finalized:
            raise RuntimeError("Cannot add splits after partition IDs have been assigned")
        if split_axis < 0:
            raise ValueError("split_axis cannot be negative")
        self._validate_node_id(node_id)
        if int(self._rows[node_id, 1]) != self.LEAF_AXIS:
            raise ValueError(f"Node {node_id} is already an internal split")

        lower_node_id = self._append_leaf()
        higher_node_id = self._append_leaf()
        self._rows[node_id] = [
            float(split_value),
            float(split_axis),
            float(lower_node_id),
            float(higher_node_id),
        ]
        return lower_node_id, higher_node_id

    def assign_partition_ids(self) -> int:
        """
        Compact the tree and replace leaves with ``-partition_id - 1``.

        Leaves are numbered by pre-order traversal from 0 to
        ``num_partitions - 1``. After this method returns, the intermediate
        explicit leaf rows have been removed and the tree is ready for search.
        """
        if self._finalized:
            return self._num_partitions

        next_partition_id = 0
        compact_rows: List[List[float]] = []

        def compile_node(node_id: int) -> int:
            nonlocal next_partition_id
            self._validate_node_id(node_id)
            split_axis = int(self._rows[node_id, 1])
            if split_axis == self.LEAF_AXIS:
                partition_ref = -next_partition_id - 1
                next_partition_id += 1
                return partition_ref
            if split_axis < 0:
                raise RuntimeError(f"Invalid split axis {split_axis} at node {node_id}")

            compact_node_id = len(compact_rows)
            compact_rows.append([float(self._rows[node_id, 0]), float(split_axis), 0.0, 0.0])

            lower_ref = compile_node(int(self._rows[node_id, 2]))
            higher_ref = compile_node(int(self._rows[node_id, 3]))
            compact_rows[compact_node_id][2] = float(lower_ref)
            compact_rows[compact_node_id][3] = float(higher_ref)
            return compact_node_id

        root_ref = compile_node(0)
        self._rows = (
            np.asarray(compact_rows, dtype=np.float64)
            if compact_rows
            else np.empty((0, 4), dtype=np.float64)
        )
        self._finalized = True
        self._num_partitions = next_partition_id
        if root_ref != 0 and compact_rows:
            raise RuntimeError("Compacted tree root was not written at row 0")
        return self._num_partitions

    def search(self, x: float, y: float) -> int:
        """Return the partition ID containing point ``(x, y)``."""
        return self.search_point((x, y))

    def search_point(self, coords) -> int:
        """Return the partition ID containing a point coordinate sequence."""
        if not self._finalized:
            raise RuntimeError("Partition IDs must be assigned before searching")
        if len(self) == 0 or len(coords) == 0:
            return 0

        node_id = 0
        while True:
            split_value = float(self._rows[node_id, 0])
            split_axis = int(self._rows[node_id, 1])
            if split_axis >= len(coords):
                raise ValueError(
                    f"Point has {len(coords)} dimensions, but split uses axis {split_axis}"
                )

            side_col = 2 if float(coords[split_axis]) <= split_value else 3
            child_ref = int(self._rows[node_id, side_col])
            if child_ref < 0:
                return -child_ref - 1
            if child_ref == 0:
                raise RuntimeError("Encountered an unfinished terminal leaf during search")
            self._validate_node_id(child_ref)
            node_id = child_ref

    def getPartitionMBR(self, partitionID: int, global_mbr: EnvelopeNDLite) -> EnvelopeNDLite:
        """Return the partition MBR implied by split paths through the tree."""
        if not self._finalized:
            raise RuntimeError("Partition IDs must be assigned before computing partition MBRs")
        if partitionID < 0 or partitionID >= self._num_partitions:
            raise IndexError(f"Partition ID {partitionID} is outside the tree")
        if len(self) == 0:
            return global_mbr.copy()

        def visit(node_id: int, mins: np.ndarray, maxs: np.ndarray) -> Optional[EnvelopeNDLite]:
            split_value = float(self._rows[node_id, 0])
            split_axis = int(self._rows[node_id, 1])

            lower_ref = int(self._rows[node_id, 2])
            lower_mins = mins.copy()
            lower_maxs = maxs.copy()
            lower_maxs[split_axis] = min(lower_maxs[split_axis], split_value)
            found = visit_ref(lower_ref, lower_mins, lower_maxs)
            if found is not None:
                return found

            higher_ref = int(self._rows[node_id, 3])
            higher_mins = mins.copy()
            higher_maxs = maxs.copy()
            higher_mins[split_axis] = max(higher_mins[split_axis], split_value)
            return visit_ref(higher_ref, higher_mins, higher_maxs)

        def visit_ref(ref: int, mins: np.ndarray, maxs: np.ndarray) -> Optional[EnvelopeNDLite]:
            if ref < 0:
                pid = -ref - 1
                if pid == partitionID:
                    return EnvelopeNDLite(mins, maxs)
                return None
            return visit(ref, mins, maxs)

        result = visit(0, global_mbr.mins.copy(), global_mbr.maxs.copy())
        if result is None:
            raise IndexError(f"Partition ID {partitionID} was not found in the tree")
        return result

    def _append_leaf(self) -> int:
        self._rows = np.vstack([self._rows, self._leaf_row()])
        return len(self) - 1

    @classmethod
    def _leaf_row(cls) -> np.ndarray:
        return np.array([[np.nan, float(cls.LEAF_AXIS), 0.0, 0.0]], dtype=np.float64)

    def _validate_node_id(self, node_id: int) -> None:
        if node_id < 0 or node_id >= len(self):
            raise IndexError(f"Node ID {node_id} is outside the tree")


# ----------------------------- Histogram API ---------------------------------

class AbstractHistogram(Protocol):
    """Minimal protocol to interoperate with computePointWeights()."""
    def getCoordinateDimension(self) -> int: ...
    def getNumBins(self) -> int: ...
    def getBinID(self, coords: np.ndarray) -> int: ...
    def getBinValue(self, bin_id: int) -> int: ...


# ----------------------------- R*-like splitter -------------------------------

def _overlap_volume(a: EnvelopeNDLite, b: EnvelopeNDLite) -> float:
    lo = np.maximum(a.mins, b.mins)
    hi = np.minimum(a.maxs, b.maxs)
    side = np.maximum(0.0, hi - lo)
    return float(np.prod(side))


def _choose_split(coords: np.ndarray,
                  start: int,
                  end: int,
                  w: Optional[np.ndarray],
                  m: float,
                  M: float,
                  fraction_min_split: float) -> Tuple[int, int, float]:
    """
    R*-style split selection on the given subset (columns in coords).
    Sorts the target slice of coords (and weights) in-place by the chosen axis and returns the
    split position within that slice.
    - Examine each axis
    - For each axis, sort by coordinate, consider candidate split positions
      respecting m (min on each side) and (M) capacity target (soft). We allow
      full range but optionally thin with fraction_min_split in (0..0.5].
    - First criterion: minimal sum of margins of the two groups
    - Tie-breaker: minimal overlap of the two MBRs
    - Tie-breaker: minimal total area
    """
    D, _ = coords.shape
    n = end - start
    assert n >= 2, "Need at least 2 points to split"

    has_weights = w is not None
    sorted_axis: Optional[int] = None  # track last axis we sorted by to avoid rescanning

    def _quicksort_inplace(axis: int):
        """Sort coords[:, start:end] (and weights) in-place using quicksort on coords[axis]."""
        nonlocal sorted_axis
        if n <= 1 or sorted_axis == axis:
            return
        coord_axis = coords[axis]
        stack: List[Tuple[int, int]] = [(0, n - 1)]

        def _median_of_three_pos(lo: int, hi: int) -> int:
            mid = (lo + hi) // 2
            a_val = coord_axis[start + lo]
            b_val = coord_axis[start + mid]
            c_val = coord_axis[start + hi]
            if (a_val <= b_val <= c_val) or (c_val <= b_val <= a_val):
                return mid
            if (b_val <= a_val <= c_val) or (c_val <= a_val <= b_val):
                return lo
            return hi

        def _swap(i_pos: int, j_pos: int):
            if i_pos == j_pos:
                return
            a, b = start + i_pos, start + j_pos
            coords[:, [a, b]] = coords[:, [b, a]]
            if has_weights:
                w[a], w[b] = w[b], w[a]

        while stack:
            lo, hi = stack.pop()
            while lo < hi:
                i, j = lo, hi
                pivot_pos = _median_of_three_pos(lo, hi)
                pivot_idx = start + pivot_pos
                pivot = coord_axis[pivot_idx]
                while i <= j:
                    while True:
                        cur_idx = start + i
                        cur_val = coord_axis[cur_idx]
                        if (cur_val < pivot) or (cur_val == pivot and cur_idx < pivot_idx):
                            i += 1
                            continue
                        break
                    while True:
                        cur_idx = start + j
                        cur_val = coord_axis[cur_idx]
                        if (cur_val > pivot) or (cur_val == pivot and cur_idx > pivot_idx):
                            j -= 1
                            continue
                        break
                    if i <= j:
                        _swap(i, j)
                        i += 1
                        j -= 1
                if (j - lo) < (hi - i):
                    if i < hi:
                        stack.append((i, hi))
                    hi = j
                else:
                    if lo < j:
                        stack.append((lo, j))
                    lo = i
        sorted_axis = axis

    def _best_split_for_axis(k_candidates: List[int]) -> Tuple[Optional[Tuple[float, float, float]], Optional[int]]:
        """
        Compute prefix/suffix MBRs once and evaluate candidate split positions.
        Returns (best_score, best_k) for the current axis.
        """
        D_local = coords.shape[0]
        left_min = np.empty((D_local, n), dtype=float)
        left_max = np.empty((D_local, n), dtype=float)
        right_min = np.empty((D_local, n), dtype=float)
        right_max = np.empty((D_local, n), dtype=float)

        first_pt = coords[:, start]
        left_min[:, 0] = first_pt
        left_max[:, 0] = first_pt
        for i in range(1, n):
            pt = coords[:, start + i]
            np.minimum(left_min[:, i - 1], pt, out=left_min[:, i])
            np.maximum(left_max[:, i - 1], pt, out=left_max[:, i])

        last_pt = coords[:, start + n - 1]
        right_min[:, -1] = last_pt
        right_max[:, -1] = last_pt
        for i in range(n - 2, -1, -1):
            pt = coords[:, start + i]
            np.minimum(right_min[:, i + 1], pt, out=right_min[:, i])
            np.maximum(right_max[:, i + 1], pt, out=right_max[:, i])

        best_axis = None
        best_axis_k = None
        for k in k_candidates:
            if k + 1 >= n:
                continue
            lmin = left_min[:, k]
            lmax = left_max[:, k]
            rmin = right_min[:, k + 1]
            rmax = right_max[:, k + 1]

            side_l = np.maximum(0.0, lmax - lmin)
            side_r = np.maximum(0.0, rmax - rmin)
            score_margin = float(np.sum(side_l) + np.sum(side_r))
            overlap_side = np.maximum(0.0, np.minimum(lmax, rmax) - np.maximum(lmin, rmin))
            score_overlap = float(np.prod(overlap_side))
            score_area = float(np.prod(side_l) + np.prod(side_r))

            cand = (score_margin, score_overlap, score_area)
            if (best_axis is None) or (cand < best_axis):
                best_axis = cand
                best_axis_k = k + 1  # split position (exclusive)

        return best_axis, best_axis_k
    # Candidate split positions must leave >= m on each side (in weight terms).
    if w is None:
        total_w = float(n)
    else:
        total_w = float(np.sum(w[start:end]))
    min_side_w = float(m)
    # Boundaries in index space: we'll use cumulative weights to honor m and M
    # Build a helper over a sorted order per axis, so thresholds translate via prefix sums.

    best = None  # (score_margin, score_overlap, score_area)
    best_axis_id = None
    best_k = None

    for axis in range(D):
        _quicksort_inplace(axis)
        # left_min, left_max, right_min, right_max = _compute_bounds()

        if has_weights:
            prefix = np.cumsum(w[start:end]).astype(float, copy=False)  # cum weights
        else:
            prefix = np.arange(1, n + 1, dtype=float)
        # valid split positions are between elements: at k means left=[0:k], right=[k:n]
        # require both sides >= min_side_w
        left_ok = prefix >= min_side_w
        right_ok = (total_w - prefix) >= min_side_w
        valid = left_ok & right_ok

        if not np.any(valid):
            # fall back to a median split if nothing valid
            k_candidates = [n // 2]
        else:
            k_valid = np.nonzero(valid)[0]  # positions 0..n-1 (split after k)
            # thin candidates per fraction_min_split (like Java's fraction)
            if fraction_min_split > 0.0:
                # keep a band around mid (e.g., 0.0=all, 0.25=middle 50%)
                lo = int((1.0 - fraction_min_split) * 0.5 * len(k_valid))
                hi = int((1.0 + fraction_min_split) * 0.5 * len(k_valid))
                if hi <= lo:  # ensure at least one candidate
                    lo = 0; hi = len(k_valid)
                k_valid = k_valid[lo:hi]
            k_candidates = k_valid.tolist()

        # Evaluate candidates on R*-criteria
        best_axis, best_axis_k = _best_split_for_axis(k_candidates)

        if best_axis is None:
            continue

        if (best is None) or (best_axis < best):
            best = best_axis
            best_axis_id = axis
            best_k = best_axis_k

    if best_axis_id is None or best_k is None:
        _quicksort_inplace(0)
        best_axis_id = 0
        best_k = n // 2

    # Ensure the slice is sorted by the chosen axis before returning split position.
    _quicksort_inplace(best_axis_id)
    split_at = start + best_k
    split_value = float(coords[best_axis_id, split_at - 1])
    return split_at, best_axis_id, split_value


def _rstar_partition_iterative(coords: np.ndarray,
                               w: Optional[np.ndarray],
                               min_cap: float,
                               max_cap: float,
                               fraction_min_split: float) -> KDPartitionTree:
    """
    Iteratively partition indices into boxes with capacity in [min_cap, max_cap]
    (capacity = count if w is None, else sum(weights) when w provided).
    """
    import logging
    logger = logging.getLogger("RSGrovePartitioner._rstar_partition_iterative")
    tree = KDPartitionTree()
    stack: List[Tuple[int, int, int]] = [(0, coords.shape[1], 0)]

    while stack:
        start, end, node_id = stack.pop()
        subset_size = end - start
        logger.debug(f"Stack pop start={start}, end={end}, node_id={node_id}, subset_size={subset_size}")
        if subset_size <= 0:
            continue

        if w is None:
            cap_here = float(subset_size)
        else:
            cap_here = float(np.sum(w[start:end]))

        logger.debug(f"Subset start={start}, end={end}: cap_here={cap_here}, max_cap={max_cap}")
        if cap_here <= max_cap:
            logger.debug(f"Subset start={start}, end={end}: within capacity, creating box.")
            continue

        split_at, split_axis, split_value = _choose_split(
            coords, start, end, w, min_cap, max_cap, fraction_min_split
        )
        logger.debug(
            "Subset start=%d, end=%d: split_at=%d axis=%d value=%s",
            start, end, split_at, split_axis, split_value,
        )

        if split_at <= start or split_at >= end:
            logger.warning(f"Subset start={start}, end={end}: Pathological split detected, splitting by median.")
            split_at = start + subset_size // 2
            split_axis = 0
            order = np.argsort(coords[split_axis, start:end], kind="mergesort")
            coords[:, start:end] = coords[:, start:end][:, order]
            if w is not None:
                w[start:end] = w[start:end][order]
            split_value = float(coords[split_axis, split_at - 1])

        lower_node_id, higher_node_id = tree.add_split(node_id, split_axis, split_value)
        stack.append((split_at, end, higher_node_id))
        stack.append((start, split_at, lower_node_id))

    tree.assign_partition_ids()
    return tree


def partition_points(coords: np.ndarray,
                     min_cap: int,
                     max_cap: int,
                     fraction_min_split: float) -> KDPartitionTree:
    import logging
    logger = logging.getLogger("RSGrovePartitioner.partition_points")
    weights = np.ones(coords.shape[1], dtype=float)
    logger.info(f"Starting partition_weighted_points with {coords.shape[1]} points (uniform weights).")
    return partition_weighted_points(coords, weights, float(min_cap), float(max_cap), fraction_min_split)


def partition_weighted_points(coords: np.ndarray,
                              weights: np.ndarray,
                              min_cap_w: float,
                              max_cap_w: float,
                              fraction_min_split: float) -> KDPartitionTree:
    """Weighted partitioning (capacities based on data sizes)."""
    _, N = coords.shape
    return _rstar_partition_iterative(coords, weights.astype(float), float(min_cap_w), float(max_cap_w),
                               fraction_min_split)


# --------------------------- Partitioner (public) -----------------------------

class RSGrovePartitioner:
    """
    Python implementation inspired by Beast's RSGrovePartitioner with R*-style splitting.

    Methods:
      - setup(mmRatio)
      - construct(summary, sample, histogram, numPartitions)
      - overlapPartitions(mbr: EnvelopeNDLite, out: Optional[IntArray]) -> IntArray
      - overlapPartition(mbr: EnvelopeNDLite) -> int
      - numPartitions() -> int
      - isDisjoint() -> bool
      - getCoordinateDimension() -> int
      - getPartitionMBR(partitionID, mbr_out: EnvelopeNDLite)
      - getEnvelope() -> EnvelopeNDLite

    Notes:
      * `summary` must expose:
          - getCoordinateDimension()
          - getMinCoord(d), getMaxCoord(d)
          - getSideLength(d)
          - (or) mins/maxs arrays; we only need the global MBR.
      * If `histogram` is provided (AbstractHistogram), weighted partitioning is used.
    """

    def __init__(self, mmRatio: float = 0.95, minSplitSize: float = 0.0):
        import logging
        self.logger = logging.getLogger("RSGrovePartitioner")
        # Config / state
        self.mMRatio: float = mmRatio
        self.fractionMinSplitSize: float = minSplitSize
        self.searchTree: KDPartitionTree = KDPartitionTree()

        self._rng = random.Random()

    # ---------- API parity ----------
    def construct(self,
                  summary,
                  sample: np.ndarray,
                  histogram: Optional[AbstractHistogram],
                  numPartitions: int):
        self.logger.info(f"Constructing partitions with numPartitions={numPartitions}")
        """
        summary: exposes coordinate dimension and bounds (min/max per dim or getMinCoord/getMaxCoord)
        sample: np.ndarray with shape (D, N)
        histogram: optional AbstractHistogram (weighted mode)
        """
        # Handle empty sample: fabricate uniform points within summary MBR (like Java)
        def _summary_dims():
            return int(summary.getCoordinateDimension())

        D = _summary_dims()
        self.logger.info(f"Summary dimension: {D}")
        # Merge summary to mbrPoints
        mins = np.array([summary.getMinCoord(d) for d in range(D)], dtype=float)
        maxs = np.array([summary.getMaxCoord(d) for d in range(D)], dtype=float)
        self.mbrPoints = EnvelopeNDLite(mins, maxs)

        if sample.size == 0:
            self.logger.warning("Sample is empty, fabricating points within summary MBR.")
            Nf = 1000
            fabricated = np.zeros((D, Nf), dtype=float)
            for d in range(D):
                fabricated[d, :] = np.random.rand(Nf) * (maxs[d] - mins[d]) + mins[d]
            sample = fabricated

        assert self.mMRatio > 0, "mMRatio cannot be zero. Call setup() first."

        if sample.shape[0] != D:
            self.logger.error(f"Sample shape mismatch: got {sample.shape[0]}, expected {D}")
            raise ValueError(f"sample must have shape (D, N) with D={D}")

        N = sample.shape[1]
        self.logger.info(f"Sample size: {N}")

        if histogram is None:
            # Unweighted mode: choose M,m from sample count
            M = int(math.ceil(N / float(numPartitions)))
            m = int(math.ceil(self.mMRatio * M))
            self.logger.info(f"Unweighted partitioning: M={M}, m={m}")
            self.logger.info(f"Calling KD-tree partitioning with sample shape {sample.shape}, min_cap={m}, max_cap={M}, fraction_min_split={self.fractionMinSplitSize}")
            weights = np.ones(sample.shape[1], dtype=float)
            self.searchTree = _rstar_partition_iterative(
                sample,
                weights,
                float(m),
                float(M),
                self.fractionMinSplitSize,
            )
        else:
            # Weighted mode: compute point weights from histogram, then split by total size
            weights = self.computePointWeights(sample, histogram)  # long[] in Java
            total_size = float(np.sum(weights))
            M = float(math.ceil(total_size / float(numPartitions)))
            m = float(total_size * self.mMRatio / float(numPartitions))
            self.logger.info(f"Weighted partitioning: total_size={total_size}, M={M}, m={m}")
            self.searchTree = _rstar_partition_iterative(
                sample,
                weights.astype(float),
                m,
                M,
                self.fractionMinSplitSize,
            )

        # Store min/max arrays
        P = self.searchTree.numPartitions()
        self.logger.info(f"Constructed {P} KD-tree partitions.")

    # ---------- Helpers (API-compatible) ----------

    @staticmethod
    def computePointWeights(sample: np.ndarray, histogram: AbstractHistogram) -> np.ndarray:
        D, N = sample.shape
        assert D == histogram.getCoordinateDimension()
        num_bins = histogram.getNumBins()
        counts = np.zeros(num_bins, dtype=int)
        # First pass: count points per bin
        tmp = np.empty(D, dtype=float)
        bin_ids = np.empty(N, dtype=int)
        for i in range(N):
            tmp[:] = sample[:, i]
            b = histogram.getBinID(tmp)
            bin_ids[i] = b
            counts[b] += 1
        # Second pass: distribute bin weight to points in the bin
        weights = np.zeros(N, dtype=np.int64)
        for i in range(N):
            b = bin_ids[i]
            bin_val = histogram.getBinValue(b)
            w = 0 if counts[b] == 0 else int(bin_val // max(1, counts[b]))
            weights[i] = max(0, w)
        return weights

    def numPartitions(self) -> int:
        return self.searchTree.numPartitions()

    def getPartitionMBR(self, partition_id: int, global_mbr: EnvelopeNDLite) -> EnvelopeNDLite:
        return self.searchTree.getPartitionMBR(partition_id, global_mbr)

    def overlapPartition(self, mbr_or_coords) -> int:
        if self.numPartitions() == 0:
            return -1
        if isinstance(mbr_or_coords, EnvelopeNDLite):
            if (
                not np.all(np.isfinite(mbr_or_coords.mins))
                or not np.all(np.isfinite(mbr_or_coords.maxs))
            ):
                coords = self.mbrPoints.mins.copy()
            else:
                coords = (mbr_or_coords.mins + mbr_or_coords.maxs) * 0.5
        else:
            coords = mbr_or_coords
        return self.searchTree.search_point(coords)
