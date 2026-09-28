"""Reference implementation of a stable incremental codebook update."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, log1p, sqrt
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


Vector = Sequence[float]
CenterMap = Mapping[str, Mapping[int, Vector]]

__all__ = [
    "CenterMap",
    "CodeAssignment",
    "ItemSnapshot",
    "UpdateConfig",
    "UpdateResult",
    "assign_leaf_ids_stably",
    "choose_cluster_stably",
    "cosine_distance",
    "fit_order_signal_ranges",
    "l3_order_score",
    "nearest_cluster",
    "select_drift_candidates",
    "update_codebook",
]


@dataclass(frozen=True)
class UpdateConfig:
    # Candidate gate: cosine distance > 0.02 (cosine similarity < 0.98).
    drift_cosine_threshold: float = 0.02

    # Confirmation gate. Distances below are squared Euclidean distances, so
    # 2.0 corresponds to an ordinary-distance ratio greater than sqrt(2).
    reassignment_distance_ratio: float = 2.0

    # Batch-level safety budget. The highest-drift candidates are retained.
    max_candidate_fraction: float = 0.05

    # Select exactly one interpretable L3 ordering signal. Use "price_level"
    # for price-aware ordering or "click_count" for popularity-aware ordering.
    l3_order_signal: str = "price_level"
    l3_order_buckets: int = 50

    # Click counts are usually log-scaled because they are heavy-tailed.
    # Set descending_order_signal=True if hot items should receive smaller IDs.
    log_scale_order_signal: bool = False
    descending_order_signal: bool = False
    max_leaf_slots: int = 2_000

    def __post_init__(self) -> None:
        if not 0.0 <= self.drift_cosine_threshold <= 2.0:
            raise ValueError("drift_cosine_threshold must be in [0, 2]")
        if self.reassignment_distance_ratio <= 0.0:
            raise ValueError("reassignment_distance_ratio must be positive")
        if not 0.0 <= self.max_candidate_fraction <= 1.0:
            raise ValueError("max_candidate_fraction must be in [0, 1]")
        if not self.l3_order_signal:
            raise ValueError("l3_order_signal must not be empty")
        if self.l3_order_buckets <= 0:
            raise ValueError("l3_order_buckets must be positive")
        if self.max_leaf_slots <= 0:
            raise ValueError("max_leaf_slots must be positive")


@dataclass(frozen=True)
class ItemSnapshot:
    item_id: str
    semantic_group: str
    embedding: Vector
    # Example keys: "price_level" and "click_count".
    l3_signals: Mapping[str, float]


@dataclass(frozen=True)
class CodeAssignment:
    semantic_group: str
    cluster_id: int
    leaf_id: int


@dataclass(frozen=True)
class UpdateResult:
    assignments: Mapping[str, CodeAssignment]
    drift_candidates: frozenset[str]
    reassigned_items: frozenset[str]


def _dot(left: Vector, right: Vector) -> float:
    return sum(x * y for x, y in zip(left, right))


def cosine_distance(left: Vector, right: Vector) -> Optional[float]:
    """Return 1 - cosine similarity, or None for incompatible vectors."""
    if not left or len(left) != len(right):
        return None
    left_norm = sqrt(_dot(left, left))
    right_norm = sqrt(_dot(right, right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    similarity = _dot(left, right) / (left_norm * right_norm)
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def squared_l2(vector: Vector, center: Vector) -> float:
    if len(vector) != len(center):
        raise ValueError("embedding and center dimensions must match")
    return sum((x - y) ** 2 for x, y in zip(vector, center))


def select_drift_candidates(
    current_items: Mapping[str, ItemSnapshot],
    previous_items: Mapping[str, ItemSnapshot],
    config: UpdateConfig,
) -> frozenset[str]:
    """Apply the cosine gate, followed by the batch-level safety budget."""
    candidates = []
    for item_id, current in current_items.items():
        previous = previous_items.get(item_id)
        if previous is None:
            continue  # A new item is assigned separately; it is not "drifted".
        distance = cosine_distance(current.embedding, previous.embedding)
        if distance is not None and distance > config.drift_cosine_threshold:
            candidates.append((distance, item_id))

    candidates.sort(key=lambda pair: (-pair[0], pair[1]))
    budget = floor(len(current_items) * config.max_candidate_fraction)
    if budget <= 0:
        return frozenset()
    if len(candidates) > budget:
        candidates = candidates[:budget]
    return frozenset(item_id for _, item_id in candidates)


def nearest_cluster(
    item: ItemSnapshot,
    centers: CenterMap,
) -> Tuple[int, float]:
    group_centers = centers.get(item.semantic_group)
    if not group_centers:
        raise ValueError(f"no centers for semantic group {item.semantic_group!r}")
    distances = [
        (squared_l2(item.embedding, center), cluster_id)
        for cluster_id, center in group_centers.items()
    ]
    best_distance, best_cluster = min(distances)
    return best_cluster, best_distance


def choose_cluster_stably(
    item: ItemSnapshot,
    previous: Optional[CodeAssignment],
    is_drift_candidate: bool,
    centers: CenterMap,
    config: UpdateConfig,
) -> Tuple[int, bool]:
    """Return (cluster_id, changed_from_previous_cluster)."""
    same_parent = (
        previous is not None
        and previous.semantic_group == item.semantic_group
        and previous.cluster_id in centers.get(item.semantic_group, {})
    )

    # Stable fast path: an unchanged item keeps its historical cluster.
    if same_parent and not is_drift_candidate:
        return previous.cluster_id, False

    nearest_id, nearest_distance = nearest_cluster(item, centers)
    if not same_parent:
        return nearest_id, previous is not None
    if nearest_id == previous.cluster_id:
        return previous.cluster_id, False

    old_center = centers[item.semantic_group][previous.cluster_id]
    old_distance = squared_l2(item.embedding, old_center)
    confident_move = (
        nearest_distance == 0.0
        or old_distance / nearest_distance
        > config.reassignment_distance_ratio
    )
    if confident_move:
        return nearest_id, True
    return previous.cluster_id, False


SignalRange = Mapping[str, Tuple[float, float]]


def fit_order_signal_ranges(
    items: Iterable[ItemSnapshot],
    config: UpdateConfig,
) -> SignalRange:
    """Fit per-semantic-group min/max values for one ordering signal."""
    observed: Dict[str, list[float]] = {}
    for item in items:
        value = float(item.l3_signals.get(config.l3_order_signal, 0.0))
        if config.log_scale_order_signal:
            value = log1p(max(0.0, value))
        observed.setdefault(item.semantic_group, []).append(value)
    return {
        group: (min(values), max(values))
        for group, values in observed.items()
    }


def l3_order_score(
    item: ItemSnapshot,
    signal_ranges: SignalRange,
    config: UpdateConfig,
) -> float:
    """Normalize one price/popularity signal within its semantic group."""
    value = float(item.l3_signals.get(config.l3_order_signal, 0.0))
    if config.log_scale_order_signal:
        value = log1p(max(0.0, value))
    minimum, maximum = signal_ranges[item.semantic_group]
    normalized = (
        (value - minimum) / (maximum - minimum)
        if maximum - minimum > 1e-8
        else 0.5
    )
    if config.descending_order_signal:
        normalized = 1.0 - normalized
    return max(0.0, min(1.0, normalized))


def _preferred_slot(order_score: float, config: UpdateConfig) -> int:
    bucket = min(
        config.l3_order_buckets - 1,
        max(0, floor(order_score * config.l3_order_buckets)),
    )
    # Map the price/popularity bucket to its interval in leaf-ID space.
    start = bucket * config.max_leaf_slots // config.l3_order_buckets
    end = (bucket + 1) * config.max_leaf_slots // config.l3_order_buckets
    return min(config.max_leaf_slots - 1, (start + end) // 2)


def assign_leaf_ids_stably(
    current_items: Mapping[str, ItemSnapshot],
    cluster_ids: Mapping[str, int],
    previous_assignments: Mapping[str, CodeAssignment],
    config: UpdateConfig,
) -> Dict[str, CodeAssignment]:
    """Keep historical IDs and bucket-place only unassigned items."""
    signal_ranges = fit_order_signal_ranges(current_items.values(), config)
    members: Dict[Tuple[str, int], list[str]] = {}
    for item_id, cluster_id in cluster_ids.items():
        item = current_items[item_id]
        members.setdefault((item.semantic_group, cluster_id), []).append(item_id)

    result: Dict[str, CodeAssignment] = {}
    for (semantic_group, cluster_id), item_ids in members.items():
        occupied: Dict[int, str] = {}
        pending = []

        # Preserve a historical leaf ID only when the full parent path is stable.
        for item_id in sorted(item_ids):
            previous = previous_assignments.get(item_id)
            can_keep = (
                previous is not None
                and previous.semantic_group == semantic_group
                and previous.cluster_id == cluster_id
                and 0 <= previous.leaf_id < config.max_leaf_slots
                and previous.leaf_id not in occupied
            )
            if can_keep:
                occupied[previous.leaf_id] = item_id
                result[item_id] = previous
            else:
                order_score = l3_order_score(
                    current_items[item_id], signal_ranges, config
                )
                pending.append((order_score, item_id))

        free_slots = set(range(config.max_leaf_slots)) - set(occupied)
        for order_score, item_id in sorted(
            pending, key=lambda pair: (pair[0], pair[1])
        ):
            if not free_slots:
                raise OverflowError(
                    f"leaf capacity exhausted for group={semantic_group!r}, "
                    f"cluster={cluster_id}"
                )
            target = _preferred_slot(order_score, config)
            leaf_id = min(free_slots, key=lambda slot: (abs(slot - target), slot))
            free_slots.remove(leaf_id)
            result[item_id] = CodeAssignment(
                semantic_group=semantic_group,
                cluster_id=cluster_id,
                leaf_id=leaf_id,
            )
    return result


def update_codebook(
    current_items: Mapping[str, ItemSnapshot],
    previous_items: Mapping[str, ItemSnapshot],
    previous_assignments: Mapping[str, CodeAssignment],
    centers: CenterMap,
    config: UpdateConfig = UpdateConfig(),
) -> UpdateResult:
    """Perform one stable incremental update using the supplied center map."""
    drift_candidates = select_drift_candidates(
        current_items, previous_items, config
    )

    cluster_ids: Dict[str, int] = {}
    reassigned = set()
    for item_id, item in current_items.items():
        cluster_id, changed = choose_cluster_stably(
            item=item,
            previous=previous_assignments.get(item_id),
            is_drift_candidate=item_id in drift_candidates,
            centers=centers,
            config=config,
        )
        cluster_ids[item_id] = cluster_id
        if changed:
            reassigned.add(item_id)

    assignments = assign_leaf_ids_stably(
        current_items=current_items,
        cluster_ids=cluster_ids,
        previous_assignments=previous_assignments,
        config=config,
    )
    return UpdateResult(
        assignments=assignments,
        drift_candidates=drift_candidates,
        reassigned_items=frozenset(reassigned),
    )
