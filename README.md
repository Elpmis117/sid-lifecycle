# Embed, Quantize, Update: Understanding the Semantic ID Lifecycle in Industrial Generative Recommendation

This is the accompanying code repository for the paper *Embed, Quantize,
Update: Understanding the Semantic ID Lifecycle in Industrial Generative
Recommendation*. It provides a compact, dependency-free implementation of the
stable incremental codebook update used in the paper's SID lifecycle study.
In particular, the repository provides the public implementation of the DISCo
stable codebook maintenance update.

Given two consecutive item snapshots, historical assignments, and the current
cluster centers, one update pass identifies material embedding drift,
selectively updates cluster assignments, and allocates stable leaf IDs for new
or moved items.

## Method

The update policy has four stages:

1. **Drift candidate selection.** For items present in both snapshots, compute
   cosine distance between the current and previous embeddings. Keep items above
   the drift threshold, subject to a deterministic per-update budget.
2. **Stable cluster assignment.** An unchanged item keeps its historical
   cluster. A drift candidate changes cluster only when the nearest current
   center is sufficiently better than its historical center under the relative
   squared-distance gate. New items are assigned to their nearest center.
3. **Signal-based leaf ordering.** Within each semantic group, normalize one
   configured ordering signal. The example supports price-like values and
   popularity-like counts, with optional log scaling and descending order.
4. **Stable leaf allocation.** An item keeps its historical leaf ID when its
   semantic group and cluster remain unchanged. New or moved items are placed in
   the preferred signal bucket and assigned the nearest available slot with a
   deterministic item-ID tie-breaker.

The default configuration uses a cosine drift threshold of `0.02`, a relative
squared-distance ratio of `2.0`, a five-percent candidate budget, 50 ordering
buckets, and 2,000 leaf slots per semantic group and cluster. These values are
provided as a reproducible reference configuration.

## API

The main entry point is:

```python
from incremental_codebook import update_codebook

result = update_codebook(
    current_items=current_items,
    previous_items=previous_items,
    previous_assignments=previous_assignments,
    centers=centers,
)
```

The input records are defined by `ItemSnapshot` and `CodeAssignment`:

```python
ItemSnapshot(
    item_id="item-1",
    semantic_group="group-a",
    embedding=(0.2, 0.8),
    l3_signals={"price_level": 3.0, "click_count": 120.0},
)

CodeAssignment(
    semantic_group="group-a",
    cluster_id=4,
    leaf_id=128,
)
```

`UpdateResult` returns the complete current assignment map together with the
drift candidates and the items whose cluster assignment changed:

```python
result.assignments
result.drift_candidates
result.reassigned_items
```

Use `UpdateConfig` to select the drift gates, candidate budget, ordering signal,
bucket count, and leaf capacity. All tie-breaking is deterministic for a fixed
input.

## Repository layout

```text
src/incremental_codebook.py          Core implementation
tests/test_incremental_codebook.py  Synthetic behavior tests
pyproject.toml                       Minimal project metadata
LICENSE                              MIT license
```

## Run the tests

Python 3.9 or newer is required. The implementation has no third-party runtime
dependencies.

```text
python -m unittest discover -s tests -v
```

The test suite covers drift selection, stable cluster reassignment, historical
leaf-ID preservation, price/popularity ordering, configuration validation, and
explicit capacity handling.

## License

Released under the MIT License. See `LICENSE`.
