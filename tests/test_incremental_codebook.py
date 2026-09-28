import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from incremental_codebook import (  # noqa: E402
    CodeAssignment,
    ItemSnapshot,
    UpdateConfig,
    cosine_distance,
    select_drift_candidates,
    update_codebook,
)


class IncrementalCodebookTests(unittest.TestCase):
    def test_cosine_distance_handles_zero_and_mismatched_vectors(self):
        self.assertAlmostEqual(cosine_distance((1.0, 0.0), (0.0, 1.0)), 1.0)
        self.assertIsNone(cosine_distance((0.0, 0.0), (1.0, 0.0)))
        self.assertIsNone(cosine_distance((1.0,), (1.0, 0.0)))

    def test_candidate_budget_keeps_highest_drift_and_zero_budget_keeps_none(self):
        previous = {
            "a": ItemSnapshot("a", "g", (1.0, 0.0), {"price_level": 1}),
            "b": ItemSnapshot("b", "g", (1.0, 0.0), {"price_level": 2}),
            "c": ItemSnapshot("c", "g", (1.0, 0.0), {"price_level": 3}),
            "d": ItemSnapshot("d", "g", (1.0, 0.0), {"price_level": 4}),
        }
        current = dict(previous)
        current["a"] = ItemSnapshot("a", "g", (0.0, 1.0), {"price_level": 1})
        current["c"] = ItemSnapshot("c", "g", (0.0, -1.0), {"price_level": 3})

        config = UpdateConfig(max_candidate_fraction=0.5)
        self.assertEqual(
            select_drift_candidates(current, previous, config), frozenset({"a", "c"})
        )
        self.assertEqual(
            select_drift_candidates(
                current, previous, UpdateConfig(max_candidate_fraction=0.1)
            ),
            frozenset(),
        )

    def test_update_preserves_stable_ids_and_moves_only_confident_drift(self):
        previous_items = {
            "stable": ItemSnapshot("stable", "g", (1.0, 0.0), {"price_level": 1}),
            "moved": ItemSnapshot("moved", "g", (1.0, 0.0), {"price_level": 8}),
        }
        current_items = {
            "stable": previous_items["stable"],
            "moved": ItemSnapshot("moved", "g", (0.0, 1.0), {"price_level": 8}),
            "new": ItemSnapshot("new", "g", (0.9, 0.1), {"price_level": 4}),
        }
        previous_assignments = {
            "stable": CodeAssignment("g", 0, 7),
            "moved": CodeAssignment("g", 0, 8),
        }
        centers = {"g": {0: (1.0, 0.0), 1: (0.0, 1.0)}}

        result = update_codebook(
            current_items,
            previous_items,
            previous_assignments,
            centers,
            UpdateConfig(max_candidate_fraction=1.0, max_leaf_slots=16),
        )

        self.assertEqual(result.drift_candidates, frozenset({"moved"}))
        self.assertEqual(result.reassigned_items, frozenset({"moved"}))
        self.assertEqual(result.assignments["stable"].leaf_id, 7)
        self.assertEqual(result.assignments["stable"].cluster_id, 0)
        self.assertEqual(result.assignments["moved"].cluster_id, 1)
        self.assertNotEqual(result.assignments["moved"].leaf_id, 8)
        self.assertEqual(result.assignments["new"].cluster_id, 0)

    def test_popularity_signal_supports_log_scale_and_descending_order(self):
        items = {
            "cold": ItemSnapshot("cold", "g", (1.0,), {"click_count": 1}),
            "hot": ItemSnapshot("hot", "g", (1.0,), {"click_count": 100}),
        }
        centers = {"g": {0: (1.0,)}}
        result = update_codebook(
            items,
            items,
            {},
            centers,
            UpdateConfig(
                l3_order_signal="click_count",
                log_scale_order_signal=True,
                descending_order_signal=True,
            ),
        )
        self.assertLess(
            result.assignments["hot"].leaf_id,
            result.assignments["cold"].leaf_id,
        )

    def test_invalid_configuration_and_capacity_are_explicit(self):
        with self.assertRaises(ValueError):
            UpdateConfig(max_candidate_fraction=1.1)

        item_a = ItemSnapshot("a", "g", (1.0,), {"price_level": 1})
        item_b = ItemSnapshot("b", "g", (1.0,), {"price_level": 2})
        with self.assertRaises(OverflowError):
            update_codebook(
                {"a": item_a, "b": item_b},
                {},
                {},
                {"g": {0: (1.0,)}},
                UpdateConfig(max_leaf_slots=1),
            )


if __name__ == "__main__":
    unittest.main()
