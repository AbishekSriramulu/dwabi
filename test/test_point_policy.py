import json
import unittest
from dataclasses import asdict

import numpy as np
from dwabi.point_policy import PointDecision, PointHistory, center


class PointPolicyTest(unittest.TestCase):
    def test_cold_even_median_and_json_roundtrip(self):
        history = PointHistory(("north", "south"))
        decision = history.freeze("new task", "new family", issue_at=10)
        restored = PointDecision(**json.loads(json.dumps(asdict(decision))))
        self.assertEqual(decision, restored)
        self.assertIsNone(decision.ordinal_weights)
        np.testing.assert_array_equal(center([[0.0, 4.0]], restored), [2.0])

    def test_guard_uses_mature_prequential_pairs_and_excludes_own_family(self):
        history = PointHistory(("alpha", "beta", "gamma", "delta", "epsilon"))

        def observe(task, family, issue, points):
            history.observe(
                task,
                family,
                origin=issue,
                issued_at=issue,
                first_target_at=issue + 1,
                available_at=issue + 2,
                points=[points],
                actual=[0.0],
                case_weights=[1.0],
            )

        for family in ("peer one", "peer two", "peer three"):
            for issue in (0, 3):
                observe((family, 1), family, issue, [8.0, 8.0, 8.0, 0.0, 0.0])
        for issue in (0, 3):
            observe(
                "same physical source",
                "recipient family",
                issue,
                [0.0, 0.0, 0.0, 8.0, 8.0],
            )
        for issue in (18, 14, 10):
            observe("recipient", "recipient family", issue, [8.0, 8.0, 8.0, 0.0, 0.0])
        history.observe(
            "recipient",
            "recipient family",
            origin="overlap",
            issued_at=13,
            first_target_at=14,
            available_at=19,
            points=[[0.0, 0.0, 0.0, 8.0, 8.0]],
            actual=[0.0],
            case_weights=[1.0],
        )

        early = history.freeze("recipient", "recipient family", issue_at=20)
        self.assertFalse(early.accept)
        self.assertEqual(early.ordinal_origins, (10, 14))
        self.assertEqual(tuple(row.origin for row in early.comparisons), (10, 14))

        later = history.freeze("recipient", "recipient family", issue_at=21)
        self.assertTrue(later.accept)
        self.assertEqual(later.peer_count, 3)
        self.assertEqual(later.operator, "median")
        self.assertEqual(later.members, ("alpha", "delta", "epsilon"))
        self.assertEqual(tuple(row.origin for row in later.comparisons), (10, 14, 18))
        self.assertTrue(
            all(
                row.reference_wape == 8.0 and row.peer_wape == 0.0
                for row in later.comparisons
            )
        )
        np.testing.assert_array_equal(center([[8.0, 8.0, 8.0, 0.0, 0.0]], later), [0.0])

    def test_ordinal_uses_worst_single_donor_deletion_after_three_origins(self):
        history = PointHistory(("alpha", "beta", "gamma"))
        for issue, points in enumerate(([0.0, 4.0, 8.0],) * 3 + ([100.0, 4.0, 8.0],)):
            history.observe(
                "task",
                "family",
                origin=issue,
                issued_at=10 * issue,
                first_target_at=10 * issue + 1,
                available_at=10 * issue + 2,
                points=[points],
                actual=[0.0],
                case_weights=[1.0],
                include_menu=False,
            )

        three = history.freeze("task", "family", issue_at=32)
        self.assertEqual(three.ordinal_origins, (0, 1, 2))
        np.testing.assert_array_equal(center([[0.0, 4.0, 8.0]], three), [0.0])

        four = history.freeze("task", "family", issue_at=33)
        self.assertEqual(four.ordinal_origins, (0, 1, 2, 3))
        self.assertEqual(four.ordinal_weights, (3, 3, 3))
        np.testing.assert_array_equal(center([[0.0, 4.0, 8.0]], four), [4.0])


if __name__ == "__main__":
    unittest.main()
