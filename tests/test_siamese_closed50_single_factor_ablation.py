import unittest

from src.siamese_closed50_single_factor_ablation import (
    BASELINE,
    build_ablation_configurations,
)


class Closed50SingleFactorAblationTests(unittest.TestCase):
    def test_each_ablation_changes_exactly_one_conceptual_factor(self):
        configurations = build_ablation_configurations(selected_gap=3)
        self.assertEqual(configurations[0], BASELINE)
        fields = ("gap_months", "reference_mode", "feature_mode")
        for config in configurations[1:]:
            changed = [
                field
                for field in fields
                if getattr(config, field) != getattr(BASELINE, field)
            ]
            self.assertEqual(changed, [config.changed_factor])

    def test_expected_single_factor_settings(self):
        by_name = {
            config.name: config
            for config in build_ablation_configurations(selected_gap=1)
        }
        self.assertEqual(by_name["only_gap"].gap_months, 1)
        self.assertEqual(by_name["only_gap"].reference_mode, "shape")
        self.assertEqual(by_name["only_gap"].feature_mode, "signed_diff")
        self.assertEqual(
            by_name["only_hybrid_distance"].aggregation,
            "inverse_hybrid_distance",
        )
        self.assertEqual(
            by_name["only_100d_feature"].feature_mode,
            "target_plus_diff",
        )


if __name__ == "__main__":
    unittest.main()
