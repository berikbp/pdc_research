"""Configuration errors should be rejected before launching benchmark processes."""

import copy
import unittest

from pdc_research.common import read_json
from pdc_research.config import (
    experiment_plan,
    load_configs,
    validate_experiment,
    validate_machine,
)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.machine, self.experiment = load_configs(
            "configs/machines/berik-wsl.json", "configs/experiments/pilot.json"
        )

    def test_pilot_budget_is_preserved(self):
        plan = experiment_plan(self.machine, self.experiment)
        self.assertEqual(plan["configurations"], 12)
        self.assertEqual(plan["aligned_inputs"], 80)
        self.assertEqual(plan["aligned_measured_requests"], 2880)

    def test_invalid_execution_settings_are_rejected(self):
        for key, value in (
            ("threads", [0]),
            ("threads", [True]),
            ("gpu_layers", [-1]),
            ("gpu_layers", [8, 8]),
            ("port", 70000),
            ("ubatch_size", 1024),
        ):
            with self.subTest(key=key, value=value):
                machine = {**self.machine, key: value}
                with self.assertRaises(ValueError):
                    validate_machine(machine)

    def test_invalid_experiment_settings_are_rejected(self):
        for key, value in (
            ("repetitions", 0),
            ("warmups", False),
            ("languages", ["en", "en"]),
            ("languages", ["../en"]),
            ("matched_tokens", [64, 64]),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    validate_experiment({**self.experiment, key: value})

    def test_empty_matched_lengths_support_aligned_only_experiments(self):
        validate_experiment({**self.experiment, "matched_tokens": []})

    def test_validation_preserves_notes_and_metadata(self):
        original = copy.deepcopy(self.machine)
        validate_machine(self.machine)
        self.assertEqual(self.machine, original)
        validate_experiment(read_json("configs/experiments/smoke.json"))


if __name__ == "__main__":
    unittest.main()
