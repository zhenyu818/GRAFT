"""Metrics collection accepts active and standby controller replicas."""

import json
import tempfile
import unittest
from unittest.mock import patch

from scripts.experiments import Suite


class ControllerMetricsTests(unittest.TestCase):
    def collect(self, active_metrics):
        def response(*arguments, **kwargs):
            if "--raw" not in arguments:
                return json.dumps(
                    {
                        "items": [
                            {"metadata": {"name": "active"}, "status": {"phase": "Running"}},
                            {"metadata": {"name": "standby"}, "status": {"phase": "Running"}},
                        ]
                    }
                )
            liveness = "go_goroutines 10\nprocess_cpu_seconds_total 0.5\n"
            if "active:8080" in arguments[-1]:
                return liveness + active_metrics
            return liveness

        with tempfile.TemporaryDirectory() as directory:
            suite = Suite(directory)
            with patch("scripts.experiments.kubectl", side_effect=response):
                suite.metrics()
            return suite.records["controller_metrics"]

    def test_active_and_standby_metrics_have_distinct_roles(self):
        snapshots = self.collect(
            'controller_runtime_reconcile_total{controller="trainingrun",result="success"} 10\n'
        )
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(sum(item["reconcile_metrics_present"] for item in snapshots), 1)

    def test_liveness_alone_does_not_establish_reconciliation(self):
        with self.assertRaises(AssertionError):
            self.collect("")


if __name__ == "__main__":
    unittest.main()
