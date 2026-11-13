"""Checkpoint integrity and state continuity tests using the standard test runner."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from graft_worker import checkpoint
from graft_worker.config import Config
from graft_worker.rl import Artifacts, Policy, collect
import asyncio


class CheckpointProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = torch.nn.Linear(3, 2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.identity = {"test": "committed-state"}

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, step):
        return checkpoint.save(
            self.root,
            0,
            step,
            0,
            1,
            self.model,
            self.optimizer,
            self.identity,
            torch.device("cpu"),
            lambda *a, **k: None,
        )

    def test_committed_generation_restores_optimizer_and_rng(self):
        self.model(torch.ones(1, 3)).sum().backward()
        self.optimizer.step()
        expected = {key: value.clone() for key, value in self.model.state_dict().items()}
        self.save(10)
        next_random = torch.rand(4)
        with torch.no_grad():
            self.model.weight.fill_(100)
        step = checkpoint.restore(
            self.root,
            0,
            1,
            self.model,
            self.optimizer,
            self.identity,
            torch.device("cpu"),
            lambda *a, **k: None,
        )
        self.assertEqual(step, 10)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, expected[key]))
        self.assertTrue(torch.equal(torch.rand(4), next_random))
        self.assertEqual(float(self.optimizer.state[self.model.weight]["step"]), 1)

    def test_partial_generation_preserves_last_committed_state(self):
        self.save(10)
        partial = self.root / "checkpoints" / "step-00000020-attempt-0"
        partial.mkdir()
        (partial / "rank-0.pt.tmp").write_bytes(b"partial")
        self.assertEqual(checkpoint.valid_generations(self.root, self.identity, 1)[0][0], 10)

    def test_checksum_mismatch_selects_previous_generation(self):
        self.save(10)
        newest = self.save(20)
        with (newest / "rank-0.pt").open("ab") as handle:
            handle.write(b"corrupted")
        self.assertEqual(checkpoint.valid_generations(self.root, self.identity, 1)[0][0], 10)

    def test_rank_set_and_world_size_are_part_of_the_commit(self):
        newest = self.save(10)
        manifest = json.loads((newest / "manifest.json").read_text())
        manifest["files"]["rank-1.pt"] = "missing"
        (newest / "manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(checkpoint.valid_generations(self.root, self.identity, 1), [])

    def test_configuration_identity_controls_restore_compatibility(self):
        self.save(10)
        self.assertEqual(checkpoint.valid_generations(self.root, {"test": "another-run"}, 1), [])


class EnvironmentProtocolTests(unittest.TestCase):
    def test_timeout_retries_keep_episode_identity_and_policy_version(self):
        torch.manual_seed(2026)
        config = Config(
            workload="rl",
            replicas=1,
            gpusPerWorker=0,
            environmentConcurrency=4,
            environmentTimeoutSeconds=0.01,
            injectEnvironmentTimeout=True,
        )
        with tempfile.TemporaryDirectory() as root:
            events = Artifacts(root, 0, 0)
            trajectories = asyncio.run(collect(Policy(), config, 0, events))
        self.assertEqual(len({item["episode_id"] for item in trajectories}), 4)
        self.assertTrue(all(item["policy_version"] == 0 for item in trajectories))
        self.assertEqual(sum(item["retries"] for item in trajectories), 1)


if __name__ == "__main__":
    unittest.main()
