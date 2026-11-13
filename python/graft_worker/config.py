"""Shared execution configuration for Kubernetes and command-line workloads."""

from dataclasses import asdict, dataclass, field, fields
import argparse
import json
from pathlib import Path


@dataclass
class Config:
    workload: str = "ddp"
    replicas: int = 2
    gpusPerWorker: int = 1
    steps: int = 80
    seed: int = 2026
    microBatch: int = 8
    accumulation: int = 2
    checkpointEvery: int = 10
    stepDelaySeconds: float = 0.0
    environmentConcurrency: int = 8
    environmentTimeoutSeconds: float = 0.1
    injectEnvironmentTimeout: bool = False
    fault: dict = field(default_factory=dict)

    @classmethod
    def read(cls, path: str):
        data = json.loads(Path(path).read_text())
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    def identity(self):
        """Identify the state evolution required for a compatible restart."""
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            in {
                "workload",
                "replicas",
                "gpusPerWorker",
                "seed",
                "microBatch",
                "accumulation",
                "environmentConcurrency",
            }
        }


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--attempt", type=int, default=0)
    return parser.parse_args()
