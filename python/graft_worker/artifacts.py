"""Atomic progress records and durable, rank-specific event streams."""

import json
import os
from pathlib import Path
import time


def atomic_json(path: Path, value: dict, durable: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        if durable:
            os.fsync(handle.fileno())
    os.replace(temporary, path)
    if durable:
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class Artifacts:
    def __init__(self, root, attempt, rank):
        self.root = Path(root)
        self.attempt = attempt
        self.rank = rank
        self.directory = self.root / "attempts" / str(attempt)
        self.directory.mkdir(parents=True, exist_ok=True)

    def event(self, event_type, **details):
        record = {
            "event": event_type,
            "time_unix": time.time(),
            "attempt": self.attempt,
            "rank": self.rank,
            **details,
        }
        with (self.directory / f"events-rank-{self.rank}.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            if event_type == "fault_injected":
                os.fsync(handle.fileno())
        print(json.dumps(record, sort_keys=True), flush=True)

    def progress(self, step):
        if self.rank == 0:
            atomic_json(
                self.directory / "progress.json",
                {"attempt": self.attempt, "step": step, "time_unix": time.time()},
            )

    def complete(self, result):
        if self.rank == 0:
            atomic_json(self.directory / "result.json", result, durable=True)
            atomic_json(
                self.directory / "complete.json",
                {"attempt": self.attempt, "step": result["step"], "time_unix": time.time()},
                durable=True,
            )
