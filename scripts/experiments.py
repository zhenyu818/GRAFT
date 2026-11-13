#!/usr/bin/env python3
"""Run reproducible Kubernetes, GPU recovery, scheduling, and learning experiments."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

NAMESPACE = "graft-system"
RESOURCE = "trainingruns.training.graft.dev"
VOLUME = Path(os.environ.get("GRAFT_ARTIFACT_ROOT", "/var/lib/graft/artifacts"))


def command(arguments, *, data=None, timeout=60):
    result = subprocess.run(arguments, input=data, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed: {' '.join(arguments)}\n{result.stderr[-6000:]}")
    return result.stdout


def kubectl(*arguments, data=None):
    return command(["kubectl", "-n", NAMESPACE, *arguments], data=data)


def resource(name):
    return json.loads(kubectl("get", RESOURCE, name, "-o", "json"))


def read(path):
    return json.loads(Path(path).read_text())


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, int((len(values) - 1) * fraction)))]


def source_digest():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    files = []
    for folder in ("api", "cmd", "internal", "python", "scripts", "config", "tests", "build"):
        files += [
            path for path in (root / folder).rglob("*") if path.is_file() and "__pycache__" not in str(path)
        ]
    files += [root / "go.mod", root / "go.sum", root / "pyproject.toml"]
    for path in sorted(files):
        if path.exists():
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


class Suite:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.prefix = "exp-" + uuid.uuid4().hex[:6]
        self.records = {}
        self.traces = {}

    def submit(self, label, **changes):
        spec = {
            "workload": "ddp",
            "image": "graft-worker:0.1.0",
            "replicas": 2,
            "gpusPerWorker": 1,
            "steps": 60,
            "seed": 2026,
            "microBatch": 8,
            "accumulation": 2,
            "checkpointEvery": 10,
            "progressTimeoutSeconds": 12,
            "startupTimeoutSeconds": 120,
            "priorityClassName": "graft-normal",
            "maxRetries": 2,
            "queue": "graft",
        }
        spec.update(changes)
        manifest = {
            "apiVersion": "training.graft.dev/v1alpha1",
            "kind": "TrainingRun",
            "metadata": {
                "name": f"{self.prefix}-{label}",
                "namespace": NAMESPACE,
                "labels": {"training.graft.dev/suite": self.prefix},
            },
            "spec": spec,
        }
        started = time.time()
        result = json.loads(kubectl("create", "-f", "-", "-o", "json", data=json.dumps(manifest)))
        result["client_created_at"] = started
        return result

    def wait(self, run, predicate=None, timeout=600):
        name = run["metadata"]["name"]
        deadline = time.monotonic() + timeout
        history = self.traces.setdefault(name, [])
        previous = None
        while time.monotonic() < deadline:
            current = resource(name)
            status = current.get("status", {})
            key = (status.get("phase"), status.get("attempt"), status.get("lastStep"))
            if key != previous:
                history.append({"time_unix": time.time(), "phase": key[0], "attempt": key[1], "step": key[2]})
                previous = key
            if status.get("phase") == "Failed":
                raise RuntimeError(f"TrainingRun {name} failed: {json.dumps(status)}")
            if (predicate and predicate(current)) or (
                predicate is None and status.get("phase") == "Succeeded"
            ):
                return current
            time.sleep(0.25)
        raise TimeoutError(f"TrainingRun {name} observation deadline exceeded")

    def summarize(self, run):
        status = run["status"]
        root = VOLUME / run["metadata"]["uid"]
        events = []
        for path in sorted((root / "attempts").glob("*/events-rank-*.jsonl")):
            events += [json.loads(line) for line in path.read_text().splitlines() if line]
        result = read(root / "attempts" / str(status["attempt"]) / "result.json")
        summary = {
            "configuration": run["spec"],
            "status": status,
            "result": result,
            "events": sorted(events, key=lambda event: event["time_unix"]),
        }
        faults = [event for event in events if event["event"] == "fault_injected"]
        if faults:
            injected = min(event["time_unix"] for event in faults)
            detected = status["recoveries"][0]["detectedAtUnixMs"] / 1000.0
            selected = [
                event for event in events if event["event"] == "checkpoint_selected" and event["attempt"] > 0
            ]
            resumed_progress = [
                event for event in events if event["event"] == "training_progress" and event["attempt"] > 0
            ]
            summary["recovery"] = {
                "injected_at_unix": injected,
                "detected_at_unix": detected,
                "detection_seconds": max(0.0, detected - injected),
                "first_resumed_step_seconds": min(event["time_unix"] for event in resumed_progress)
                - injected,
                "restored_step": min(event["step"] for event in selected),
                "replayed_completed_steps": max(
                    0,
                    faults[0]["step"]
                    - int(faults[0]["kind"] != "checkpoint")
                    - min(event["step"] for event in selected),
                ),
            }
        return summary

    def correctness(self, reference, candidate):
        import torch
        from graft_worker.checkpoint import valid_generations
        from graft_worker.config import Config

        def generation(run):
            config = Config(
                **{key: value for key, value in run["spec"].items() if key in Config.__dataclass_fields__}
            )
            return valid_generations(
                VOLUME / run["metadata"]["uid"], config.identity(), run["spec"]["replicas"]
            )[0][2]

        left, right = generation(reference), generation(candidate)
        comparisons = []

        def maximum_difference(a, b):
            if isinstance(a, torch.Tensor):
                return float((a.to(torch.float64) - b.to(torch.float64)).abs().max()) if a.numel() else 0.0
            if isinstance(a, dict):
                if set(a) != set(b):
                    raise AssertionError("State dictionaries must have identical keys")
                return max((maximum_difference(a[key], b[key]) for key in a), default=0.0)
            if isinstance(a, (list, tuple)):
                if len(a) != len(b):
                    raise AssertionError("State sequences must have identical lengths")
                return max((maximum_difference(x, y) for x, y in zip(a, b)), default=0.0)
            if isinstance(a, (int, float)):
                return abs(float(a) - float(b))
            if a != b:
                raise AssertionError("State values must match")
            return 0.0

        for rank in range(reference["spec"]["replicas"]):
            a = torch.load(left / f"rank-{rank}.pt", map_location="cpu", weights_only=True)
            b = torch.load(right / f"rank-{rank}.pt", map_location="cpu", weights_only=True)
            comparison = {
                "rank": rank,
                "model_max_abs_difference": maximum_difference(a["model"], b["model"]),
                "optimizer_max_abs_difference": maximum_difference(a["optimizer"], b["optimizer"]),
                "cpu_rng_equal": torch.equal(a["torch_rng"], b["torch_rng"]),
                "cuda_rng_equal": torch.equal(a["cuda_rng"], b["cuda_rng"]),
                "data_cursor_equal": a["data_cursor"] == b["data_cursor"],
            }
            assert comparison["model_max_abs_difference"] <= 1e-7, comparison
            assert comparison["optimizer_max_abs_difference"] <= 1e-7, comparison
            assert (
                comparison["cpu_rng_equal"]
                and comparison["cuda_rng_equal"]
                and comparison["data_cursor_equal"]
            ), comparison
            comparisons.append(comparison)
        return comparisons

    def recovery(self):
        baseline = self.wait(self.submit("baseline"))
        self.records["baseline"] = self.summarize(baseline)
        for kind, step in (("exit", 25), ("hang", 25), ("checkpoint", 30)):
            run = self.wait(self.submit(f"recovery-{kind}", fault={"kind": kind, "step": step, "rank": 1}))
            assert run["status"]["attempt"] == 1, run["status"]
            record = self.summarize(run)
            record["state_comparison"] = self.correctness(baseline, run)
            self.records[f"recovery_{kind}"] = record
            print(
                f"Recovery verified: {kind}, restored step {record['recovery']['restored_step']}", flush=True
            )
            self.save()

    def pod_replacement(self):
        run = self.submit("pod-replacement", stepDelaySeconds=0.05)
        active = self.wait(run, lambda item: item.get("status", {}).get("lastStep", 0) >= 15)
        pods = json.loads(
            kubectl(
                "get", "pods", "-l", f"training.graft.dev/attempt={active['status']['jobName']}", "-o", "json"
            )
        )["items"]
        target = next(
            item for item in pods if item["metadata"]["labels"]["training.graft.dev/role"] == "worker"
        )
        injection = time.time()
        kubectl("delete", "pod", target["metadata"]["name"], "--grace-period=0", "--force", "--wait=false")
        final = self.wait(run)
        assert final["status"]["attempt"] == 1, final["status"]
        record = self.summarize(final)
        record["pod_deleted_at_unix"] = injection
        # Step delay changes only wall-clock timing, preserving numerical state.
        baseline_name = next(name for name in self.traces if name.endswith("-baseline"))
        record["state_comparison"] = self.correctness(resource(baseline_name), final)
        self.records["pod_replacement"] = record
        self.save()

    def failover(self):
        run = self.submit("controller-failover", steps=100, stepDelaySeconds=0.15)
        self.wait(run, lambda item: item.get("status", {}).get("lastStep", 0) >= 10)
        lease = json.loads(kubectl("get", "lease", "graft-controller.training.graft.dev", "-o", "json"))
        old_holder = lease["spec"]["holderIdentity"]
        pods = json.loads(kubectl("get", "pods", "-l", "app=graft-controller", "-o", "json"))["items"]
        target = next(pod for pod in pods if old_holder.startswith(pod["metadata"]["name"]))
        injected = time.time()
        kubectl("delete", "pod", target["metadata"]["name"], "--grace-period=0", "--force", "--wait=false")
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            lease = json.loads(kubectl("get", "lease", "graft-controller.training.graft.dev", "-o", "json"))
            if lease["spec"].get("holderIdentity") and lease["spec"]["holderIdentity"] != old_holder:
                break
            time.sleep(0.25)
        else:
            raise TimeoutError("A replacement manager must acquire the Lease")
        elected = time.time()
        final = self.wait(run)
        assert final["status"]["attempt"] == 0 and not final["status"].get("recoveries"), final["status"]
        record = self.summarize(final)
        record["leader_replacement_seconds"] = elected - injected
        record["worker_group_attempts"] = 1
        self.records["controller_failover"] = record
        self.save()

    def scheduling(self):
        blocker = self.submit("priority-blocker", steps=90, stepDelaySeconds=0.15)
        self.wait(blocker, lambda item: item.get("status", {}).get("lastStep", 0) >= 5)
        low = self.submit("priority-low", steps=20, priorityClassName="graft-low")
        high = self.submit("priority-high", steps=20, priorityClassName="graft-high")
        self.wait(low, lambda item: item.get("status", {}).get("phase") == "Queued")
        self.wait(high, lambda item: item.get("status", {}).get("phase") == "Queued")
        completed_blocker = self.wait(blocker)
        completed_high = self.wait(high)
        completed_low = self.wait(low)
        high_time = completed_high["status"]["startedAt"]
        low_time = completed_low["status"]["startedAt"]
        assert high_time < low_time, (high_time, low_time)
        self.records["priority_scheduling"] = {
            "submission_order": ["blocker", "low", "high"],
            "execution_order": ["blocker", "high", "low"],
            "runs": {
                label: self.summarize(item)
                for label, item in (
                    ("blocker", completed_blocker),
                    ("high", completed_high),
                    ("low", completed_low),
                )
            },
        }
        oversized = self.submit("gang-three", replicas=3, steps=10)
        self.wait(oversized, lambda item: item.get("status", {}).get("phase") == "Queued")
        fitted = self.wait(self.submit("gang-two", steps=10))
        current = resource(oversized["metadata"]["name"])
        pods = json.loads(
            kubectl(
                "get",
                "pods",
                "-l",
                f"training.graft.dev/run-uid={oversized['metadata']['uid']}",
                "-o",
                "json",
            )
        )["items"]
        assert current["status"]["phase"] == "Queued", current["status"]
        assert all(
            pod.get("status", {}).get("phase") == "Pending" and not pod.get("spec", {}).get("nodeName")
            for pod in pods
        ), pods
        self.records["gang_admission"] = {
            "cluster_gpu_capacity": 2,
            "oversized_group_requested_gpus": 3,
            "oversized_group_scheduled_pods": sum(bool(pod.get("spec", {}).get("nodeName")) for pod in pods),
            "fitting_group_requested_gpus": 2,
            "fitting_group_phase": fitted["status"]["phase"],
        }
        kubectl("delete", RESOURCE, oversized["metadata"]["name"], "--wait=true", "--timeout=90s")
        self.save()

    def concurrency(self, count=100):
        started = time.time()

        def create(index):
            before = time.time()
            run = self.submit(f"admission-{index:03d}", suspend=True, steps=5)
            return run, before, time.time() - before

        with ThreadPoolExecutor(max_workers=16) as executor:
            submissions = list(executor.map(create, range(count)))
        requested = {run["metadata"]["name"] for run, _, _ in submissions}
        first_observed = {}
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            all_runs = json.loads(
                kubectl("get", RESOURCE, "-l", f"training.graft.dev/suite={self.prefix}", "-o", "json")
            )["items"]
            for run in all_runs:
                if run["metadata"]["name"] in requested and run.get("status", {}).get("phase") == "Suspended":
                    assert run["status"]["observedGeneration"] == run["metadata"]["generation"]
                    first_observed.setdefault(run["metadata"]["name"], time.time())
            if len(first_observed) == count:
                break
            time.sleep(0.25)
        assert len(first_observed) == count, len(first_observed)
        uids = {run["metadata"]["uid"] for run, _, _ in submissions}
        jobs = json.loads(kubectl("get", "jobs.batch.volcano.sh", "-o", "json"))["items"]
        created_jobs = sum(
            job["metadata"].get("labels", {}).get("training.graft.dev/run-uid") in uids for job in jobs
        )
        assert created_jobs == 0, created_jobs
        latencies = [first_observed[run["metadata"]["name"]] - before for run, before, _ in submissions]
        self.records["concurrent_admission"] = {
            "submissions": count,
            "client_threads": 16,
            "observed_resources": len(first_observed),
            "created_worker_groups": created_jobs,
            "wall_seconds": time.time() - started,
            "create_request_p50_seconds": statistics.median(item[2] for item in submissions),
            "create_request_p95_seconds": percentile([item[2] for item in submissions], 0.95),
            "observed_state_p95_seconds": percentile(latencies, 0.95),
            "observation_interval_seconds": 0.25,
            "mode": "held admission; controller and API-server workload",
        }
        for run, _, _ in submissions:
            kubectl("delete", RESOURCE, run["metadata"]["name"], "--wait=false")
        self.save()

    def reinforcement_learning(self):
        run = self.wait(
            self.submit(
                "navigation",
                workload="rl",
                replicas=1,
                gpusPerWorker=0,
                steps=30,
                environmentConcurrency=8,
                environmentTimeoutSeconds=0.05,
                injectEnvironmentTimeout=True,
            )
        )
        record = self.summarize(run)
        result = record["result"]
        assert result["episodes"] == 240 and result["environment_retries"] == 1, result
        assert len([event for event in record["events"] if event["event"] == "policy_updated"]) == 30
        self.records["reinforcement_learning"] = record
        self.save()

    def optimization(self):
        configurations = (("micro8-acc4", 8, 4), ("micro16-acc2", 16, 2), ("micro32-acc1", 32, 1))
        records = []
        for label, micro, accumulation in configurations:
            for repeat in range(3):
                run = self.wait(
                    self.submit(
                        f"{label}-r{repeat}",
                        steps=80,
                        microBatch=micro,
                        accumulation=accumulation,
                        checkpointEvery=0,
                    )
                )
                record = self.summarize(run)
                record["repeat"] = repeat
                records.append(record)
                self.records["batch_optimization"] = records
                self.save()
        assert len({item["result"]["global_batch"] for item in records}) == 1

    def environment(self):
        import numpy
        import torch

        gpu = command(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]
        )
        node = json.loads(command(["kubectl", "get", "nodes", "-o", "json"]))["items"][0]
        return {
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "gpus": [line.strip() for line in gpu.splitlines()],
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "numpy": numpy.__version__,
            "cuda": torch.version.cuda,
            "nccl": list(torch.cuda.nccl.version())
            if isinstance(torch.cuda.nccl.version(), tuple)
            else torch.cuda.nccl.version(),
            "kubernetes": node["status"]["nodeInfo"]["kubeletVersion"],
            "container_runtime": node["status"]["nodeInfo"]["containerRuntimeVersion"],
            "node_count": 1,
            "source_sha256": source_digest(),
            "worker_image": command(
                ["docker", "image", "inspect", "graft-worker:0.1.0", "--format", "{{.Id}}"]
            ).strip(),
        }

    def metrics(self):
        pods = json.loads(kubectl("get", "pods", "-l", "app=graft-controller", "-o", "json"))["items"]
        snapshots = []
        for pod in pods:
            name = pod["metadata"]["name"]
            if pod.get("status", {}).get("phase") != "Running":
                continue
            raw = kubectl("get", "--raw", f"/api/v1/namespaces/{NAMESPACE}/pods/{name}:8080/proxy/metrics")
            selected = [
                line
                for line in raw.splitlines()
                if line.startswith(
                    (
                        "graft_recoveries_total",
                        "controller_runtime_reconcile_total",
                        "workqueue_depth",
                        "workqueue_retries_total",
                        "go_goroutines",
                        "process_cpu_seconds_total",
                    )
                )
            ]
            assert selected, "Each running manager must expose liveness metrics"
            active = any(line.startswith("controller_runtime_reconcile_total") for line in selected)
            snapshots.append({"replica": name, "reconcile_metrics_present": active, "samples": selected})
        assert any(snapshot["reconcile_metrics_present"] for snapshot in snapshots), (
            "The active controller must expose reconcile metrics"
        )
        self.records["controller_metrics"] = snapshots

    def save(self):
        payload = {
            "suite": "GRAFT dual-GPU integration experiments",
            "environment": self.provenance,
            "experiments": self.records,
            "state_observation_traces": self.traces,
        }
        temporary = self.output / "results.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.output / "results.json")

    def run(self):
        self.provenance = self.environment()
        self.recovery()
        self.pod_replacement()
        self.failover()
        self.scheduling()
        self.concurrency()
        self.reinforcement_learning()
        self.optimization()
        self.metrics()
        self.save()
        print(f"Experiment suite completed: {self.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/dual-v100")
    args = parser.parse_args()
    Suite(args.output).run()
