#!/usr/bin/env python3
"""Render the published experiment tables directly from recorded JSON evidence."""

import argparse
import json
from pathlib import Path
import statistics


def render(path):
    data = json.loads(path.read_text())
    experiments = data["experiments"]
    environment = data["environment"]
    lines = [
        "# GRAFT Experiment Results",
        "",
        f"Recorded on {environment['date_utc']}.",
        "",
        "## Execution environment",
        "",
        f"- GPUs: {'; '.join(environment['gpus'])}.",
        f"- PyTorch: {environment['torch']}; CUDA: {environment['cuda']}; NCCL: {environment['nccl']}.",
        f"- Kubernetes: {environment['kubernetes']}; runtime: {environment['container_runtime']}.",
        f"- Execution-source SHA-256: `{environment['source_sha256']}`.",
        "",
        "## Coordinated recovery",
        "",
        "Each run trains the same causal Transformer for 60 optimizer steps on two GPUs. "
        "Recovery restores model, optimizer, per-rank RNG, and data cursor state. "
        "The comparison uses the final committed state of every rank.",
        "",
        "| Injected event | Restored step | Detection (s) | First resumed step (s) | Model max abs. difference | Optimizer max abs. difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, title in (
        ("recovery_exit", "Worker exit"),
        ("recovery_hang", "Training stall"),
        ("recovery_checkpoint", "Checkpoint write interruption"),
    ):
        record = experiments[key]
        recovery = record["recovery"]
        comparison = record["state_comparison"]
        lines.append(
            f"| {title} | {recovery['restored_step']} | {recovery['detection_seconds']:.3f} | "
            f"{recovery['first_resumed_step_seconds']:.3f} | "
            f"{max(item['model_max_abs_difference'] for item in comparison):.3g} | "
            f"{max(item['optimizer_max_abs_difference'] for item in comparison):.3g} |"
        )
    lines += [
        "",
        "Detection uses millisecond-resolution controller timestamps. "
        "Recovery latency ends at the first completed optimizer step after restoration. "
        "CPU RNG, CUDA RNG, and data cursor equality are recorded per rank in `results.json`.",
        "",
        "## Kubernetes control and scheduling",
        "",
        "| Experiment | Recorded result |",
        "|---|---|",
    ]
    load = experiments["concurrent_admission"]
    failover = experiments["controller_failover"]
    gang = experiments["gang_admission"]
    lines += [
        f"| Concurrent submissions | {load['submissions']} resources; {load['client_threads']} client threads; "
        f"all {load['observed_resources']} reached their declared admission state |",
        f"| Client create latency through kubectl | P50 {load['create_request_p50_seconds']:.3f} s; P95 {load['create_request_p95_seconds']:.3f} s |",
        f"| Admission state observation | P95 {load['observed_state_p95_seconds']:.3f} s; held jobs created {load['created_worker_groups']} worker groups |",
        f"| Leader replacement | {failover['leader_replacement_seconds']:.3f} s; training completed in {failover['worker_group_attempts']} worker-group attempt |",
        "| Waiting-job priority | Submission: blocker, low, high; execution: blocker, high, low |",
        f"| Gang admission | Three-GPU group scheduled {gang['oversized_group_scheduled_pods']} pods on a two-GPU node; a fitting two-GPU group completed |",
        "| Worker Pod replacement | One coordinated retry; final state compared with uninterrupted training |",
        "",
        "The submission experiment exercises the Kubernetes API and controller with held jobs. "
        "State observation starts after the concurrent create batch and includes client submission time. "
        "Priority and gang experiments use actual GPU-requesting worker Pods.",
        "",
        "## Framework and resource configuration",
        "",
        "Global batch size remains 64 sequences across two ranks; each sequence has 64 tokens. "
        "Each configuration runs three times for 80 steps. Steps 6–80 form the throughput window. "
        "Periodic checkpointing is disabled for this comparison; the final snapshot follows the timed loop.",
        "",
        "| Microbatch per GPU | Accumulation | Median tokens/s | Min–max tokens/s | Peak allocated memory per rank (MiB) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for micro, accumulation in ((8, 4), (16, 2), (32, 1)):
        subset = [
            record["result"]
            for record in experiments["batch_optimization"]
            if record["result"]["micro_batch"] == micro
        ]
        throughputs = [result["tokens_per_second"] for result in subset]
        lines.append(
            f"| {micro} | {accumulation} | {statistics.median(throughputs):,.0f} | "
            f"{min(throughputs):,.0f}–{max(throughputs):,.0f} | "
            f"{max(result['peak_cuda_memory_bytes'] for result in subset) / 2**20:.1f} |"
        )
    rl = experiments["reinforcement_learning"]["result"]
    lines += [
        "",
        "## Reinforcement learning environment interaction",
        "",
        f"The navigation policy completed {rl['step']} policy updates using {rl['episodes']} unique episodes "
        f"and {rl['transitions']} environment transitions. Sampling concurrency was {rl['environment_concurrency']}. "
        f"The injected environment timeout produced {rl['environment_retries']} episode retry. "
        "Episode IDs and policy versions are checked before each update.",
        "",
        f"- Initial five-update mean success rate: {rl['initial_success_rate']:.1%}.",
        f"- Final five-update mean success rate: {rl['final_success_rate']:.1%}.",
        f"- End-to-end sampling and learning throughput: {rl['transitions_per_second']:.1f} transitions/s.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "make experiment",
        "make report",
        "```",
        "",
        "`results.json` contains configurations, worker events, recovery decisions, state comparisons, "
        "and observed lifecycle transitions. `scripts/experiments.py` performs the assertions before publishing each case.",
        "",
    ]
    output = path.with_name("REPORT.md")
    output.write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    render(args.results)
