# GRAFT Experiment Results

Recorded on 2026-11-13T09:02:48.322314+00:00.

## Execution environment

- GPUs: Tesla V100-SXM2-32GB, 32768, 550.127.05; Tesla V100-SXM2-32GB, 32768, 550.127.05.
- PyTorch: 2.5.0+cu124; CUDA: 12.4; NCCL: [2, 21, 5].
- Kubernetes: v1.32.11+k3s1; runtime: containerd://2.1.5-k3s1.32.
- Execution-source SHA-256: `3464db08138ad5b7a68904c99c9092862fd31b7e6111c26e362e17ca1b69510e`.

## Coordinated recovery

Each run trains the same causal Transformer for 60 optimizer steps on two GPUs. Recovery restores model, optimizer, per-rank RNG, and data cursor state. The comparison uses the final committed state of every rank.

| Injected event | Restored step | Detection (s) | First resumed step (s) | Model max abs. difference | Optimizer max abs. difference |
|---|---:|---:|---:|---:|---:|
| Worker exit | 20 | 2.721 | 19.263 | 0 | 0 |
| Training stall | 20 | 12.780 | 29.756 | 0 | 0 |
| Checkpoint write interruption | 20 | 2.621 | 18.235 | 0 | 0 |

Detection uses millisecond-resolution controller timestamps. Recovery latency ends at the first completed optimizer step after restoration. CPU RNG, CUDA RNG, and data cursor equality are recorded per rank in `results.json`.

## Kubernetes control and scheduling

| Experiment | Recorded result |
|---|---|
| Concurrent submissions | 100 resources; 16 client threads; all 100 reached their declared admission state |
| Client create latency through kubectl | P50 0.274 s; P95 0.371 s |
| Admission state observation | P95 7.327 s; held jobs created 0 worker groups |
| Leader replacement | 16.362 s; training completed in 1 worker-group attempt |
| Waiting-job priority | Submission: blocker, low, high; execution: blocker, high, low |
| Gang admission | Three-GPU group scheduled 0 pods on a two-GPU node; a fitting two-GPU group completed |
| Worker Pod replacement | One coordinated retry; final state compared with uninterrupted training |

The submission experiment exercises the Kubernetes API and controller with held jobs. State observation starts after the concurrent create batch and includes client submission time. Priority and gang experiments use actual GPU-requesting worker Pods.

## Framework and resource configuration

Global batch size remains 64 sequences across two ranks; each sequence has 64 tokens. Each configuration runs three times for 80 steps. Steps 6–80 form the throughput window. Periodic checkpointing is disabled for this comparison; the final snapshot follows the timed loop.

| Microbatch per GPU | Accumulation | Median tokens/s | Min–max tokens/s | Peak allocated memory per rank (MiB) |
|---:|---:|---:|---:|---:|
| 8 | 4 | 101,645 | 101,274–103,605 | 83.4 |
| 16 | 2 | 183,344 | 180,187–186,454 | 95.6 |
| 32 | 1 | 282,086 | 274,950–288,835 | 118.8 |

## Reinforcement learning environment interaction

The navigation policy completed 30 policy updates using 240 unique episodes and 1570 environment transitions. Sampling concurrency was 8. The injected environment timeout produced 1 episode retry. Episode IDs and policy versions are checked before each update.

- Initial five-update mean success rate: 67.5%.
- Final five-update mean success rate: 100.0%.
- End-to-end sampling and learning throughput: 1247.7 transitions/s.

## Reproduction

```bash
make experiment
make report
```

`results.json` contains configurations, worker events, recovery decisions, state comparisons, and observed lifecycle transitions. `scripts/experiments.py` performs the assertions before publishing each case.
