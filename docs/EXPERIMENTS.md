# Experiment Methodology

## Workload and state comparison

The language workload is a two-layer, width-128 causal Transformer with four heads,
sequence length 64, vocabulary size 128, dropout, and AdamW. Input tokens are
generated from seeded cyclic sequences. The logical data cursor and global rank
determine each local batch; no dataset download is required.

Recovery cases use identical seeds, microbatches, accumulation, world size, and
optimizer settings. The baseline reaches step 60. The candidate runs inject one
first-attempt fault and also reach step 60. Final committed model and optimizer
tensors are compared on every rank with a maximum absolute difference threshold
of 1e-7. CPU RNG, CUDA RNG, and data cursor are compared for equality.

## Failure scenarios

- **Worker exit:** rank one exits with code 86 before optimizer step 25.
- **Training stall:** rank one sleeps indefinitely before step 25 while its peer
  reaches collective synchronization. Progress supervision initiates recovery.
- **Checkpoint write interruption:** rank one writes a partial temporary file and
  exits during the step-30 save. Restore selects the preceding committed manifest.
- **Worker Pod replacement:** the experiment deletes a worker Pod after step 15.
  The Operator observes the missing or changed worker identity and replaces the group.
- **Controller replacement:** the active Lease holder is deleted during a longer
  run. A replacement controller acquires leadership and observes the existing group.

Injected faults are application/process and Kubernetes lifecycle events. Their event
records identify the exact injection step and rank. Detection latency uses the
millisecond-resolution controller timestamp. Recovery latency ends at the first
completed optimizer step after checkpoint restoration.

## Scheduling and concurrency

The priority experiment occupies both GPUs with a running job, then submits a low
priority job followed by a high priority job. Their recorded start times establish
the order after the blocker releases its resources.

The gang experiment submits a three-GPU group to a two-GPU queue and verifies that
none of its Pods receives a node assignment. A fitting two-GPU group completes
while the oversized group remains queued. The oversized declaration is then deleted.

The concurrent-submission experiment sends 100 held TrainingRuns using 16 client
threads. It checks that all resources reach their declared `Suspended` state at the
current observed generation and create zero worker groups. Metrics record API
request latency and client-observed state latency. Observation begins after the
create batch; the latency includes that batching interval. GPU execution is measured
separately by the training scenarios.

## Throughput and memory

Three microbatch/accumulation configurations preserve global batch 64 across two
GPUs: 8×4, 16×2, and 32×1 per rank. Each configuration runs three times for 80
optimizer steps. Steps 1–5 warm up the workload; steps 6–80 form the throughput
window. CUDA synchronization bounds the window. The metric counts all global tokens
processed in that interval, including routine progress publication. Periodic
checkpoints are disabled in this comparison and the final state is saved afterward.

Rank zero publishes the throughput and peak allocated CUDA tensor memory from its
GPU. The memory table takes its maximum over the three repetitions for each
configuration. This metric describes PyTorch's active tensor allocation.

## Environment interaction

The navigation task uses a position-based policy with two actions, an eight-state
environment, a twenty-step episode limit, and discounted rewards. Eight environment
sessions run concurrently. Thirty collection/update rounds generate 240 accepted
episodes. One delayed environment step exercises the timeout and retry path.
The report records transition count, accepted episode uniqueness, retries, policy
updates, and the first/last five-round mean success rates for the recorded seed.

## Evidence and reproduction

`scripts/experiments.py` creates a unique suite label, performs the assertions, and
incrementally records configurations, lifecycle observations, worker event streams,
recovery history, state comparisons, and performance measurements. Its source
fingerprint covers execution code, manifests, dependency files, and experiment
scripts. Run `make report` to render tables from the recorded JSON.
