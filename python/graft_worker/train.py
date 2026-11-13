"""Fixed-world-size DDP training with coordinated checkpoint recovery."""

from contextlib import nullcontext
from datetime import timedelta
import hashlib
import os
from pathlib import Path
import time

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from . import checkpoint
from .artifacts import Artifacts
from .config import Config, arguments
from .model import CausalTransformer, SEQUENCE_LENGTH, batch


def run(config, root, attempt):
    torch.set_num_threads(1)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != config.replicas:
        raise ValueError("The configured replicas must equal WORLD_SIZE")
    use_cuda = config.gpusPerWorker > 0
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("A GPU workload requires a visible CUDA device")
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    backend = "nccl" if use_cuda else "gloo"
    if "RANK" in os.environ:
        dist.init_process_group(backend, timeout=timedelta(seconds=90))
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    events = Artifacts(root, attempt, rank)
    model = CausalTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    network = (
        DistributedDataParallel(model, device_ids=[local_rank] if use_cuda else None)
        if dist.is_initialized()
        else model
    )
    # Restore after model construction and DDP synchronization, including RNG state.
    resumed = checkpoint.restore(
        root, rank, world_size, model, optimizer, config.identity(), device, events.event
    )
    events.event(
        "worker_started",
        world_size=world_size,
        backend=backend,
        device=str(device),
        resumed_step=resumed,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
    )
    events.progress(resumed)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    measured_started = started
    measured_from = max(resumed, min(5, config.steps))
    losses = []
    checkpoint_seconds = 0.0
    last_checkpoint = resumed
    for step in range(resumed + 1, config.steps + 1):
        selected = attempt == 0 and step == config.fault.get("step") and rank == config.fault.get("rank", 1)
        if selected and config.fault.get("kind") in {"exit", "hang"}:
            events.event("fault_injected", kind=config.fault["kind"], step=step)
            if config.fault["kind"] == "exit":
                os._exit(86)
            while True:
                time.sleep(1)
        inputs, targets = batch(config.seed, rank, step, config.microBatch * config.accumulation, device)
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for micro in range(config.accumulation):
            synchronize = micro == config.accumulation - 1 or not dist.is_initialized()
            context = nullcontext() if synchronize else network.no_sync()
            with context:
                lower = micro * config.microBatch
                upper = lower + config.microBatch
                predictions = network(inputs[lower:upper])
                loss = nn.functional.cross_entropy(predictions.flatten(0, 1), targets[lower:upper].flatten())
                (loss / config.accumulation).backward()
                step_loss += loss.detach().item() / config.accumulation
        optimizer.step()
        losses.append(step_loss)
        if config.checkpointEvery and step % config.checkpointEvery == 0:
            saving = time.perf_counter()
            checkpoint.save(
                root,
                attempt,
                step,
                rank,
                world_size,
                model,
                optimizer,
                config.identity(),
                device,
                events.event,
                interrupt=selected and config.fault.get("kind") == "checkpoint",
            )
            checkpoint_seconds += time.perf_counter() - saving
            last_checkpoint = step
        if config.stepDelaySeconds:
            time.sleep(config.stepDelaySeconds)
        events.progress(step)
        if step == measured_from:
            if use_cuda:
                torch.cuda.synchronize(device)
            measured_started = time.perf_counter()
        if step == resumed + 1 or step % 10 == 0:
            events.event("training_progress", step=step, loss=step_loss)
    if use_cuda:
        torch.cuda.synchronize(device)
    loop_seconds = time.perf_counter() - started
    measured_seconds = time.perf_counter() - measured_started
    if last_checkpoint != config.steps:
        checkpoint.save(
            root,
            attempt,
            config.steps,
            rank,
            world_size,
            model,
            optimizer,
            config.identity(),
            device,
            events.event,
        )
    if dist.is_initialized():
        dist.barrier()
    model_hash = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        model_hash.update(key.encode())
        model_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    result = {
        "workload": "ddp",
        "attempt": attempt,
        "step": config.steps,
        "resumed_from_step": resumed,
        "world_size": world_size,
        "backend": backend,
        "global_batch": config.microBatch * config.accumulation * world_size,
        "micro_batch": config.microBatch,
        "accumulation": config.accumulation,
        "sequence_length": SEQUENCE_LENGTH,
        "measured_steps": config.steps - measured_from,
        "loop_seconds": loop_seconds,
        "measured_seconds": measured_seconds,
        "tokens_per_second": (
            (config.steps - measured_from)
            * config.microBatch
            * config.accumulation
            * world_size
            * SEQUENCE_LENGTH
            / max(measured_seconds, 1e-9)
        ),
        "checkpoint_seconds": checkpoint_seconds,
        "checkpoint_every": config.checkpointEvery,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if use_cuda else 0,
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "model_sha256": model_hash.hexdigest(),
    }
    events.complete(result)
    events.event("worker_completed", step=config.steps, model_sha256=result["model_sha256"])
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return result


if __name__ == "__main__":
    args = arguments()
    run(Config.read(args.config), Path(args.run_dir), args.attempt)
