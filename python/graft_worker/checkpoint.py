"""Immutable generations with checksums and a manifest published after all ranks.

File replacement and directory fsync implement the POSIX shared-volume protocol.
Each attempt writes a new generation, including when it repeats a previous step.
"""

import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .artifacts import atomic_json


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def valid_generations(root, identity, world_size):
    valid = []
    for path in (Path(root) / "checkpoints").glob("step-*-attempt-*"):
        try:
            manifest = json.loads((path / "manifest.json").read_text())
            if manifest["identity"] != identity or manifest["world_size"] != world_size:
                continue
            expected = {f"rank-{rank}.pt" for rank in range(world_size)}
            if set(manifest["files"]) != expected:
                continue
            if all(digest(path / name) == checksum for name, checksum in manifest["files"].items()):
                valid.append((int(manifest["step"]), int(manifest["attempt"]), path))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(valid, reverse=True)


def save(root, attempt, step, rank, world_size, model, optimizer, identity, device, event, interrupt=False):
    directory = Path(root) / "checkpoints" / f"step-{step:08d}-attempt-{attempt}"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"rank-{rank}.pt"
    temporary = directory / f"rank-{rank}.pt.tmp"
    state = {
        "step": step,
        "data_cursor": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }
    if interrupt:
        with temporary.open("wb") as handle:
            handle.write(b"GRAFT interrupted checkpoint write\n")
            handle.flush()
            os.fsync(handle.fileno())
        event("fault_injected", kind="checkpoint", step=step)
        os._exit(86)
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        files = {f"rank-{index}.pt": digest(directory / f"rank-{index}.pt") for index in range(world_size)}
        atomic_json(
            directory / "manifest.json",
            {
                "format_version": 1,
                "step": step,
                "attempt": attempt,
                "world_size": world_size,
                "identity": identity,
                "files": files,
            },
            durable=True,
        )
    if dist.is_initialized():
        dist.barrier()
    event("checkpoint_committed", step=step, generation=directory.name)
    return directory


def restore(root, rank, world_size, model, optimizer, identity, device, event):
    selection = [None]
    if rank == 0:
        generations = valid_generations(root, identity, world_size)
        if generations:
            selection[0] = str(generations[0][2])
    if dist.is_initialized():
        dist.broadcast_object_list(selection, src=0, device=device if device.type == "cuda" else None)
    if selection[0] is None:
        event("checkpoint_selected", step=0, generation=None)
        return 0
    directory = Path(selection[0])
    state = torch.load(directory / f"rank-{rank}.pt", map_location="cpu", weights_only=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    if state["step"] != manifest["step"] or state["data_cursor"] != state["step"]:
        raise ValueError("Checkpoint step and data cursor must match the committed manifest")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"], device)
    event("checkpoint_selected", step=state["step"], generation=directory.name)
    return state["step"]
