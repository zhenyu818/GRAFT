"""Versioned, bounded-concurrency environment sampling and policy-gradient learning."""

import asyncio
import time

import torch
from torch import nn

from . import checkpoint
from .artifacts import Artifacts
from .config import Config, arguments


class NavigationEnvironment:
    """Move from a seeded initial position to position seven within twenty steps."""

    def __init__(self, seed):
        self.initial = 1 + seed % 4

    def reset(self):
        return self.initial

    @staticmethod
    async def step(position, action, delayed=False, timeout=0.1):
        await asyncio.sleep(timeout * 3 if delayed else 0)
        next_position = max(0, min(7, position + (1 if action == 1 else -1)))
        terminated = next_position == 7
        return next_position, 1.0 if terminated else -0.02, terminated


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Embedding(8, 32), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2))

    def forward(self, position):
        return self.network(position)


async def collect(policy, config, version, events):
    async def episode(index):
        episode_id = version * config.environmentConcurrency + index
        for retry in range(2):
            env = NavigationEnvironment(config.seed + episode_id)
            position = env.reset()
            logs, rewards = [], []
            terminated, timed_out = False, False
            for step in range(20):
                distribution = torch.distributions.Categorical(logits=policy(torch.tensor(position)))
                action = distribution.sample()
                delayed = (
                    config.injectEnvironmentTimeout
                    and version == 0
                    and index == 0
                    and retry == 0
                    and step == 0
                )
                try:
                    position, reward, terminated = await asyncio.wait_for(
                        env.step(position, int(action), delayed, config.environmentTimeoutSeconds),
                        timeout=config.environmentTimeoutSeconds,
                    )
                except asyncio.TimeoutError:
                    timed_out = True
                    events.event(
                        "environment_timeout", episode_id=episode_id, policy_version=version, retry=retry
                    )
                    break
                logs.append(distribution.log_prob(action))
                rewards.append(reward)
                if terminated:
                    break
            else:
                terminated = False
            if not timed_out and logs:
                return {
                    "episode_id": episode_id,
                    "policy_version": version,
                    "logs": logs,
                    "rewards": rewards,
                    "success": terminated,
                    "retries": retry,
                }
        raise RuntimeError("The environment retry budget was consumed")

    trajectories = await asyncio.gather(*(episode(index) for index in range(config.environmentConcurrency)))
    ids = [item["episode_id"] for item in trajectories]
    if len(ids) != len(set(ids)) or any(item["policy_version"] != version for item in trajectories):
        raise RuntimeError("Each episode must be consumed once under the sampled policy version")
    return trajectories


def run(config, root, attempt):
    if config.replicas != 1:
        raise ValueError("The environment workload uses one coordinator per job")
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    events = Artifacts(root, attempt, 0)
    resumed = checkpoint.restore(
        root, 0, 1, policy, optimizer, config.identity(), torch.device("cpu"), events.event
    )
    events.progress(resumed)
    started = time.perf_counter()
    successes, consumed, samples, retries = [], set(), 0, 0
    for iteration in range(resumed, config.steps):
        trajectories = asyncio.run(collect(policy, config, iteration, events))
        objective = []
        iteration_successes = 0
        for trajectory in trajectories:
            if trajectory["episode_id"] in consumed:
                raise RuntimeError("An episode was consumed twice")
            consumed.add(trajectory["episode_id"])
            returns = []
            discounted = 0.0
            for reward in reversed(trajectory["rewards"]):
                discounted = reward + 0.95 * discounted
                returns.append(discounted)
            returns.reverse()
            objective.append(-(torch.stack(trajectory["logs"]) * torch.tensor(returns)).sum())
            iteration_successes += int(trajectory["success"])
            samples += len(returns)
            retries += trajectory["retries"]
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack(objective).mean()
        loss.backward()
        optimizer.step()
        successes.append(iteration_successes / len(trajectories))
        step = iteration + 1
        events.progress(step)
        if config.checkpointEvery and step % config.checkpointEvery == 0:
            checkpoint.save(
                root,
                attempt,
                step,
                0,
                1,
                policy,
                optimizer,
                config.identity(),
                torch.device("cpu"),
                events.event,
            )
        events.event(
            "policy_updated",
            step=step,
            policy_version=iteration,
            episodes=len(trajectories),
            success_rate=successes[-1],
            loss=float(loss.detach()),
        )
    elapsed = time.perf_counter() - started
    result = {
        "workload": "rl",
        "attempt": attempt,
        "step": config.steps,
        "resumed_from_step": resumed,
        "episodes": len(consumed),
        "unique_episodes": len(consumed),
        "transitions": samples,
        "environment_retries": retries,
        "environment_concurrency": config.environmentConcurrency,
        "seconds": elapsed,
        "transitions_per_second": samples / max(elapsed, 1e-9),
        "initial_success_rate": sum(successes[:5]) / max(len(successes[:5]), 1),
        "final_success_rate": sum(successes[-5:]) / max(len(successes[-5:]), 1),
    }
    events.complete(result)
    events.event("worker_completed", step=config.steps)
    return result


if __name__ == "__main__":
    args = arguments()
    run(Config.read(args.config), args.run_dir, args.attempt)
