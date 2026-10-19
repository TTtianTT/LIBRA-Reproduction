"""Evaluation and SB3 fitting for a frozen, finite-budget simulation study."""
from pathlib import Path
import time

import numpy as np

from .training import ObservationTransform, StudyEnv


def chunks(counts, episode_seconds=21600):
    return [np.asarray(counts[i:i+episode_seconds], dtype=np.int64)
            for i in range(0, len(counts), episode_seconds)]


def aggregate(summaries):
    keys = ("total_cost_usd", "vm_cost_usd", "faas_cost_usd", "gateway_cost_usd", "requests",
            "vm_requests", "faas_requests", "expected_violations", "vm_seconds", "starting_vm_seconds",
            "cancelled_startup_vm_seconds", "unfinished_startup_vm_seconds")
    result = {key: sum(row[key] for row in summaries) for key in keys}
    result["violation_rate"] = result["expected_violations"] / max(1, result["requests"])
    result["episodes"] = len(summaries)
    result["max_target_vms"] = max(row.get("max_target_vms", 0) for row in summaries)
    return result


def evaluate_agent(agent, counts, *, model, forecast_model, max_vms, penalty,
                   decision_seconds=60, action_mode="residual", hide_forecast=False,
                   episode_seconds=21600, latency_mode=None):
    # Batch inference over independent finite episodes; no shared VM resource state.
    envs, observations = [], []
    for block in chunks(counts, episode_seconds):
        env = ObservationTransform(StudyEnv(block, model=model, decision_seconds=decision_seconds,
                                            episode_seconds=None, action_mode=action_mode,
                                            max_vms=max_vms, forecast_model=forecast_model, penalty=penalty),
                                   hide_forecast=hide_forecast)
        observations.append(env.reset(seed=0, options={"start": 0})[0])
        envs.append(env)
    active = list(range(len(envs)))
    maxima = [0] * len(envs)
    while active:
        actions, _ = agent.predict(np.stack([observations[i] for i in active]), deterministic=True)
        next_active = []
        for i, action in zip(active, actions):
            observation, _, terminated, truncated, info = envs[i].step(action)
            maxima[i] = max(maxima[i], info["target_vms"])
            observations[i] = observation
            if not (terminated or truncated):
                next_active.append(i)
        active = next_active
    return aggregate([{**env.unwrapped.summary(), "max_target_vms": maximum}
                      for env, maximum in zip(envs, maxima)])


def train_agent(train, validation, *, job, protocol, model, forecast_model, max_vms,
                output, fingerprint):
    """Four post-update checkpoints, selected exclusively on full validation."""
    import json
    import hashlib
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    run_id = job["id"]
    quantum = protocol["ppo"]["n_steps"]
    checkpoints = [int(round(job["samples"] * f)) for f in protocol["checkpoint_fractions"]]
    if any((b - a) % quantum for a, b in zip([0] + checkpoints[:-1], checkpoints)):
        raise ValueError("checkpoint increments must be exact multiples of the PPO rollout length")
    results_path = output / "runs" / f"{run_id}.json"
    model_path = output / "models" / f"{run_id}.zip"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        result = json.loads(results_path.read_text())
        if result["fingerprint"] != fingerprint:
            raise ValueError(f"stale completed run: {run_id}; choose a new output directory")
        if not model_path.exists() or hashlib.sha256(model_path.read_bytes()).hexdigest() != result["checkpoint_sha256"]:
            raise ValueError(f"missing or modified local checkpoint: {run_id}")
        return result
    env = Monitor(ObservationTransform(StudyEnv(
        train, model=model, forecast_model=forecast_model, max_vms=max_vms,
        decision_seconds=job["decision_seconds"], action_mode=job["action_mode"],
        augment=job["augment"], penalty=job["penalty"],
        episode_seconds=protocol["episode_seconds"], epsilon=protocol["epsilon"],
        cost_normalizer=protocol["cost_normalizer_usd"]), hide_forecast=job["hide_forecast"]))
    agent = PPO("MlpPolicy", env, seed=job["seed"], device="cpu", verbose=0,
                policy_kwargs={"net_arch": protocol["network"]}, **protocol["ppo"])
    learning, best_score, selected_step, selected_summary = [], float("inf"), None, None
    fit_seconds = eval_seconds = 0.0
    wall = time.perf_counter()
    previous = 0
    for fraction in protocol["checkpoint_fractions"]:
        target = int(round(job["samples"] * fraction))
        start = time.perf_counter()
        agent.learn(total_timesteps=target - previous, reset_num_timesteps=False)
        fit_seconds += time.perf_counter() - start
        previous = target
        start = time.perf_counter()
        metrics = evaluate_agent(agent, validation, model=model, forecast_model=forecast_model,
                                 max_vms=max_vms, penalty=job["penalty"],
                                 decision_seconds=job["decision_seconds"], action_mode=job["action_mode"],
                                 hide_forecast=job["hide_forecast"], episode_seconds=protocol["episode_seconds"])
        eval_seconds += time.perf_counter() - start
        score = metrics["total_cost_usd"] + job["penalty"] * metrics["expected_violations"]
        learning.append({"samples": agent.num_timesteps, "split": "validation", "summary": metrics,
                         "selection_objective_usd": score,
                         "training_episode_return_mean": float(np.mean([r["r"] for r in agent.ep_info_buffer]))
                         if agent.ep_info_buffer else None,
                         "fit_seconds": fit_seconds, "evaluation_seconds": eval_seconds})
        if score < best_score:
            best_score, selected_step, selected_summary = score, int(agent.num_timesteps), metrics
            agent.save(model_path)
    result = {"job": job, "fingerprint": fingerprint, "selected_step": selected_step,
              "checkpoint_selection_split": "validation", "validation": selected_summary,
              "learning_curve": learning, "fit_seconds": fit_seconds, "evaluation_seconds": eval_seconds,
              "wall_seconds": time.perf_counter() - wall, "samples": int(agent.num_timesteps),
              "simulated_training_seconds": int(agent.num_timesteps) * job["decision_seconds"],
              "checkpoint": str(model_path.relative_to(output)),
              "checkpoint_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
    results_path.write_text(json.dumps(result, indent=2) + "\n")
    env.close()
    return result
