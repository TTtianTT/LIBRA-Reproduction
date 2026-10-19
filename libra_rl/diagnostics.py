"""Reusable learning diagnostics with nested budgets and separate rule fallbacks."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .diagnostic_env import DiagnosticEnv, initialize_rule_prior
from .study import aggregate, chunks
from .training import ObservationTransform


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    temporary.replace(path)


def make_env(counts, *, model, teacher, max_vms, forecast, penalty,
             augment=False, episode_seconds=None):
    return ObservationTransform(DiagnosticEnv(counts,model=model,rule_rho=teacher["rho"],
        rule_window=teacher["window"],max_vms=max_vms,forecast_model=forecast,penalty=penalty,
        augment=augment,episode_seconds=episode_seconds,cost_normalizer=0.01))


def evaluate(agent, counts, *, model, teacher, max_vms, forecast, penalty,
             episode_seconds=21600, fixed=None):
    envs, observations = [], []
    for block in chunks(counts,episode_seconds):
        env = make_env(block,model=model,teacher=teacher,max_vms=max_vms,forecast=forecast,penalty=penalty)
        observations.append(env.reset(seed=0,options={"start":0})[0])
        envs.append(env)
    active, maxima = list(range(len(envs))), [0]*len(envs)
    action_counts = np.zeros(envs[0].action_space.n,dtype=np.int64)
    behavioral_teacher, decisions = 0, 0
    while active:
        if fixed is not None:
            actions = [envs[i].unwrapped.teacher_action if fixed=="teacher" else
                       envs[i].unwrapped.residual_choices for i in active]
        else:
            actions,_ = agent.predict(np.stack([observations[i] for i in active]),deterministic=True)
        remaining = []
        for i,action in zip(active,actions):
            base = envs[i].unwrapped.rule_target()
            observation,_,terminated,truncated,info = envs[i].step(action)
            action_counts[int(action)] += 1
            decisions += 1
            behavioral_teacher += int(info["target_vms"]==base and
                                      (base==0 or abs(info["rho"]-teacher["rho"])<1e-12))
            maxima[i] = max(maxima[i],info["target_vms"])
            observations[i] = observation
            if not (terminated or truncated):
                remaining.append(i)
        active = remaining
    result = aggregate([{**env.unwrapped.summary(),"max_target_vms":n} for env,n in zip(envs,maxima)])
    result.update(action_counts=action_counts.tolist(),decisions=decisions,
                  behavioral_teacher_fraction=behavioral_teacher/max(1,decisions),
                  explicit_faas_action_fraction=float(action_counts[-1]/max(1,decisions)))
    return result


def train_nested(train, validation, *, model, forecast, teacher, penalty, max_vms,
                  job, protocol, output, fingerprint):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    output = Path(output)
    result_path = output/"runs"/(job["id"]+".json")
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous["fingerprint"] != fingerprint:
            raise ValueError("stale diagnostic training result")
        for point in previous["curve"]:
            p = output/point["checkpoint"]
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=point["checkpoint_sha256"]:
                raise ValueError("missing/modified diagnostic checkpoint")
        return previous
    env = Monitor(make_env(train,model=model,teacher=teacher,max_vms=max_vms,forecast=forecast,
                           penalty=penalty,augment=job["augment"],episode_seconds=job["episode_seconds"]))
    agent = PPO("MlpPolicy",env,seed=job["seed"],device="cpu",verbose=0,
                policy_kwargs={"net_arch":protocol["network"]},**protocol["ppo"])
    if job["initialization"]=="rule_prior":
        initialize_rule_prior(agent,env.unwrapped.teacher_action,protocol["rule_prior_probability"])
    elif job["initialization"]!="random":
        raise ValueError("unknown initialization")
    checkpoints = [n for n in protocol["checkpoints"] if n<=job["samples"]]
    if not checkpoints or checkpoints[-1]!=job["samples"]:
        raise ValueError("training budget must end at a declared checkpoint")
    quantum = protocol["ppo"]["n_steps"]
    if any((b-a)%quantum for a,b in zip([0]+checkpoints[:-1],checkpoints)):
        raise ValueError("checkpoint increments must match the rollout length")
    directory = output/"models"/job["id"]
    directory.mkdir(parents=True,exist_ok=True)
    evaluation_args = dict(model=model,teacher=teacher,max_vms=max_vms,forecast=forecast,
                            penalty=penalty,episode_seconds=job["episode_seconds"])
    initial = evaluate(agent,validation,**evaluation_args)
    teacher_result = evaluate(None,validation,fixed="teacher",**evaluation_args)
    faas_result = evaluate(None,validation,fixed="faas",**evaluation_args)
    if job["initialization"]=="rule_prior":
        if abs(initial["total_cost_usd"]-teacher_result["total_cost_usd"])>1e-9:
            raise AssertionError("greedy rule prior does not reproduce teacher billing")
    curve, fit_seconds, evaluation_seconds, previous = [],0.0,0.0,0
    for target in checkpoints:
        start = time.perf_counter()
        agent.learn(total_timesteps=target-previous,reset_num_timesteps=False)
        fit_seconds += time.perf_counter()-start
        if agent.num_timesteps!=target:
            raise AssertionError("SB3 silently overshot the declared sample budget")
        start = time.perf_counter()
        validation_result = evaluate(agent,validation,**evaluation_args)
        evaluation_seconds += time.perf_counter()-start
        path = directory/f"step{target}.zip"
        agent.save(path)
        logger = agent.logger.name_to_value
        diagnostics = {k:float(logger[k]) for k in ("train/entropy_loss","train/approx_kl", "train/value_loss",
                       "train/explained_variance","train/policy_gradient_loss")
                       if k in logger and np.isfinite(logger[k])}
        curve.append({"samples":target,"split":"validation","summary":validation_result,
                      "objective_usd":validation_result["total_cost_usd"]+penalty*validation_result["expected_violations"],
                      "fit_seconds":fit_seconds,"evaluation_seconds":evaluation_seconds,
                      "optimizer_updates":int(agent._n_updates),"optimizer_diagnostics":diagnostics,
                      "checkpoint":str(path.relative_to(output)),
                      "checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        previous = target
    selections = []
    budgets = [n for n in protocol["budgets"] if n<=job["samples"]]
    if job["case"]=="toy":
        budgets = [job["samples"]]
    for budget in budgets:
        allowed = {budget//4,budget//2,3*budget//4,budget}
        candidates = [p for p in curve if p["samples"] in allowed]
        if len(candidates)!=4:
            raise AssertionError("each nested budget must have four post-update candidates")
        selected = min(candidates,key=lambda p:p["objective_usd"])
        selection = {"budget":budget,"selected_samples":selected["samples"],"summary":selected["summary"],
                     "checkpoint":selected["checkpoint"],"checkpoint_sha256":selected["checkpoint_sha256"],
                     "objective_usd":selected["objective_usd"],"selection_split":"validation"}
        deploy_candidates = [("teacher",teacher_result),("all_faas",faas_result)]
        deploy_candidates += [(f"learned_step{p['samples']}",p["summary"]) for p in candidates]
        feasible = [(label,s) for label,s in deploy_candidates if s["violation_rate"]<=protocol["epsilon"]]
        label,summary = min(feasible,key=lambda item:item[1]["total_cost_usd"])
        selection["validation_feasible_fallback"] = {"source":label,"summary":summary,
            "meaning":"reported separately; retaining a fixed teacher is not an RL improvement or a test guarantee"}
        selections.append(selection)
    result = {"fingerprint":fingerprint,"job":job,"teacher":teacher,"penalty":penalty,
              "initial_validation":initial,"teacher_validation":teacher_result,"faas_validation":faas_result,
              "curve":curve,"selections":selections,"samples":int(agent.num_timesteps),
              "fit_seconds":fit_seconds,"evaluation_seconds":evaluation_seconds,
              "selection_split":"validation","test_evaluated":False}
    write_json(result_path,result)
    env.close()
    return result
