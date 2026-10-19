from itertools import product
import math

import numpy as np
import pytest

from libra_rl.diagnostic_env import DiagnosticEnv, base_targets, initialize_rule_prior
from libra_rl.evaluation import evaluate_targets
from libra_rl.model import CostModel
from libra_rl.queue_validation import simulate_queue
from libra_rl.restricted_dp import residual_bound
from libra_rl.dp import evaluate_schedule
from libra_rl.training import ObservationTransform


def test_nested_budgets_are_one_training_trajectory(tmp_path):
    import torch
    from libra_rl.diagnostics import train_nested
    torch.set_num_threads(1)
    protocol = dict(network=[8,8], ppo=dict(n_steps=128,batch_size=64,n_epochs=1),
        checkpoints=[128,256,384,512,1024,1536,2048],budgets=[512,2048],
        epsilon=0.001,rule_prior_probability=0.9)
    job = dict(id="unit",case="unit",initialization="rule_prior",seed=0,
               augment=False,episode_seconds=600,samples=2048)
    args = dict(model=CostModel(),forecast=None,teacher=dict(rho=1.0,window=300),
                penalty=0,max_vms=8,job=job,protocol=protocol,output=tmp_path,fingerprint="unit")
    result = train_nested(np.arange(1800)%50,np.arange(600)%40,**args)
    assert result["samples"] == 2048
    assert len(result["curve"]) == 7
    assert [s["budget"] for s in result["selections"]] == [512,2048]
    assert result["initial_validation"]["total_cost_usd"] == result["teacher_validation"]["total_cost_usd"]
    assert train_nested(np.arange(1800)%50,np.arange(600)%40,**args) == result
    assert result["test_evaluated"] is False


@pytest.mark.parametrize("k", [60,120,300])
@pytest.mark.parametrize("allow_faas", [False,True])
def test_restricted_dp_matches_exhaustive_and_replay(k, allow_faas):
    x = np.arange(2*k-7)%75
    bases = [4,5]
    model = CostModel(version="v2",service_seconds=0.7)
    rho_grid, penalty = (0.6,1.0), 0.0001
    result = residual_bound(x,bases,model=model,decision_seconds=k,max_vms=6,
                             rho_grid=rho_grid,allow_faas=allow_faas,penalty=penalty,return_plan=True)
    choices = [set(np.clip(b+np.arange(-3,4),0,6)) | ({0} if allow_faas else set()) for b in bases]
    best = math.inf
    for targets in product(*choices):
        for rhos in product(rho_grid,repeat=2):
            replay = evaluate_schedule(x,targets,model=model,decision_seconds=k,rhos=rhos)
            best = min(best,replay["total_cost_usd"]+penalty*replay["expected_violations"])
    assert result["objective_usd"] == pytest.approx(best,abs=1e-11)
    replay = evaluate_schedule(x,result["targets"],model=model,decision_seconds=k,rhos=result["rhos"])
    assert replay["total_cost_usd"] == pytest.approx(result["total_cost_usd"],abs=1e-11)
    assert replay["expected_violations"] == pytest.approx(result["expected_violations"],abs=1e-9)
    assert all(t in allowed for t,allowed in zip(result["targets"],choices))


def test_faas_expands_domain_and_guarantees_feasibility():
    x = np.full(360,300)
    bases = np.full(6,20)
    model = CostModel(version="v2",service_seconds=0.7)
    old = residual_bound(x,bases,model=model,rho_grid=(0.6,0.8,1.0),penalty=0.01)
    new = residual_bound(x,bases,model=model,rho_grid=(0.6,0.8,1.0),penalty=0.01,allow_faas=True)
    assert new["objective_usd"] <= old["objective_usd"]+1e-12
    minimum = residual_bound(x,bases,model=model,cost_weight=0,penalty=1,allow_faas=True)
    assert minimum["expected_violations"] == 0
    assert minimum["objective_usd"] == 0


@pytest.mark.parametrize("service,rho",[(1.0,1.0),(0.5,0.7),(0.7,0.6)])
def test_teacher_and_full_faas_replay_exact_accounting(service,rho):
    model = CostModel(version="v1" if service==1 else "v2",service_seconds=service)
    x = np.arange(1003)%150
    targets = base_targets(x,model,rho=rho)
    ref = evaluate_targets(x,targets,model=model,rhos=rho)
    env = DiagnosticEnv(x,model=model,rule_rho=rho,episode_seconds=None,max_vms=512)
    env.reset()
    while True:
        _,_,done,_,_ = env.step(env.teacher_action)
        if done:
            break
    for key in ("total_cost_usd","expected_violations","vm_seconds"):
        assert env.summary()[key] == pytest.approx(ref[key],abs=1e-9)
    env.reset()
    env.step(env.teacher_action)
    _,_,_,_,info = env.step(env.residual_choices)
    assert info["target_vms"] == 0
    assert info["vm_requests"] == info["expected_violations"] == 0
    assert info["faas_requests"] == sum(x[60:120])


def test_rule_logit_prior_is_exact_greedy_teacher():
    sb3 = pytest.importorskip("stable_baselines3")
    import torch
    torch.set_num_threads(1)
    env = ObservationTransform(DiagnosticEnv([32]*300,episode_seconds=None))
    agent = sb3.PPO("MlpPolicy",env,n_steps=8,batch_size=8,seed=0,device="cpu")
    teacher = env.unwrapped.teacher_action
    initialize_rule_prior(agent,teacher,0.9)
    obs,_ = env.reset(seed=0)
    tensor,_ = agent.policy.obs_to_tensor(obs)
    distribution = agent.policy.get_distribution(tensor)
    assert float(distribution.distribution.probs[0,teacher].detach()) == pytest.approx(0.9,abs=1e-6)
    assert int(agent.predict(obs,deterministic=True)[0]) == teacher


def test_queue_simulation_matches_known_single_server_limits():
    mm = simulate_queue(utilization=0.7,service_seconds=0.5,distribution="exponential",
                          servers=1,warmup=10000,measured=100000,seed=2)
    assert mm["violation_rate"] == pytest.approx(math.exp(-0.6),abs=0.02)
    md = simulate_queue(utilization=0.5,service_seconds=0.5,distribution="deterministic",
                          servers=1,warmup=10000,measured=100000,seed=3)
    assert md["mean_response_seconds"] == pytest.approx(0.75,abs=0.02)
