import heapq
from itertools import product

import numpy as np
import pytest

from libra_rl.physical import ArrivalStream,QueueFleet,_serve,serve,evaluate_plan,staggered_copies,actions
from libra_rl.physical_planning import cost_relaxation,clairvoyant_search,perfect_window


def test_slot_ring_matches_independent_heap_queue():
    rng = np.random.default_rng(12)
    arrivals = np.cumsum(rng.exponential(0.02,12000))
    destinations = rng.integers(0,3,len(arrivals))
    available,pointers = np.zeros((3,16)),np.zeros(3,dtype=np.int64)
    heaps = [[0.0]*16 for _ in range(3)]
    expected = [0,0,0.0,0.0]
    for a,vm in zip(arrivals,destinations):
        start = max(a,heaps[vm][0]); departure = start+0.7
        heapq.heapreplace(heaps[vm],departure)
        response = departure-a
        expected[0] += int(response>1+1e-10)
        expected[1] += int(start>a+1e-10)
        expected[2] += response
        expected[3] = max(expected[3],response)
    observed = serve(arrivals,destinations,available,pointers,0.7,1.0)
    np.testing.assert_allclose(observed,expected,rtol=0,atol=1e-8)
    np.testing.assert_allclose(np.sort(available),np.sort(np.array(heaps)),atol=1e-10)


def test_full_load_carries_work_to_next_second_and_drains_on_scale_in():
    stream = ArrivalStream(np.array([32,32]),np.array([0,32,64]),
        np.r_[np.full(32,0.99),np.full(32,1.0)],np.tile(np.arange(32),2),np.zeros(64))
    fleet = QueueFleet(stream,service_seconds=0.5,cold_start_seconds=0)
    first = fleet.step(1,1.0,1)
    assert first["violations"]==0
    assert first["backlog"]==32
    second = fleet.step(1,1.0,1)
    assert second["violations"]==32
    assert fleet.summary(terminal=True)["terminal_drain_vm_seconds"]==pytest.approx(0.99)
    other = QueueFleet(stream,service_seconds=0.5,cold_start_seconds=0)
    other.step(1,1.0,1)
    other.step(0,1.0,1)
    assert other.summary()["drain_vm_seconds"]==pytest.approx(0.99)
    assert other.summary()["faas_requests"]==32
    assert other.summary()["violations"]==0


def test_lookup_rejects_saturation_instead_of_returning_one():
    from libra_rl.physical_lookup import MD16Lookup
    table = MD16Lookup({"0.5":[dict(utilization=0.0,monotone_estimate=0.0),
                               dict(utilization=0.9,monotone_estimate=0.025)]})
    assert table.probability(0.9,0.5)==0.025
    with pytest.raises(ValueError,match="transient"):
        table.probability(1.0,0.5)


def test_split_and_combined_steps_match_and_clone_isolated():
    stream = ArrivalStream.generate(np.arange(181)%65,seed=4)
    one = QueueFleet(stream,service_seconds=0.5)
    split = one.clone()
    one.step(3,0.9,181)
    for _ in range(3):
        split.step(3,0.9,60)
    split.step(3,0.9,1)
    for key in one.summary(terminal=True):
        assert one.summary(terminal=True)[key]==pytest.approx(split.summary(terminal=True)[key],abs=1e-9)
    original = split.availability.copy()
    clone = split.clone(); clone.availability[:]=999
    np.testing.assert_array_equal(split.availability,original)


def test_staggered_workload_never_wraps_or_crosses_partition():
    x = np.arange(10)
    np.testing.assert_array_equal(staggered_copies(x,[0,2,3]),x[3:]+x[1:8]+x[:7])
    a,b = np.arange(20),np.arange(20)
    b[10:]=9999
    np.testing.assert_array_equal(staggered_copies(a,[0,2])[:8],staggered_copies(b,[0,2])[:8])
    with pytest.raises(ValueError):
        staggered_copies(x,[10])


def test_controller_does_not_see_the_current_second_and_quota_is_causal():
    from scripts.physical_headroom import bases_for
    a = np.zeros(1200,dtype=int)
    a[540:600]=33
    b = a.copy(); b[600]=100000
    # An inclusive t window would cross a VM threshold on this counterfactual.
    np.testing.assert_array_equal(bases_for(a,600,60,0.5),bases_for(b,600,60,0.5))
    stream = ArrivalStream.generate([100],seed=4)
    fleet = QueueFleet(stream,service_seconds=0.5,cold_start_seconds=0)
    fleet.step(1,0.5,1)
    summary = fleet.summary(terminal=True)
    assert summary["vm_requests"]==16 and summary["faas_requests"]==84
    assert summary["terminal_drain_vm_seconds"]==0


def test_exact_queue_state_dp_matches_enumeration_and_lower_bound():
    stream = ArrivalStream.generate(np.full(180,45),seed=8)
    bases = np.array([0,0,0])
    best = float("inf")
    for choices in product(actions(0),repeat=3):
        result = evaluate_plan(stream,[m for m,_ in choices],[r for _,r in choices],service_seconds=0.5)
        if result["violation_rate"]<=0.001:
            best = min(best,result["total_cost_usd"])
    dp = clairvoyant_search(stream,bases,service_seconds=0.5,incumbent_targets=[0,0,0],
                            incumbent_rhos=[1,1,1],exact=True)
    assert dp["exact"]
    assert dp["summary"]["total_cost_usd"]==pytest.approx(best,abs=1e-9)
    assert dp["cost_lower_bound_usd"]<=best+1e-9
    replay = evaluate_plan(stream,dp["targets"],dp["rhos"],service_seconds=0.5)
    assert replay==dp["summary"]


def test_beam_labels_pruning_and_keeps_feasible_incumbent():
    stream = ArrivalStream.generate(np.arange(240)%60,seed=6)
    bases = np.array([2,2,2,2])
    result = clairvoyant_search(stream,bases,service_seconds=0.7,incumbent_targets=[0]*4,
                                incumbent_rhos=[1.0]*4,beam_width=2)
    assert not result["exact"] and result["pruned_states"]>0
    assert result["summary"]["violation_rate"]<=0.001
    assert result["cost_lower_bound_usd"]<=result["summary"]["total_cost_usd"]+1e-9
    oracle = perfect_window(stream,bases,service_seconds=0.7)
    assert oracle["summary"]["violation_rate"]<=0.001
    assert evaluate_plan(stream,oracle["targets"],oracle["rhos"],service_seconds=0.7)==oracle["summary"]


def test_anytime_exact_search_certifies_or_returns_valid_interval():
    from libra_rl.physical_exact import optimal_interval
    stream = ArrivalStream.generate(np.full(180,45),seed=9)
    bases = np.array([0,0,0])
    expected = clairvoyant_search(stream,bases,service_seconds=0.5,incumbent_targets=[0]*3,
                                  incumbent_rhos=[1.0]*3,exact=True)
    exact = optimal_interval(stream,bases,service_seconds=0.5,incumbent_targets=[0]*3,
                             incumbent_rhos=[1.0]*3,max_seconds=10)
    assert exact["exact"]
    assert exact["cost_lower_bound_usd"]==pytest.approx(expected["summary"]["total_cost_usd"],abs=1e-9)
    limited = optimal_interval(stream,bases,service_seconds=0.5,incumbent_targets=[0]*3,
                               incumbent_rhos=[1.0]*3,max_expansions=0)
    assert limited["cost_lower_bound_usd"]<=exact["feasible_upper_bound_usd"]+1e-9
    assert limited["feasible_upper_bound_usd"]>=exact["cost_lower_bound_usd"]-1e-9


def test_phase_sla_dual_cannot_exceed_independent_constrained_enumeration():
    from libra_rl.physical_dual import phase_dual_tables
    stream = ArrivalStream.generate(np.full(180,100),seed=11)
    bases = np.zeros(3,dtype=int)
    best = float("inf")
    for choices in product(actions(0),repeat=3):
        result = evaluate_plan(stream,[m for m,_ in choices],[r for _,r in choices],service_seconds=0.7)
        if result["violation_rate"]<=0.001:
            best = min(best,result["total_cost_usd"])
    penalties,tables = phase_dual_tables(stream,bases,0.7)
    gateway = 180*0.0116/3600
    dual = max(v[0][0]-p*0.001*int(stream.counts.sum())+gateway for p,v in zip(penalties,tables))
    assert dual<=best+1e-9
    assert dual>=cost_relaxation(stream.counts,bases,0.7)["lower_bound_usd"]-1e-9
