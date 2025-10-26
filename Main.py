import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# Simulation parameter
T = 300
capacity = 100
conflict_sets = ["hot", "cold"]
alpha_grid = np.linspace(0.0, 0.6, 7)
burst_prob = 0.2
burst_multiplier = 2.5
local_fee_k = 0.8
runs = 5

lambda_L = 80
lambda_M = 40
lambda_S = 40

def draw_values(n, mean, std):
    return np.maximum(rng.lognormal(mean=np.log(mean), sigma=std, size=n), 0.01)

def local_fee_cost(demand, cap):
    load = demand / max(cap, 1)
    return local_fee_k * (load ** 2)

records = []

for alpha in alpha_grid:
    for r in range(runs):
        revenues = []
        failures = []
        surplus_total = []
        surplus_L = []
        surplus_S = []
        revenue_var_tracker = []
        
        for t in range(T):
            is_burst = rng.random() < burst_prob
            burst = burst_multiplier if is_burst else 1.0
            
            n_L = rng.poisson(lambda_L * burst)
            n_M = rng.poisson(lambda_M * burst)
            n_S = rng.poisson(lambda_S)
            
            def assign_sets(n):
                k_hot = int(round(hot_share * n))
                k_cold = n - k_hot
                sets = ["hot"] * k_hot + ["cold"] * k_cold
                rng.shuffle(sets)
                return sets
            
            sets_L = assign_sets(n_L)
            sets_M = assign_sets(n_M)
            sets_S = assign_sets(n_S)
            
            vals_L = draw_values(n_L, mean=5.0, std=0.6)
            vals_M = draw_values(n_M, mean=6.0, std=0.7)
            vals_S = draw_values(n_S, mean=4.0, std=0.5)
            
            cap_AOT = int(round(alpha * capacity))
            cap_JIT = capacity - cap_AOT
            
            revenue_slot = 0.0
            failures_slot = 0
            surplus_slot = 0.0
            surplus_L_slot = 0.0
            surplus_S_slot = 0.0
            
            for cs in conflict_sets:
                idx_S = [i for i, s in enumerate(sets_S) if s == cs]
                bids_S = vals_S[idx_S] if len(idx_S) > 0 else np.array([])
                
                lf_A = local_fee_cost(len(idx_S), capacity)
                eff_bids_S = bids_S - lf_A
                eff_bids_S = eff_bids_S[eff_bids_S > 0]
                
                q_A = min(cap_AOT, len(eff_bids_S))
                if q_A > 0:
                    sorted_bids = np.sort(eff_bids_S)[::-1]
                    pA = sorted_bids[q_A - 1]
                    winners = sorted_bids[:q_A]
                    revenue_slot += pA * q_A
                    true_winners = np.sort(bids_S)[::-1][:q_A] if len(bids_S) >= q_A else bids_S
                    surplus = np.sum(true_winners - (pA + lf_A))
                    surplus_slot += max(surplus, 0.0)
                    surplus_S_slot += max(surplus, 0.0)
                
                idx_L = [i for i, s in enumerate(sets_L) if s == cs]
                idx_M = [i for i, s in enumerate(sets_M) if s == cs]
                bids_L = vals_L[idx_L] if len(idx_L) > 0 else np.array([])
                bids_M = vals_M[idx_M] if len(idx_M) > 0 else np.array([])
                
                jit_demand = len(bids_L) + len(bids_M)
                lf_J = local_fee_cost(jit_demand, capacity)
                
                eff_bids_J = np.concatenate([bids_L, bids_M]) - lf_J
                eff_bids_J = eff_bids_J[eff_bids_J > 0]
                
                q_J = min(cap_JIT, len(eff_bids_J))
                if q_J > 0:
                    sorted_e = np.sort(eff_bids_J)[::-1]
                    pJ = sorted_e[q_J - 1]
                    revenue_slot += pJ * q_J
                    combined_true = np.sort(np.concatenate([bids_L, bids_M]))[::-1]
                    true_top = combined_true[:q_J] if len(combined_true) >= q_J else combined_true
                    surplus_true = np.sum(true_top - (pJ + lf_J))
                    surplus_true = max(surplus_true, 0.0)
                    surplus_slot += surplus_true
                    if q_J > 0 and len(bids_L) > 0:
                        share_L = len(bids_L) / max(len(bids_L) + len(bids_M), 1)
                        surplus_L_slot += surplus_true * share_L
                
                failures_slot += max(len(bids_S) - q_A, 0)
                failures_slot += max(jit_demand - q_J, 0)
            
            revenues.append(revenue_slot)
            failures.append(failures_slot)
            surplus_total.append(surplus_slot)
            surplus_L.append(surplus_L_slot)
            surplus_S.append(surplus_S_slot)
            revenue_var_tracker.append(revenue_slot)
        
        records.append({
            "alpha": alpha,
            "run": r,
            "mean_revenue": np.mean(revenues),
            "revenue_variance": np.var(revenue_var_tracker),
            "failed_fraction": np.sum(failures) / (T * (lambda_L + lambda_M + lambda_S)),
            "mean_surplus_total": np.mean(surplus_total),
            "mean_surplus_L": np.mean(surplus_L),
            "mean_surplus_S": np.mean(surplus_S),
        })

df = pd.DataFrame(records)
summary = df.groupby("alpha").agg({
    "mean_revenue": "mean",
    "revenue_variance": "mean",
    "failed_fraction": "mean",
    "mean_surplus_total": "mean",
    "mean_surplus_L": "mean",
    "mean_surplus_S": "mean",
}).reset_index()

print(summary.to_string(index=False))

# Simple plots
plt.figure()
plt.plot(summary["alpha"], summary["failed_fraction"], marker="o")
plt.xlabel("AOT reserve α")
plt.ylabel("Failed transaction fraction")
plt.title("Failures vs AOT reserve")
plt.show()

plt.figure()
plt.plot(summary["alpha"], summary["revenue_variance"], marker="o")
plt.xlabel("AOT reserve α")
plt.ylabel("Validator revenue variance")
plt.title("Revenue variance vs AOT reserve")
plt.show()

plt.figure()
plt.plot(summary["alpha"], summary["mean_revenue"], marker="o")
plt.xlabel("AOT reserve α")
plt.ylabel("Mean validator revenue")
plt.title("Mean revenue vs AOT reserve")
plt.show()
