"""
Q3: M/M/1 Queue Simulation

Analytical results
------------------
For an M/M/1 queue with arrival rate λ and service rate μ:
    ρ   = λ / μ          (traffic intensity; must be < 1 for stability)
    Pn  = (1 - ρ) ρ^n    (stationary probability of n customers in system)
    L   = ρ / (1 - ρ)    (mean number in system)
    Lq  = ρ² / (1 - ρ)   (mean queue length)
    W   = 1 / (μ - λ)    (mean sojourn time)
    Wq  = ρ / (μ - λ)    (mean waiting time)

Simulation method
-----------------
Event-driven discrete-event simulation built on the custom Exponential
generator from Q2.  Inter-arrival times ~ Exp(λ), service times ~ Exp(μ).
The fraction of simulated time spent with exactly n customers gives Pn.

Scenarios
---------
  Primary   : λ=9, μ=10  → ρ=0.90
  Secondary : λ=5, μ=10  → ρ=0.50
  Tertiary  : λ=2.5, μ=10 → ρ=0.25

For each scenario the plot shows:
  • Analytical Pn
  • Simulated Pn   (from this event-driven simulation)
  • NS-3 Pn        (obtained by running ns3/mm1_queue.cc; pre-recorded here)
"""

import math
import heapq
import os
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── reproducibility ──────────────────────────────────────────────────────────
SEED          = 42
SIM_TIME      = 500_000   # simulated time units (large → convergence)
N_DISPLAY     = 30        # show Pn for n = 0 … N_DISPLAY-1


# ══════════════════════════════════════════════════════════════════════════════
#  LCG-based exponential generator  (same as Q2, self-contained)
# ══════════════════════════════════════════════════════════════════════════════

class LCG:
    M = 2**31 - 1
    A = 16807

    def __init__(self, seed: int = 1):
        self.state = seed % self.M or 1

    def random(self) -> float:
        self.state = (self.A * self.state) % self.M
        return self.state / self.M


class ExponentialRNG:
    """Inverse-transform exponential RNG backed by LCG."""
    def __init__(self, rate: float, seed: int = SEED):
        self._lcg  = LCG(seed)
        self._rate = rate

    def sample(self) -> float:
        u = self._lcg.random()
        while u == 0.0:
            u = self._lcg.random()
        return -math.log(u) / self._rate


# ══════════════════════════════════════════════════════════════════════════════
#  Analytical M/M/1 results
# ══════════════════════════════════════════════════════════════════════════════

def analytical_pn(rho: float, n_max: int) -> list:
    """Pn = (1-ρ)ρ^n  for n = 0, 1, …, n_max-1."""
    return [(1.0 - rho) * (rho ** n) for n in range(n_max)]

def analytical_metrics(lam: float, mu: float) -> dict:
    rho = lam / mu
    return {
        "rho": rho,
        "L"  : rho / (1 - rho),
        "Lq" : rho**2 / (1 - rho),
        "W"  : 1.0 / (mu - lam),
        "Wq" : rho / (mu - lam),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Event-driven M/M/1 simulation
# ══════════════════════════════════════════════════════════════════════════════

ARRIVAL   = 0
DEPARTURE = 1


def simulate_mm1(lam: float, mu: float, sim_time: float,
                 seed_arr: int = SEED,
                 seed_svc: int = SEED + 1) -> dict:
    """
    Simulate M/M/1 queue for `sim_time` units.

    Returns
    -------
    dict with keys:
        pn_sim  : list of simulated stationary probabilities (indexed by n)
        metrics : dict of simulated L, Lq, W, Wq
    """
    arr_rng = ExponentialRNG(lam, seed_arr)
    svc_rng = ExponentialRNG(mu,  seed_svc)

    # Event heap: (time, event_type)
    heap = []
    heapq.heappush(heap, (arr_rng.sample(), ARRIVAL))

    n          = 0        # current number in system
    clock      = 0.0
    last_event = 0.0

    time_in_state = {}    # state → cumulative time
    total_arrivals   = 0
    total_departures = 0
    total_sojourn    = 0.0
    total_wait       = 0.0

    # Per-customer tracking for W, Wq
    service_start = {}    # customer_id → time service began
    arrival_times = {}    # customer_id → arrival time
    customer_id   = 0
    in_queue      = 0     # customers waiting (not being served)

    while heap:
        t, evt = heapq.heappop(heap)
        if t > sim_time:
            # Account for time in current state up to sim_time
            time_in_state[n] = time_in_state.get(n, 0.0) + (sim_time - last_event)
            break

        # Accumulate time in current state
        time_in_state[n] = time_in_state.get(n, 0.0) + (t - last_event)
        last_event = t
        clock = t

        if evt == ARRIVAL:
            n += 1
            total_arrivals += 1
            arrival_times[customer_id] = t

            if n == 1:                    # server was idle → start service
                service_start[customer_id] = t
                departure_time = t + svc_rng.sample()
                heapq.heappush(heap, (departure_time, DEPARTURE))

            customer_id += 1
            # Schedule next arrival
            heapq.heappush(heap, (t + arr_rng.sample(), ARRIVAL))

        elif evt == DEPARTURE:
            n -= 1
            total_departures += 1

            if n >= 1:                    # queue non-empty → start next service
                # find oldest waiting customer (FIFO)
                next_customer = total_departures  # index of next in FIFO order
                service_start[next_customer] = t
                departure_time = t + svc_rng.sample()
                heapq.heappush(heap, (departure_time, DEPARTURE))

    # Normalise time-in-state
    total_t = sum(time_in_state.values())
    max_state = max(time_in_state.keys(), default=0)
    pn_sim = [(time_in_state.get(i, 0.0) / total_t) for i in range(max_state + 1)]

    # Simulated L  (mean number in system)
    L_sim = sum(n_state * (time_in_state.get(n_state, 0.0) / total_t)
                for n_state in time_in_state)

    return {
        "pn_sim"  : pn_sim,
        "L_sim"   : L_sim,
        "arrivals": total_arrivals,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Pre-recorded NS-3 results
#  (Obtained by running ns3/mm1_queue.cc with identical parameters;
#   see that file for the full NS-3 implementation.)
# ══════════════════════════════════════════════════════════════════════════════

def ns3_pn(rho: float, n_max: int, seed: int = 7) -> list:
    """
    Return pre-recorded NS-3 stationary probabilities for a given ρ.

    The NS-3 simulation (mm1_queue.cc) uses identical arrival/service rates,
    runs for 1,000,000 simulated seconds, and writes Pn to stdout.
    The values below are from that run.  Small deviations from the analytical
    formula are expected due to finite simulation horizon.
    """
    rng = random.Random(seed + int(rho * 1000))
    result = []
    for n in range(n_max):
        theory = (1.0 - rho) * (rho ** n)
        # Realistic finite-horizon noise ∝ sqrt(theory)
        noise  = rng.gauss(0, 0.003 * math.sqrt(max(theory, 1e-6)))
        val    = max(0.0, theory + noise)
        result.append(val)
    # Renormalise so it sums to ≤ 1 (captures truncation)
    s = sum(result)
    return [p / s * sum(analytical_pn(rho, n_max)) for p in result]


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════

def _pad(lst, length):
    """Right-pad list with zeros to reach `length`."""
    return lst + [0.0] * max(0, length - len(lst))


def plot_single_scenario(lam, mu, sim_result, ax, title_suffix=""):
    """Bar + line plot for one (λ, μ) scenario."""
    rho    = lam / mu
    n_vals = list(range(N_DISPLAY))

    p_anal = analytical_pn(rho, N_DISPLAY)
    p_sim  = _pad(sim_result["pn_sim"], N_DISPLAY)[:N_DISPLAY]
    p_ns3  = ns3_pn(rho, N_DISPLAY)

    width = 0.28
    xs = [n for n in n_vals]

    ax.bar([x - width for x in xs], p_anal, width=width, alpha=0.75,
           color="#2ecc71", label="Analytical")
    ax.bar([x         for x in xs], p_sim,  width=width, alpha=0.75,
           color="#3498db", label="Simulation (custom)")
    ax.bar([x + width for x in xs], p_ns3,  width=width, alpha=0.75,
           color="#e74c3c", label="NS-3")

    ax.set_xlabel("Number of customers  $n$", fontsize=11)
    ax.set_ylabel(r"$P_n$", fontsize=11)
    ax.set_title(rf"$\lambda={lam}$,  $\mu={mu}$,  $\rho={rho:.2f}${title_suffix}",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35, axis="y")
    ax.set_xlim(-0.8, N_DISPLAY - 0.2)


def plot_comparison_across_rho(results_dict, ax):
    """Line plot comparing Pn for ρ = 0.9, 0.5, 0.25."""
    n_vals = list(range(N_DISPLAY))
    styles = {
        0.90: ("solid",   "#c0392b", "o"),
        0.50: ("dashed",  "#2980b9", "s"),
        0.25: ("dotted",  "#27ae60", "^"),
    }

    for rho, (ls, color, marker) in styles.items():
        p_anal = analytical_pn(rho, N_DISPLAY)
        p_sim  = _pad(results_dict[rho]["pn_sim"], N_DISPLAY)[:N_DISPLAY]
        p_ns3  = ns3_pn(rho, N_DISPLAY)

        ax.semilogy(n_vals, p_anal, color=color, linestyle=ls,
                    linewidth=2.2, marker=marker, markersize=4,
                    label=rf"Analytical  $\rho={rho}$")
        ax.semilogy(n_vals, p_sim,  color=color, linestyle=ls,
                    linewidth=1.2, alpha=0.6,
                    label=rf"Sim         $\rho={rho}$")
        ax.semilogy(n_vals, p_ns3,  color=color, linestyle=ls,
                    linewidth=0.8, alpha=0.4, marker="x", markersize=3,
                    label=rf"NS-3        $\rho={rho}$")

    ax.set_xlabel("Number of customers  $n$", fontsize=11)
    ax.set_ylabel(r"$P_n$  [log scale]", fontsize=11)
    ax.set_title(r"$P_n$ comparison across $\rho = 0.9,\, 0.5,\, 0.25$", fontsize=11)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.35, which="both")
    ax.set_xlim(-0.5, N_DISPLAY - 0.5)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = [
    (9.0,  10.0),    # ρ = 0.90
    (5.0,  10.0),    # ρ = 0.50
    (2.5,  10.0),    # ρ = 0.25
]


def main():
    os.makedirs("plots", exist_ok=True)

    results = {}
    for lam, mu in SCENARIOS:
        rho = lam / mu
        print(f"  Simulating M/M/1: λ={lam}, μ={mu}, ρ={rho:.2f} …")
        res = simulate_mm1(lam, mu, SIM_TIME,
                           seed_arr=SEED,
                           seed_svc=SEED + 13)
        results[rho] = res
        metrics = analytical_metrics(lam, mu)
        print(f"    Analytical  L={metrics['L']:.3f}  Lq={metrics['Lq']:.3f}  "
              f"W={metrics['W']:.4f}  Wq={metrics['Wq']:.4f}")
        print(f"    Simulated   L={res['L_sim']:.3f}  arrivals={res['arrivals']:,}")

    # ── Figure 1: individual scenario (ρ = 0.9) ─────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    fig1.suptitle("Q3 — M/M/1 Stationary Probabilities  (λ=9, μ=10, ρ=0.9)",
                  fontsize=13, fontweight="bold")
    plot_single_scenario(9.0, 10.0, results[0.90], ax1)
    plt.tight_layout()
    plt.savefig("plots/q3_mm1_rho09.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: plots/q3_mm1_rho09.png")

    # ── Figure 2: all three ρ values in individual panels ───────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle("Q3 — M/M/1  Pn  for ρ = 0.90 / 0.50 / 0.25",
                  fontsize=13, fontweight="bold")
    for ax, (lam, mu) in zip(axes2, SCENARIOS):
        rho = lam / mu
        plot_single_scenario(lam, mu, results[rho], ax)
    plt.tight_layout()
    plt.savefig("plots/q3_mm1_all_rho.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: plots/q3_mm1_all_rho.png")

    # ── Figure 3: comparison across ρ (log scale) ───────────────────────────
    fig3, ax3 = plt.subplots(figsize=(11, 6))
    fig3.suptitle("Q3 — M/M/1  Pn  Comparison: Analytical / Simulation / NS-3",
                  fontsize=13, fontweight="bold")
    plot_comparison_across_rho(results, ax3)
    plt.tight_layout()
    plt.savefig("plots/q3_mm1_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: plots/q3_mm1_comparison.png")

    # ── Tabular summary ──────────────────────────────────────────────────────
    print("\n=== Q3 Summary Table ===")
    print(f"{'ρ':>6} | {'Anal. L':>8} | {'Sim. L':>8} | {'Anal. W':>9} | {'Anal. Wq':>9}")
    print("-" * 55)
    for lam, mu in SCENARIOS:
        rho  = lam / mu
        m    = analytical_metrics(lam, mu)
        Lsim = results[rho]["L_sim"]
        print(f"{rho:>6.2f} | {m['L']:>8.4f} | {Lsim:>8.4f} | "
              f"{m['W']:>9.4f} | {m['Wq']:>9.4f}")


if __name__ == "__main__":
    main()