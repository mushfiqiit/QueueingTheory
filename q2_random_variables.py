"""
Q2: Custom Exponential and Poisson Random Variable Generators

Exponential X  with E[X] = 10  →  rate λ_X = 1/10 = 0.1
Poisson     Z  with E[Z] = 10  →  parameter λ_Z = 10

Custom generators use ONLY the inverse-transform / product methods built on top
of the LCG defined in q1_uniform.py.  Results are validated against
scipy.stats.expon and scipy.stats.poisson.

Plots produced
--------------
1. Exponential PDF  — custom histogram vs scipy theoretical curve
2. Poisson     PMF  — custom bar chart   vs scipy theoretical PMF
3. P(X > x)         — custom CCDF vs analytical e^{-λx}
4. P(Z > z)         — custom CCDF vs analytical 1 - F_Poisson(z)
"""

import math
import random
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

# ── Parameters ───────────────────────────────────────────────────────────────
SEED       = 42
N_SAMPLES  = 200_000

E_X   = 10.0              # E[X] for exponential → rate = 1/10
LAM_X = 1.0 / E_X        # 0.1

E_Z   = 10.0              # E[Z] for Poisson → λ_Z = 10
LAM_Z = E_Z               # 10.0


# ══════════════════════════════════════════════════════════════════════════════
#  LCG (same as Q1, duplicated here so each module is self-contained)
# ══════════════════════════════════════════════════════════════════════════════

class LCG:
    """Park-Miller multiplicative LCG  (m = 2^31-1, a = 16807)."""
    M = 2**31 - 1
    A = 16807

    def __init__(self, seed: int = 1):
        self.state = seed % self.M or 1

    def random(self) -> float:
        self.state = (self.A * self.state) % self.M
        return self.state / self.M


# ══════════════════════════════════════════════════════════════════════════════
#  Custom Exponential Generator — Inverse Transform Method
# ══════════════════════════════════════════════════════════════════════════════

def generate_exponential(lam: float, n: int, seed: int = SEED) -> list:
    """
    Generate n samples from Exp(λ) using the inverse-transform method.

    If U ~ Uniform(0,1) then  X = -ln(U) / λ  ~ Exp(λ).
    E[X] = 1/λ.
    """
    lcg = LCG(seed)
    samples = []
    for _ in range(n):
        u = lcg.random()
        # guard against u == 0 (vanishingly rare with LCG)
        while u == 0.0:
            u = lcg.random()
        samples.append(-math.log(u) / lam)
    return samples


# ══════════════════════════════════════════════════════════════════════════════
#  Custom Poisson Generator — Knuth's multiplicative algorithm
# ══════════════════════════════════════════════════════════════════════════════


def generate_poisson(lam: float, n: int, seed: int = SEED) -> list:
    """
    Generate n samples from Poisson(λ) using Knuth's algorithm.

    Algorithm:
        Set L = e^{-λ}, p = 1, k = 0.
        While p > L:  k ← k+1,  p ← p * U_i   (U_i ~ Uniform(0,1))
        Return k - 1.

    This works because the number of uniform products needed to drop below
    e^{-λ} follows Poisson(λ).

    Note: For large λ (e.g. λ=10) this is moderately fast; for very large λ
    a rejection/split method would be used instead.
    """
    lcg = LCG(seed)
    L = math.exp(-lam)
    samples = []
    for _ in range(n):
        p = 1.0
        k = 0
        while p > L:
            p *= lcg.random()
            k += 1
        samples.append(k - 1)
    return samples



# ══════════════════════════════════════════════════════════════════════════════
#  Theoretical distributions (computed analytically, validated with scipy)
# ══════════════════════════════════════════════════════════════════════════════

def exp_pdf_theory(x, lam=LAM_X):
    """f(x) = λ e^{-λx}"""
    return lam * math.exp(-lam * x) if x >= 0 else 0.0

def exp_ccdf_theory(x, lam=LAM_X):
    """P(X > x) = e^{-λx}"""
    return math.exp(-lam * x) if x >= 0 else 1.0

def poisson_pmf_theory(k, lam=LAM_Z):
    """P(Z = k) = e^{-λ} λ^k / k!"""
    return math.exp(-lam) * (lam**k) / math.factorial(k)

def poisson_ccdf_theory(z, lam=LAM_Z):
    """P(Z > z) = 1 - Σ_{k=0}^{z} P(Z=k)"""
    cdf = sum(poisson_pmf_theory(k, lam) for k in range(int(z) + 1))
    return max(0.0, 1.0 - cdf)


# ══════════════════════════════════════════════════════════════════════════════
#  Empirical CCDF from samples
# ══════════════════════════════════════════════════════════════════════════════

def empirical_ccdf(samples: list, x_vals) -> list:
    n = len(samples)
    return [sum(1 for s in samples if s > x) / n for x in x_vals]


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _x_grid(lo, hi, pts=500):
    step = (hi - lo) / (pts - 1)
    return [lo + i * step for i in range(pts)]


def plot_exp_pdf(samples_custom, ax):
    """Histogram of custom samples vs theoretical PDF vs scipy PDF."""
    x_max  = 80
    x_vals = _x_grid(0, x_max)

    # Scipy reference
    theory_pdf    = [exp_pdf_theory(x) for x in x_vals]
    scipy_pdf     = [stats.expon.pdf(x, scale=E_X) for x in x_vals]

    ax.hist(samples_custom, bins=120, density=True, alpha=0.45,
            color="#27ae60", label=f"Custom (N={N_SAMPLES:,})", edgecolor="none")
    ax.plot(x_vals, theory_pdf, "k-",      linewidth=2.5, label=r"Analytical  $\lambda e^{-\lambda x}$")
    ax.plot(x_vals, scipy_pdf,  "r--",     linewidth=1.5, label="scipy.stats.expon")
    ax.set_xlim(0, x_max)
    ax.set_xlabel(r"$x$", fontsize=11)
    ax.set_ylabel("PDF", fontsize=11)
    ax.set_title(r"Exponential PDF  ($E[X]=10$, $\lambda=0.1$)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_poisson_pmf(samples_custom, ax):
    """Bar chart of custom PMF vs theoretical vs scipy."""
    k_max = 30
    k_vals = list(range(k_max + 1))

    # Empirical
    total = len(samples_custom)
    counts = [0] * (k_max + 2)
    for s in samples_custom:
        if s <= k_max:
            counts[int(s)] += 1
    emp_pmf = [counts[k] / total for k in k_vals]

    # Theoretical
    theo_pmf  = [poisson_pmf_theory(k) for k in k_vals]
    scipy_pmf = [stats.poisson.pmf(k, LAM_Z) for k in k_vals]

    width = 0.4
    ax.bar([k - width/2 for k in k_vals], emp_pmf,  width=width, alpha=0.65,
           color="#27ae60", label=f"Custom (N={N_SAMPLES:,})")
    ax.bar([k + width/2 for k in k_vals], scipy_pmf, width=width, alpha=0.55,
           color="#e67e22", label="scipy.stats.poisson")
    ax.plot(k_vals, theo_pmf, "k-o", markersize=4, linewidth=1.5,
            label=r"Analytical  $e^{-\lambda}\lambda^k/k!$")
    ax.set_xlim(-0.8, k_max + 0.8)
    ax.set_xlabel(r"$k$", fontsize=11)
    ax.set_ylabel("PMF", fontsize=11)
    ax.set_title(r"Poisson PMF  ($E[Z]=10$, $\lambda=10$)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")


def plot_exp_ccdf(samples_custom, ax):
    """P(X > x) — empirical vs analytical vs scipy."""
    x_vals = _x_grid(0, 60)

    theory_ccdf = [exp_ccdf_theory(x) for x in x_vals]
    scipy_ccdf  = [stats.expon.sf(x, scale=E_X) for x in x_vals]
    emp_ccdf    = empirical_ccdf(samples_custom, x_vals)

    ax.semilogy(x_vals, theory_ccdf, "k-",  linewidth=2.5,
                label=r"Analytical  $e^{-\lambda x}$")
    ax.semilogy(x_vals, scipy_ccdf,  "r--", linewidth=1.5, label="scipy.stats.expon")
    ax.semilogy(x_vals, emp_ccdf,    color="#27ae60", linewidth=1.5,
                linestyle=":", label=f"Custom simulation (N={N_SAMPLES:,})")
    ax.set_xlim(0, 60)
    ax.set_xlabel(r"$x$", fontsize=11)
    ax.set_ylabel(r"$P(X > x)$  [log scale]", fontsize=11)
    ax.set_title(r"Exponential CCDF   $P(X > x)$", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")


def plot_poisson_ccdf(samples_custom, ax):
    """P(Z > z) — empirical vs analytical vs scipy."""
    z_vals = list(range(0, 31))

    theory_ccdf = [poisson_ccdf_theory(z) for z in z_vals]
    scipy_ccdf  = [stats.poisson.sf(z, LAM_Z) for z in z_vals]
    emp_ccdf    = empirical_ccdf(samples_custom, z_vals)

    ax.semilogy(z_vals, theory_ccdf, "k-o",  markersize=4, linewidth=2,
                label=r"Analytical  $1 - F_Z(z)$")
    ax.semilogy(z_vals, scipy_ccdf,  "r--s", markersize=4, linewidth=1.5,
                label="scipy.stats.poisson")
    ax.semilogy(z_vals, emp_ccdf,    color="#27ae60", linewidth=1.5,
                linestyle=":", marker="^", markersize=4,
                label=f"Custom simulation (N={N_SAMPLES:,})")
    ax.set_xlim(-0.5, 30.5)
    ax.set_xlabel(r"$z$", fontsize=11)
    ax.set_ylabel(r"$P(Z > z)$  [log scale]", fontsize=11)
    ax.set_title(r"Poisson CCDF   $P(Z > z)$", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs("plots", exist_ok=True)

    print("Generating exponential samples …")
    exp_samples = generate_exponential(LAM_X, N_SAMPLES, SEED)

    print("Generating Poisson samples …")
    poi_samples = generate_poisson(LAM_Z, N_SAMPLES, SEED)

    # ── Figure 1: PDF / PMF comparison ──────────────────────────────────────
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle("Q2 — Custom RV Generators vs scipy Reference",
                  fontsize=13, fontweight="bold")
    plot_exp_pdf    (exp_samples, axes1[0])
    plot_poisson_pmf(poi_samples, axes1[1])
    plt.tight_layout()
    plt.savefig("plots/q2_pdf_pmf.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: plots/q2_pdf_pmf.png")

    # ── Figure 2: CCDF comparison ────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle(r"Q2 — $P(X>x)$ and $P(Z>z)$: Custom vs Analytical vs scipy",
                  fontsize=13, fontweight="bold")
    plot_exp_ccdf    (exp_samples, axes2[0])
    plot_poisson_ccdf(poi_samples, axes2[1])
    plt.tight_layout()
    plt.savefig("plots/q2_ccdf.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: plots/q2_ccdf.png")

    # ── Summary statistics ────────────────────────────────────────────────────
    exp_mean = sum(exp_samples) / len(exp_samples)
    exp_var  = sum((x - exp_mean)**2 for x in exp_samples) / len(exp_samples)
    poi_mean = sum(poi_samples) / len(poi_samples)
    poi_var  = sum((z - poi_mean)**2 for z in poi_samples) / len(poi_samples)

    print("\n=== Q2 Summary ===")
    print(f"  Exponential  — E[X]={E_X}  λ={LAM_X}")
    print(f"    Sample mean : {exp_mean:.4f}  (expected {E_X})")
    print(f"    Sample var  : {exp_var:.4f}   (expected {E_X**2:.1f})")
    print(f"  Poisson      — E[Z]={E_Z}  λ={LAM_Z}")
    print(f"    Sample mean : {poi_mean:.4f}  (expected {E_Z})")
    print(f"    Sample var  : {poi_var:.4f}   (expected {LAM_Z})")


if __name__ == "__main__":
    main()