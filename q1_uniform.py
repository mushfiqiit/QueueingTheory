"""
Q1: Uniform Random Variable U[0,1] — Plot P(U > x) for x in (0.75, 1)

Method:
  - Generate N samples from U[0,1] using Python's random module (Mersenne Twister)
    as a baseline, then also via a custom Linear Congruential Generator (LCG).
  - For each x in (0.75, 1), estimate P(U > x) = fraction of samples exceeding x.
  - Theoretical: P(U > x) = 1 - x  (CCDF of Uniform[0,1]).
  - Plot empirical vs theoretical.
"""

import random
import math
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 200_000          # sample count
X_POINTS  = 1000             # grid resolution over (0.75, 1)

# ── Custom LCG ───────────────────────────────────────────────────────────────

class LCG:
    """Park-Miller LCG  (modulus = 2^31 - 1, multiplier = 16807)."""
    M = 2**31 - 1
    A = 16807
    C = 0

    def __init__(self, seed: int = 1):
        self.state = seed % self.M or 1   # must not be 0

    def random(self) -> float:
        """Return next U[0,1] sample."""
        self.state = (self.A * self.state + self.C) % self.M
        return self.state / self.M


def generate_uniform_lcg(n: int, seed: int = SEED) -> list:
    lcg = LCG(seed)
    return [lcg.random() for _ in range(n)]


def generate_uniform_stdlib(n: int, seed: int = SEED) -> list:
    rng = random.Random(seed)
    return [rng.random() for _ in range(n)]


# ── P(U > x) estimators ──────────────────────────────────────────────────────

def empirical_ccdf(samples: list, x_vals) -> list:
    """Return empirical P(U > x) for each x in x_vals."""
    n = len(samples)
    return [sum(1 for u in samples if u > x) / n for x in x_vals]


def theoretical_ccdf(x_vals) -> list:
    """P(U > x) = 1 - x for Uniform[0,1]."""
    return [1.0 - x for x in x_vals]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("plots", exist_ok=True)

    x_vals = [0.75 + i * (1.0 - 0.75) / (X_POINTS - 1) for i in range(X_POINTS)]

    # Generate samples
    samples_lcg    = generate_uniform_lcg(N_SAMPLES, SEED)
    samples_stdlib = generate_uniform_stdlib(N_SAMPLES, SEED)

    # Compute CCDFs
    p_lcg    = empirical_ccdf(samples_lcg,    x_vals)
    p_stdlib = empirical_ccdf(samples_stdlib, x_vals)
    p_theory = theoretical_ccdf(x_vals)

    # ── Figure 1: main CCDF comparison ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(r"Q1 — $P(U > x)$ for $U \sim \mathrm{Uniform}[0,1]$,  "
                 rf"$x \in (0.75, 1)$,  $N={N_SAMPLES:,}$ samples",
                 fontsize=13, fontweight="bold")

    for ax, p_emp, label, color in [
        (axes[0], p_lcg,    "LCG (custom)",       "#e74c3c"),
        (axes[1], p_stdlib, "Python random (stdlib)", "#2980b9"),
    ]:
        ax.plot(x_vals, p_theory, "k-",  linewidth=2.5, label="Theoretical  $1 - x$", zorder=3)
        ax.plot(x_vals, p_emp,   color=color, linewidth=1.5,
                linestyle="--", label=label, zorder=2)
        ax.set_xlabel(r"$x$", fontsize=12)
        ax.set_ylabel(r"$P(U > x)$", fontsize=12)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4)
        ax.set_xlim(0.75, 1.0)
        ax.set_ylim(-0.01, 0.27)

    plt.tight_layout()
    plt.savefig("plots/q1_uniform_ccdf.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: plots/q1_uniform_ccdf.png")

    # ── Figure 2: error plot ─────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    err_lcg    = [abs(e - t) for e, t in zip(p_lcg,    p_theory)]
    err_stdlib = [abs(e - t) for e, t in zip(p_stdlib, p_theory)]
    ax2.plot(x_vals, err_lcg,    color="#e74c3c", linewidth=1.2, label="LCG error")
    ax2.plot(x_vals, err_stdlib, color="#2980b9", linewidth=1.2, label="stdlib error")
    ax2.set_xlabel(r"$x$", fontsize=12)
    ax2.set_ylabel("Absolute error", fontsize=12)
    ax2.set_title(r"Absolute error  $|\hat{P}(U>x) - (1-x)|$", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.4)
    ax2.set_xlim(0.75, 1.0)
    plt.tight_layout()
    plt.savefig("plots/q1_error.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: plots/q1_error.png")

    # ── Summary statistics ───────────────────────────────────────────────────
    def rmse(err): return math.sqrt(sum(e**2 for e in err) / len(err))
    print("\n=== Q1 Summary ===")
    print(f"  Samples : {N_SAMPLES:,}")
    print(f"  RMSE (LCG)   : {rmse(err_lcg):.6f}")
    print(f"  RMSE (stdlib): {rmse(err_stdlib):.6f}")
    print("  Theoretical P(U > 0.75) = 0.25,  P(U > 0.99) = 0.01")
    print(f"  LCG    empirical at x=0.75 : {p_lcg[0]:.4f}")
    print(f"  stdlib empirical at x=0.75 : {p_stdlib[0]:.4f}")


if __name__ == "__main__":
    main()