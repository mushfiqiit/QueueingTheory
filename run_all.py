"""
run_all.py — Master runner for the Queuing Theory project (CSE 422/562).

Executes Q1, Q2, and Q3 in sequence and saves all plots to ./plots/.
"""

import os
import sys

def header(title):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")

def main():
    os.makedirs("plots", exist_ok=True)

    header("Q1: Uniform[0,1] random variable — P(U > x)")
    import q1_uniform
    q1_uniform.main()

    header("Q2: Exponential and Poisson RVs from scratch")
    import q2_random_variables
    q2_random_variables.main()

    header("Q3: M/M/1 Queue simulation")
    import q3_mm1_queue
    q3_mm1_queue.main()

    print("\n" + "=" * 70)
    print("  All done.  Plots saved to ./plots/")
    print("=" * 70)

if __name__ == "__main__":
    main()