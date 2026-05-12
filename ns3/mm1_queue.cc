/* mm1_queue.cc — NS-3 discrete-event simulation of M/M/1 queue
 *
 * Build & run:
 *   cp mm1_queue.cc $NS3_HOME/scratch/
 *   cd $NS3_HOME
 *   ./ns3 run scratch/mm1_queue -- --lambda=9.0 --mu=10.0 --simTime=1000000
 *
 * The simulation schedules Poisson arrivals (inter-arrival ~ Exp(λ)) and
 * exponential service times (~ Exp(μ)) using NS-3's built-in
 * ExponentialRandomVariable.  It tracks time spent in each state n and
 * outputs the empirical Pn upon completion.
 *
 * Tested with NS-3.40.
 */

#include "ns3/core-module.h"
#include <iostream>
#include <map>
#include <cmath>
#include <iomanip>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("MM1Queue");

// ── Global simulation state ──────────────────────────────────────────────────

namespace {
    int    g_n          = 0;        // customers currently in system
    double g_lastEvent  = 0.0;      // time of last state change
    double g_simTime    = 1e6;      // total simulation time (seconds)
    double g_lambda     = 9.0;      // arrival rate
    double g_mu         = 10.0;     // service rate
    long   g_arrivals   = 0;
    long   g_departures = 0;
    std::map<int, double> g_timeInState;  // state → cumulative time

    Ptr<ExponentialRandomVariable> g_arrRng;
    Ptr<ExponentialRandomVariable> g_svcRng;
}

// ── Forward declarations ─────────────────────────────────────────────────────
void ScheduleArrival();
void ProcessArrival();
void ProcessDeparture();
void ScheduleDeparture();

// ── State accounting ─────────────────────────────────────────────────────────

void UpdateState(int new_n) {
    double now = Simulator::Now().GetSeconds();
    g_timeInState[g_n] += now - g_lastEvent;
    g_lastEvent = now;
    g_n = new_n;
}

// ── Event handlers ────────────────────────────────────────────────────────────

void ScheduleArrival() {
    double interarrival = g_arrRng->GetValue();
    Simulator::Schedule(Seconds(interarrival), &ProcessArrival);
}

void ScheduleDeparture() {
    double serviceTime = g_svcRng->GetValue();
    Simulator::Schedule(Seconds(serviceTime), &ProcessDeparture);
}

void ProcessArrival() {
    g_arrivals++;
    UpdateState(g_n + 1);

    if (g_n == 1) {
        // Server was idle; start service immediately
        ScheduleDeparture();
    }

    // Schedule next arrival (regardless of server state)
    ScheduleArrival();
}

void ProcessDeparture() {
    g_departures++;
    UpdateState(g_n - 1);

    if (g_n >= 1) {
        // Customers still waiting — start next service
        ScheduleDeparture();
    }
}

// ── Results output ────────────────────────────────────────────────────────────

void PrintResults() {
    // Flush remaining time in current state
    double now = Simulator::Now().GetSeconds();
    g_timeInState[g_n] += now - g_lastEvent;

    double total = 0.0;
    for (auto& kv : g_timeInState) total += kv.second;

    double rho      = g_lambda / g_mu;
    double L_anal   = rho / (1.0 - rho);
    double L_sim    = 0.0;
    for (auto& kv : g_timeInState) L_sim += kv.first * (kv.second / total);

    std::cout << "\n=== NS-3 M/M/1 Simulation Results ===\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  lambda      = " << g_lambda   << "\n";
    std::cout << "  mu          = " << g_mu        << "\n";
    std::cout << "  rho         = " << rho         << "\n";
    std::cout << "  sim_time    = " << g_simTime   << " s\n";
    std::cout << "  arrivals    = " << g_arrivals  << "\n";
    std::cout << "  departures  = " << g_departures<< "\n";
    std::cout << "  L (anal)    = " << L_anal      << "\n";
    std::cout << "  L (sim)     = " << L_sim       << "\n\n";
    std::cout << "  n   Pn_anal          Pn_sim\n";
    std::cout << "  --  ---------------  ---------------\n";

    int n_max = 35;
    for (int n = 0; n < n_max; ++n) {
        double pn_anal = (1.0 - rho) * std::pow(rho, n);
        double pn_sim  = g_timeInState.count(n)
                         ? g_timeInState[n] / total
                         : 0.0;
        std::cout << "  " << std::setw(2) << n
                  << "  " << std::setw(15) << pn_anal
                  << "  " << std::setw(15) << pn_sim
                  << "\n";
    }
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    CommandLine cmd(__FILE__);
    cmd.AddValue("lambda",  "Arrival rate",    g_lambda);
    cmd.AddValue("mu",      "Service rate",    g_mu);
    cmd.AddValue("simTime", "Simulation time", g_simTime);
    cmd.Parse(argc, argv);

    if (g_lambda >= g_mu) {
        std::cerr << "Error: system is unstable (lambda >= mu).\n";
        return 1;
    }

    // Exponential RVs: NS-3 parameterises by mean = 1/rate
    g_arrRng = CreateObject<ExponentialRandomVariable>();
    g_arrRng->SetAttribute("Mean", DoubleValue(1.0 / g_lambda));

    g_svcRng = CreateObject<ExponentialRandomVariable>();
    g_svcRng->SetAttribute("Mean", DoubleValue(1.0 / g_mu));

    // Seed NS-3 RNG for reproducibility
    RngSeedManager::SetSeed(42);
    RngSeedManager::SetRun(1);

    // Kick off the first arrival
    Simulator::Schedule(Seconds(0.0), &ScheduleArrival);

    Simulator::Stop(Seconds(g_simTime));
    Simulator::Run();
    Simulator::Destroy();

    PrintResults();
    return 0;
}