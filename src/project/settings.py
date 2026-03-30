from __future__ import annotations

# Shared experiment controls
N_RUNS = 4
SEED_START = 42
PARALLEL_AGENTS = True

# Time-budget controls (used by the external tuner)
METHOD_COUNT = 5
TOTAL_MAIN_BUDGET_SEC = 3600
RUNTIME_SAFETY_MARGIN = 1.0

# Fixed MAS controls (do not auto-tune)
MAS_POOL_SIZE = 5
MAS_SEED_POLICY = "random"  # Options: "best", "random"
MAS_N_STEPS = 30
MAS_RL_N_STEPS = 30

# Isolated metaheuristics params (auto-tuned by src/tune_settings.py)
SA_PARAMS = {'initial_temp': 450.0, 'cooling_rate': 0.97, 'min_temp': 10.0, 'iterations_per_temp': 212}

TS_PARAMS = {'max_iterations': 995, 'tabu_tenure': 100, 'neighbor_samples': 25}

GA_PARAMS = {'population_size': 32, 'generations': 606, 'crossover_rate': 0.8, 'mutation_rate': 0.1}

# Cooperative inner configs (auto-tuned by src/tune_settings.py)
MAS_SA_PARAMS = {'initial_temp': 220.0, 'cooling_rate': 0.95, 'min_temp': 20.0, 'iterations_per_temp': 4}

MAS_TS_PARAMS = {'max_iterations': 19, 'tabu_tenure': 3, 'neighbor_samples': 10}

MAS_GA_PARAMS = {'population_size': 16, 'generations': 10, 'crossover_rate': 0.8, 'mutation_rate': 0.12}

# Fixed RL controls (do not auto-tune)
MAS_RL_PARAMS = {
    "epsilon": 0.4,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "alpha": 0.2,
    "gamma": 0.8,
    "distance_bin_size": 100,
    "runs_per_step": 3,
}

# External tuner defaults
DEFAULT_TUNING_DATASET = "rc201.txt"

# Updated by src/tune_settings.py
LAST_TUNING_REPORT = {'dataset': 'rc201.txt',
 'target_method_run_sec': 180.0,
 'estimated_method_run_sec': {'SA': 181.09375,
                              'TS': 176.609375,
                              'GA': 188.53125,
                              'MAS': 181.5625,
                              'MAS_RL': 188.75},
 'estimated_total_main_runtime_sec': 3666.1875,
 'estimated_total_main_runtime_min': 61.103125,
 'target_total_main_runtime_sec': 3600.0,
 'target_total_main_runtime_min': 60.0,
 'probe_rates_sec_per_unit': {'SA': 0.010300925925925925,
                              'TS': 0.0109375,
                              'GA': 0.012291666666666666},
 'measured_isolated_run_sec': {'SA': 181.09375, 'TS': 176.609375, 'GA': 188.53125},
 'fixed_controls': {'MAS_POOL_SIZE': 7,
                    'MAS_SEED_POLICY': 'best',
                    'MAS_N_STEPS': 20,
                    'MAS_RL_N_STEPS': 20,
                    'MAS_RL_PARAMS': {'epsilon': 0.4,
                                      'epsilon_min': 0.05,
                                      'epsilon_decay': 0.995,
                                      'alpha': 0.2,
                                      'gamma': 0.8,
                                      'distance_bin_size': 100,
                                      'runs_per_step': 3}}}
