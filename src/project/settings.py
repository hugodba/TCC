from __future__ import annotations

# Experiment controls
N_RUNS = 1
SEED_START = 1
PARALLEL_AGENTS = True

# Time-budget controls (used by the external tuner)
METHOD_COUNT = 6
TOTAL_MAIN_BUDGET_SEC = 60*60*24 # 24 hours

# MAS controls
MAS_POOL_SIZE = 5
MAS_SEED_POLICY = "random"  # Options: "best", "random"
MAS_N_STEPS = 200

# Isolated metaheuristics params (auto-tuned by src/tune_settings.py)
# Tunnable params are: iterations_per_temp | max_iterations, tabu_tenure | generations
SA_PARAMS = {'initial_temp': 1000.0, 'cooling_rate': 0.95, 'min_temp': 10.0, 'iterations_per_temp': 949}
TS_PARAMS = {'max_iterations': 1738, 'tabu_tenure': 243, 'neighbor_samples': 50}
GA_PARAMS = {'population_size': 64, 'generations': 1063, 'crossover_rate': 0.8, 'mutation_rate': 0.1}

# Cooperative inner configs (auto-tuned by src/tune_settings.py)
MAS_SA_PARAMS = {'initial_temp': 300.0, 'cooling_rate': 0.95, 'min_temp': 10.0, 'iterations_per_temp': 1}
MAS_TS_PARAMS = {'max_iterations': 3, 'tabu_tenure': 0, 'neighbor_samples': 20}
MAS_GA_PARAMS = {'population_size': 16, 'generations': 2, 'crossover_rate': 0.8, 'mutation_rate': 0.1}

# RL controls
MAS_RL_PARAMS = {
    # Probabilidade inicial de exploração no epsilon-greedy
    "epsilon": 0.8,
    # Valor mínimo de epsilon (limite inferior da exploração)
    "epsilon_min": 0.05,
    # Decaimento multiplicativo de epsilon aplicado a cada step
    "epsilon_decay": 0.992,
    # Taxa de aprendizado (quanto o Q-valor novo substitui o antigo)
    "alpha": 0.2,
    # Fator de desconto da recompensa futura no Q-Learning
    "gamma": 0.8,
    # Tamanho do bin para discretizar distância no estado: d_bin = int(d / bin_size)
    "distance_bin_size": 100,
    # Quantas ações (execuções de agentes) são feitas por step de RL
    # Também é usado pelo tuner para estimar orçamento por solver interno
    "runs_per_step": 3,
}

# Comparison controls
D_GAPS = [1.0, 1.5, 2.0, 2.5, 3.0] #1.0 means +100%, 3.0 means +300%.
K_GAPS = [1.0, 1.5, 2.0, 2.5, 3.0]
TTT_TARGETS = {"vehicles": True, "distance": True,}
METHOD_LABELS = ["SA", "TS", "GA", "MAS", "MAS_RL", "MAS_RL_WARM"]
INSTANCE_GROUPS = {
    "C1": [
        "c101.txt",
        "c102.txt",
        #"c103.txt",
        #"c104.txt",
        #"c105.txt"
    ],
    "C2": [
        "c201.txt",
        "c202.txt",
        #"c203.txt",
        #"c204.txt",
        #"c205.txt"
    ],
    "R1": [
        "r101.txt",
        "r102.txt",
        #"r103.txt",
        #"r104.txt",
        #"r105.txt"
    ],
    "R2": [
        "r201.txt",
        "r202.txt",
        #"r204.txt",
        #"r206.txt",
        #"r208.txt"
    ],
    "RC1": [
        "rc102.txt",
        "rc103.txt",
        #"rc104.txt",
        #"rc105.txt",
        #"rc106.txt"
    ],
    "RC2": [
        "rc201.txt",
        "rc204.txt",
        #"rc205.txt",
        #"rc206.txt",
        #"rc207.txt"
    ],
}

# Calculations
INSTANCE_COUNT = sum([len(group_instances) for group_instances in INSTANCE_GROUPS.values()])
TARGET_METHOD_RUNS = METHOD_COUNT * INSTANCE_COUNT * N_RUNS
TARGET_METHOD_RUN_SEC = TOTAL_MAIN_BUDGET_SEC / TARGET_METHOD_RUNS


# External tuner defaults
DEFAULT_TUNING_DATASET = "rc201.txt"

# Updated by src/tune_settings.py
LAST_TUNING_REPORT = {'dataset': 'rc201.txt',
 'method_runs': 2,
 'total_main_budegt_sec': 86400,
 'mas_n_steps': 300,
 'method_count': 6,
 'instance_count': 13,
 'target_method_runs': 156,
 'target_method_run_sec': 553.8461538461538,
 'estimated_method_run_sec': {'SA': 551.1875,
                              'TS': 561.140625,
                              'GA': 549.046875,
                              'MAS': 679.6875,
                              'MAS_RL': 679.6875,
                              'MAS_RL_WARM': 679.6875},
 'estimated_total_main_runtime_sec': 96211.375,
 'estimated_total_main_runtime_min': 1603.5229166666666,
 'fixed_controls': {'MAS_POOL_SIZE': 5,
                    'MAS_SEED_POLICY': 'random',
                    'MAS_RL_PARAMS': {'epsilon': 0.8,
                                      'epsilon_min': 0.05,
                                      'epsilon_decay': 0.992,
                                      'alpha': 0.2,
                                      'gamma': 0.8,
                                      'distance_bin_size': 100,
                                      'runs_per_step': 3}}}
