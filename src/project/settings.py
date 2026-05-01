from __future__ import annotations

# Experiment controls
N_RUNS = 2
SEED_START = 1
PARALLEL_AGENTS = True

# MAS controls
MAS_POOL_SIZE = 7
ELITE_POOL_DIVERSITY_TRESHOLD = 10
MAS_SEED_POLICY = "random"  # Options: "best", "random"
MAS_N_STEPS = 80

# # Isolated metaheuristics params (auto-tuned by src/tune_settings.py)
# SA_PARAMS = {'initial_temp': 1800, 'cooling_rate': 0.99, 'min_temp': 3, 'iterations_per_temp': 800} # iterations per temp * math.log(min_temp / initial_temp) / math.log(cooling_rate)
# TS_PARAMS = {'max_iterations': 2000, 'tabu_tenure': 20, 'neighbor_samples': 300} # max iters * neighbor samples
# GA_PARAMS = {'population_size': 70, 'generations': 10_000, 'crossover_rate': 0.7, 'mutation_rate': 0.05} # generations * population size

# # Cooperative inner configs (auto-tuned by src/tune_settings.py)
# MAS_SA_PARAMS = {'initial_temp': 1800.0, 'cooling_rate': 0.99, 'min_temp': 3.0, 'iterations_per_temp': 50}
# MAS_TS_PARAMS = {'max_iterations': 500, 'tabu_tenure': 20, 'neighbor_samples': 75}
# MAS_GA_PARAMS = {'population_size': 70, 'generations': 500, 'crossover_rate': 0.7, 'mutation_rate': 0.05}


# Isolated metaheuristics params (auto-tuned by src/tune_settings.py)
SA_PARAMS = {'initial_temp': 1800, 'cooling_rate': 0.99, 'min_temp': 3, 'iterations_per_temp': 20} # iterations per temp * math.log(min_temp / initial_temp) / math.log(cooling_rate)
TS_PARAMS = {'max_iterations': 100, 'tabu_tenure': 20, 'neighbor_samples': 100} # max iters * neighbor samples
GA_PARAMS = {'population_size': 70, 'generations': 40, 'crossover_rate': 0.7, 'mutation_rate': 0.05} # generations * population size

# Cooperative inner configs (auto-tuned by src/tune_settings.py)
MAS_SA_PARAMS = {'initial_temp': 1800.0, 'cooling_rate': 0.99, 'min_temp': 3.0, 'iterations_per_temp': 20}
MAS_TS_PARAMS = {'max_iterations': 100, 'tabu_tenure': 20, 'neighbor_samples': 75}
MAS_GA_PARAMS = {'population_size': 70, 'generations': 20, 'crossover_rate': 0.7, 'mutation_rate': 0.05}

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
INSTANCES = ["rc201.txt", "rc204.txt", "rc205.txt", "rc206.txt", "rc207.txt"]

# External tuner defaults
DEFAULT_TUNING_DATASET = "rc201.txt"

# Updated by src/tune_settings.py
LAST_TUNING_REPORT = {'tuner': 'optuna_single_objective',
 'objective_strategy': 'vehicles_scaled_priority',
 'objective_formula': 'vehicles * 1000000 + distance',
 'dataset': 'rc201.txt',
 'seed_start': 1,
 'eval_runs': 1,
 'n_trials_per_method': 30,
 'methods': {'SA': {'best_params': {'initial_temp': 222.51446048745515,
                                    'cooling_rate': 0.9834211564571398,
                                    'min_temp': 1.1524825557934042,
                                    'iterations_per_temp': 571},
                    'selected_objective': {'scalar': 9001832.606042597,
                                           'avg_vehicles': 9.0,
                                           'avg_distance': 1832.606042597375,
                                           'avg_scalar_objective': 9001832.606042597},
                    'selected_trial_number': 3,
                    'n_trials': 30,
                    'best_params_run_time_sec': 1348.8125,
                    'best_metrics': {'avg_vehicles': 9.0,
                                     'avg_distance': 1832.606042597375,
                                     'avg_scalar_objective': 9001832.606042597,
                                     'avg_runtime_sec': 1348.8125,
                                     'best_run_vehicles': 9,
                                     'best_run_distance': 1832.606042597375}},
             'TS': {'best_params': {'max_iterations': 1316,
                                    'tabu_tenure': 321,
                                    'neighbor_samples': 25},
                    'selected_objective': {'scalar': 12003108.070439514,
                                           'avg_vehicles': 12.0,
                                           'avg_distance': 3108.07043951299,
                                           'avg_scalar_objective': 12003108.070439514},
                    'selected_trial_number': 23,
                    'n_trials': 30,
                    'best_params_run_time_sec': 284.78125,
                    'best_metrics': {'avg_vehicles': 12.0,
                                     'avg_distance': 3108.07043951299,
                                     'avg_scalar_objective': 12003108.070439514,
                                     'avg_runtime_sec': 284.78125,
                                     'best_run_vehicles': 12,
                                     'best_run_distance': 3108.07043951299}},
             'GA': {'best_params': {'population_size': 79,
                                    'generations': 445,
                                    'crossover_rate': 0.5508609045034749,
                                    'mutation_rate': 0.21642252201152065},
                    'selected_objective': {'scalar': 12003414.607561821,
                                           'avg_vehicles': 12.0,
                                           'avg_distance': 3414.607561821397,
                                           'avg_scalar_objective': 12003414.607561821},
                    'selected_trial_number': 27,
                    'n_trials': 30,
                    'best_params_run_time_sec': 323.953125,
                    'best_metrics': {'avg_vehicles': 12.0,
                                     'avg_distance': 3414.607561821397,
                                     'avg_scalar_objective': 12003414.607561821,
                                     'avg_runtime_sec': 323.953125,
                                     'best_run_vehicles': 12,
                                     'best_run_distance': 3414.607561821397}}},
 'generated_at': '2026-04-27 19:38:23'}

