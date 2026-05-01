import optuna 
from optuna.samplers import TPESampler
import math

def objective(trial):

    x = trial.suggest_float("x", -10, 10)
    return x**2

study = optuna.create_study(sampler=TPESampler())
study.optimize(objective, n_trials=100)


initial_temp = 1250
cooling_rate = 0.99
min_temp = 3
iterations_per_temp = 280

# T_final = T_initial * (cooling_rate ^ num_steps)
# num_steps = log(T_final / T_initial) / log(cooling_rate)

num_steps = math.log(min_temp / initial_temp) / math.log(cooling_rate)
total_iterations = num_steps * iterations_per_temp

print(f"{num_steps=}")
print(f"{total_iterations=}")