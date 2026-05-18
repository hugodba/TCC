from __future__ import annotations

import ast
import math
import pprint
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

import optuna

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
SETTINGS_PATH = SRC_DIR / "project" / "settings.py"

# Method-specific guardrails to avoid very slow trials.
MAX_ESTIMATED_WORK_UNITS = {
    "SA": 600_000,
    "TS": 700_000,
    "GA": 900_000,
}
STAGE1_MAX_RUNTIME_SEC = {
    "SA": 120.0,
    "TS": 90.0,
    "GA": 120.0,
}
STAGE_RUNTIME_GROWTH_LIMIT = {
    "SA": 1.25,
    "TS": 1.30,
    "GA": 1.30,
}

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project import Dataset
from project import settings as cfg
from project.Metaheuristics import GeneticAlgorithm, Metaheuristic, SimulatedAnnealing, TabuSearch


def _status(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[tune_settings {stamp}] {message}", flush=True)


def _format_assignment(name: str, value: Any) -> list[str]:
    rendered = pprint.pformat(value, width=100, sort_dicts=False)
    text = f"{name} = {rendered}"
    return text.splitlines()


def _replace_assignments(path: Path, updates: dict[str, Any]) -> None:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)

    nodes: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    nodes[target.id] = node
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                nodes[node.target.id] = node

    replacements: list[tuple[int, int, list[str]]] = []
    for name, value in updates.items():
        if name not in nodes:
            raise KeyError(f"Could not find assignment for '{name}' in {path}")
        node = nodes[name]
        start = node.lineno - 1
        end = node.end_lineno
        replacements.append((start, end, _format_assignment(name, value)))

    for start, end, new_lines in sorted(replacements, key=lambda item: item[0], reverse=True):
        lines[start:end] = new_lines

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scalarized_objective(vehicles: int, distance: float) -> float:
    # Single-objective score with vehicles prioritized over distance.
    return float(vehicles) * 1_000_000.0 + float(distance)


def _estimate_sa_temperature_levels(
    initial_temp: float,
    cooling_rate: float,
    min_temp: float,
) -> int:
    initial = max(1e-9, float(initial_temp))
    minimum = max(1e-9, float(min_temp))
    cooling = float(cooling_rate)

    if cooling <= 0.0 or cooling >= 1.0:
        return 10**9
    if minimum >= initial:
        return 1

    try:
        levels = int(math.ceil(math.log(minimum / initial) / math.log(cooling)))
    except (ValueError, ZeroDivisionError):
        return 10**9

    return max(1, levels)


def _estimate_work_units(
    method_name: str,
    params: dict[str, float | int],
) -> int:
    if method_name == "SA":
        levels = _estimate_sa_temperature_levels(
            initial_temp=float(params["initial_temp"]),
            cooling_rate=float(params["cooling_rate"]),
            min_temp=float(params["min_temp"]),
        )
        return int(levels * int(params["iterations_per_temp"]))

    if method_name == "TS":
        max_iterations = int(params["max_iterations"])
        neighbor_samples = int(params["neighbor_samples"])
        return int(max_iterations * neighbor_samples)

    if method_name == "GA":
        population_size = int(params["population_size"])
        generations = int(params["generations"])
        return int(population_size * generations)

    return 0


def _scaled_params_by_budget(
    method_name: str,
    params: dict[str, float | int],
    budget_ratio: float,
) -> dict[str, float | int]:
    scaled = dict(params)
    ratio = min(1.0, max(0.05, float(budget_ratio)))

    if method_name == "SA":
        base_iterations = int(params["iterations_per_temp"])
        scaled["iterations_per_temp"] = max(1, int(round(base_iterations * ratio)))

        initial_temp = max(1e-9, float(params["initial_temp"]))
        base_min_temp = min(initial_temp, max(1e-9, float(params["min_temp"])))

        # Keep early stages cheap by shortening the cooling depth.
        scaled_min_temp = initial_temp * ((base_min_temp / initial_temp) ** ratio)
        scaled["min_temp"] = float(min(initial_temp, max(base_min_temp, scaled_min_temp)))
    elif method_name == "TS":
        base_iterations = int(params["max_iterations"])
        base_tabu_tenure = int(params["tabu_tenure"])

        scaled_iterations = max(1, int(round(base_iterations * ratio)))
        scaled_tabu_tenure = max(1, int(round(base_tabu_tenure * ratio)))

        scaled["max_iterations"] = scaled_iterations
        scaled["tabu_tenure"] = min(scaled_iterations, scaled_tabu_tenure)
    elif method_name == "GA":
        base = int(params["generations"])
        scaled["generations"] = max(1, int(round(base * ratio)))

    return scaled


def _evaluate_solver_params(
    dataset: Dataset,
    algo_cls: type[Metaheuristic],
    params: dict[str, float | int],
    seed_start: int,
    eval_runs: int,
) -> dict[str, Any]:
    vehicles_values: list[int] = []
    distance_values: list[float] = []
    scalar_values: list[float] = []
    runtime_values: list[float] = []
    objective_pairs: list[tuple[int, float]] = []

    for run_idx in range(eval_runs):
        seed = seed_start + run_idx
        solver = algo_cls(dataset=dataset, seed=seed, **params)

        start = time.perf_counter()
        best_route = solver.solve()
        elapsed_sec = time.perf_counter() - start

        vehicles, distance = best_route.cost_function()
        scalar_value = _scalarized_objective(vehicles, distance)

        vehicles_values.append(int(vehicles))
        distance_values.append(float(distance))
        scalar_values.append(float(scalar_value))
        runtime_values.append(float(elapsed_sec))
        objective_pairs.append((int(vehicles), float(distance)))

    best_pair = min(objective_pairs)

    return {
        "avg_vehicles": float(fmean(vehicles_values)),
        "avg_distance": float(fmean(distance_values)),
        "avg_scalar_objective": float(fmean(scalar_values)),
        "avg_runtime_sec": float(fmean(runtime_values)),
        "best_run_vehicles": int(best_pair[0]),
        "best_run_distance": float(best_pair[1]),
    }


def _suggest_sa_params(trial: optuna.Trial) -> dict[str, float | int]:
    initial_temp = trial.suggest_float("initial_temp", 100.0, 5_000.0, log=True)
    cooling_rate = trial.suggest_float("cooling_rate", 0.90, 0.995)
    min_temp_upper = min(250.0, max(1.0, initial_temp * 0.8))
    min_temp = trial.suggest_float("min_temp", 1.0, min_temp_upper, log=True)
    iterations_per_temp = trial.suggest_int("iterations_per_temp", 20, 3_000, log=True)

    return {
        "initial_temp": float(initial_temp),
        "cooling_rate": float(cooling_rate),
        "min_temp": float(min_temp),
        "iterations_per_temp": int(iterations_per_temp),
    }


def _suggest_ts_params(trial: optuna.Trial) -> dict[str, float | int]:
    max_iterations = trial.suggest_int("max_iterations", 50, 5_000, log=True)
    max_tabu_tenure = max(2, min(500, max_iterations // 2))
    tabu_tenure = trial.suggest_int("tabu_tenure", 2, max_tabu_tenure)
    neighbor_samples = trial.suggest_int("neighbor_samples", 5, 250, log=True)

    return {
        "max_iterations": int(max_iterations),
        "tabu_tenure": int(tabu_tenure),
        "neighbor_samples": int(neighbor_samples),
    }


def _suggest_ga_params(trial: optuna.Trial) -> dict[str, float | int]:
    population_size = trial.suggest_int("population_size", 16, 512, log=True)
    generations = trial.suggest_int("generations", 20, 4_000, log=True)
    crossover_rate = trial.suggest_float("crossover_rate", 0.5, 1.0)
    mutation_rate = trial.suggest_float("mutation_rate", 0.01, 0.5)

    return {
        "population_size": int(population_size),
        "generations": int(generations),
        "crossover_rate": float(crossover_rate),
        "mutation_rate": float(mutation_rate),
    }


def _optimize_method(
    method_name: str,
    dataset: Dataset,
    algo_cls: type[Metaheuristic],
    suggest_params_fn: Callable[[optuna.Trial], dict[str, float | int]],
    seed_start: int,
    eval_runs: int,
    n_trials: int,
    optuna_n_jobs: int,
) -> dict[str, Any]:
    stage_ratios = (0.25, 0.5, 1.0)
    resolved_optuna_n_jobs = max(1, int(optuna_n_jobs))
    max_estimated_work_units = int(MAX_ESTIMATED_WORK_UNITS.get(method_name, 0))
    stage1_runtime_limit_sec = float(STAGE1_MAX_RUNTIME_SEC.get(method_name, 0.0))
    runtime_growth_limit = float(STAGE_RUNTIME_GROWTH_LIMIT.get(method_name, 1.25))

    _status(
        f"Optuna tuning {method_name} | trials={n_trials}, "
        f"eval_runs={eval_runs}, seed_start={seed_start}, "
        f"pruning=Hyperband, stage_budgets={stage_ratios}, "
        f"optuna_jobs={resolved_optuna_n_jobs}, "
        f"max_est_work={max_estimated_work_units}, "
        f"stage1_max_runtime={stage1_runtime_limit_sec:.1f}s, "
        f"runtime_growth_limit=x{runtime_growth_limit:.2f}"
    )

    sampler = optuna.samplers.TPESampler(seed=seed_start)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=len(stage_ratios),
        reduction_factor=2,
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"{method_name}_tuning",
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params_fn(trial)
        final_metrics: dict[str, Any] | None = None
        previous_stage_objective: float | None = None
        previous_stage_runtime_sec: float | None = None
        estimated_work_units = _estimate_work_units(method_name=method_name, params=params)

        trial.set_user_attr("estimated_work_units", int(estimated_work_units))

        if max_estimated_work_units > 0 and estimated_work_units > max_estimated_work_units:
            raise optuna.TrialPruned(
                f"Estimated {method_name} effort too high "
                f"({estimated_work_units} > {max_estimated_work_units})"
            )

        if method_name == "SA":
            estimated_levels = _estimate_sa_temperature_levels(
                initial_temp=float(params["initial_temp"]),
                cooling_rate=float(params["cooling_rate"]),
                min_temp=float(params["min_temp"]),
            )

            trial.set_user_attr("sa_estimated_temp_levels", int(estimated_levels))
            trial.set_user_attr("sa_estimated_neighbor_evals", int(estimated_work_units))
        elif method_name == "TS":
            trial.set_user_attr("ts_estimated_neighbor_evals", int(estimated_work_units))
        elif method_name == "GA":
            trial.set_user_attr("ga_estimated_population_updates", int(estimated_work_units))

        for stage_idx, stage_ratio in enumerate(stage_ratios, start=1):
            stage_params = _scaled_params_by_budget(
                method_name=method_name,
                params=params,
                budget_ratio=stage_ratio,
            )

            stage_start = time.perf_counter()
            stage_metrics = _evaluate_solver_params(
                dataset=dataset,
                algo_cls=algo_cls,
                params=stage_params,
                seed_start=seed_start,
                eval_runs=eval_runs,
            )
            stage_runtime_sec = time.perf_counter() - stage_start
            stage_objective = float(stage_metrics["avg_scalar_objective"])

            trial.set_user_attr(f"stage_{stage_idx}_ratio", float(stage_ratio))
            trial.set_user_attr(f"stage_{stage_idx}_objective", float(stage_objective))
            trial.set_user_attr(f"stage_{stage_idx}_runtime_sec", float(stage_runtime_sec))

            if stage_idx == 1 and stage1_runtime_limit_sec > 0.0 and stage_runtime_sec > stage1_runtime_limit_sec:
                raise optuna.TrialPruned(
                    f"Stage 1 runtime too high for {method_name} "
                    f"({stage_runtime_sec:.2f}s > {stage1_runtime_limit_sec:.2f}s)"
                )

            trial.report(stage_objective, stage_idx)

            if (
                previous_stage_objective is not None
                and previous_stage_runtime_sec is not None
                and stage_idx < len(stage_ratios)
                and stage_objective >= previous_stage_objective
                and stage_runtime_sec > previous_stage_runtime_sec * runtime_growth_limit
            ):
                raise optuna.TrialPruned(
                    f"No objective improvement by stage {stage_idx}; "
                    f"runtime grew by more than x{runtime_growth_limit:.2f}"
                )

            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned by Hyperband at stage {stage_idx}")

            previous_stage_objective = stage_objective
            previous_stage_runtime_sec = stage_runtime_sec
            final_metrics = stage_metrics

        assert final_metrics is not None
        trial.set_user_attr("avg_vehicles", final_metrics["avg_vehicles"])
        trial.set_user_attr("avg_distance", final_metrics["avg_distance"])
        trial.set_user_attr("avg_scalar_objective", final_metrics["avg_scalar_objective"])
        trial.set_user_attr("avg_runtime_sec", final_metrics["avg_runtime_sec"])
        return float(final_metrics["avg_scalar_objective"])

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=resolved_optuna_n_jobs,
        show_progress_bar=(resolved_optuna_n_jobs == 1),
    )

    selected_trial = study.best_trial
    best_params = suggest_params_fn(optuna.trial.FixedTrial(selected_trial.params))
    best_metrics = _evaluate_solver_params(
        dataset=dataset,
        algo_cls=algo_cls,
        params=best_params,
        seed_start=seed_start,
        eval_runs=eval_runs,
    )

    _status(
        f"{method_name} done | selected_trial={selected_trial.number}, "
        f"selected_objective={float(selected_trial.value):.3f}, "
        f"selected_avg_k={float(selected_trial.user_attrs.get('avg_vehicles', float('nan'))):.3f}, "
        f"selected_avg_d={float(selected_trial.user_attrs.get('avg_distance', float('nan'))):.3f}, "
        f"pruned_trials={len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}, "
        f"best_params_run_time={best_metrics['avg_runtime_sec']:.3f}s"
    )

    return {
        "best_params": best_params,
        "selected_objective": {
            "scalar": float(selected_trial.value),
            "avg_vehicles": float(selected_trial.user_attrs.get("avg_vehicles", float("nan"))),
            "avg_distance": float(selected_trial.user_attrs.get("avg_distance", float("nan"))),
            "avg_scalar_objective": float(
                selected_trial.user_attrs.get("avg_scalar_objective", float("nan"))
            ),
        },
        "selected_trial_number": int(selected_trial.number),
        "n_trials": int(len(study.trials)),
        "best_params_run_time_sec": float(best_metrics["avg_runtime_sec"]),
        "best_metrics": best_metrics,
    }


def main(
    dataset_name: str | None = None,
    n_trials_per_method: int = 30,
    eval_runs: int = 1,
    optuna_n_jobs: int = 1,
) -> int:
    #optuna.logging.set_verbosity(optuna.logging.WARNING)

    resolved_dataset = dataset_name or cfg.DEFAULT_TUNING_DATASET
    resolved_trials = max(1, n_trials_per_method)
    resolved_eval_runs = max(1, eval_runs)
    resolved_optuna_jobs = max(1, int(optuna_n_jobs))
    seed_start = int(cfg.SEED_START)

    dataset_path = REPO_ROOT / "Datasets" / resolved_dataset

    dataset = Dataset(dataset_path)
    _status(f"Starting Optuna tune | dataset={resolved_dataset} | settings={SETTINGS_PATH}")

    methods: list[tuple[str, type[Metaheuristic], Callable[[optuna.Trial], dict[str, float | int]]]] = [
        ("SA", SimulatedAnnealing, _suggest_sa_params),
        ("TS", TabuSearch, _suggest_ts_params),
        ("GA", GeneticAlgorithm, _suggest_ga_params),
    ]

    results: dict[str, dict[str, Any]] = {}
    for method_name, algo_cls, suggest_fn in methods:
        results[method_name] = _optimize_method(
            method_name=method_name,
            dataset=dataset,
            algo_cls=algo_cls,
            suggest_params_fn=suggest_fn,
            seed_start=seed_start,
            eval_runs=resolved_eval_runs,
            n_trials=resolved_trials,
            optuna_n_jobs=resolved_optuna_jobs,
        )

    for method_name in ("SA", "TS", "GA"):
        method_time_sec = float(results[method_name]["best_params_run_time_sec"])
        _status(
            f"{method_name} final best-params run time: {method_time_sec:.3f}s"
        )

    sa_params = results["SA"]["best_params"]
    ts_params = results["TS"]["best_params"]
    ga_params = results["GA"]["best_params"]

    report = {
        "tuner": "optuna_single_objective",
        "objective_strategy": "vehicles_scaled_priority",
        "objective_formula": "vehicles * 1000000 + distance",
        "dataset": resolved_dataset,
        "seed_start": seed_start,
        "eval_runs": resolved_eval_runs,
        "n_trials_per_method": resolved_trials,
        "methods": {
            "SA": results["SA"],
            "TS": results["TS"],
            "GA": results["GA"],
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    updates: dict[str, Any] = {
        "SA_PARAMS": sa_params,
        "TS_PARAMS": ts_params,
        "GA_PARAMS": ga_params,
        "LAST_TUNING_REPORT": report,
    }

    _status("Writing tuned parameters into settings.py...")
    _replace_assignments(SETTINGS_PATH, updates)
    _status("settings.py updated successfully")
    return 0


if __name__ == "__main__":
    main()
    