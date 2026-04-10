from __future__ import annotations

import ast
import pprint
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
SETTINGS_PATH = SRC_DIR / "project" / "settings.py"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project import Dataset
from project import settings as cfg
from project.Metaheuristics import GeneticAlgorithm, Metaheuristic, SimulatedAnnealing, TabuSearch
from project.mesa_model import VRPOptimizationModel
from project.mesa_model_rl import VRPOptimizationModelRL


def _status(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[tune_settings {stamp}] {message}", flush=True)

def _sa_temperature_levels(initial_temp: float, cooling_rate: float, min_temp: float) -> int:
    '''
    Calculate the number of temperature levels in SA given the cooling schedule.
    This basically estimates how many iterations the SA will run for, since each level runs a fixed number of iterations.
     - If the initial temperature is already at or below the minimum, or if the cooling rate is invalid, we consider it as 1 level (no cooling).
     - Otherwise, we calculate the number of levels until the temperature drops below the minimum using the formula:
       levels = ceil(log(min_temp / initial_temp) / log(cooling_rate))
    '''
    if initial_temp <= min_temp or not (0.0 < cooling_rate < 1.0):
        return 1
    levels = int(np.ceil(np.log(min_temp / initial_temp) / np.log(cooling_rate)))
    return max(1, levels)


def _sa_work_units(params: dict[str, float | int]) -> float:
    levels = _sa_temperature_levels(
        float(params["initial_temp"]),
        float(params["cooling_rate"]),
        float(params["min_temp"]),
    )
    return float(levels * int(params["iterations_per_temp"]))


def _ts_work_units(params: dict[str, float | int]) -> float:
    return float(int(params["max_iterations"]) * int(params["neighbor_samples"]))


def _ga_work_units(params: dict[str, float | int]) -> float:
    return float(int(params["population_size"]) * int(params["generations"]))


def _measure_solver_runtime(
    dataset: Dataset,
    algo_cls: type[Metaheuristic],
    params: dict[str, float | int],
    seed: int,
) -> float:
    solver = algo_cls(dataset=dataset, seed=seed, **params)
    # Use CPU time so tuner and notebook runner measure runtime consistently.
    start = time.process_time()
    _ = solver.solve()
    return time.process_time() - start


def _safe_rate(elapsed_sec: float, work_units: float) -> float:
    return max(1e-9, elapsed_sec / max(1.0, work_units))


def _scale_sa_params(
    params: dict[str, float | int],
    scale: float,
) -> dict[str, float | int]:
    scaled = dict(params)
    scaled["iterations_per_temp"] = round(params["iterations_per_temp"] * scale)
    return scaled

def _scale_ts_params(
    params: dict[str, float | int],
    scale: float,
) -> dict[str, float | int]:
    scaled = dict(params)
    scaled["max_iterations"] = round(params["max_iterations"] * scale)
    scaled["tabu_tenure"] = round(scaled["max_iterations"] * 0.14)
    return scaled

def _scale_ga_params(
    params: dict[str, float | int],
    scale: float,
) -> dict[str, float | int]:
    scaled = dict(params)
    scaled["generations"] = round(params["generations"] * scale)
    return scaled


def _feedback_tune_isolated(
    name: str,
    dataset: Dataset,
    algo_cls: type[Metaheuristic],
    params: dict[str, float | int],
    target_sec: float,
    tolerance: float,
    seed: int,
    max_iters: int,
    scale_fn: Callable[[dict[str, float | int], float], dict[str, float | int]],
) -> tuple[dict[str, float | int], float]:
    """Tune one isolated method using measured one-run feedback."""
    current = dict(params)
    measured_sec = float("nan")

    for iter_idx in range(1, max_iters + 1):
        measured_sec = _measure_solver_runtime(dataset, algo_cls, current, seed=seed)
        error_sec = measured_sec - target_sec

        _status(
            f"{name} feedback iter {iter_idx} | "
            f"measured={measured_sec:.3f}s, target={target_sec:.3f}s, "
            f"error={error_sec:+.3f}s"
        )

        over_target_ratio = abs(error_sec) / target_sec
        if 0.0 <= over_target_ratio <= tolerance:
            if iter_idx == 1:
                _status(
                    f"{name} already within +{tolerance * 100.0:.1f}% over target; "
                    "skipping further adjustments"
                )
            else:
                _status(f"{name} converged within +{tolerance * 100.0:.1f}% over target")
            break

        scale = target_sec / measured_sec
        candidate = scale_fn(current, scale)

        if candidate == current:
            _status(f"{name} parameters reached bounds; cannot adjust further")
            break

        _status(f"{name} adjust -> scale={scale:.3f}")
        current = candidate

    return current, measured_sec

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


def main(
    tolerance: float = 0.03,
    max_iters: int = 4,
) -> int:
    _status("Timing metric: CPU process time (time.process_time)")

    dataset_name = cfg.DEFAULT_TUNING_DATASET
    dataset_path = REPO_ROOT / "Datasets" / dataset_name
    dataset = Dataset(dataset_path)
    _status(f"Starting tune run | dataset={dataset_name} | settings={SETTINGS_PATH}")

    instance_count = cfg.INSTANCE_COUNT
    total_budget_sec = cfg.TOTAL_MAIN_BUDGET_SEC
    target_method_runs = cfg.TARGET_METHOD_RUNS
    target_method_run_sec = cfg.TARGET_METHOD_RUN_SEC
    _status(
        "Budget and controls | "
        f"total={total_budget_sec:.1f}s ({total_budget_sec / 60.0:.1f} min), "
        f"runs={int(cfg.N_RUNS)}, methods={int(cfg.METHOD_COUNT)}, "
        f"instances={instance_count}, method-runs={target_method_runs}, "
        f"target/method-run={target_method_run_sec:.2f}s"
    )

    tolerance = tolerance
    max_iters = max_iters
    seed = cfg.SEED_START
    _status(
        "Feedback controls | "
        f"tolerance=+{tolerance * 100.0:.1f}% over target, "
        f"max_iters={max_iters}, seed={seed}"
    )

    _status("Tuning isolated SA/TS/GA parameters...")

    sa_params = cfg.SA_PARAMS
    ts_params = cfg.TS_PARAMS
    ga_params = cfg.GA_PARAMS

    _status(f"SA initial params from probe-rates: {sa_params}")
    _status(f"TS initial params from probe-rates: {ts_params}")
    _status(f"GA initial params from probe-rates: {ga_params}")

    _status("Running isolated feedback tuning ...")
    sa_params, sa_measured_sec = _feedback_tune_isolated(
        "SA",
        dataset,
        SimulatedAnnealing,
        sa_params,
        target_method_run_sec,
        tolerance,
        seed=cfg.SEED_START,
        max_iters=max_iters,
        scale_fn=_scale_sa_params,
    )
    ts_params, ts_measured_sec = _feedback_tune_isolated(
        "TS",
        dataset,
        TabuSearch,
        ts_params,
        target_method_run_sec,
        tolerance,
        seed=cfg.SEED_START,
        max_iters=max_iters,
        scale_fn=_scale_ts_params,
    )
    ga_params, ga_measured_sec = _feedback_tune_isolated(
        "GA",
        dataset,
        GeneticAlgorithm,
        ga_params,
        target_method_run_sec,
        tolerance,
        seed=cfg.SEED_START,
        max_iters=max_iters,
        scale_fn=_scale_ga_params,
    )
    _status(f"SA final params after feedback: {sa_params}")
    _status(f"TS final params after feedback: {ts_params}")
    _status(f"GA final params after feedback: {ga_params}")


    _status("Tunning MAS family parameters with feedback from inner solvers...")
    # Tune MAS/MAS_RL runtime using inner params only (step controls are fixed).
    mas_steps = cfg.MAS_N_STEPS

    target_mas_run_sec = target_method_run_sec
    target_mas_step_sec = target_mas_run_sec / mas_steps

    # With CPU-time measurement, parallel SA/TS/GA costs are additive.
    parallel_method_runs = cfg.MAS_RL_PARAMS['runs_per_step']
    inner_solver_target_sec = target_mas_step_sec / parallel_method_runs
    _status(
        "Running MAS and MAS_RL feedback tunning | "
        f"target_run_mas={target_mas_run_sec:.3f}s, "
        f"fixed_steps_mas={mas_steps}, "
        f"target_step_mas={target_mas_step_sec:.3f}s, "
        f"parallel_method_runs={parallel_method_runs}, "
        f"inner_solver_target={inner_solver_target_sec:.3f}s, "
        f"parallel_agents={cfg.PARALLEL_AGENTS}"
    )

    mas_sa_params = cfg.MAS_SA_PARAMS
    mas_ts_params = cfg.MAS_TS_PARAMS
    mas_ga_params = cfg.MAS_GA_PARAMS

    _status(f"MAS family initial params from probe-rates: SA={mas_sa_params}, TS={mas_ts_params}, GA={mas_ga_params}")

    _status("Running feedback tuning for MAS agents...")
    mas_sa_params, mas_sa_measured_sec = _feedback_tune_isolated(
        "MAS SA",
        dataset,
        SimulatedAnnealing,
        mas_sa_params,
        inner_solver_target_sec,
        tolerance,
        seed=cfg.SEED_START,
        max_iters=max_iters,
        scale_fn=_scale_sa_params,
    )
    mas_ts_params, mas_ts_measured_sec = _feedback_tune_isolated(
        "MAS TS",
        dataset,
        TabuSearch,
        mas_ts_params,
        inner_solver_target_sec,
        tolerance,
        seed=cfg.SEED_START,
        max_iters=max_iters,
        scale_fn=_scale_ts_params,
    )
    mas_ga_params, mas_ga_measured_sec = _feedback_tune_isolated(
        "MAS GA",
        dataset,
        GeneticAlgorithm,
        mas_ga_params,
        inner_solver_target_sec,
        tolerance,
        seed=cfg.SEED_START,
        max_iters=max_iters,
        scale_fn=_scale_ga_params,
    )
    _status(f"MAS SA final params after feedback: {mas_sa_params}")
    _status(f"MAS TS final params after feedback: {mas_ts_params}")
    _status(f"MAS GA final params after feedback: {mas_ga_params}")

    est_sa = sa_measured_sec
    est_ts = ts_measured_sec
    est_ga = ga_measured_sec
    est_mas = (mas_sa_measured_sec + mas_ts_measured_sec + mas_ga_measured_sec) * cfg.MAS_N_STEPS
    est_total = (
        cfg.N_RUNS
        * instance_count
        * (est_sa + est_ts + est_ga + est_mas + est_mas)
    )

    report = {
        "dataset": dataset_name,
        "method_runs": cfg.N_RUNS,
        "total_main_budegt_sec": cfg.TOTAL_MAIN_BUDGET_SEC,
        "mas_n_steps": cfg.MAS_N_STEPS,

        "method_count": cfg.METHOD_COUNT,
        "instance_count": cfg.INSTANCE_COUNT,
        "target_method_runs": cfg.TARGET_METHOD_RUNS,
        "target_method_run_sec": cfg.TARGET_METHOD_RUN_SEC,

        "estimated_method_run_sec": {
            "SA": float(est_sa),
            "TS": float(est_ts),
            "GA": float(est_ga),
            "MAS": float(est_mas),
            "MAS_RL": float(est_mas),
        },
        "estimated_total_main_runtime_sec": float(est_total),
        "estimated_total_main_runtime_min": float(est_total / 60.0),

        "fixed_controls": {
            "MAS_POOL_SIZE": cfg.MAS_POOL_SIZE,
            "MAS_SEED_POLICY": cfg.MAS_SEED_POLICY,
            "MAS_RL_PARAMS": cfg.MAS_RL_PARAMS,
        },
    }

    updates: dict[str, Any] = {
        "SA_PARAMS": sa_params,
        "TS_PARAMS": ts_params,
        "GA_PARAMS": ga_params,
        "MAS_SA_PARAMS": mas_sa_params,
        "MAS_TS_PARAMS": mas_ts_params,
        "MAS_GA_PARAMS": mas_ga_params,
        "LAST_TUNING_REPORT": report,
    }

    _status("Writing tuned parameters into settings.py...")
    _replace_assignments(SETTINGS_PATH, updates)
    _status("settings.py updated successfully")

if __name__ == "__main__":
    main()
    

    
