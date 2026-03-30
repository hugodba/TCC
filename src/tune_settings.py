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


def _clamp_int(value: float, low: int, high: int) -> int:
    return int(np.clip(round(value), low, high))


def _sa_temperature_levels(initial_temp: float, cooling_rate: float, min_temp: float) -> int:
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


def _measure_avg_step_runtime(model: Any, n_steps: int = 2) -> float:
    # Keep the same timing metric used in notebook experiments.
    start = time.process_time()
    for _ in range(n_steps):
        model.step()
    elapsed = time.process_time() - start
    return elapsed / max(1, n_steps)


def _scaled_mas_params(
    mas_sa_params: dict[str, float | int],
    mas_ts_params: dict[str, float | int],
    mas_ga_params: dict[str, float | int],
    scale: float,
) -> tuple[dict[str, float | int], dict[str, float | int], dict[str, float | int]]:
    scaled_sa = dict(mas_sa_params)
    scaled_ts = dict(mas_ts_params)
    scaled_ga = dict(mas_ga_params)

    scaled_sa["iterations_per_temp"] = _clamp_int(
        int(mas_sa_params["iterations_per_temp"]) * scale,
        1,
        300,
    )
    scaled_ts["max_iterations"] = _clamp_int(
        int(mas_ts_params["max_iterations"]) * scale,
        2,
        1600,
    )
    scaled_ts["tabu_tenure"] = _clamp_int(
        int(scaled_ts["max_iterations"]) * 0.14,
        2,
        60,
    )
    scaled_ga["generations"] = _clamp_int(
        int(mas_ga_params["generations"]) * scale,
        2,
        1200,
    )

    return scaled_sa, scaled_ts, scaled_ga


def _scale_sa_params(
    params: dict[str, float | int],
    scale: float,
) -> dict[str, float | int]:
    scaled = dict(params)
    scaled["iterations_per_temp"] = _clamp_int(
        int(params["iterations_per_temp"]) * scale,
        5,
        1200,
    )
    return scaled


def _scale_ts_params(
    params: dict[str, float | int],
    scale: float,
) -> dict[str, float | int]:
    scaled = dict(params)
    scaled["max_iterations"] = _clamp_int(
        int(params["max_iterations"]) * scale,
        20,
        5000,
    )
    scaled["tabu_tenure"] = _clamp_int(
        int(scaled["max_iterations"]) * 0.1,
        7,
        120,
    )
    return scaled


def _scale_ga_params(
    params: dict[str, float | int],
    scale: float,
) -> dict[str, float | int]:
    scaled = dict(params)
    scaled["generations"] = _clamp_int(
        int(params["generations"]) * scale,
        20,
        4000,
    )
    return scaled


def _feedback_tune_isolated(
    name: str,
    dataset: Dataset,
    algo_cls: type[Metaheuristic],
    params: dict[str, float | int],
    target_sec: float,
    tolerance_sec: float,
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

        if abs(error_sec) <= tolerance_sec:
            if iter_idx == 1:
                _status(f"{name} already within ±{tolerance_sec:.1f}s; skipping further adjustments")
            else:
                _status(f"{name} converged within ±{tolerance_sec:.1f}s")
            break

        if measured_sec <= 1e-9:
            _status(f"{name} measured time is too small to scale safely; stopping")
            break

        raw_scale = target_sec / measured_sec
        scale = float(np.clip(raw_scale, 0.5, 2.0))
        candidate = scale_fn(current, scale)
        if candidate == current:
            _status(f"{name} parameters reached bounds; cannot adjust further")
            break

        _status(f"{name} adjust -> scale={scale:.3f} (raw={raw_scale:.3f})")
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
    dataset_name: str | None = None,
    total_seconds: float | None = None,
    probe_steps: int = 1,
    isolated_tolerance_sec: float = 5.0,
    isolated_max_iters: int = 4,
) -> int:
    _status("Timing metric: CPU process time (time.process_time)")

    dataset_name = dataset_name or cfg.DEFAULT_TUNING_DATASET
    dataset_path = REPO_ROOT / "Datasets" / dataset_name
    _status(f"Starting tune run | dataset={dataset_name} | settings={SETTINGS_PATH}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    total_budget_sec = float(total_seconds) if total_seconds is not None else float(cfg.TOTAL_MAIN_BUDGET_SEC)
    target_run_sec = max(
        5.0,
        (total_budget_sec / (int(cfg.METHOD_COUNT) * int(cfg.N_RUNS))) * float(cfg.RUNTIME_SAFETY_MARGIN),
    )
    isolated_tolerance_sec = max(0.1, float(isolated_tolerance_sec))
    isolated_max_iters = max(1, int(isolated_max_iters))
    isolated_seed = int(cfg.SEED_START)
    _status(
        "Budget and controls | "
        f"total={total_budget_sec:.1f}s ({total_budget_sec / 60.0:.1f} min), "
        f"runs={int(cfg.N_RUNS)}, methods={int(cfg.METHOD_COUNT)}, "
        f"target/method-run={target_run_sec:.2f}s, safety_margin={float(cfg.RUNTIME_SAFETY_MARGIN):.3f}"
    )
    _status(
        "Isolated feedback controls | "
        f"tolerance=±{isolated_tolerance_sec:.1f}s, "
        f"max_iters={isolated_max_iters}, seed={isolated_seed}"
    )

    dataset = Dataset(dataset_path)
    _status(f"Loaded dataset: {dataset.name}")

    # Probe each isolated method for seconds per work unit.
    probe_sa = {
        "initial_temp": 200.0,
        "cooling_rate": 0.95,
        "min_temp": 20.0,
        "iterations_per_temp": 6,
    }
    probe_ts = {
        "max_iterations": 30,
        "tabu_tenure": 10,
        "neighbor_samples": 6,
    }
    probe_ga = {
        "population_size": 10,
        "generations": 15,
        "crossover_rate": 0.8,
        "mutation_rate": 0.1,
    }

    _status("Probing isolated solvers (SA/TS/GA) for sec-per-work-unit rates...")

    sa_probe_sec = _measure_solver_runtime(dataset, SimulatedAnnealing, probe_sa, seed=int(cfg.SEED_START) + 11)
    ts_probe_sec = _measure_solver_runtime(dataset, TabuSearch, probe_ts, seed=int(cfg.SEED_START) + 22)
    ga_probe_sec = _measure_solver_runtime(dataset, GeneticAlgorithm, probe_ga, seed=int(cfg.SEED_START) + 33)
    _status(
        f"Probe elapsed | SA={sa_probe_sec:.3f}s, TS={ts_probe_sec:.3f}s, GA={ga_probe_sec:.3f}s"
    )

    sa_rate = _safe_rate(sa_probe_sec, _sa_work_units(probe_sa))
    ts_rate = _safe_rate(ts_probe_sec, _ts_work_units(probe_ts))
    ga_rate = _safe_rate(ga_probe_sec, _ga_work_units(probe_ga))
    _status(
        "Probe rates sec/unit | "
        f"SA={sa_rate:.8f}, TS={ts_rate:.8f}, GA={ga_rate:.8f}"
    )

    # Tune isolated SA / TS / GA.
    _status("Tuning isolated SA/TS/GA parameters...")
    sa_base = {
        "initial_temp": float(cfg.SA_PARAMS["initial_temp"]),
        "cooling_rate": float(cfg.SA_PARAMS["cooling_rate"]),
        "min_temp": float(cfg.SA_PARAMS["min_temp"]),
    }
    sa_levels = _sa_temperature_levels(
        sa_base["initial_temp"],
        sa_base["cooling_rate"],
        sa_base["min_temp"],
    )
    sa_work_target = target_run_sec / sa_rate
    sa_iterations = _clamp_int(sa_work_target / sa_levels, 5, 900)
    sa_params = {**sa_base, "iterations_per_temp": sa_iterations}

    ts_neighbor_samples = int(cfg.TS_PARAMS["neighbor_samples"])
    ts_work_target = target_run_sec / ts_rate
    ts_max_iterations = _clamp_int(ts_work_target / ts_neighbor_samples, 20, 4000)
    ts_tabu_tenure = _clamp_int(ts_max_iterations * 0.1, 7, 80)
    ts_params = {
        "max_iterations": ts_max_iterations,
        "tabu_tenure": ts_tabu_tenure,
        "neighbor_samples": ts_neighbor_samples,
    }

    ga_population_size = int(cfg.GA_PARAMS["population_size"])
    ga_work_target = target_run_sec / ga_rate
    ga_generations = _clamp_int(ga_work_target / ga_population_size, 20, 3000)
    ga_params = {
        "population_size": ga_population_size,
        "generations": ga_generations,
        "crossover_rate": float(cfg.GA_PARAMS["crossover_rate"]),
        "mutation_rate": float(cfg.GA_PARAMS["mutation_rate"]),
    }
    _status(f"SA initial params from probe-rates: {sa_params}")
    _status(f"TS initial params from probe-rates: {ts_params}")
    _status(f"GA initial params from probe-rates: {ga_params}")

    _status("Running isolated one-run feedback tuning (same measurement style as main)...")
    sa_params, sa_measured_sec = _feedback_tune_isolated(
        "SA",
        dataset,
        SimulatedAnnealing,
        sa_params,
        target_run_sec,
        isolated_tolerance_sec,
        seed=isolated_seed,
        max_iters=isolated_max_iters,
        scale_fn=_scale_sa_params,
    )
    ts_params, ts_measured_sec = _feedback_tune_isolated(
        "TS",
        dataset,
        TabuSearch,
        ts_params,
        target_run_sec,
        isolated_tolerance_sec,
        seed=isolated_seed,
        max_iters=isolated_max_iters,
        scale_fn=_scale_ts_params,
    )
    ga_params, ga_measured_sec = _feedback_tune_isolated(
        "GA",
        dataset,
        GeneticAlgorithm,
        ga_params,
        target_run_sec,
        isolated_tolerance_sec,
        seed=isolated_seed,
        max_iters=isolated_max_iters,
        scale_fn=_scale_ga_params,
    )
    _status(f"SA final params after feedback: {sa_params}")
    _status(f"TS final params after feedback: {ts_params}")
    _status(f"GA final params after feedback: {ga_params}")

    # Tune only MAS inner metaheuristic params, keeping MAS controls and MAS_RL_PARAMS fixed.
    target_mas_step_sec = target_run_sec / max(1, int(cfg.MAS_N_STEPS))
    target_mas_rl_step_sec = target_run_sec / max(1, int(cfg.MAS_RL_N_STEPS))
    target_step_sec = min(target_mas_step_sec, target_mas_rl_step_sec)

    if bool(cfg.PARALLEL_AGENTS):
        inner_solver_target_sec = target_step_sec * 0.70
    else:
        inner_solver_target_sec = target_step_sec / 3.3
    _status(
        "Tuning MAS inner params with fixed MAS controls/RL params | "
        f"target_step_mas={target_mas_step_sec:.3f}s, "
        f"target_step_mas_rl={target_mas_rl_step_sec:.3f}s, "
        f"inner_solver_target={inner_solver_target_sec:.3f}s, "
        f"parallel_agents={bool(cfg.PARALLEL_AGENTS)}"
    )

    mas_sa_base = {
        "initial_temp": float(cfg.MAS_SA_PARAMS["initial_temp"]),
        "cooling_rate": float(cfg.MAS_SA_PARAMS["cooling_rate"]),
        "min_temp": float(cfg.MAS_SA_PARAMS["min_temp"]),
    }
    mas_sa_levels = _sa_temperature_levels(
        mas_sa_base["initial_temp"],
        mas_sa_base["cooling_rate"],
        mas_sa_base["min_temp"],
    )
    mas_sa_iters = _clamp_int((inner_solver_target_sec / sa_rate) / mas_sa_levels, 1, 240)
    mas_sa_params = {**mas_sa_base, "iterations_per_temp": mas_sa_iters}

    mas_ts_neighbor_samples = int(cfg.MAS_TS_PARAMS["neighbor_samples"])
    mas_ts_max_iters = _clamp_int(
        (inner_solver_target_sec / ts_rate) / mas_ts_neighbor_samples,
        2,
        1200,
    )
    mas_ts_params = {
        "max_iterations": mas_ts_max_iters,
        "tabu_tenure": _clamp_int(mas_ts_max_iters * 0.14, 2, 40),
        "neighbor_samples": mas_ts_neighbor_samples,
    }

    mas_ga_population = int(cfg.MAS_GA_PARAMS["population_size"])
    mas_ga_generations = _clamp_int(
        (inner_solver_target_sec / ga_rate) / mas_ga_population,
        2,
        900,
    )
    mas_ga_params = {
        "population_size": mas_ga_population,
        "generations": mas_ga_generations,
        "crossover_rate": float(cfg.MAS_GA_PARAMS["crossover_rate"]),
        "mutation_rate": float(cfg.MAS_GA_PARAMS["mutation_rate"]),
    }

    probe_steps = max(1, int(probe_steps))

    def _build_mas() -> VRPOptimizationModel:
        return VRPOptimizationModel(
            dataset,
            pool_size=int(cfg.MAS_POOL_SIZE),
            seed=int(cfg.SEED_START) + 1001,
            total_steps=probe_steps,
            ga_params=mas_ga_params,
            sa_params=mas_sa_params,
            ts_params=mas_ts_params,
            seed_policy=str(cfg.MAS_SEED_POLICY),
            parallel_agents=bool(cfg.PARALLEL_AGENTS),
        )

    def _build_mas_rl() -> VRPOptimizationModelRL:
        return VRPOptimizationModelRL(
            dataset,
            pool_size=int(cfg.MAS_POOL_SIZE),
            seed=int(cfg.SEED_START) + 2001,
            total_steps=probe_steps,
            ga_params=mas_ga_params,
            sa_params=mas_sa_params,
            ts_params=mas_ts_params,
            seed_policy=str(cfg.MAS_SEED_POLICY),
            parallel_agents=bool(cfg.PARALLEL_AGENTS),
            **dict(cfg.MAS_RL_PARAMS),
        )

    target_combo = (target_mas_step_sec + target_mas_rl_step_sec) / 2.0
    mas_step_sec = 0.0
    mas_rl_step_sec = 0.0

    for iter_idx in range(1, 6):
        mas_step_sec = _measure_avg_step_runtime(_build_mas(), n_steps=probe_steps)
        mas_rl_step_sec = _measure_avg_step_runtime(_build_mas_rl(), n_steps=probe_steps)

        observed_combo = (mas_step_sec + mas_rl_step_sec) / 2.0
        if observed_combo <= 0.0:
            _status(
                f"MAS calibration iter {iter_idx}: observed combo <= 0, stopping early"
            )
            break

        ratio = target_combo / observed_combo
        _status(
            "MAS calibration iter "
            f"{iter_idx} | step_mas={mas_step_sec:.3f}s, step_mas_rl={mas_rl_step_sec:.3f}s, "
            f"observed_combo={observed_combo:.3f}s, target_combo={target_combo:.3f}s, ratio={ratio:.3f}"
        )
        if 0.94 <= ratio <= 1.06:
            _status(f"MAS calibration iter {iter_idx}: converged within tolerance")
            break

        candidate = _scaled_mas_params(
            mas_sa_params,
            mas_ts_params,
            mas_ga_params,
            float(np.clip(ratio, 0.35, 1.65)),
        )
        if candidate == (mas_sa_params, mas_ts_params, mas_ga_params):
            _status(
                f"MAS calibration iter {iter_idx}: params unchanged at bounds, stopping"
            )
            break

        mas_sa_params, mas_ts_params, mas_ga_params = candidate
        _status(
            "MAS inner params updated | "
            f"SA iters/temp={int(mas_sa_params['iterations_per_temp'])}, "
            f"TS max_iter={int(mas_ts_params['max_iterations'])}, "
            f"TS tenure={int(mas_ts_params['tabu_tenure'])}, "
            f"GA generations={int(mas_ga_params['generations'])}"
        )

    # Final probe with final params used for saving.
    mas_step_sec = _measure_avg_step_runtime(_build_mas(), n_steps=probe_steps)
    mas_rl_step_sec = _measure_avg_step_runtime(_build_mas_rl(), n_steps=probe_steps)
    _status(
        f"Final MAS probe | step_mas={mas_step_sec:.3f}s, step_mas_rl={mas_rl_step_sec:.3f}s"
    )

    est_sa = float(sa_measured_sec)
    est_ts = float(ts_measured_sec)
    est_ga = float(ga_measured_sec)
    est_mas = mas_step_sec * int(cfg.MAS_N_STEPS)
    est_mas_rl = mas_rl_step_sec * int(cfg.MAS_RL_N_STEPS)
    est_total = int(cfg.N_RUNS) * (est_sa + est_ts + est_ga + est_mas + est_mas_rl)

    report = {
        "dataset": dataset_name,
        "target_method_run_sec": float(target_run_sec),
        "estimated_method_run_sec": {
            "SA": float(est_sa),
            "TS": float(est_ts),
            "GA": float(est_ga),
            "MAS": float(est_mas),
            "MAS_RL": float(est_mas_rl),
        },
        "estimated_total_main_runtime_sec": float(est_total),
        "estimated_total_main_runtime_min": float(est_total / 60.0),
        "target_total_main_runtime_sec": float(total_budget_sec),
        "target_total_main_runtime_min": float(total_budget_sec / 60.0),
        "probe_rates_sec_per_unit": {
            "SA": float(sa_rate),
            "TS": float(ts_rate),
            "GA": float(ga_rate),
        },
        "measured_isolated_run_sec": {
            "SA": float(sa_measured_sec),
            "TS": float(ts_measured_sec),
            "GA": float(ga_measured_sec),
        },
        "fixed_controls": {
            "MAS_POOL_SIZE": int(cfg.MAS_POOL_SIZE),
            "MAS_SEED_POLICY": str(cfg.MAS_SEED_POLICY),
            "MAS_N_STEPS": int(cfg.MAS_N_STEPS),
            "MAS_RL_N_STEPS": int(cfg.MAS_RL_N_STEPS),
            "MAS_RL_PARAMS": dict(cfg.MAS_RL_PARAMS),
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
    if total_seconds is not None:
        updates["TOTAL_MAIN_BUDGET_SEC"] = int(round(total_budget_sec))

    _status("Writing tuned parameters into settings.py...")
    _replace_assignments(SETTINGS_PATH, updates)
    _status("settings.py updated successfully")

    print("Updated settings.py with tuned parameters.")
    print(f"Dataset: {dataset_name}")
    print(f"Target per method run: {target_run_sec:.2f}s")
    print(
        "Measured one-run (main style) | "
        f"SA={sa_measured_sec:.1f}s, TS={ts_measured_sec:.1f}s, GA={ga_measured_sec:.1f}s"
    )
    print(
        "Estimated per-run | "
        f"SA={est_sa:.1f}s, TS={est_ts:.1f}s, GA={est_ga:.1f}s, "
        f"MAS={est_mas:.1f}s, MAS-RL={est_mas_rl:.1f}s"
    )
    print(
        f"Estimated total for N_RUNS={int(cfg.N_RUNS)}: {est_total / 60.0:.1f} min "
        f"(target={total_budget_sec / 60.0:.1f} min)"
    )
    return 0


if __name__ == "__main__":
    main()
    
