from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
import random

from mesa import Agent, Model
from mesa.agent import AgentSet

from .Route import Route
from .Metaheuristics import GeneticAlgorithm, SimulatedAnnealing, TabuSearch, Metaheuristic

logger = logging.getLogger(__name__)


@dataclass
class ElitePool:
    """Shared memory of best solutions (blackboard)."""

    max_size: int
    items: List[Route]

    def add(self, route: Route) -> None:
        """Add a cloned route to the pool and keep only the best max_size."""
        self.items.append(route.clone())
        self.items.sort(key=lambda r: r.cost_function())
        if len(self.items) > self.max_size:
            del self.items[self.max_size :]

    def best(self) -> Optional[Route]:
        """Return the best route in the pool, or None if empty."""
        if not self.items:
            return None
        return self.items[0]

    def sample(self, rng: random.Random) -> Optional[Route]:
        """Return a random route from the pool using the provided RNG.""" # use best sol instead of random?
        if not self.items:
            return None
        return rng.choice(self.items)


class VRPOptimizationModel(Model):
    """Mesa model managing heuristic agents and a shared elite pool."""

    def __init__(
        self,
        dataset,
        pool_size: int = 10,
        seed: int | None = None,
        total_steps: int = 5,
        ga_params: Dict[str, Any] | None = None,
        sa_params: Dict[str, Any] | None = None,
        ts_params: Dict[str, Any] | None = None,
        seed_policy: str = "best",
        parallel_agents: bool = False,
    ) -> None:
        """Initialize the model, RNG, agent set, and shared elite pool."""
        super().__init__()
        self.dataset = dataset
        self.random = random.Random(seed)
        self.agent_set = AgentSet([], random=self.random)
        self.elite_pool = ElitePool(max_size=pool_size, items=[])

        self.total_steps = total_steps
        if seed_policy not in {"best", "random"}:
            raise ValueError("seed_policy must be 'best' or 'random'")
        self.seed_policy = seed_policy
        self.parallel_agents = parallel_agents

        ga_seed = None if seed is None else seed + 101
        sa_seed = None if seed is None else seed + 202
        ts_seed = None if seed is None else seed + 303

        self.ga_agent = HeuristicAgent(self, "GA", ga_seed, solver_params=ga_params)
        self.sa_agent = HeuristicAgent(self, "SA", sa_seed, solver_params=sa_params)
        self.ts_agent = HeuristicAgent(self, "TS", ts_seed, solver_params=ts_params)

        self.agent_set.add(self.ga_agent)
        self.agent_set.add(self.sa_agent)
        self.agent_set.add(self.ts_agent)

    def step(self) -> None:
        """Run one model tick by stepping all agents once."""
        if self.parallel_agents:
            self._step_parallel()
        else:
            self.agent_set.shuffle_do("step")
        if len(self.elite_pool.items) >= 2:
            self.run_path_relinking()

    def _step_parallel(self) -> None:
        """Run one solver from each agent concurrently and merge outcomes."""
        agents = list(self.agent_set)
        self.random.shuffle(agents)

        seeds = [self.get_seed() for _ in agents]
        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            futures = [
                executor.submit(agent.solve_once, seed_route)
                for agent, seed_route in zip(agents, seeds)
            ]
            results = [future.result() for future in futures]

        for agent, improved in zip(agents, results):
            if improved is None:
                continue
            k, d = improved.cost_function()
            logger.info(
                "Agent %s result -> vehicles: %s, distance: %.2f",
                agent.algo,
                k,
                d,
            )
            self.add_to_pool(improved)

    def add_to_pool(self, route: Route) -> None:
        """Add a route to the elite pool if it is valid."""
        if route is None:
            return
        if not route.is_feasible():
            logger.debug("Skipping infeasible route during pool update")
            return
        self.elite_pool.add(route)

    def get_seed(self) -> Optional[Route]:
        """Sample a seed route from the elite pool for warm starting."""
        if self.seed_policy == "best":
            return self.elite_pool.best()
        return self.elite_pool.sample(self.random)

    def run_path_relinking(self) -> None:
        """Perform path relinking between two elite solutions."""
        source = self.elite_pool.best()
        if source is None or len(self.elite_pool.items) < 2:
            logger.info("==> PR skipped: elite pool has fewer than 2 solutions")
            return

        source_perm = self._route_to_permutation(source)

        # pick the most diverse target by Hamming distance on permutations
        target = None
        max_dist = -1
        for candidate in self.elite_pool.items:
            if candidate is source:
                continue
            cand_perm = self._route_to_permutation(candidate)
            if len(cand_perm) != len(source_perm):
                continue
            dist = sum(a != b for a, b in zip(source_perm, cand_perm))
            if dist > max_dist:
                max_dist = dist
                target = candidate

        if target is None:
            logger.info("==> PR skipped: no valid target found")
            return

        decoder = self._get_decoder()
        if decoder is None:
            logger.info("==> PR skipped: decoder unavailable")
            return

        src_k, src_d = source.cost_function()
        tgt_k, tgt_d = target.cost_function()
        logger.info(
            "==> PR start: source (k=%s, d=%.2f) -> target (k=%s, d=%.2f), diversity=%s",
            src_k, src_d, tgt_k, tgt_d, max_dist
        )

        def relink(src: Route, tgt: Route) -> Route:
            src_perm = self._route_to_permutation(src)
            tgt_perm = self._route_to_permutation(tgt)
            current = list(src_perm)
            pos = {cust: idx for idx, cust in enumerate(current)}
            best_route = src
            best_k, best_d = src.cost_function()
            for i, desired in enumerate(tgt_perm):
                if current[i] == desired:
                    continue
                j = pos[desired]
                cust_i = current[i]
                cust_j = current[j]
                current[i], current[j] = cust_j, cust_i
                pos[cust_i] = j
                pos[cust_j] = i
                candidate = decoder._decode(current)
                cand_k, cand_d = candidate.cost_function()
                if (cand_k, cand_d) < (best_k, best_d):
                    best_route = candidate
                    best_k, best_d = cand_k, cand_d
            return best_route

        best_route = relink(source, target)
        best_route_rev = relink(target, source)

        fwd_k, fwd_d = best_route.cost_function()
        rev_k, rev_d = best_route_rev.cost_function()
        logger.info(
            ' '*4 + f"PR candidates -> forward (k={fwd_k}, d={fwd_d:.2f}), reverse (k={rev_k}, d={rev_d:.2f})"
        )

        if best_route_rev.cost_function() < best_route.cost_function():
            best_route = best_route_rev

        if best_route is not source:
            k, d = best_route.cost_function()
            logger.info(' '*4 + f"PR improved elite pool -> (k={k}, d={d:.2f})\n")
            self.add_to_pool(best_route)
        else:
            logger.info(' '*4 + "PR finished: no improvement over source\n")

    def _route_to_permutation(self, route: Route) -> List[int]:
        """Convert a route into a customer permutation, excluding depots."""
        perm: List[int] = []
        for sub in route.subroutes:
            for cust in sub:
                if cust != 0:
                    perm.append(cust)
        return perm

    def _get_decoder(self) -> Optional[Metaheuristic]:
        """Return a solver instance to reuse its decoder, if any agent exists."""
        for agent in self.agent_set:
            return agent.solver
        return None


class HeuristicAgent(Agent):
    """Agent that runs a specific metaheuristic and updates the elite pool."""

    def __init__(
        self,
        model: VRPOptimizationModel,
        algo: str,
        seed: int | None,
        solver_params: Dict[str, Any] | None = None,
    ) -> None:
        """Create an agent bound to a specific metaheuristic solver."""
        super().__init__(model)
        self.algo = algo
        params = dict(solver_params or {})
        params.pop("seed", None)
        if seed is not None:
            params["seed"] = seed

        if algo == "GA":
            self.solver = GeneticAlgorithm(model.dataset, **params)
        elif algo == "SA":
            self.solver = SimulatedAnnealing(model.dataset, **params)
        elif algo == "TS":
            self.solver = TabuSearch(model.dataset, **params)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

    def step(self) -> None:
        """Run the solver once and push any improvement to the elite pool."""
        self.model: VRPOptimizationModel  # type hint for self.model

        seed_route = self.model.get_seed()
        improved = self.solve_once(seed_route)
        k, d = improved.cost_function()

        logger.info(
            "Agent %s result -> vehicles: %s, distance: %.2f",
            self.algo,
            k,
            d,
        )

        self.model.add_to_pool(improved)

    def solve_once(self, seed_route: Optional[Route]) -> Route:
        """Run solver once from an optional warm-start seed route."""
        seed_perm = self._route_to_permutation(seed_route) if seed_route else None
        return self.solver.solve(initial_permutation=seed_perm)

    def _route_to_permutation(self, route: Route) -> List[int]:
        """Convert a route into a customer permutation, excluding depots."""
        perm: List[int] = []
        for sub in route.subroutes:
            for cust in sub:
                if cust != 0:
                    perm.append(cust)
        return perm
