from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import random

from mesa import Agent, Model
from mesa.time import RandomActivation

from .Route import Route
from .Metaheuristics import GeneticAlgorithm, SimulatedAnnealing, TabuSearch


@dataclass
class ElitePool:
    """Shared memory of best solutions (blackboard)."""

    max_size: int
    items: List[Route]

    def add(self, route: Route) -> None:
        self.items.append(route.clone())
        self.items.sort(key=lambda r: r.cost_function())
        if len(self.items) > self.max_size:
            del self.items[self.max_size :]

    def best(self) -> Optional[Route]:
        if not self.items:
            return None
        return self.items[0]

    def sample(self, rng: random.Random) -> Optional[Route]:
        if not self.items:
            return None
        return rng.choice(self.items)


class VRPOptimizationModel(Model):
    """Mesa model managing heuristic agents and a shared elite pool."""

    def __init__(self, dataset, pool_size: int = 10, seed: int | None = None) -> None:
        super().__init__()
        self.dataset = dataset
        self.random = random.Random(seed)
        self.schedule = RandomActivation(self)
        self.elite_pool = ElitePool(max_size=pool_size, items=[])

        self.schedule.add(HeuristicAgent(self.next_id(), self, "GA"))
        self.schedule.add(HeuristicAgent(self.next_id(), self, "SA"))
        self.schedule.add(HeuristicAgent(self.next_id(), self, "TS"))

    def step(self) -> None:
        self.schedule.step()
        if len(self.elite_pool.items) >= 2:
            self.run_path_relinking()

    def add_to_pool(self, route: Route) -> None:
        if route is None:
            return
        self.elite_pool.add(route)

    def get_seed(self) -> Optional[Route]:
        return self.elite_pool.sample(self.random)

    def run_path_relinking(self) -> None:
        # TODO: implement path relinking between elite solutions
        return


class HeuristicAgent(Agent):
    """Agent that runs a specific metaheuristic and updates the elite pool."""

    def __init__(self, unique_id: int, model: VRPOptimizationModel, algo: str) -> None:
        super().__init__(unique_id, model)
        self.algo = algo
        if algo == "GA":
            self.solver = GeneticAlgorithm(model.dataset)
        elif algo == "SA":
            self.solver = SimulatedAnnealing(model.dataset)
        elif algo == "TS":
            self.solver = TabuSearch(model.dataset)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

    def step(self) -> None:
        seed_route = self.model.get_seed()
        seed_perm = self._route_to_permutation(seed_route) if seed_route else None

        improved = self.solver.solve(initial_permutation=seed_perm)
        self.model.add_to_pool(improved)

    def _route_to_permutation(self, route: Route) -> List[int]:
        perm: List[int] = []
        for sub in route.subroutes:
            for cust in sub:
                if cust != 0:
                    perm.append(cust)
        return perm
