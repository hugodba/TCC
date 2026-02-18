"""Simulated Annealing for VRPTW.

- A trajectory-based metaheuristic inspired by the metallurgical process of heating 
    and controlled cooling of materials.
- Mechanism: Starts with a high 'temperature' and a single solution. In each step, it 
    probabilistically accepts worse solutions to 'jump' out of local optima. As the 
    temperature drops, the search becomes more restrictive.
- Strength: Balanced approach that transitions from global exploration to intensive 
    local refinement as the system 'cools'.

Representation:
    - State: a permutation of all customers.
    - Decoding: split the permutation into feasible subroutes with depot (0).
Neighborhood:
    - Swap two customer positions in the permutation.
Temperature:
    - Controls acceptance of worse moves; higher temperature accepts more.
Acceptance:
    - Always accept improving moves; accept worse moves with probability based on
      cost difference and current temperature.
Schedule:
    - Temperature decreases by a cooling rate until a minimum threshold.
Fitness:
    - Hierarchical objective (vehicles K, then total distance D) scalarized
      for acceptance.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence

from ..Dataset import Dataset
from ..Route import Route
from .Metaheuristic import Metaheuristic


class SimulatedAnnealing(Metaheuristic):
    """Simulated Annealing for VRPTW using permutation + split encoding."""

    def __init__(
        self,
        dataset: Dataset,
        initial_temp: float = 200.0,
        cooling_rate: float = 0.98,
        min_temp: float = 10.0,
        iterations_per_temp: int = 10,
        seed: int | None = None,
    ) -> None:
        """Initialize SA hyperparameters and RNG seed.

        Args:
            initial_temp: Starting temperature; higher values accept worse moves
                more often early in the search.
            cooling_rate: Multiplier applied each step; values closer to 1.0
                cool more slowly and take longer.
            min_temp: Stopping temperature; higher values stop earlier.
            iterations_per_temp: Neighbors evaluated at each temperature;
                higher values improve search but increase runtime.
            seed: Optional RNG seed for reproducibility.
        """
        super().__init__(dataset, seed)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.iterations_per_temp = iterations_per_temp

    def solve(self, initial_permutation: Sequence[int] | None = None) -> Route:
        """Run SA and return the best solution found.

        Args:
            initial_permutation: Optional customer order used to seed
                the initial state.
        """
        current_perm = (
            list(initial_permutation)
            if initial_permutation is not None
            else self._random_state()
        )
        current_route = self._decode(current_perm)
        best_route = current_route.clone()
        best_cost = self._scalar_cost(best_route)

        self.best_route = best_route
        self.route_historic.append(best_route.clone())

        temperature = self.initial_temp
        while temperature > self.min_temp:
            for _ in range(self.iterations_per_temp):
                neighbor_perm = self._neighbor_swap(current_perm)
                neighbor_route = self._decode(neighbor_perm)

                if self._accept(current_route, neighbor_route, temperature):
                    current_perm = neighbor_perm
                    current_route = neighbor_route

                current_cost = self._scalar_cost(current_route)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_route = current_route.clone()

            self.best_route = best_route
            self.route_historic.append(best_route.clone())
            temperature *= self.cooling_rate

        return best_route
    
    def _random_state(self):
        return super()._random_instance()

    def _neighbor_swap(self, permutation: Sequence[int]) -> List[int]:
        """Create a neighbor by swapping two positions."""
        neighbor = list(permutation)
        if len(neighbor) < 2:
            return neighbor
        i, j = self.random.sample(range(len(neighbor)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor

    def _scalar_cost(self, route: Route) -> float:
        """Scalarize hierarchical cost for SA acceptance."""
        k, d = route.cost_function()
        return k * 1_000_000.0 + d

    def _accept(self, current: Route, candidate: Route, temperature: float) -> bool:
        """Accept candidate based on cost and temperature."""
        c_cost = self._scalar_cost(current)
        n_cost = self._scalar_cost(candidate)
        if n_cost < c_cost:
            return True

        delta = n_cost - c_cost
        return self.random.random() < math.exp(-delta / temperature)
