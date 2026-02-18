"""Tabu Search for VRPTW.

- A trajectory-based metaheuristic that utilizes 'memory' structures to guide the 
    search process.
- Mechanism: It aggressively explores neighborhoods by always moving to the best 
    available neighbor. To avoid cycling (falling back into the same local optima), it 
    records recent moves in a 'Tabu List', forbidding them for a set number of iterations.
- Strength: Superior at 'Exploitation' (Intensification), forcing the algorithm to 
    exhaustively investigate new regions of the solution space by blocking previously 
    visited paths.

Representation:
    - State: a permutation of all customers.
    - Decoding: split the permutation into feasible subroutes with depot (0).
Neighborhood:
    - Swap two customer positions in the permutation.
Tabu list:
    - Stores recent swaps to prevent cycling; aspiration allows overriding
      when a move improves the global best.
Fitness:
    - Hierarchical objective (vehicles K, then total distance D).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..Dataset import Dataset
from ..Route import Route
from .Metaheuristic import Metaheuristic


class TabuSearch(Metaheuristic):
    """Tabu Search for VRPTW using permutation + split encoding."""

    def __init__(
        self,
        dataset: Dataset,
        max_iterations: int = 100,
        tabu_tenure: int = 20,
        neighbor_samples: int = 10,
        seed: int | None = None,
    ) -> None:
        """
        Initialize Tabu Search hyperparameters and RNG seed.

        Args:
            max_iterations: Number of iterations in the search loop.
            tabu_tenure: How many iterations a move (swap) stays forbidden. Larger avoids cycling but can block good moves.
            neighbor_samples: How many random neighbors you evaluate per iteration. Larger improves search quality but costs more time.
            seed: Optional RNG seed for reproducibility.
        """
        super().__init__(dataset, seed)
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure
        self.neighbor_samples = neighbor_samples

    def solve(self, initial_permutation: Sequence[int] | None = None) -> Route:
        """Run Tabu Search and return the best solution found.

        Args:
            initial_permutation: Optional customer order used to seed
                the initial state.
        """
        current_perm = (
            list(initial_permutation)
            if initial_permutation is not None
            else self._random_instance()
        )
        current_route = self._decode(current_perm)
        best_route = current_route.clone()

        tabu: Dict[Tuple[int, int], int] = {}
        self.best_route = best_route
        self.route_historic.append(best_route.clone())

        for _ in range(self.max_iterations):
            candidate_perm, candidate_route, move = self._best_neighbor(
                current_perm, best_route, tabu
            )

            if candidate_perm is not None:
                current_perm = candidate_perm
                current_route = candidate_route

                tabu[move] = self.tabu_tenure

            best_route = self._select_best(best_route, current_route)
            self.best_route = best_route
            self.route_historic.append(best_route.clone())

            self._decrease_tabu(tabu)

        return best_route

    def _best_neighbor(
        self,
        permutation: Sequence[int],
        best_route: Route,
        tabu: Dict[Tuple[int, int], int],
    ) -> tuple[List[int] | None, Route | None, Tuple[int, int]]:
        """Return the best admissible neighbor and its move signature."""
        best_perm = None
        best_route_candidate = None
        best_move = (0, 0)

        n = len(permutation)
        if n < 2:
            return None, None, best_move

        for _ in range(self.neighbor_samples):
            i, j = self.random.sample(range(n), 2)
            move = self._normalize_move(i, j)

            neighbor_perm = list(permutation)
            neighbor_perm[i], neighbor_perm[j] = neighbor_perm[j], neighbor_perm[i]
            neighbor_route = self._decode(neighbor_perm)

            if move in tabu:
                if self._better_than(neighbor_route, best_route):
                    pass
                else:
                    continue

            if best_route_candidate is None:
                best_perm = neighbor_perm
                best_route_candidate = neighbor_route
                best_move = move
                continue

            if self._better_than(neighbor_route, best_route_candidate):
                best_perm = neighbor_perm
                best_route_candidate = neighbor_route
                best_move = move

        return best_perm, best_route_candidate, best_move

    def _better_than(self, a: Route, b: Route) -> bool:
        """Compare two routes by hierarchical objective (K, D)."""
        return a.cost_function() < b.cost_function()

    def _select_best(self, best: Route, candidate: Route) -> Route:
        """Choose the better route between current best and candidate."""
        return candidate if self._better_than(candidate, best) else best

    def _normalize_move(self, i: int, j: int) -> Tuple[int, int]:
        """Canonicalize a swap move as an ordered tuple."""
        return (i, j) if i < j else (j, i)

    def _decrease_tabu(self, tabu: Dict[Tuple[int, int], int]) -> None:
        """Decrease tabu tenure counters and remove expired moves."""
        to_delete = []
        for move in tabu:
            tabu[move] -= 1
            if tabu[move] <= 0:
                to_delete.append(move)
        for move in to_delete:
            del tabu[move]
