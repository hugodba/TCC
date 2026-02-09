"""
Genetic Algorithm for VRPTW.

- A population-based metaheuristic inspired by Darwinian natural selection.
- Mechanism: Maintains a pool of solutions (population). It selects 'parents' based on 
    their fitness to produce 'offspring' via Crossover (recombining routes) and Mutation 
    (randomly altering routes).
- Strength: Excellent for 'Exploration' (Diversification), as it searches multiple 
    areas of the solution space simultaneously by sharing information between individuals.

Representation:
    - Gene: a customer index.
    - Individual: a permutation of all customers.
    - Decoding: split permutation into feasible subroutes with depot (0).
Operators:
    - Selection: tournament.
    - Crossover: order crossover (OX).
    - Mutation: swap.
Fitness:
    - Hierarchical objective (vehicles K, then total distance D).
"""

from __future__ import annotations

import random
from typing import List, Sequence, Optional

from ..Dataset import Dataset
from ..Route import Route
from .Metaheuristic import Metaheuristic


class GeneticAlgorithm(Metaheuristic):
    """Simple Genetic Algorithm for VRPTW using permutation + split encoding."""

    def __init__(
        self,
        dataset: Dataset,
        population_size: int = 30,
        generations: int = 200,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.2,
        seed: int | None = None,
    ) -> None:
        """
        Initialize GA hyperparameters and RNG seed.

        Args:
            population_size: Number of solutions per generation. Larger
                populations explore more but take longer per generation.
            generations: Number of evolutionary iterations. More generations
                allow more refinement but increase runtime.
            crossover_rate: Probability of applying crossover to parents.
                Higher values increase mixing between solutions.
            mutation_rate: Probability of mutating an offspring. Higher values
                add exploration but can introduce more randomness.
            seed: Optional RNG seed for reproducibility.
        """
        super().__init__(dataset, seed)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def solve(self, initial_permutation: Sequence[int] | None = None) -> Route:
        """Run the GA loop and return the best solution found.

        Args:
            initial_permutation: Optional customer order used to seed
                the initial population.
        """
        population = [self._random_individual() for _ in range(self.population_size)]
        if initial_permutation is not None:
            population[0] = list(initial_permutation)
        routes = [self._decode(ind) for ind in population]

        self.best_route = routes[0]
        for r in routes[1:]:
            self.best_route = self._compare_routes(r, self.best_route)
        self.route_historic.append(self.best_route.clone())

        for _ in range(self.generations):
            new_population: List[List[int]] = []

            while len(new_population) < self.population_size:
                p1 = self._tournament_select(population, routes)
                p2 = self._tournament_select(population, routes)

                if self.random.random() < self.crossover_rate:
                    c1, c2 = self._order_crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]

                if self.random.random() < self.mutation_rate:
                    self._swap_mutation(c1)
                if self.random.random() < self.mutation_rate:
                    self._swap_mutation(c2)

                new_population.extend([c1, c2])

            population = new_population[: self.population_size]
            routes = [self._decode(ind) for ind in population]

            for r in routes:
                self.best_route = self._compare_routes(r, self.best_route)
            self.route_historic.append(self.best_route.clone())

        return self.best_route
    
    def _random_individual(self) -> List[int]:
        '''Generate a random permutation of customers.'''
        return super()._random_instance()
    
    def _compare_routes(self, candidate: Route, current: Optional[Route] = None) -> Route:
        """Return the better route based on hierarchical objective (K, D)."""
        best = current if current is not None else self.best_route
        if best is None:
            return candidate

        return candidate if candidate.cost_function() < best.cost_function() else best

    def _append_customer(
        self,
        route: List[int],
        cust: int,
        current_load: float,
        current_time: float,
    ) -> tuple[float, float]:
        """Append a customer and update load/time state."""
        last = route[-1]
        travel = float(self.dataset.distance_matrix[last][cust])
        arrival = current_time + travel

        ready = float(self.dataset.customers_df.loc[cust, "ready_time"])
        service = float(self.dataset.customers_df.loc[cust, "service_time"])

        start_service = max(arrival, ready)
        depart = start_service + service

        route.append(cust)
        current_load += float(self.dataset.customers_df.loc[cust, "demand"])
        current_time = depart

        return current_load, current_time

    def _tournament_select(
        self, population: List[List[int]], routes: List[Route], k: int = 3
    ) -> List[int]:
        """Select one parent using tournament selection.

        Picks k random candidates and returns a copy of the best one
        according to the hierarchical cost (K, D).
        """
        idxs = [self.random.randrange(len(population)) for _ in range(k)]
        best_idx = idxs[0]
        for i in idxs[1:]:
            if routes[i].cost_function() < routes[best_idx].cost_function():
                best_idx = i
        return population[best_idx][:]

    def _order_crossover(
        self, parent1: Sequence[int], parent2: Sequence[int]
    ) -> tuple[List[int], List[int]]:
        """
        Order crossover (OX) operator for permutations.
        
        Creates two children by exchanging ordered segments between parents.
        """
        size = len(parent1)
        a, b = sorted(self.random.sample(range(size), 2))

        child1 = [-1] * size
        child2 = [-1] * size

        child1[a:b] = parent1[a:b]
        child2[a:b] = parent2[a:b]

        self._fill_ox(child1, parent2, b)
        self._fill_ox(child2, parent1, b)

        return child1, child2

    def _fill_ox(self, child: List[int], donor: Sequence[int], start: int) -> None:
        """
        Fill remaining genes in an OX child using donor order.

        Iterates donor genes and inserts those not yet present in the child,
        starting from the cut point and wrapping around.
        """
        size = len(child)
        idx = start
        for gene in donor:
            if gene in child:
                continue
            while child[idx % size] != -1:
                idx += 1
            child[idx % size] = gene
            idx += 1

    def _swap_mutation(self, individual: List[int]) -> None:
        """
        Swap-mutation operator for permutations.

        Randomly selects two positions and swaps their genes.
        """
        if len(individual) < 2:
            return
        i, j = self.random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
