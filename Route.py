from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import numpy as np
from Dataset import Dataset
from matplotlib import pyplot as plt

@dataclass
class Route:
    """Represents a solution for VRPTW metaheuristics.

    Attributes:
        subroutes: A list of subroutes; each subroute is a list of customer indices.
        total_distance: Total traveled distance for all subroutes (the cost).
        feasibility: Whether the solution satisfies all constraints.
        metadata: Optional dictionary for algorithm-specific data.
    """

    dataset: Dataset
    subroutes: List[List[int]] = field(default_factory=list)
    total_distance: float = 0.0
    feasibility: bool = True

    # vehicles used
    def num_subroutes(self) -> int:
        return len(self.subroutes)

    def total_customers(self) -> int:
        return sum(len(route) for route in self.subroutes)

    def add_subroute(self, subroute: Sequence[int]) -> None:
        self.subroutes.append(list(subroute))

    def clone(self) -> "Route":
        return Route(
            dataset=self.dataset,
            subroutes=[list(r) for r in self.subroutes],
            total_distance=self.total_distance,
            feasibility=self.feasibility,
        )

    def is_feasible(self) -> bool:
        # implement
        return self.feasibility
    
    # cost function
    def get_total_distance(self) -> float:
        """Return the total distance for a given list of customer indices."""
        if not self.subroutes:
            self.total_distance = 0.0
            return 0.0

        total_distance = 0.0
        dist = self.dataset.distance_matrix
        for route in self.subroutes:
            if len(route) < 2:
                continue
            r = np.asarray(route, dtype=int)
            total_distance += dist[r[:-1], r[1:]].sum()

        self.total_distance = float(total_distance)
        return self.total_distance
    
    def read_solution(self, path: Path, include_depot: bool = True) -> None:
        """Read a solution file and update route attributes.

        Expected format (example):
            Route 1 : 42 36 39
            Route 2 : 65 14 47
        """
        self.subroutes = []

        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line.startswith("Route"):
                    continue

                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue

                customers_str = parts[1].strip()
                if not customers_str:
                    continue

                customers = [int(x) for x in customers_str.split()]
                if include_depot:
                    route = [0, *customers, 0]
                else:
                    route = customers

                self.subroutes.append(route)

        self.total_distance = self.get_total_distance()
        self.feasibility = True

    def plot_routes(self) -> None:
        """Visualize the routes on a 2D plot."""
        plt.figure(figsize=(10, 7))
        # Plot customers
        plt.scatter(
            self.dataset.customers_df.x[1:],
            self.dataset.customers_df.y[1:],
            c='blue',
            label='Customers',
            alpha=0.6
        )
        # Plot depot
        plt.scatter(
            self.dataset.customers_df.x[0],
            self.dataset.customers_df.y[0],
            c='red',
            label='Depot',
            marker='s',
            s=100
        )

        # Plot each subroute
        for idx, route in enumerate(self.subroutes):
            route_coords = self.dataset.customers_df.iloc[route][['x', 'y']].values
            plt.plot(
                route_coords[:, 0],
                route_coords[:, 1],
                label=f'Route {idx + 1}'
            )

        plt.title('Vehicle Routes')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.show()
