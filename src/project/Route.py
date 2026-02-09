from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from .Dataset import Dataset
from matplotlib import pyplot as plt
import copy

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

    feasibility: bool = True

    total_distance: float = 0.0
    total_vehicles: int = 0
    
    total_waiting_time: float = 0.0
    total_service_time: float = 0.0
    total_time: float = 0.0

    def clone(self) -> "Route":
        return copy.deepcopy(self)
    
    def is_feasible(self) -> bool:
        if not self.subroutes:
            self.feasibility = False
            return self.feasibility

        n_customers = len(self.dataset.customers_df)
        all_customers = set(range(1, n_customers))
        visited: list[int] = []

        for route in self.subroutes:
            if len(route) < 2:
                self.feasibility = False
                return self.feasibility

            if route[0] != 0 or route[-1] != 0:
                self.feasibility = False
                return self.feasibility

            customers = [c for c in route if c != 0]
            visited.extend(customers)

            # capacity constraint
            demand = self.dataset.customers_df.loc[customers, "demand"].sum()
            if demand > self.dataset.capacity:
                self.feasibility = False
                return self.feasibility

            # time window constraint
            time = 0.0
            for prev_idx, curr_idx in zip(route[:-1], route[1:]):
                time += float(self.dataset.distance_matrix[prev_idx][curr_idx])

                ready = float(self.dataset.customers_df.loc[curr_idx, "ready_time"])
                due = float(self.dataset.customers_df.loc[curr_idx, "due_date"])
                service = float(self.dataset.customers_df.loc[curr_idx, "service_time"])

                if time < ready:
                    time = ready
                if time > due:
                    self.feasibility = False
                    return self.feasibility

                time += service

        if len(visited) != len(all_customers):
            self.feasibility = False
            return self.feasibility

        if set(visited) != all_customers:
            self.feasibility = False
            return self.feasibility

        self.feasibility = True
        return self.feasibility
    
    def calc_total_vehicles(self) -> int:
        self.total_vehicles = len(self.subroutes)
        return self.total_vehicles
    
    def calc_total_distance(self) -> float:
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

    def calc_total_waiting_time(self) -> float:
        """Return the total waiting time across all subroutes."""
        if not self.subroutes:
            self.total_waiting_time = 0.0
            return 0.0

        total_wait = 0.0
        for route in self.subroutes:
            if len(route) < 2:
                continue

            time = 0.0
            for prev_idx, curr_idx in zip(route[:-1], route[1:]):
                time += float(self.dataset.distance_matrix[prev_idx][curr_idx])

                if curr_idx == 0:
                    continue

                ready = float(self.dataset.customers_df.loc[curr_idx, "ready_time"])
                service = float(self.dataset.customers_df.loc[curr_idx, "service_time"])

                if time < ready:
                    total_wait += ready - time
                    time = ready

                time += service

        self.total_waiting_time = float(total_wait)
        return self.total_waiting_time

    def calc_total_service_time(self) -> float:
        """Return the total service time across all subroutes."""
        if not self.subroutes:
            self.total_service_time = 0.0
            return 0.0

        total_service = 0.0
        for route in self.subroutes:
            for cust in route:
                if cust == 0:
                    continue
                total_service += float(self.dataset.customers_df.loc[cust, "service_time"])

        self.total_service_time = float(total_service)
        return self.total_service_time

    def calc_total_time(self) -> float:
        """Return total time = travel + waiting + service."""
        travel = self.calc_total_distance()
        waiting = self.calc_total_waiting_time()
        service = self.calc_total_service_time()
        self.total_time = float(travel + waiting + service)
        return self.total_time

    def cost_function(self) -> tuple[int, float]:
        """Hierarchical objective: minimize vehicles, then total distance."""
        k = self.calc_total_vehicles()
        d = self.calc_total_distance()
        return k, d
    
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

        self.total_distance = self.calc_total_distance()
        self.feasibility = True

    def plot_routes(self, algo_name: str) -> None:
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

        plt.title(f'{algo_name} - Vehicle Routes')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.show()
