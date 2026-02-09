from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Sequence
import matplotlib.pyplot as plt
import random

from ..Dataset import Dataset
from ..Route import Route


class Metaheuristic(ABC):
    """Base class for VRPTW metaheuristics."""

    def __init__(self, dataset: Dataset, seed: int | None = None) -> None:
        self.dataset = dataset
        self.best_route: Optional[Route] = None
        self.route_historic: list[Route] = []    
        self.customers: List[int] = list(range(1, len(self.dataset.customers_df)))
        self.random = random.Random(seed)

    def history_costs(self) -> Tuple[List[int], List[float]]:
        """Return (vehicles, distance) history from stored best routes."""
        vehicles_hist = [r.calc_total_vehicles() for r in self.route_historic]
        distance_hist = [r.calc_total_distance() for r in self.route_historic]
        return vehicles_hist, distance_hist

    def _random_instance(self) -> List[int]:
        """Generate a random permutation of customers."""
        ind = self.customers[:]
        self.random.shuffle(ind)
        return ind

    def _decode(self, permutation: Sequence[int]) -> Route:
        """Split a customer permutation into feasible subroutes."""
        subroutes: List[List[int]] = []
        current_route: List[int] = [0]
        current_load = 0.0
        current_time = 0.0

        for cust in permutation:
            if self._can_append(current_route, cust, current_load, current_time):
                current_load, current_time = self._append_customer(
                    current_route, cust, current_load, current_time
                )
                continue

            current_route.append(0)
            subroutes.append(current_route)

            current_route = [0]
            current_load = 0.0
            current_time = 0.0

            current_load, current_time = self._append_customer(
                current_route, cust, current_load, current_time
            )

        current_route.append(0)
        subroutes.append(current_route)

        route = Route(dataset=self.dataset, subroutes=subroutes)
        route.calc_total_distance()
        route.calc_total_vehicles()
        route.is_feasible()
        return route

    def _can_append(
        self,
        route: List[int],
        cust: int,
        current_load: float,
        current_time: float,
    ) -> bool:
        """Check capacity/time-window feasibility for appending a customer."""
        demand = float(self.dataset.customers_df.loc[cust, "demand"])
        if current_load + demand > self.dataset.capacity:
            return False

        last = route[-1]
        travel = float(self.dataset.distance_matrix[last][cust])
        arrival = current_time + travel

        due = float(self.dataset.customers_df.loc[cust, "due_date"])
        if arrival > due:
            return False

        return True

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

    def plot_cost_history(self) -> None:
        """Plot historic cost curves (vehicles and distance)."""
        if not self.route_historic:
            return

        vehicles_hist, distance_hist = self.history_costs()
        iterations = list(range(len(self.route_historic)))

        ax1: plt.Axes
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(iterations, vehicles_hist, color="tab:blue", label="Vehicles (K)")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Vehicles (K)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(iterations, distance_hist, color="tab:orange", label="Distance (D)")
        ax2.set_ylabel("Distance (D)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        ax1.set_title(f"{self.__class__.__name__} - Cost Function History")
        fig.tight_layout()
        plt.show()

    @abstractmethod
    def solve(self) -> Route:
        """Run the metaheuristic and return the best route found."""
        raise NotImplementedError

    
