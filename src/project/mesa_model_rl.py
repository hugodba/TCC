from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import logging
import random

from mesa import Agent, Model
from mesa.agent import AgentSet

from .Route import Route
from .Metaheuristics import GeneticAlgorithm, SimulatedAnnealing, TabuSearch, Metaheuristic

logger = logging.getLogger("TCC")


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
        """Return a random route from the pool using the provided RNG.""" 
        if not self.items:
            return None
        return rng.choice(self.items)


class VRPOptimizationModelRL(Model):
    """Mesa model managing heuristic agents with Q-Learning as a Hyper-heuristic."""

    def __init__(self, dataset, pool_size: int = 10, seed: int | None = None, total_steps: int = 100) -> None:
        """Initialize the model, RL parameters, and agents."""
        super().__init__()
        self.dataset = dataset
        self.random = random.Random(seed)
        self.agent_set = AgentSet([], random=self.random)
        self.elite_pool = ElitePool(max_size=pool_size, items=[])
        self.total_steps = total_steps

        # --- Reinforcement Learning Parameters ---
        self.q_table: Dict[Tuple[int, int], List[float]] = {} # Map state -> [Q_SA, Q_TS, Q_GA]
        self.action_count_table: Dict[Tuple[int, int], List[int]] = {} # Map state -> [N_SA, N_TS, N_GA]
        self.epsilon = 0.4          # Exploration rate
        self.alpha = 0.2              # Learning rate
        self.gamma = 0.8            # Discount factor
        self.distance_bin_size = 100 # Discretization factor for state

        # Agents mapping (Action Space)
        # 0: SA, 1: TS, 2: GA
        self.sa_agent = HeuristicAgent(self, "SA", seed)
        self.ts_agent = HeuristicAgent(self, "TS", seed)
        self.ga_agent = HeuristicAgent(self, "GA", seed)

        self.agents_list = [self.sa_agent, self.ts_agent, self.ga_agent]
        
        # Add to Mesa set for compatibility
        self.agent_set.add(self.ga_agent)
        self.agent_set.add(self.sa_agent)
        self.agent_set.add(self.ts_agent)

    def get_state(self) -> Tuple[int, int]:
        """
        Define the environment state. 
        State = (Number of Vehicles, Discrete Distance Band)
        """
        best_sol = self.elite_pool.best()
        if best_sol is None:
            # Default state if pool is empty (arbitrary high numbers)
            return (999, 999)
        
        k, d = best_sol.cost_function()
        # Discretize distance to reduce state space size
        d_bin = int(d / self.distance_bin_size)
        return (k, d_bin)

    def get_q_values(self, state: Tuple[int, int]) -> List[float]:
        """Retrieve Q-values for a state, initializing if unknown."""
        if state not in self.q_table:
            # Initialize with 0.0 for all 3 actions [SA, TS, GA]
            self.q_table[state] = [0.0, 0.0, 0.0]
        return self.q_table[state]

    def get_action_counts(self, state: Tuple[int, int]) -> List[int]:
        """Retrieve action selection counts for a state, initializing if unknown."""
        if state not in self.action_count_table:
            self.action_count_table[state] = [0, 0, 0]
        return self.action_count_table[state]

    def select_action(self, state: Tuple[int, int]) -> int:
        """Epsilon-Greedy selection of the metaheuristic to apply."""
        if self.random.random() < self.epsilon:
            # Explore: choose random agent
            return self.random.randint(0, 2)
        else:
            # Exploit: choose best Q-value
            q_values = self.get_q_values(state)
            # Find index of max value (manual argmax)
            return q_values.index(max(q_values))

    def calculate_reward(self, prev_best: Optional[Route], curr_best: Optional[Route]) -> float:
        """
        Hierarchical Reward Function:
        1. Huge reward for reducing Vehicle count (Primary Objective).
        2. Moderate reward for reducing Distance (Secondary Objective).
        3. Small negative reward for stagnation.
        """
        if prev_best is None or curr_best is None:
            return 0.0
        
        prev_k, prev_d = prev_best.cost_function()
        curr_k, curr_d = curr_best.cost_function()

        reward = 0.0

        # Primary Objective: Vehicles
        if curr_k < prev_k:
            reward += 200  # Massive bonus for dropping a vehicle
            reward += (prev_d - curr_d) # Plus distance gain
        # Secondary Objective: Distance (only if vehicles didn't increase)
        elif curr_k == prev_k:
            if curr_d < prev_d:
                reward += (prev_d - curr_d) # Reward = magnitude of improvement
            else:
                reward -= 1 # Small penalty for stagnation/time wasting
        else:
            reward -= 20 # Penalty for worsening solution (rare, but possible)

        return reward

    def step(self) -> None:
        """
        RL Step Logic (3 runs per model step):
        - For each run, use epsilon-greedy RL to choose an agent.
        - Execute chosen agent, compute reward, and update Q-table.
        - Same agent may be chosen multiple times.
        """
        for run_idx in range(3):
            # 1) Observe state before action
            current_state = self.get_state()
            prev_best = self.elite_pool.best()

            # 2) RL action selection (always)
            action_idx = self.select_action(current_state)
            selected_agent = self.agents_list[action_idx]
            self.get_action_counts(current_state)[action_idx] += 1
            logger.info(
                f"==> RL Run {run_idx + 1}/3 | State {current_state} -> Selected Action: {selected_agent.algo}"
            )

            # 3) Execute selected action
            selected_agent.step()

            # 4) Observe transition and reward
            new_state = self.get_state()
            curr_best = self.elite_pool.best()
            reward = self.calculate_reward(prev_best, curr_best)

            # 5) Q-learning update
            q_values_current = self.get_q_values(current_state)
            q_values_next = self.get_q_values(new_state)
            max_q_next = max(q_values_next)

            current_q = q_values_current[action_idx]
            new_q = current_q + self.alpha * (reward + self.gamma * max_q_next - current_q)
            self.q_table[current_state][action_idx] = new_q

            logger.info(
                ' '*4 + f"Update: Reward={reward:.2f} | "
                f"Q({selected_agent.algo}) of state{current_state} updated to {new_q:.2f}"
            )

        if len(self.elite_pool.items) >= 2:
            self.run_path_relinking()

    def add_to_pool(self, route: Route) -> None:
        """Add a route to the elite pool if it is valid."""
        if route is None:
            return
        self.elite_pool.add(route)

    def get_seed(self) -> Optional[Route]:
        """Sample a seed route from the elite pool for warm starting."""
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

        decoder = self._get_decoder()

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
        for agent in self.agents_list:
             return agent.solver
        return None


class HeuristicAgent(Agent):
    """Agent that runs a specific metaheuristic and updates the elite pool."""

    def __init__(self, model: VRPOptimizationModelRL, algo: str, seed: int | None) -> None:
        """Create an agent bound to a specific metaheuristic solver."""
        super().__init__(model)
        self.algo = algo
        if algo == "GA":
            self.solver = GeneticAlgorithm(model.dataset, seed=seed)
        elif algo == "SA":
            self.solver = SimulatedAnnealing(model.dataset, seed=seed)
        elif algo == "TS":
            self.solver = TabuSearch(model.dataset, seed=seed)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

    def step(self) -> None:
        """Run the solver once and push any improvement to the elite pool."""
        self.model: VRPOptimizationModelRL  # type hint for self.model

        seed_route = self.model.get_seed()
        seed_perm = None
        if seed_route:
            # Helper to extract perm from route inside agent context
            seed_perm = []
            for sub in seed_route.subroutes:
                for cust in sub:
                    if cust != 0:
                        seed_perm.append(cust)

        improved = self.solver.solve(initial_permutation=seed_perm)

        k, d = improved.cost_function()
        logger.info(' '*4 + f"Agent {self.algo} step result -> vehicles: {k}, distance: {d:.2f}")

        self.model.add_to_pool(improved)