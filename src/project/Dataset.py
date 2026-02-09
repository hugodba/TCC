import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io

class Dataset:
    def __init__(self, path: Path):
        self.path: Path = path
        self.name: str = ""
        self.max_vehicles: int = 0
        self.capacity: int = 0
        self.customers_df: pd.DataFrame = None
        self.distance_matrix: np.ndarray = None
        
        self._load_data()
        self._compute_distance_matrix()

    def _load_data(self):
        """
        Sets attributes reading data from dataset txt file
        - self.name
        - self.max_vehicles
        - self.capacity
        - self.customers_df
        """
        with open(self.path, 'r') as file:
            lines = file.readlines()

        self.name = lines[0].strip()

        # Vehicle parsing
        for i, line in enumerate(lines):
            if "NUMBER" in line:
                parts = lines[i+1].split()
                self.max_vehicles = int(parts[0])
                self.capacity = int(parts[1])
                break

        # Customer parsing
        start_idx = 0
        for i, line in enumerate(lines):
            if "CUST NO." in line:
                start_idx = i + 1
                break
        
        data_body = "".join(lines[start_idx:])
        cols = ["cust_no", "x", "y", "demand", "ready_time", "due_date", "service_time"]
        self.customers_df = pd.read_csv(io.StringIO(data_body), sep='\s+', names=cols)

    def _compute_distance_matrix(self):
        """Compute the Euclidean distance matrix with double precision."""
        coords = self.customers_df[['x', 'y']].values
        n = len(coords)
        self.distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                # Euclidean distance (Solomon rule)
                d = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
                self.distance_matrix[i][j] = d

    def plot_data(self):
        """Visualize customers and the depot."""
        plt.figure(figsize=(10, 7))
        # Customers
        plt.scatter(self.customers_df.x[1:], self.customers_df.y[1:], c='blue', label='Customers', alpha=0.6)
        # Depot
        plt.scatter(self.customers_df.x[0], self.customers_df.y[0], c='red', marker='s', s=100, label='Depot')
        
        plt.title(f"Instance {self.name} - Geographic Distribution")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    
    