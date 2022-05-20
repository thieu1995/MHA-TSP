#!/usr/bin/env python
# Created by "Thieu" at 01:15, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

### 1. Import libraries
import numpy as np
from mealpy.swarm_based import WOA
from models.tsp_model import TravellingSalesmanProblem
from models.tsp_solution import generate_stable_solution, generate_unstable_solution


### 2. Define data and problem
np.random.seed(10)
N_CITIES = 15
CITY_POSITIONS = np.random.rand(N_CITIES, 2)
TSP = TravellingSalesmanProblem(n_cities=N_CITIES, city_positions=CITY_POSITIONS)
TSP.plot_cities(pathsave="./results/GA-1-TSP", filename="cities_map")

LB = [0, ] * TSP.n_cities
UB = [(TSP.n_cities - 0.01), ] * TSP.n_cities

problem = {
    "fit_func": TSP.fitness_function,
    "lb": LB,
    "ub": UB,
    "minmax": "min",        # Trying to find the minimum distance
    "log_to": "console",
    "amend_position": generate_stable_solution
}

### 3. Call the model
model = WOA.BaseWOA(problem, epoch=100, pop_size=50, pc = 0.9, pm = 0.05, selection="roulette", crossover="multi_points")

### 4. Train the model
best_position, best_fitness = model.solve()

### 5. Show the results
print(f"Best solution: {best_position}, Obj = Total Distance: {best_fitness}")


print(len(model.history.list_global_best))
dict_solutions = {}
for idx, g_best in enumerate(model.history.list_global_best):
    dict_solutions[idx] = [g_best[0], g_best[1][0]]  # Final solution and fitness
# TSP.plot_animate(dict_solutions, filename="GA-1-TSP-results", pathsave="./results/GA-1-TSP")
TSP.plot_solutions(dict_solutions, filename="g-best-solutions-after-epochs", pathsave="./results/GA-1-TSP")

