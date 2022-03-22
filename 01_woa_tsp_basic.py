#!/usr/bin/env python
# Created by "Thieu" at 12:17, 05/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

#### Let's try to use Whale Optimization Algorithm (WOA)
### 1. Import libraries
### 2. Define data
### 3. Design fitness function (objective function)
### 4. Design problem dictionary
### 5. Call the model
### 6. Train the model
### 7. Show the results


##################################################################################

### 1. Import libraries
import numpy as np
from mealpy.swarm_based import WOA
from models.tsp_model import TravellingSalesmanProblem
from models.tsp_solution import generate_stable_solution, generate_unstable_solution


### 2. Define data
np.random.seed(10)
N_CITIES = 15
CITY_POSITIONS = np.random.rand(N_CITIES, 2)
TSP = TravellingSalesmanProblem(n_cities=N_CITIES, city_positions=CITY_POSITIONS)
TSP.plot_cities(pathsave="./results/WOA-TSP", filename="cities_map")


### 3. Design fitness function
# For this simple example, we take the defined fitness function inside the class TravellingSalesmanProblem


### 4. Design problem dictionary
## A solution is a permutation of cities index --> Lower bound = 0, Upper bound = N cities - 1

## We are using ROUNDING OFF method, therefore:
## 0 -> 0.99: Represent city 1
## 1 -> 1.99: Represent city 2
## ....
## N-1 -> N-0.01: Represent city N
##
## We need to choice the amend_position function for model, because this is discrete problem
## Let's take the function that can generate stable solution

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

solution = np.array([1, 5, 9, 7, 8, 0, 2, 4, 6, 3])
fit_value = TSP.fitness_function(solution)
print(f"Objective value / fitness value / distance value (in this case): {fit_value}")

optimal_solution = np.array([6, 4, 0, 10, 2, 8, 12, 13, 14, 7, 9, 5, 1, 11, 3])
optimal_dist = TSP.fitness_function(optimal_solution)
print(f"Optimal Objective value / fitness value / distance value (in this case): {optimal_dist}")

### 5. Call the model
model = WOA.BaseWOA(problem, epoch=10, pop_size=50)


### 6. Train the model
best_position, best_fitness = model.solve()


### 7. Show the results
print(f"Best solution: {best_position}, Obj = Total Distance: {best_fitness}")


print(len(model.history.list_global_best))
dict_solutions = {}
for idx, g_best in enumerate(model.history.list_global_best):
    dict_solutions[idx] = [g_best[0], g_best[1][0]]  # Final solution and fitness
TSP.plot_animate(dict_solutions, filename="WOA-TSP-results", pathsave="./results/WOA-TSP")
TSP.plot_solutions(dict_solutions, filename="g-best-solutions-after-epochs", pathsave="./results/WOA-TSP")

