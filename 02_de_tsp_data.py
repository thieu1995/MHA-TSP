#!/usr/bin/env python
# Created by "Thieu" at 18:16, 06/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

#### Let's try to use Differential Evolution (DE) this time
### 1. Import libraries
### 2. Define data
### 3. Design fitness function (objective function)
### 4. Design problem dictionary
### 5. Call the model
### 6. Train the model
### 7. Show the results


from mealpy.evolutionary_based import DE
from models.tsp_model import TravellingSalesmanProblem
from models.tsp_solution import generate_unstable_solution
import numpy as np

## https://developers.google.com/optimization/routing/tsp

DIST_MATRIX = np.array([
    [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
    [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
    [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
    [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
    [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
    [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
    [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
    [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
    [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
    [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
    [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
    [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
    [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
])
CITIES_DICT = {
    0: "New York", 1: "Los angeles", 2: "Chicago", 3: "Minneapolis", 4: "Denver", 5: "Dallas", 6: "Seattle",
    7: "Boston", 8: "San Francisco", 9: "St. Louis", 10: "Houston", 11: "Phoenix", 12: "Salt Lake City"
}
# I tried to design position of cities by hand, but it may not effect the real coordinate due to the distance among cities.
# You can also try to locate city's position randomly for better visualization
CITIES_POS = {
    0: [1, 1],
    1: [3, 4],
    2: [6, 2],
    3: [10, 3],
    4: [2, 6],
    5: [5, 5],
    6: [7, 5],
    7: [4, 7],
    8: [8, 7],
    9: [9, 6],
    10: [1, 9],
    11: [5, 8],
    12: [8, 9]
}

# np.random.seed(10)
N_CITIES = DIST_MATRIX.shape[0]
TSP = TravellingSalesmanProblem(n_cities=N_CITIES, city_positions=np.array(list(CITIES_POS.values())))
TSP.plot_cities(pathsave="./results/DE-TSP-2", filename="cities_map")

LB = [1, ] * (N_CITIES - 1)
UB = [(N_CITIES - 0.01), ] * (N_CITIES - 1)
## For this problem, depot always start at 0 --> problem size = n_dims = N_CITIES - 1
## The value from 0-0.99 is city index 0, but we remove it because it always start at 0
## https://developers.google.com/optimization/routing/tsp


def correct_full_solution(solution_optimization):
    return np.insert(solution_optimization, 0, 0)        # Now we need to insert city index 0 to the generated solution


def fitness_function(solution_optimization):
    ## solution_optimization is the solution get from Optimization process.
    # It it not the full solution we need to calculate fitness value.
    pos = correct_full_solution(solution_optimization)
    dist = 0
    for idx_city in range(0, N_CITIES):
        idx_next = idx_city + 1
        if idx_city == N_CITIES - 1:
            idx_next = 0
        dist += DIST_MATRIX[pos[idx_city]][pos[idx_next]]
    return dist


## Example
solution_optimization = np.random.uniform(LB, UB)
print(solution_optimization)
solution_amended = generate_unstable_solution(solution_optimization, LB, UB)
print(solution_amended)
# solution_full = correct_full_solution(solution_amended)
# print(solution_full)
distance = fitness_function(solution_amended)   # correct_full_solution is already called in fitness_function
print(f"Example distance: {distance}")


## Define problem dictionary
problem = {
    "fit_func": fitness_function,
    "lb": LB,
    "ub": UB,
    "minmax": "min",
    "amend_position": generate_unstable_solution
}

## Run the algorithm
model = DE.SHADE(problem, epoch=5, pop_size=50)
best_position, best_fitness = model.solve()
print(f"Best position: {best_position}, Best fit: {best_fitness}")


## Show the results
dict_solutions = {}
for idx, g_best in enumerate(model.history.list_global_best):
    dict_solutions[idx] = [correct_full_solution(g_best[0]), g_best[1][0]]      # Final solution and fitness
TSP.plot_animate(dict_solutions, filename="DE-TSP-results", pathsave="./results/DE-TSP-2")
TSP.plot_solutions(dict_solutions, filename="g-best-solutions-after-epochs", pathsave="./results/DE-TSP-2")
