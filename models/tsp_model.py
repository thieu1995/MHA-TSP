#!/usr/bin/env python
# Created by "Thieu" at 20:43, 05/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import platform

if platform.system() == "Linux":  # Linux: "Linux", Mac: "Darwin", Windows: "Windows"
    import matplotlib

    matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.


class TravellingSalesmanProblem:
    def __init__(self, n_cities, city_positions):
        self.n_cities = n_cities
        self.city_positions = city_positions

    def fitness_function(self, solution):
        ## For Travelling Salesman Problem, the solution should be a permutation
        ## Lowerbound: [0, 0,...]
        ## Upperbound: [N_cities - 1.11, ....]

        ## Objective for this problem is the sum of distance between all cities that salesman has passed
        ## This can be change depend on your problem
        city_coord = self.city_positions[solution]
        line_x = city_coord[:, 0]
        line_y = city_coord[:, 1]
        total_distance = np.sum(np.sqrt(np.square(np.diff(line_x)) + np.square(np.diff(line_y))))
        return total_distance

    def __get_space__(self):
        x_min, x_max = np.min(self.city_positions[:, 0:]), np.max(self.city_positions[:, 0:])
        y_min, y_max = np.min(self.city_positions[:, 1:]), np.max(self.city_positions[:, 1:])
        text_space_x = (x_min + x_max) / 50
        text_space_y = (y_min + y_max) / 20
        space_x = np.mean(self.city_positions[:, 0:]) / 5
        space_y = np.mean(self.city_positions[:, 1:]) / 5
        return x_min, x_max, y_min, y_max, text_space_x, text_space_y, space_x, space_y

    def plot_cities(self, filename: str, pathsave: str, exts=(".png", ".pdf"), size=100, show_id=True):
        plt.scatter(self.city_positions[:, 0].T, self.city_positions[:, 1].T, s=size, c='k')
        # add text annotation
        x_min, x_max, y_min, y_max, text_space_x, text_space_y, space_x, space_y = self.__get_space__()
        if show_id:
            for city in range(0, self.n_cities):
                plt.text(self.city_positions[city][0] + text_space_x, self.city_positions[city][1] - text_space_y,
                         f"{city}", size='medium', color='black', weight='semibold')
        plt.xlim((x_min - space_x, x_max + space_x))
        plt.ylim((y_min - space_y, y_max + space_y))
        plt.title(f"Cities Map")
        Path(pathsave).mkdir(parents=True, exist_ok=True)
        for idx, ext in enumerate(exts):
            plt.savefig(f"{pathsave}/{filename}{ext}", bbox_inches='tight')
        if platform.system() != "Linux":
            plt.show()
        plt.close()

    def plot_solutions(self, dict_solutions, filename: str, pathsave: str, exts=(".png", ".pdf"),
                       size=100, show_id=True):
        x_min, x_max, y_min, y_max, text_space_x, text_space_y, space_x, space_y = self.__get_space__()
        for idx_pos, solution in enumerate(dict_solutions.values()):
            obj_value = solution[1]
            city_coord = self.city_positions[solution[0]]
            line_x = city_coord[:, 0]
            line_y = city_coord[:, 1]
            plt.scatter(self.city_positions[:, 0].T, self.city_positions[:, 1].T, s=size, c='k')
            # add text annotation
            if show_id:
                for city in range(0, self.n_cities):
                    plt.text(self.city_positions[city][0] + text_space_x, self.city_positions[city][1] - text_space_y,
                             f"{city}", size='medium', color='black', weight='semibold')
            plt.plot(line_x.T, line_y.T, 'r-')
            plt.text(x_min-2*space_x, y_min-2*space_y, f"Total distance: {obj_value:.2f}", fontdict={'size': 12, 'color': 'red'})
            plt.xlim((x_min - space_x, x_max + space_x))
            plt.ylim((y_min - space_y, y_max + space_y))
            plt.title(f"Solution: {idx_pos+1}, GBest: {solution[1]}")

            Path(pathsave).mkdir(parents=True, exist_ok=True)
            for idx, ext in enumerate(exts):
                plt.savefig(f"{pathsave}/{filename}-id{idx_pos+1}{ext}", bbox_inches='tight')
            if platform.system() != "Linux":
                plt.show()
            plt.close()

    def plot_animate(self, dict_solutions, filename: str, pathsave: str, exts=(".gif", ".mp4"),
                     size=100, show_id=True):
        # 1. https://www.youtube.com/watch?v=F57_0XPdhD8
        # 2. https://towardsdatascience.com/how-to-create-animated-graphs-in-python-bb619cc2dec1
        # 3. https://towardsdatascience.com/basics-of-gifs-with-pythons-matplotlib-54dd544b6f30

        x_min, x_max, y_min, y_max, text_space_x, text_space_y, space_x, space_y = self.__get_space__()

        def animate(solution):
            text_total_dist.set_text(f"Total distance: {solution[1]:.2f}")
            text_title.set_text(f"Solution: {solution[0]}")
            city_coord = self.city_positions[solution[0]]
            line_x = city_coord[:, 0]
            line_y = city_coord[:, 1]
            line.set_data(line_x.T, line_y.T)
            return line, text_total_dist, text_title,

        fig, ax = plt.subplots()
        ax.set(xlim=(x_min - space_x, x_max + space_x), ylim=(y_min - space_y, y_max + space_y))
        ax.scatter(self.city_positions[:, 0].T, self.city_positions[:, 1].T, s=size, c='k')
        # add text annotation
        if show_id:
            for city in range(0, self.n_cities):
                plt.text(self.city_positions[city][0] + text_space_x, self.city_positions[city][1] - text_space_y,
                         f"{city}", size='medium', color='black', weight='semibold')
        line, = ax.plot([], [], 'r-')
        text_total_dist = ax.text(x_min-2*space_x, y_min-2*space_y, "", fontdict={'size': 14, 'color': 'red'})
        text_title = ax.set_title("")

        # Pass to FuncAnimation
        Path(pathsave).mkdir(parents=True, exist_ok=True)
        anim = FuncAnimation(fig, animate, frames=dict_solutions.values(), interval=500, blit=True)
        for idx, ext in enumerate(exts):
            anim.save(f"{pathsave}/{filename}{ext}")
        plt.close()
