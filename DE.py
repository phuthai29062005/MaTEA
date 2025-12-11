import numpy as np
import random

from benchmark import *

def crossover(tried, v1, j_rand, crossover_rate):

    for j in range(len(v1)):

        rate = np.random.uniform(0, 1)
        if rate <= crossover_rate or j == j_rand:
            tried[j] = v1[j]

    return tried

def DE(x, task, fitness_arr, scalar_factor, crossover_rate):

    new_par = task.copy()
    shift, dim, bounds = get_task_info(f"T{x}")
    
    for i in range(len(task)):

        u1 = np.random.randint(0, len(task))
        while(u1 == i):
            u1 = np.random.randint(0, len(task))
        v1 = task[i] + scalar_factor * (task[u1] - task[i])
        v1 = np.clip(v1, 0, 1)

        tried = task[i].copy()
        j_rand = np.random.randint(0, len(tried))
        tried = crossover(tried, v1, j_rand, crossover_rate)

        real_gen_new = bounds[0] + tried[:dim] * (bounds[1] - bounds[0])

        new_fitness = calculate_objective_function(f"T{x}", real_gen_new, shift)
        if new_fitness < fitness_arr[i]:
            new_par[i] = tried
            fitness_arr[i] = new_fitness
    
    return new_par

        