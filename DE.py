import numpy as np
import random

from benchmark import *
def get_rand(i, pop_size):
    
    candidates = list(range(pop_size))
    candidates.remove(i)

    idxs = np.random.choice(candidates, 3, replace=False)
    return idxs[0], idxs[1], idxs[2]

def DE(x, task, scalar_factor, crossover_rate):

    new_par = task.copy()
    shift, dim, bounds = get_task_info(f"T{x}")

    fitness_arr = np.zeros(len(task))
    for k in range(len(task)):
         real_gen = bounds[0] + new_par[k] * (bounds[1] - bounds[0])
         fitness_arr[k] = calculate_objective_function(f"T{x}", real_gen, shift)

    for i in range(len(task)):

        u1, u2, u3 = get_rand(i, len(task))
        v1 = new_par[u1] + scalar_factor * (new_par[u2] - new_par[u3])
        v1 = np.clip(v1, 0, 1)

        tried = new_par[i].copy()
        j_rand = np.random.randint(0, len(tried))

        for j in range(len(v1)):

            rate = np.random.uniform(0, 1)
            if rate <= crossover_rate or j == j_rand:
                tried[j] = v1[j]
            
        real_gen_new = bounds[0] + tried *(bounds[1] - bounds[0])
        
        new_fitness = calculate_objective_function(f"T{x}", real_gen_new, shift)
        if new_fitness < fitness_arr[i]:
            new_par[i] = tried
            fitness_arr[i] = new_fitness
    
    return new_par

        