import numpy as np
import random

from benchmark import *
X = np.random.uniform(0, 1, (2, 3))
print(X)
def MaTEA():

    num_task = 10
    population = 100
    reward_rate = 0.8
    p = 0.8
    transfer_rate = 0.1
    update_rate = 0.2
    crossover_rate = np.random.uniform(0.1, 0.9)
    scalar_factor = np.random.uniform(0.1, 2)
    Achive = []
    archive_size = 300
    task = [[] for _ in range(num_task + 1)]
    dim_max = 0
    rewards = np.ones((num_task + 1, num_task + 1))
    for i in range(1, num_task + 1):
        _, dim, _ = get_task_info(f"T{i}")
        if dim > dim_max:
            dim_max = dim
    
    for i in range(1, num_task + 1):
        task[i] = np.random.uniform(0, 1, (population, dim_max))
        Achive.append(task[i])

