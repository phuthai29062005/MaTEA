import numpy as np
import random

from DE import *
from KLD import caculate_similarity
from benchmark import *



def log_to_file(fitness):
    with open("MaTEA.txt", "a") as f:
        f.write(f"{fitness}\n")

def clear_old_logs():
    with open("MaTEA.txt", "w") as f:
        f.write("Fitness\n")

    
def calculate_fitness(task, i_idx):
    real_gen = task.copy()
    shift, dim, bounds = get_task_info(f"T{i_idx}")
    real_gen = bounds[0] + real_gen[:dim] * (bounds[1] - bounds[0])
    fitness = calculate_objective_function(f"T{i_idx}", real_gen, shift)
    return fitness

def MaTEA():
    
    clear_old_logs()
    num_task = 10
    population = 100
    reward_rate = 0.8
    p = 0.8
    transfer_rate = 0.1
    update_rate = 0.2
    
    Achive = [[] for _ in range(num_task + 1)]
    archive_size = 300
    task = [[] for _ in range(num_task + 1)]
    dim_max = 0
    fitness_arr = [np.zeros(population) for _ in range(num_task + 1)]
    
    rewards = np.ones((num_task + 1, num_task + 1))
    scores = np.zeros((num_task + 1, num_task + 1))
    history_fitness = [float('inf') for _ in range(num_task + 1)]
    
    for i in range(1, num_task + 1):
        _, dim, _ = get_task_info(f"T{i}")
        if dim > dim_max:
            dim_max = dim
    
    for i in range(num_task + 1):
        _, dim, _ = get_task_info(f"T{i}")
        task[i] = np.random.uniform(0, 1, (population, dim_max))
        if i > 0:
            for j in range(population):
                fitness = calculate_fitness(task[i][j], i)
                fitness_arr[i][j] = fitness
                if fitness < history_fitness[i]:
                    history_fitness[i] = fitness
        
        Achive[i] = [ind.copy() for ind in task[i]]

    for gen in range(1, 1000):
        clean_history = [float(x) for x in history_fitness]
        log_to_file(clean_history[1:])
        if gen % 50 == 0:     
            print(f"Generation {gen} completed.")
        
        crossover_rate = np.random.uniform(0.1, 0.9)
        scalar_factor = np.random.uniform(0.1, 2)
        
        for i_idx in range(1, num_task + 1):

            rand_rate = np.random.uniform(0, 1)
            if rand_rate > transfer_rate:
                task[i_idx], best_fit = DE(i_idx, task[i_idx], fitness_arr[i_idx], scalar_factor, crossover_rate)
                if best_fit < history_fitness[i_idx]:
                    history_fitness[i_idx] = best_fit
            else:
                Probability = []
                
                for j_idx in range(1, num_task + 1):
                    if i_idx == j_idx:
                        continue
                    matrix_i = np.array(Achive[i_idx])
                    matrix_j = np.array(Achive[j_idx])
                    
                    sim = caculate_similarity(matrix_i, matrix_j)
                    scores[i_idx][j_idx] = p * scores[i_idx][j_idx] + rewards[i_idx][j_idx] * np.exp(-sim)
                    if scores[i_idx][j_idx] < 0:
                        scores[i_idx][j_idx] = 0.0
                    
                total_score = np.sum(scores[i_idx])
                
                for j_idx in range(1, num_task + 1):
                    if i_idx == j_idx:
                        Probability.append(0)
                    else:
                        Probability.append(scores[i_idx][j_idx] / total_score)
                        
                selected_task = np.random.choice(range(1, num_task + 1), p=Probability)
                while selected_task == i_idx:
                    selected_task = np.random.choice(range(1, num_task + 1), p=Probability)

                best_fitness = 1e9
                
                for i in range(population):
                    v1 = np.random.randint(0, population)
                    j_rand = np.random.randint(0, min(len(task[selected_task][0]), len(task[i_idx][0])))
                    new_child = crossover(task[i_idx][i].copy(), task[selected_task][v1], j_rand, crossover_rate)
                    fitness_new = calculate_fitness(new_child, i_idx)
                    if fitness_new < fitness_arr[i_idx][i]:
                        task[i_idx][i] = new_child
                        fitness_arr[i_idx][i] = fitness_new
                    
                    if fitness_arr[i_idx][i] < best_fitness:
                        best_fitness = fitness_arr[i_idx][i]
                
                if best_fitness < history_fitness[i_idx]:
                    rewards[i_idx][selected_task] /= reward_rate
                    history_fitness[i_idx] = best_fitness
                else:
                    rewards[i_idx][selected_task] *= reward_rate
                    

            # Cập nhật Achive
            for ind in task[i_idx]:
                if np.random.rand() < update_rate:
                    current_archive = Achive[i_idx] # Đây là List
                    
                    if len(current_archive) < archive_size:
                        # Nếu chưa đầy -> Thêm mới
                        current_archive.append(ind.copy())
                    else:
                        # Nếu đầy -> Thay thế ngẫu nhiên một phần tử cũ
                        replace_idx = np.random.randint(0, archive_size)
                        current_archive[replace_idx] = ind.copy()
                        
        
    
    
if __name__ == "__main__":
    MaTEA()

    
                
           
                
