from compactsolver import *
import time
import random
from setup import *
import numpy as np
from test import *

# **** Solve ****
data = pd.DataFrame({
    'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
    'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
    'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
})

random.seed()
def generate_cost(num_days, phys, K):
    cost = {}
    shifts = range(1, K + 1)
    for day in range(1, num_days + 1):
        num_costs = phys
        for shift in shifts[:-1]:
            shift_cost = random.randrange(0, num_costs)
            cost[(day, shift)] = shift_cost
            num_costs -= shift_cost
        cost[(day, shifts[-1])] = num_costs
    return cost

demand_dict = generate_cost(14, len(I), len(K))

# Parameter
time_Limit = 3600
max_itr = 25
output_len = 98
mue = 1e-4
eps = 0.18

# **** Compact Solver ****
problem_t0 = time.time()
problem = Problem(data, demand_dict, eps, Min_WD_i, Max_WD_i)
problem.buildLinModel()
problem.updateModel()
problem.model.Params.LogFile = "./test.log"
problem.model.optimize()

file='./test.log'

results, timeline = glt.get_dataframe([file], timelines=True)

# Plot
default_run = timeline["nodelog"]

print(default_run)

plot(default_run)