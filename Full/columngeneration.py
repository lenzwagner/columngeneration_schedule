import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from masterproblem import *
from plots import *
import seaborn as sns
from gcutil import *
import random
from subproblem import *
from compactsolver import *

# Set of indices
I, T, K = [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [1, 2, 3]

# Create Dataframes
data = pd.DataFrame({
    'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
    'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
    'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
})

# Additional Sets
Min_WD_i = [2, 3, 2, 3]
Max_WD_i = [5, 5, 5, 6]
S_T = {1: 8, 2: 8, 3: 8}
I_T = {1: 40, 2: 40, 3: 40, 4: 40}
W_I = {1: 1, 2: 1, 3: 1, 4: 1}

# Zipping
Min_WD_i = {a: f for a, f in zip(I, Min_WD_i)}
Max_WD_i = {a: g for a, g in zip(I, Max_WD_i)}

# Demand Dict
demand_dict1 = {(1, 1): 2, (1, 2): 1, (1, 3): 0, (2, 1): 1, (2, 2): 2, (2, 3): 0, (3, 1): 1, (3, 2): 1, (3, 3): 1, (4, 1): 1, (4, 2): 2, (4, 3): 0,
               (5, 1): 2, (5, 2): 0, (5, 3): 1, (6, 1): 1, (6, 2): 1, (6, 3): 1, (7, 1): 0, (7, 2): 3, (7, 3): 0, (8, 1): 2, (8, 2): 1, (8, 3): 0,
               (9, 1): 0, (9, 2): 3, (9, 3): 0, (10, 1): 1, (10, 2): 1, (10, 3): 1, (11, 1): 3, (11, 2): 0, (11, 3): 0, (12, 1): 0, (12, 2): 2, (12, 3): 1,
               (13, 1): 1, (13, 2): 1, (13, 3): 1, (14, 1): 2, (14, 2): 1, (14, 3): 0}

random.seed(13333)
def generate_cost(num_days, phys):
    cost = {}
    shifts = [1, 2, 3]
    for day in range(1, num_days + 1):
        num_costs = phys
        for shift in shifts[:-1]:
            shift_cost = random.randrange(0, num_costs)
            cost[(day, shift)] = shift_cost
            num_costs -= shift_cost
        cost[(day, shifts[-1])] = num_costs
    return cost

demand_dict = generate_cost(len(T), len(I))

# Parameter
time_Limit = 3600
max_itr = 20
output_len = 98
mue = 1e-4
threshold = 5e-7
eps = 0.1

# **** Compact Solver ****
problem_t0 = time.time()
problem = Problem(data, demand_dict, eps)
problem.buildLinModel()
problem.updateModel()
problem.solveModel()

obj_val_problem = round(problem.model.objval, 3)
time_problem = time.time() - problem_t0
vals_prob = problem.get_final_values()
print(obj_val_problem)



problem_start = Problem(data, demand_dict, eps)
problem_start.model.update()
problem_start.model.optimize()


# **** Column Generation ****
# Prerequisites
modelImprovable = True
reached_max_itr = False

# Get Starting Solutions
problem_start = Problem(data, demand_dict, eps)
problem_start.buildLinModel()
problem_start.model.Params.MIPFocus = 1
problem_start.model.Params.Heuristics = 1
problem_start.model.Params.NoRelHeurTime = 100
problem_start.model.Params.RINS = 10
problem_start.model.Params.MIPGap = 0.99
problem_start.model.update()
problem_start.model.optimize()
start_values_perf = {}
for i in I:
    for t in T:
        for s in K:
            start_values_perf[(i, t, s)] = problem_start.perf[i, t, s].x

start_values_p = {}
for i in I:
    for t in T:
        start_values_p[(i, t)] = problem_start.p[i, t].x

start_values_x = {}
for i in I:
    for t in T:
        for s in K:
            start_values_x[(i, t, s)] = problem_start.x[i, t, s].x

while True:
    # Initialize iterations
    itr = 0
    t0 = time.time()
    last_itr = 0

    # Create empty results lists
    objValHistSP = []
    timeHist = []
    objValHistRMP = []
    avg_rc_hist = []
    avg_sp_time = []
    gap_rc_hist = []

    X_schedules = {}
    for index in I:
        X_schedules[f"Physician_{index}"] = []

    start_values_perf_dict = {}
    for i in I:
        start_values_perf_dict[f"Physician_{i}"] = {(i, t, s): start_values_perf[(i, t, s)] for t in T for s in K}

    Perf_schedules = {}
    for index in I:
        Perf_schedules[f"Physician_{index}"] = [start_values_perf_dict[f"Physician_{index}"]]

    start_values_p_dict = {}
    for i in I:
        start_values_p_dict[f"Physician_{i}"] = {(i, t): start_values_p[(i, t)] for t in T}

    P_schedules = {}
    for index in I:
        P_schedules[f"Physician_{index}"] = [start_values_p_dict[f"Physician_{index}"]]

    start_values_x_dict = {}
    for i in I:
        start_values_x_dict[f"Physician_{i}"] = {(i, t, s): start_values_x[(i, t, s)] for t in T for s in K}

    X1_schedules = {}
    for index in I:
        X1_schedules[f"Physician_{index}"] = [start_values_x_dict[f"Physician_{index}"]]


    master = MasterProblem(data, demand_dict, max_itr, itr, last_itr, output_len, start_values_perf)
    master.buildModel()
    print("*" * (output_len + 2))
    print("*{:^{output_len}}*".format("Restricted Master Problem successfully built!", output_len=output_len))
    print("*" * (output_len + 2))

    # Initialize and solve relaxed model
    master.setStartSolution()
    master.updateModel()
    master.solveRelaxModel()

    # Retrieve dual values
    duals_i0 = master.getDuals_i()
    duals_ts0 = master.getDuals_ts()

    # Start time count
    t0 = time.time()

    while (modelImprovable) and itr < max_itr:
        # Start
        itr += 1

        # Solve RMP
        master.current_iteration = itr + 1
        master.solveRelaxModel()
        objValHistRMP.append(master.model.objval)

        # Get Duals
        duals_i = master.getDuals_i()
        duals_ts = master.getDuals_ts()
        print(f"DualsI: {duals_i}")
        print(f"DualsTs: {duals_ts}")

        # Save current optimality gap
        gap_rc = round(((round(master.model.objval, 3) - round(obj_val_problem, 3)) / round(master.model.objval, 3)), 3)
        gap_rc_hist.append(gap_rc)

        # Solve SPs
        modelImprovable = False
        for index in I:
            # Build SP
            subproblem = Subproblem(duals_i, duals_ts, data, index, itr, eps)
            subproblem.buildModel()

            # Save time to solve SP
            sub_t0 = time.time()
            subproblem.solveModel(time_Limit)
            sub_totaltime = time.time() - sub_t0
            timeHist.append(sub_totaltime)

            # Get optimal values
            optx_values = subproblem.getOptX()
            X_schedules[f"Physician_{index}"].append(optx_values)
            optPerf_values = subproblem.getOptPerf()
            Perf_schedules[f"Physician_{index}"].append(optPerf_values)
            optP_values = subproblem.getOptP()
            P_schedules[f"Physician_{index}"].append(optP_values)
            optx1_values = subproblem.getOptX()
            X1_schedules[f"Physician_{index}"].append(optx1_values)

            # Check if SP is solvable
            status = subproblem.getStatus()
            if status != 2:
                raise Exception("*{:^{output_len}}*".format("Pricing-Problem can not reach optimality!", output_len=output_len))

            # Save ObjVal History
            reducedCost = subproblem.model.objval
            objValHistSP.append(reducedCost)

            # Increase latest used iteration
            last_itr = itr + 1

            # Generate and add columns with reduced cost
            if reducedCost < -threshold:
                Schedules = subproblem.getNewSchedule()
                master.addColumn(index, itr, Schedules)
                master.addLambda(index, itr)
                master.updateModel()
                modelImprovable = True

        # Update Model
        master.updateModel()

        # Calculate Metrics
        avg_rc = sum(objValHistSP) / len(objValHistSP)
        avg_rc_hist.append(avg_rc)
        objValHistSP.clear()

        avg_time = sum(timeHist)/len(timeHist)
        avg_sp_time.append(avg_time)
        timeHist.clear()

        print("*{:^{output_len}}*".format(f"End CG iteration {itr}", output_len=output_len))

        if not modelImprovable:
            master.model.write("Final_LP.sol")
            print("*{:^{output_len}}*".format("", output_len=output_len))
            print("*{:^{output_len}}*".format("No more improvable columns found.", output_len=output_len))
            print("*{:^{output_len}}*".format("", output_len=output_len))
            print("*" * (output_len + 2))

            break

    if modelImprovable and itr == max_itr:
        print("*{:^{output_len}}*".format("More iterations needed. Increase max_itr and restart the process.",
                                          output_len=output_len))
        max_itr *= 2
    else:
        break

print(f" List of objvals relaxed {objValHistRMP}")
# Solve Master Problem with integrality restored
master.finalSolve(time_Limit)
master.model.write("Final_IP.sol")
objValHistRMP.append(master.model.objval)

# Capture total time and objval
total_time_cg = time.time() - t0
final_obj_cg = master.model.objval


print(f" Final IP: {objValHistRMP[-1]}")
print(f" Final LP: {objValHistRMP[-2]}")

gap = ((objValHistRMP[-1]-objValHistRMP[-2])/objValHistRMP[-2])*100
print(f"Gap: {round(gap,4)}%")

# Print Results
printResults(itr, total_time_cg, time_problem, output_len, final_obj_cg, objValHistRMP[-2])

# Plots
plot_obj_val(objValHistRMP, 'obj_val_plot')
plot_avg_rc(avg_rc_hist, 'rc_vals_plot')
#print(Perf_schedules)
performancePlot(plotPerformanceList(master.printLambdas(), P_schedules, I ,max_itr), len(T), 'perf_over_time')

# Rest
#dicts = create_perf_dict(plotPerformanceList(master.printLambdas(), X1_schedules, I ,max_itr), len(I), len(T), len(K))
#print(dicts)