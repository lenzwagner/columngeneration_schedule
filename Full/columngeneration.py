import pandas as pd
import numpy as np
import time
from masterproblem import *
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
demand_dict = {(1, 1): 2, (1, 2): 1, (1, 3): 0, (2, 1): 1, (2, 2): 2, (2, 3): 0, (3, 1): 1, (3, 2): 1, (3, 3): 1, (4, 1): 1, (4, 2): 2, (4, 3): 0,
               (5, 1): 2, (5, 2): 0, (5, 3): 1, (6, 1): 1, (6, 2): 1, (6, 3): 1, (7, 1): 0, (7, 2): 3, (7, 3): 0, (8, 1): 2, (8, 2): 1, (8, 3): 0,
               (9, 1): 0, (9, 2): 3, (9, 3): 0, (10, 1): 1, (10, 2): 1, (10, 3): 1, (11, 1): 3, (11, 2): 0, (11, 3): 0, (12, 1): 0, (12, 2): 2, (12, 3): 1,
               (13, 1): 1, (13, 2): 1, (13, 3): 1, (14, 1): 2, (14, 2): 1, (14, 3): 0}

# Parameter
time_Limit = 3600
max_itr = 30
seed = 123
output_len = 98
mue = 1e-4
eps = 0.33

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


# **** Column Generation ****
# Prerequisites
modelImprovable = True
reached_max_itr = False
t0 = time.time()
itr = 0
last_itr = 0

# Create empty results lists
objValHistSP = []
timeHist = []
objValHistRMP = []
avg_rc_hist = []
avg_sp_time = []
gap_rc_hist = []

Iter_schedules = {}
for index in I:
    Iter_schedules[f"Physician_{index}"] = []

master = MasterProblem(data, demand_dict, max_itr, itr, last_itr, output_len)
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
        Iter_schedules[f"Physician_{index}"].append(optx_values)

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
        if reducedCost < -1e-6:
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

    if itr == max_itr - 1:
        reached_max_itr = True
        break

    if not modelImprovable:
        print("*{:^{output_len}}*".format("No more improvable columns found.", output_len=output_len))

# Solve Master Problem with integrality restored
master.finalSolve(time_Limit)

# Capture total time and objval
total_time_cg = time.time() - t0
final_obj_cg = master.model.objval


# Define function
def printResults(itr, total_time, time_problem, obj_val_problem, final_obj_cg, nr):
    print("*" * (nr + 2))
    print("*{:^{nr}}*".format("***** Results *****", nr=nr))
    print("*{:^{nr}}*".format("", nr=nr))
    print("*{:^{nr}}*".format("Total Column Generation iterations: " + str(itr), nr=nr))
    print("*{:^{nr}}*".format("Total elapsed time: " + str(round((total_time), 4)) + " seconds", nr=nr))
    print("*{:^{nr}}*".format("Final Column Generation solution: " + str(round(final_obj_cg, 3)), nr=nr))
    print("*{:^{nr}}*".format("", nr=nr))
    print("*{:^{nr}}*".format("The optimal solution found by compact solver is: " + str(round(obj_val_problem, 3)), nr=nr))
    print("*{:^{nr}}*".format("The optimal solution found by the Column Generation solver is: " + str(round(final_obj_cg, 3)), nr=nr))
    gap = round(((round(final_obj_cg, 3)-round(obj_val_problem, 3))/round(final_obj_cg, 1))*100, 3)
    gap_str = f"{gap}%"
    if round(final_obj_cg, 3)-round(obj_val_problem, 3) == 0:
        print("*{:^{nr}}*".format("The Optimality-GAP is " + str(gap_str), nr=nr))
    else:
        print("*{:^{nr}}*".format("The Optimality-GAP is " + str(gap_str), nr=nr))
        print("*{:^{nr}}*".format("Column Generation does not provide the global optimal solution!", nr=nr))
    print("*{:^{nr}}*".format("", nr=nr))
    print("*{:^{nr}}*".format("Solving Times:", nr=nr))
    print("*{:^{nr}}*".format(f"Time Column Generation: {round(total_time, 4)} seconds", nr=nr))
    print("*{:^{nr}}*".format(f"Time Compact Solver: {round(time_problem, 4)} seconds", nr=nr))
    print("*{:^{nr}}*".format("", nr=nr))
    if round((total_time), 4) < time_problem:
        print("*{:^{nr}}*".format(
            "Column Generation is faster by " + str(round((time_problem - round((total_time), 4)), 3)) + " seconds,", nr=nr))
        print("*{:^{nr}}*".format(
            "which is " + str(round((time_problem/ round(total_time, 4)), 3)) + "x times faster.", nr=nr))
    elif round((total_time), 4) > time_problem:
        print("*{:^{nr}}*".format(
            "Compact solver is faster by " + str(round((round((total_time), 4) - time_problem), 3)) + " seconds,", nr=nr))
        print("*{:^{nr}}*".format(
            "which is " + str(round((round(total_time, 4)/ time_problem), 3)) + "x times faster.", nr=nr))
    else:
        print("*{:^{nr}}*".format("Column Generation and compact solver are equally fast: " + str(time_problem) + " seconds", nr=nr))
    print("*" * (nr + 2))

printResults(itr, total_time_cg, time_problem, obj_val_problem, final_obj_cg, output_len)
