import numpy as np
import time
from gcutil import *
import matplotlib.pyplot as plt
import seaborn as sns
from setup import *
from subproblem import *
from compactsolver import *


# Create Dataframes
data = pd.DataFrame({
    'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
    'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
    'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
})

# Parameter
time_Limit = 3600
max_itr = 200
output_len = 98
mue = 1e-4
eps = 0.2
threshold = 5e-7

# **** Compact Solver ****
problem_t0 = time.time()
problem = Problem(data, Demand_Dict, eps, Min_WD_i, Max_WD_i)
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

# Get Starting Solutions
problem_start = Problem(data, Demand_Dict, eps, Min_WD_i, Max_WD_i)
problem_start.buildLinModel()
problem_start.model.Params.MIPFocus = 1
problem_start.model.Params.Heuristics = 1
problem_start.model.Params.NoRelHeurTime = 100
problem_start.model.Params.RINS = 10
problem_start.model.Params.MIPGap = 0.5
problem_start.model.update()
problem_start.model.optimize()
start_values = {}
for i in I:
    for t in T:
        for s in K:
            start_values[(i, t, s)] = problem_start.perf[i ,t, s].x

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

Iter_schedules = {}
for index in I:
    Iter_schedules[f'Physician_{index}'] = []

master = MasterProblem(data, demand_dict, max_itr, itr, last_itr, output_len, 1, 0.1, 0.9, 0.01, start_values)
master.buildModel()
print('*' * (output_len + 2))
print('*{:^{output_len}}*'.format('Restricted Master Problem successfully built!', output_len=output_len))
print('*' * (output_len + 2))

# Initialize and solve relaxed model
master.setStartSolution()
master.updateModel()
master.solveRelaxModel()

# Start time count
t0 = time.time()

while True:
    while (modelImprovable):
        # Start
        itr += 1

        # Solve RMP
        master.current_iteration = itr + 1
        master.solveRelaxModel()
        objValHistRMP.append(master.model.objval)

        # Get Duals
        duals_i = master.getDuals_i()
        duals_ts = master.getDuals_ts()

        # Update delta
        master.updateDeltaMinus(duals_ts)
        master.updateDeltaPlus(duals_ts)
        master.updateZetaPlus()
        master.updateZetaMinus()

        # Save current optimality gap
        gap_rc = round(((round(master.model.objval, 3) - round(obj_val_problem, 3)) / round(master.model.objval, 3)), 3)
        gap_rc_hist.append(gap_rc)

        # Solve SPs
        modelImprovable = False
        for index in I:
            # Build SP
            subproblem = Subproblem(duals_i, duals_ts, data, index, itr, eps, Min_WD_i, Max_WD_i)
            subproblem.buildModel()

            # Save time to solve SP
            sub_t0 = time.time()
            subproblem.solveModel(time_Limit)
            sub_totaltime = time.time() - sub_t0
            timeHist.append(sub_totaltime)

            # Get optimal values
            optx_values = subproblem.getOptX()
            Iter_schedules[f'Physician_{index}'].append(optx_values)

            # Check if SP is solvable
            status = subproblem.getStatus()
            if status != 2:
                raise Exception('*{:^{output_len}}*'.format('Pricing-Problem can not reach optimality!', output_len=output_len))

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

        print('*{:^{output_len}}*'.format(f'End CG iteration {itr}', output_len=output_len))

    if not modelImprovable:
        if all(value < master.zetal for value in master.zeta_plus.values()) and \
                all(value < master.zetal for value in master.zeta_minus.values()):
            print('*{:^{output_len}}*'.format('', output_len=output_len))
            print('*{:^{output_len}}*'.format('Final iteration completed!', output_len=output_len))
            print('*{:^{output_len}}*'.format('', output_len=output_len))
            break
        else:
            print('*{:^{output_len}}*'.format('', output_len=output_len))
            print('*{:^{output_len}}*'.format('No more improvable columns found.', output_len=output_len))
            print('*{:^{output_len}}*'.format('Updating Parameters.....', output_len=output_len))
            print('*{:^{output_len}}*'.format('', output_len=output_len))
            print('*{:^{output_len}}*'.format('Parameter succesfully updated!', output_len=output_len))
            print('*{:^{output_len}}*'.format('', output_len=output_len))
            master.updateDeltaPlus(duals_ts)
            master.updateDeltaMinus(duals_ts)
            master.updateZetaPlus()
            print('*{:^{output_len}}*'.format(f'Old zeta {master.zeta_plus}', output_len=output_len))
            master.updateZetaMinus()
            print('*{:^{output_len}}*'.format(f'New zeta {master.zeta_plus}', output_len=output_len))

            modelImprovable = True
            continue


# Solve Master Problem with integrality restored
master.finalSolve(time_Limit)
master.model.write('final.lp')
master.model.write('final.sol')


# Capture total time and objval
total_time_cg = time.time() - t0
final_obj_cg = master.model.objval


# Define function
def printResults(itr, total_time, time_problem, obj_val_problem, final_obj_cg, nr):
    print('*' * (nr + 2))
    print('*{:^{nr}}*'.format('***** Results *****', nr=nr))
    print('*{:^{nr}}*'.format('', nr=nr))
    print('*{:^{nr}}*'.format('Total Column Generation iterations: ' + str(itr), nr=nr))
    print('*{:^{nr}}*'.format('Total elapsed time: ' + str(round((total_time), 4)) + ' seconds', nr=nr))
    print('*{:^{nr}}*'.format('Final Column Generation solution: ' + str(round(final_obj_cg, 3)), nr=nr))
    print('*{:^{nr}}*'.format('', nr=nr))
    print('*{:^{nr}}*'.format('The optimal solution found by compact solver is: ' + str(round(obj_val_problem, 3)), nr=nr))
    print('*{:^{nr}}*'.format('The optimal solution found by the Column Generation solver is: ' + str(round(final_obj_cg, 3)), nr=nr))
    gap = round(((round(final_obj_cg, 3)-round(obj_val_problem, 3))/round(final_obj_cg, 1))*100, 3)
    gap_str = f'{gap}%'
    if round(final_obj_cg, 3)-round(obj_val_problem, 3) == 0:
        print('*{:^{nr}}*'.format('The Optimality-GAP is ' + str(gap_str), nr=nr))
    else:
        print('*{:^{nr}}*'.format('The Optimality-GAP is ' + str(gap_str), nr=nr))
        print('*{:^{nr}}*'.format('Column Generation does not provide the global optimal solution!', nr=nr))
    print('*{:^{nr}}*'.format('', nr=nr))
    print('*{:^{nr}}*'.format('Solving Times:', nr=nr))
    print('*{:^{nr}}*'.format(f'Time Column Generation: {round(total_time, 4)} seconds', nr=nr))
    print('*{:^{nr}}*'.format(f'Time Compact Solver: {round(time_problem, 4)} seconds', nr=nr))
    print('*{:^{nr}}*'.format('', nr=nr))
    if round((total_time), 4) < time_problem:
        print('*{:^{nr}}*'.format(
            'Column Generation is faster by ' + str(round((time_problem - round((total_time), 4)), 3)) + ' seconds,', nr=nr))
        print('*{:^{nr}}*'.format(
            'which is ' + str(round((time_problem/ round(total_time, 4)), 3)) + 'x times faster.', nr=nr))
    elif round((total_time), 4) > time_problem:
        print('*{:^{nr}}*'.format(
            'Compact solver is faster by ' + str(round((round((total_time), 4) - time_problem), 3)) + ' seconds,', nr=nr))
        print('*{:^{nr}}*'.format(
            'which is ' + str(round((round(total_time, 4)/ time_problem), 3)) + 'x times faster.', nr=nr))
    else:
        print('*{:^{nr}}*'.format('Column Generation and compact solver are equally fast: ' + str(time_problem) + ' seconds', nr=nr))
    print('*' * (nr + 2))

printResults(itr, total_time_cg, time_problem, obj_val_problem, final_obj_cg, output_len)

def plot_obj_val(objValHistRMP):
    sns.set(style='darkgrid')
    sns.scatterplot(x=list(range(len(objValHistRMP))), y=objValHistRMP, marker='o')
    sns.lineplot(x=list(range(len(objValHistRMP))), y=objValHistRMP)
    plt.xlabel('CG Iterations')
    plt.xticks(range(0, len(objValHistRMP)))
    plt.ylabel('Objective function value')
    title = 'Optimal objective value: ' + str(round(objValHistRMP[-1], 2))
    plt.title(title)
    plt.show()

def plot_avg_rc(avg_rc_hist):
    sns.set(style='darkgrid')
    sns.scatterplot(x=list(range(1, len(avg_rc_hist) + 1)), y=avg_rc_hist, marker='o')
    sns.lineplot(x=list(range(1, len(avg_rc_hist) + 1)), y=avg_rc_hist)
    plt.xlabel('CG Iterations')
    plt.xticks(range(1, len(avg_rc_hist)+1))
    plt.ylabel('Reduced Cost')
    title = 'Final reduced cost: ' + str(round(avg_rc_hist[-1], 2))
    plt.title(title)
    plt.show()

plot_obj_val(objValHistRMP)
plot_avg_rc(avg_rc_hist)