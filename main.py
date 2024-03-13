import plotly.io as pio
import pandas as pd
import os
import time
from plots import plot_obj_val, plot_avg_rc, plot_together, optimalityplot, visualize_schedule
from utilitiy import get_nurse_schedules, ListComp, is_Opt, remove_vars
from results import printResults
from standard import Problem
from master import MasterProblem
from sub import Subproblem
import random

clear = lambda: os.system('cls')
clear()

# General Prerequisites
for file in os.listdir():
    if file.endswith('.lp') or file.endswith('.sol') or file.endswith('.txt'):
        os.remove(file)

# Create Dataframes
I_list = [1, 2, 3]
T_list = [1, 2, 3, 4, 5, 6, 7]
K_list = [1, 2, 3]
I_list1 = pd.DataFrame(I_list, columns=['I'])
T_list1 = pd.DataFrame(T_list, columns=['T'])
K_list1 = pd.DataFrame(K_list, columns=['K'])
DataDF = pd.concat([I_list1, T_list1, K_list1], axis=1)
Demand_Dict = {(1, 1): 2, (1, 2): 1, (1, 3): 0, (2, 1): 1, (2, 2): 2, (2, 3): 0, (3, 1): 1, (3, 2): 1, (3, 3): 1,
               (4, 1): 1, (4, 2): 2, (4, 3): 0, (5, 1): 2, (5, 2): 0, (5, 3): 1, (6, 1): 1, (6, 2): 1, (6, 3): 1,
               (7, 1): 0, (7, 2): 3, (7, 3): 0}

# Generate Alpha
def gen_alpha(seed):
    random.seed(seed)
    alpha = {(i, t): round(random.random(), 3) for i in I_list for t in T_list}
    return alpha

def get_alpha_lists(I_list, alpha_dict):
  alpha_lists = {}
  for i in I_list:
    alpha_list = []
    for t in T_list:
      alpha_list.append(alpha_dict[(i,t)])
    alpha_lists[f"Nurse_{i}"] = alpha_list

  return alpha_lists


# General Parameter
time_Limit = 3600
max_itr = 10
seed = 123
output_len = 98
mue = 1e-4


### Compact Solver ###
problem = Problem(DataDF, Demand_Dict, gen_alpha(seed))
problem.buildModel()
problem.solveModel(time_Limit)
obj_val_problem = round(problem.model.objval, 3)
time_problem = round(problem.getTime(), 4)
vals_prob = problem.get_final_values()


#### Column Generation ###
# CG Prerequisites
modelImprovable = True
t0 = time.time()
itr = 0
last_itr = 0

# Lists
objValHistSP = []
timeHist = []
objValHistRMP = []
avg_rc_hist = []
avg_sp_time = []
gap_rc_hist = []

# Build & Solve MP
master = MasterProblem(DataDF, Demand_Dict, max_itr, itr, last_itr, output_len)
master.buildModel()
print("*" * (output_len + 2))
print("*{:^{output_len}}*".format("", output_len=output_len))
print("*{:^{output_len}}*".format("Restricted Master Problem successfully built!", output_len=output_len))
print("*{:^{output_len}}*".format("", output_len=output_len))
print("*" * (output_len + 2))
master.setStartSolution()
master.File2Log()
master.updateModel()
master.solveRelaxModel()
#master.model.write("Initial.lp")
#master.model.write(f"Sol-{itr}.sol")

# Get Duals from MP
duals_i = master.getDuals_i()
duals_ts = master.getDuals_ts()

print("*" * (output_len + 2))
print("*{:^{output_len}}*".format("", output_len=output_len))
print("*{:^{output_len}}*".format("***** Starting Column Generation *****", output_len=output_len))
print("*{:^{output_len}}*".format("", output_len=output_len))
print("*" * (output_len + 2))
print("*{:^{output_len}}*".format("", output_len=output_len))

Iter_schedules = {}
for index in I_list:
    Iter_schedules[f"Nurse_{index}"] = []
t0 = time.time()

while (modelImprovable) and itr < max_itr:
    # Start
    itr += 1
    print("*{:^{output_len}}*".format(f"Current CG iteration: {itr}", output_len=output_len))
    # Solve RMP
    master.current_iteration = itr + 1
    master.solveRelaxModel()
    objValHistRMP.append(master.model.objval)
    print("*{:^{output_len}}*".format(f"Current RMP ObjVal: {objValHistRMP}", output_len=output_len))

    # Get Duals
    duals_i = master.getDuals_i()
    print("*{:^{output_len}}*".format(f"Duals in Iteration {itr}: {duals_i}", output_len=output_len))
    duals_ts = master.getDuals_ts()

    gap_rc = round(((round(master.model.objval, 3) - round(obj_val_problem, 3)) / round(master.model.objval, 3)), 3)
    gap_rc_hist.append(gap_rc)

    # Solve SPs
    modelImprovable = False
    for index in I_list:
        subproblem = Subproblem(duals_i, duals_ts, DataDF, index, 1e6, itr, gen_alpha(seed))
        subproblem.buildModel()

        sub_t0 = time.time()
        subproblem.solveModel(time_Limit)
        sub_totaltime = time.time() - sub_t0
        timeHist.append(sub_totaltime)

        optx_values = subproblem.getOptX()
        Iter_schedules[f"Nurse_{index}"].append(optx_values)
        print("*{:^{output_len}}*".format(f"Optimal Values Iteration {itr} for SP {index}: {subproblem.getOptX()}", output_len=output_len))

        status = subproblem.getStatus()
        if status != 2:
            raise Exception("*{:^{output_len}}*".format("Pricing-Problem can not reach optimality!", output_len=output_len))

        reducedCost = subproblem.model.objval
        objValHistSP.append(reducedCost)
        last_itr = itr + 1
        print("*{:^{output_len}}*".format(f"Reduced cost in Iteration {itr}: {reducedCost}", output_len=output_len))
        if reducedCost < -1e-6:
            Schedules = subproblem.getNewSchedule()
            master.addColumn(index, itr, Schedules)
            master.addLambda(index, itr)
            master.updateModel()
            modelImprovable = True
            print("*{:^{output_len}}*".format(f"Reduced-cost < 0 columns found...", output_len=output_len))
    master.updateModel()
    #master.model.write(f"LP-Iteration-{itr}.lp")

    avg_rc = sum(objValHistSP) / len(objValHistSP)
    avg_rc_hist.append(avg_rc)
    objValHistSP.clear()

    avg_time = sum(timeHist)/len(timeHist)
    avg_sp_time.append(avg_time)
    timeHist.clear()

    print("*{:^{output_len}}*".format("", output_len=output_len))
    print("*{:^{output_len}}*".format(f"End CG iteration {itr}", output_len=output_len))
    print("*{:^{output_len}}*".format("", output_len=output_len))
    print("*" * (output_len + 2))

    if not modelImprovable:
        print("*{:^{output_len}}*".format("", output_len=output_len))
        print("*{:^{output_len}}*".format("No more improvable columns found.", output_len=output_len))
        print("*{:^{output_len}}*".format("", output_len=output_len))
        print("*" * (output_len + 2))


# Remove Variables
remove_vars(master, I_list, T_list, K_list, last_itr, max_itr)
master.finalSolve(time_Limit)
total_time_cg = time.time() - t0
final_obj_cg = master.model.objval
lambda_values = master.printLambdas()
for i in master.nurses:
   for r in master.roster:
      if lambda_values[(i,r)] == 1:
          print("*{:^{output_len}}*".format(f"For nurse {i}, Iteration {r-1} is used.", output_len=output_len))
print("*{:^{output_len}}*".format("", output_len=output_len))

# Print Plots
plot_obj_val(objValHistRMP, 'plot_obj')
plot_avg_rc(avg_rc_hist, 'plot_rc')
plot_together(objValHistRMP, avg_rc_hist, 'plot_together')
optimalityplot(objValHistRMP, gap_rc_hist, last_itr, 'optimality_plot')


# Results
printResults(itr, total_time_cg, time_problem, obj_val_problem, final_obj_cg, output_len)

# Roster Check
ListComp(get_nurse_schedules(Iter_schedules, master.printLambdas(), I_list), problem.get_final_values(), output_len)

# Optimality check
is_Opt(seed, final_obj_cg, obj_val_problem, output_len)

# SchedulePlot
fig = visualize_schedule(problem.get_final_values_dict(), len(T_list), round(final_obj_cg, 3))
pio.write_image(fig, f'G:/Meine Ablage/Doktor/Dissertation/Paper 1/Data/Pics/physician_schedules.png',
                scale=1, width=1000, height=800,
                engine='kaleido')

print(f"Average time per SP: {avg_sp_time}")