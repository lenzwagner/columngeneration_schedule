from master import *
from sub import *
from standard import *
from plots import *
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
seed = 12345
output_len = 98



optimal_results = {}
gap_results = {}
time_compact = {}
time_cg = {}

for seed in range(100, 201):

    problem = Problem(DataDF, Demand_Dict, gen_alpha(seed))
    problem.buildModel()
    problem.solveModel(time_Limit)
    obj_val_problem = round(problem.model.objval, 3)
    time_problem = round(problem.getTime(), 4)
    vals_prob = problem.get_final_values()


    #### Column Generation
    # CG Prerequisites
    modelImprovable = True
    t0 = time.time()
    itr = 0
    last_itr = 0

    problem_start = Problem(DataDF, Demand_Dict, gen_alpha(seed))
    problem_start.buildModel()
    problem_start.model.Params.MIPGap = 0.9
    problem_start.model.update()
    problem_start.model.optimize()
    start_values = {}
    for i in I_list:
        for t in T_list:
            for s in K_list:
                start_values[(i, t, s)] = problem_start.motivation[i, t, s].x


    # Build & Solve MP
    master = MasterProblem(DataDF, Demand_Dict, max_itr, itr, last_itr, 88, start_values)
    master.buildModel()
    master.setStartSolution()
    master.File2Log()
    master.updateModel()
    master.solveRelaxModel(3600)

    # Get Duals from MP
    duals_i = master.getDuals_i()
    duals_ts = master.getDuals_ts()

    Iter_schedules = {}
    for index in I_list:
        Iter_schedules[f"Nurse_{index}"] = []
    t0 = time.time()

    while (modelImprovable) and itr < max_itr:

        # Lists
        objValHistSP = []
        objValHistRMP = []
        avg_rc_hist = []

        # Start
        itr += 1
        # Solve RMP
        master.current_iteration = itr + 1
        master.solveRelaxModel(3500)
        objValHistRMP.append(master.model.objval)

        # Get Duals
        duals_i = master.getDuals_i()
        duals_ts = master.getDuals_ts()

        # Solve SPs
        modelImprovable = False
        for index in I_list:
            subproblem = Subproblem(duals_i, duals_ts, DataDF, index, 1e6, itr, gen_alpha(seed))
            subproblem.buildModel()
            subproblem.solveModel(time_Limit)

            optx_values = subproblem.getOptX()
            Iter_schedules[f"Nurse_{index}"].append(optx_values)

            status = subproblem.getStatus()
            reducedCost = subproblem.model.objval
            objValHistSP.append(reducedCost)
            if reducedCost < -1e-6:
                Schedules = subproblem.getNewSchedule()
                master.addColumn(index, itr, Schedules)
                master.addLambda(index, itr)
                master.updateModel()
                modelImprovable = True
        master.updateModel()


        avg_rc = sum(objValHistSP) / len(objValHistSP)
        avg_rc_hist.append(avg_rc)
        objValHistSP.clear()


    # Solve MP
    master.finalSolve(time_Limit)
    total_time_cg = time.time() - t0
    final_obj_cg = master.model.objval

    gap_rc = round(((round(master.model.objval, 2) - round(obj_val_problem, 2)) / round(master.model.objval, 2)) * 100, 3)

    if gap_rc > 0:
        gap_rc_value = gap_rc
    else:
        gap_rc_value = 0.0

    def is_Opt(final_obj_cg, obj_val_problem):
        diff = round(final_obj_cg, 3) - round(obj_val_problem, 3)
        if diff == 0:
            is_optimal = 1
        else:
            is_optimal = 0

        return is_optimal

    # Optimality check
    optimal_results[seed] = is_Opt(final_obj_cg, obj_val_problem)
    gap_results[seed] = gap_rc_value

    time_compact[seed] = time_problem
    time_cg[seed] = round(total_time_cg, 2)


# Get Pie-Chart
pie_chart(optimal_results)

# Violin Plots
optBoxplot([value for value in gap_results.values() if value > 1e-8])
violinplots(list(sorted(time_cg.values())), list(sorted(time_compact.values())))
medianplots(list(sorted(time_cg.values())), list(sorted(time_compact.values())))
