from masterproblem import *
import time
from plots import *
from setup import *
from gcutil import *
from subproblem import *
from compactsolver import Problem

# **** Prerequisites ****
# Create Dataframes
data = pd.DataFrame({
    'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
    'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
    'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
})

# Remove unused files
for file in os.listdir():
    if file.endswith('.lp') or file.endswith('.sol') or file.endswith('.txt'):
        os.remove(file)

# Parameter
random.seed(13338)
time_Limit = 3600
max_itr = 20
output_len = 98
mue = 1e-4
threshold = 5e-7
eps = 0.1

# Demand Dict
demand_dict = generate_cost(len(T), len(I), len(K))


# **** Compact Solver ****
problem_t0 = time.time()
problem = Problem(data, demand_dict, eps, Min_WD_i, Max_WD_i)
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
problem_start = Problem(data, demand_dict, eps, Min_WD_i, Max_WD_i)
problem_start.buildLinModel()
problem_start.model.Params.MIPFocus = 1
problem_start.model.Params.Heuristics = 1
problem_start.model.Params.NoRelHeurTime = 100
problem_start.model.Params.RINS = 10
problem_start.model.Params.MIPGap = 0.4
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
    sum_rc_hist = []
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

        # Get and Print Duals
        duals_i = master.getDuals_i()
        duals_ts = master.getDuals_ts()
        #print(f"DualsI: {duals_i}")
        #print(f"DualsTs: {duals_ts}")

        # Save current optimality gap
        gap_rc = round(((round(master.model.objval, 3) - round(obj_val_problem, 3)) / round(master.model.objval, 3)), 3)
        gap_rc_hist.append(gap_rc)

        # Solve SPs
        modelImprovable = False
        for index in I:
            print("*{:^{output_len}}*".format(f"Begin Column Generation Iteration {itr}", output_len=output_len))

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
        sum_rc = sum(objValHistSP)
        avg_rc_hist.append(avg_rc)
        sum_rc_hist.append(sum_rc)
        objValHistSP.clear()

        avg_time = sum(timeHist)/len(timeHist)
        avg_sp_time.append(avg_time)
        timeHist.clear()

        print("*{:^{output_len}}*".format(f"End Column Generation Iteration {itr}", output_len=output_len))

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

# Calculate Gap
# Relative to the lower bound (best possible achievable solution)
gap = ((objValHistRMP[-1]-objValHistRMP[-2])/objValHistRMP[-2])*100

# Lagragian Bound
# Only yields feasible results if the SPs are solved to optimality
lagranigan_bound = round((objValHistRMP[-2] + sum_rc_hist[-1]), 3)

# Print Results
printResults(itr, total_time_cg, time_problem, output_len, final_obj_cg, objValHistRMP[-2], lagranigan_bound, obj_val_problem, eps)

# Plots
plot_obj_val(objValHistRMP, 'obj_val_plot')
plot_avg_rc(avg_rc_hist, 'rc_vals_plot')
performancePlot(plotPerformanceList(master.printLambdas(), P_schedules, I ,max_itr), len(T), len(I), 'perf_over_time')
