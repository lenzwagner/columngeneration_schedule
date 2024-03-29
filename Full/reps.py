import time
from masterproblem import *
import random
from plots import *
from subproblem import *
from gcutil import create_individual_working_list
from compactsolver import *

# Set of indices
I, T, K = list(range(1, 10)), list(range(1, 15)), list(range(1, 4))

# Create Dataframes
data = pd.DataFrame({
    'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
    'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
    'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
})

Max_WD_i = create_individual_working_list(len(I), 5, 6, 5)
Min_WD_i = create_individual_working_list(len(I), 3, 4, 3)
Min_WD_i = {a: f for a, f in zip(I, Min_WD_i)}
Max_WD_i = {a: g for a, g in zip(I, Max_WD_i)}

# Empty Dicts
optimal_results = {}
gap_results = {}
time_compact = {}
time_cg = {}

# Start Reps
for seed in range(110, 121):
    random.seed(seed)
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
    problem_start.model.Params.MIPGap = 0.5
    problem_start.model.update()
    problem_start.model.optimize()
    start_values = {}
    for i in I:
        for t in T:
            for s in K:
                start_values[(i, t, s)] = problem_start.perf[i, t, s].x

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

        Iter_schedules = {}
        for index in I:
            Iter_schedules[f"Physician_{index}"] = []

        master = MasterProblem(data, demand_dict, max_itr, itr, last_itr, output_len, start_values)
        master.buildModel()

        # Initialize and solve relaxed model
        master.setStartSolution()
        master.updateModel()
        master.solveRelaxModel()

        # Retrieve dual values
        duals_i0 = master.getDuals_i()
        duals_ts0 = master.getDuals_ts()

        # Start time count
        t0 = time.time()

        print("*" * (output_len + 2))
        print("*{:^{output_len}}*".format("", output_len=output_len))
        print("*{:^{output_len}}*".format("***** Starting Column Generation *****", output_len=output_len))
        print("*{:^{output_len}}*".format("", output_len=output_len))
        print("*" * (output_len + 2))

        while (modelImprovable) and itr < max_itr:
            # Start
            itr += 1
            print("*{:^{output_len}}*".format(f"Current CG iteration: {itr}", output_len=output_len))
            print("*{:^{output_len}}*".format("", output_len=output_len))

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
                subproblem = Subproblem(duals_i, duals_ts, data, index, itr, eps, Min_WD_i, Max_WD_i)
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


            if not modelImprovable:
                break

        if modelImprovable and itr == max_itr:
            max_itr *= 2
        else:
            break

    # Solve Master Problem with integrality restored
    master.finalSolve(time_Limit)

    # Capture total time and objval
    total_time_cg = time.time() - t0
    print("*{:^{output_len}}*".format(f"Total time: {total_time_cg}", output_len=output_len))

    final_obj_cg = master.model.objval

    gap_rc = round(((round(final_obj_cg, 2) - round(obj_val_problem, 2)) / round(final_obj_cg, 3)) * 100, 2)

    if gap_rc > 0:
        gap_rc_value = gap_rc
    else:
        gap_rc_value = 0.0

    def is_Opt(final_obj_cg, obj_val_problem):
        diff = round(final_obj_cg, 2) - round(obj_val_problem, 2)
        if diff == 0:
            is_optimal = 1
        else:
            is_optimal = 0

        return is_optimal

    # Optimality check
    optimal_results[seed] = is_Opt(final_obj_cg, obj_val_problem)
    gap_results[seed] = gap_rc_value

    time_compact[seed] = time_problem
    time_cg[seed] = round(total_time_cg, 4)


# Get Pie-Chart
pie_chart(optimal_results)

# Violin Plots
optBoxplot([value for value in gap_results.values() if value > 1e-8])
violinplots(list(sorted(time_cg.values())), list(sorted(time_compact.values())))

# Boxplots
medianplots(list(sorted(time_cg.values())), list(sorted(time_compact.values())))