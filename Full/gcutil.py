from itertools import chain
import random

# **** Print Results Table ****
def printResults(itr, total_time, time_problem, nr, optimal_ip, optimal_lp, lagranigan_bound, compact_obj, step):
    lb = analytical_lb(optimal_lp, step, optimal_ip)
    gap_percentage = round((optimal_ip - compact_obj) / compact_obj, 2) * 100
    gap_percentage_str = str(gap_percentage) if gap_percentage != -0.0 else "0.0"

    print("*" * (nr + 2))
    print("*{:^{nr}}*".format("******* Results *******", nr=nr))
    print("*{:^{nr}}*".format("", nr=nr))
    print("*{:^{nr}}*".format("Total Column Generation iterations: " + str(itr), nr=nr))
    print("*{:^{nr}}*".format("Total elapsed time: " + str(round((total_time), 4)) + " seconds", nr=nr))
    print("*{:^{nr}}*".format("Final Integer Column Generation solution: " + str(round(optimal_ip, 4)), nr=nr))
    print("*{:^{nr}}*".format("Final Compact solution: " + str(round(compact_obj, 4)), nr=nr))
    print("*{:^{nr}}*".format("IP-Optimality Gap: " + gap_percentage_str+ "%", nr=nr))

    print("*{:^{nr}}*".format("", nr=nr))
    print("*{:^{nr}}*".format("The LP Relaxation (Lower Bound) is: " + str(round(optimal_lp, 4)), nr=nr))
    print("*{:^{nr}}*".format("The Analytical Lower Bound is: " + str(round(lb, 4)), nr=nr))
    print("*{:^{nr}}*".format("The Lagrangian Bound is: " + str(round(lagranigan_bound, 4)), nr=nr))
    gap = round((((optimal_ip-optimal_lp) / optimal_lp) * 100),3)
    gap_str = f"{gap}%"
    if gap == 0:
        print("*{:^{nr}}*".format("LP-Optimality GAP: " + str(gap_str), nr=nr))
    else:
        print("*{:^{nr}}*".format("LP-Optimality GAP: " + str(gap_str), nr=nr))
        print("*{:^{nr}}*".format("Column Generation does not prove or provide the global optimal solution!", nr=nr))
    print("*{:^{nr}}*".format("", nr=nr))
    print("*{:^{nr}}*".format("Solving Times:", nr=nr))
    print("*{:^{nr}}*".format(f"Time Column Generation: {round(total_time, 4)} seconds", nr=nr))
    print("*{:^{nr}}*".format(f"Time Compact Solver: {round(time_problem, 4)} seconds", nr=nr))
    print("*{:^{nr}}*".format("", nr=nr))
    if round((total_time), 4) < time_problem:
        print("*{:^{nr}}*".format(
            "Column Generation is faster by " + str(round((time_problem - round((total_time), 4)), 4)) + " seconds,", nr=nr))
        print("*{:^{nr}}*".format(
            "which is " + str(round((time_problem/ round(total_time, 4)), 3)) + "x times faster.", nr=nr))
    elif round((total_time), 4) > time_problem:
        print("*{:^{nr}}*".format(
            "Compact solver is faster by " + str(round((round((total_time), 4) - time_problem), 4)) + " seconds,", nr=nr))
        print("*{:^{nr}}*".format(
            "which is " + str(round((round(total_time, 4)/ time_problem), 4)) + "x times faster.", nr=nr))
    else:
        print("*{:^{nr}}*".format("Column Generation and compact solver are equally fast: " + str(time_problem) + " seconds", nr=nr))
    print("*" * (nr + 2))
    return gap


# **** Compare Roster ****
def ListComp(list1, list2, num):
    if list1 == list2:
        print("*" * (num + 2))
        print("*{:^{num}}*".format(f"***** Roster Check *****", num = num))
        print("*{:^{num}}*".format(f"Roster are the same!", num = num))
        print("*" * (num + 2))
    else:
        print("*" * (num + 2))
        print("*{:^{num}}*".format(f"***** Roster Check *****", num = num))
        print("*{:^{num}}*".format(f"Roster are not the same!", num = num))
        print("*" * (num + 2))

# **** Get x-values ****
def get_physician_schedules(Iter_schedules, lambdas, I):
    physician_schedules = []
    flat_physician_schedules = []

    for i in I:
        physician_schedule = []
        for r, schedule in enumerate(Iter_schedules[f"Physician_{i}"]):
            if (i, r + 2) in lambdas and lambdas[(i, r + 2)] == 1:
                physician_schedule.append(schedule)
        physician_schedules.append(physician_schedule)
        flat_physician_schedules.extend(physician_schedule)

    flat_x = list(chain(*flat_physician_schedules))
    return flat_x

# **** Get perf-values ****
def get_physician_perf_schedules(Iter_perf_schedules, lambdas, I):
    physician_schedules = []
    flat_physician_schedules = []

    for i in I:
        physician_schedule = []
        for r, schedule in enumerate(Iter_perf_schedules[f"Physician_{i}"]):
            if (i, r + 1) in lambdas and lambdas[(i, r + 1)] == 1:
                physician_schedule.append(schedule)
        physician_schedules.append(physician_schedule)
        flat_physician_schedules.extend(physician_schedule)

    flat_perf = list(chain(*flat_physician_schedules))
    return flat_perf


def get_nurse_schedules(Iter_schedules, lambdas, I_list):
    nurse_schedules = []
    flat_nurse_schedules = []

    for i in I_list:
        nurse_schedule = []
        for r, schedule in enumerate(Iter_schedules[f"Physician_{i}"]):
            if (i, r + 1) in lambdas and lambdas[(i, r + 1)] == 1:
                nurse_schedule.append(schedule)
        nurse_schedules.append(nurse_schedule)
        flat_nurse_schedules.extend(nurse_schedule)

    flat = list(chain(*flat_nurse_schedules))
    return flat

# **** List comparison ****
def list_diff_sum(list1, list2):
    result = []

    for i in range(len(list1)):
        diff = list1[i] - list2[i]
        if diff == 0:
            result.append(0)
        else:
            result.append(1)

    return result

# **** Optimality Check ****
def is_Opt(seed, final_obj_cg, obj_val_problem, nr):
    is_optimal = {}
    diff = round(final_obj_cg, 3) - round(obj_val_problem, 3)

    if diff == 0:
        is_optimal[(seed)] = 1
    else:
        is_optimal[(seed)] = 0

    print("*" * (nr + 2))
    print("*{:^{nr}}*".format("Is optimal?", nr=nr))
    print("*{:^{nr}}*".format("1: Yes ", nr=nr))
    print("*{:^{nr}}*".format("0: No", nr=nr))
    print("*{:^{nr}}*".format("", nr=nr))
    print("*{:^{nr}}*".format(f" {is_optimal}", nr=nr))
    print("*" * (nr + 2))

    return is_optimal

# **** Remove unnecessary variables ****
def remove_vars(master, I_list, T_list, K_list, last_itr, max_itr):
    for i in I_list:
        for t in T_list:
            for s in K_list:
                for r in range(last_itr + 1, max_itr + 2):
                    var_name = f"motivation_i[{i},{t},{s},{r}]"
                    var = master.model.getVarByName(var_name)
                    master.model.remove(var)
                    master.model.update()

def create_demand_dict(num_days, total_demand):
    demand_dict = {}

    for day in range(1, num_days + 1):
        remaining_demand = total_demand
        shifts = [0, 0, 0]

        while remaining_demand > 0:
            shift_idx = random.randint(0, 2)
            shift_demand = min(remaining_demand, random.randint(0, remaining_demand))
            shifts[shift_idx] += shift_demand
            remaining_demand -= shift_demand

        for i in range(3):
            shifts[i] = round(shifts[i])
            demand_dict[(day, i + 1)] = shifts[i]

    return demand_dict

# **** Generate random pattern ****
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


def plotPerformanceList(dicts, dict_phys, I, max_itr):
    final_list = []

    for i in I:
        r_selected = None
        for r in range(1, max_itr + 2):

            if dicts.get((i, r)) == 1.0:
                r_selected = r - 1
                break

        if r_selected is not None:
            person_key = f'Physician_{i}'
            dict_selected = dict_phys[person_key][r_selected]
            final_list.extend(list(dict_selected.values()))

    return final_list

def create_perf_dict(lst, index, days, shift):
    sublist_length = len(lst) // index
    sublists = [lst[i * sublist_length:(i + 1) * sublist_length] for i in range(index)]
    result_dict = {}

    for i, sublist in enumerate(sublists):
        for d in range(1, days + 1):
            for s in range(1, shift + 1):
                index_key = (i + 1, d, s)

                value = sublist[(d - 1) * shift + (s - 1)]

                result_dict[index_key] = value

    return result_dict

def create_individual_working_list(phys, min_val, max_val, mean_val):
    random_list = []

    for _ in range(phys):
        values = list(range(min_val, max_val + 1))
        probs = [1 / (abs(val - mean_val) + 1) for val in values]
        norm_probs = [prob / sum(probs) for prob in probs]

        random_value = random.choices(values, weights=norm_probs)[0]

        random_list.append(random_value)

    return random_list

def analytical_lb(optimal_lp, step, optimal_ip):
    current_value = optimal_ip
    while current_value > optimal_lp:
        current_value -= step
        if current_value <= optimal_lp:
            return current_value + step
    return optimal_ip

