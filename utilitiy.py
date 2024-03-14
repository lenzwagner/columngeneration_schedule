from itertools import chain
import random

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

def get_nurse_schedules(Iter_schedules, lambdas, I_list):
    nurse_schedules = []
    flat_nurse_schedules = []

    for i in I_list:
        nurse_schedule = []
        for r, schedule in enumerate(Iter_schedules[f"Nurse_{i}"]):
            if (i, r + 2) in lambdas and lambdas[(i, r + 2)] == 1:
                nurse_schedule.append(schedule)
        nurse_schedules.append(nurse_schedule)
        flat_nurse_schedules.extend(nurse_schedule)

    flat = list(chain(*flat_nurse_schedules))
    return flat


def list_diff_sum(list1, list2):
    result = []

    for i in range(len(list1)):
        diff = list1[i] - list2[i]
        if diff == 0:
            result.append(0)
        else:
            result.append(1)

    return result

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

def generate_cost(num_days, phys):
    cost = {}
    shifts = [1, 2, 3]
    for day in range(1, num_days + 2):
        num_costs = phys
        for shift in shifts[:-1]:
            shift_cost = random.randrange(0, num_costs)
            cost[(day, shift)] = shift_cost
            num_costs -= shift_cost
        cost[(day, shifts[-1])] = num_costs
    return cost
