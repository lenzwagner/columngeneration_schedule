from itertools import chain

def ListComp(list1, list2):
    if list1 == list2:
        print("*" * 90)
        print("*{:^88}*".format(f"***** Roster Check *****"))
        print("*{:^88}*".format(f"Roster are the same!"))
        print("*" * 90)
    else:
        print("*" * 90)
        print("*{:^88}*".format(f"***** Roster Check *****"))
        print("*{:^88}*".format(f"Roster are not the same!"))
        print("*" * 90)

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

def is_Opt(seed, final_obj_cg, obj_val_problem):
    is_optimal = {}
    diff = round(final_obj_cg, 3) - round(obj_val_problem, 3)

    if diff == 0:
        is_optimal[(seed)] = 1
    else:
        is_optimal[(seed)] = 0

    print(is_optimal)

    return is_optimal