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