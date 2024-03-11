def printResults(itr, total_time, time_problem, obj_val_problem, final_obj_cg, nr):
    print("*" * (nr + 2))
    print("*{:^{nr}}*".format("***** Results *****", nr=nr))
    print("*{:^{nr}}*".format("", nr=nr))
    print("*{:^{nr}}*".format("Total CG iterations: " + str(itr), nr=nr))
    print("*{:^{nr}}*".format("Total elapsed time: " + str(round((total_time), 4)) + " seconds", nr=nr))
    print("*{:^{nr}}*".format("Final CG solution: " + str(round(final_obj_cg, 3)), nr=nr))
    print("*{:^{nr}}*".format("", nr=nr))
    print("*{:^{nr}}*".format("The optimal solution found by compact solver is: " + str(round(obj_val_problem, 3)), nr=nr))
    print("*{:^{nr}}*".format("The optimal solution found by the CG solver is: " + str(round(final_obj_cg, 3)), nr=nr))
    gap = round(((round(final_obj_cg, 1)-round(obj_val_problem, 1))/round(final_obj_cg, 1))*100, 3)
    gap_str = f"{gap}%"
    if round(final_obj_cg, 3)-round(obj_val_problem, 3) == 0:
        print("*{:^{nr}}*".format("The Optimality-GAP is " + str(gap_str), nr=nr))
    else:
        print("*{:^{nr}}*".format("The Optimality-GAP is " + str(gap_str), nr=nr))
        print("*{:^{nr}}*".format("CG does not provide the global optimal solution!", nr=nr))
    print("*{:^{nr}}*".format("", nr=nr))
    print("*{:^{nr}}*".format("", nr=nr))
    if round((total_time), 4) < time_problem:
        print("*{:^{nr}}*".format("CG is faster by " + str(time_problem - round((total_time), 4)) + " seconds", nr=nr))
    elif round((total_time), 4) > time_problem:
        print("*{:^{nr}}*".format("Compact solver is faster by " + str(round((total_time), 4) - time_problem) + " seconds", nr=nr))
    else:
        print("*{:^{nr}}*".format("CG and compact solver are equally fast: " + str(time_problem) + " seconds", nr=nr))
    print("*" * (nr + 2))