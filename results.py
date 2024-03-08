# Results

def printResults(itr, total_time, time_problem, obj_val_problem, final_obj_cg):
    print("*" * 90)
    print("*{:^88}*".format("***** Results *****"))
    print("*{:^88}*".format(""))
    print("*{:^88}*".format("Total CG iterations: " + str(itr)))
    print("*{:^88}*".format("Total elapsed time: " + str(round((total_time), 4)) + " seconds"))
    print("*{:^88}*".format("Final CG solution: " + str(round(final_obj_cg, 3))))
    print("*{:^88}*".format(""))
    print("*{:^88}*".format("The optimal solution found by compact solver is: " + str(round(obj_val_problem, 3))))
    print("*{:^88}*".format("The optimal solution found by the CG solver is: " + str(round(final_obj_cg, 3))))
    gap = round(((round(final_obj_cg, 1)-round(obj_val_problem, 1))/round(final_obj_cg, 1))*100, 3)
    gap_str = f"{gap}%"
    if round(final_obj_cg, 3)-round(obj_val_problem, 3) == 0:
        print("*{:^88}*".format("The Optimality-GAP is " + str(gap_str)))
    else:
        print("*{:^88}*".format("The Optimality-GAP is " + str(gap_str)))
        print("*{:^88}*".format("CG does not provide the global optimal solution!"))
    print("*{:^88}*".format(""))
    print("*{:^88}*".format(""))
    if round((total_time), 4) < time_problem:
        print("*{:^88}*".format("CG is faster by " + str(time_problem - round((total_time), 4)) + " seconds"))
    elif round((total_time), 4) > time_problem:
        print("*{:^88}*".format("Compact solver is faster by " + str(round((total_time), 4) - time_problem) + " seconds"))
    else:
        print("*{:^88}*".format("CG and compact solver are equally fast: " + str(time_problem) + " seconds"))
    print("*" * 90)
