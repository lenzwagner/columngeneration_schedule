from gurobipy import *
import gurobipy as gu
import pandas as pd
import os
import time
from plots import plot_obj_val, plot_avg_rc, plot_together
from utilitiy import get_nurse_schedules, ListComp, is_Opt
from results import printResults
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

class MasterProblem:
    def __init__(self, dfData, DemandDF, max_iteration, current_iteration):
        self.iteration = current_iteration
        self.max_iteration = max_iteration
        self.nurses = dfData['I'].dropna().astype(int).unique().tolist()
        self.days = dfData['T'].dropna().astype(int).unique().tolist()
        self.shifts = dfData['K'].dropna().astype(int).unique().tolist()
        self._current_iteration = current_iteration
        self.roster = [i for i in range(1, self.max_iteration + 2)]
        self.rosterinitial = [i for i in range(1, 2)]
        self.demand = DemandDF
        self.model = gu.Model("MasterProblem")
        self.cons_demand = {}
        self.newvar = {}
        self.cons_lmbda = {}

    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.model.update()
        self.generateObjective()
        self.model.update()

    def generateVariables(self):
        self.slack = self.model.addVars(self.days, self.shifts, vtype=gu.GRB.CONTINUOUS, lb=0, name='slack')
        self.motivation_i = self.model.addVars(self.nurses, self.days, self.shifts, self.roster,
                                               vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name='motivation_i')
        self.lmbda = self.model.addVars(self.nurses, self.roster, vtype=gu.GRB.BINARY, lb=0, name='lmbda')

    def generateConstraints(self):
        for i in self.nurses:
            self.cons_lmbda[i] = self.model.addLConstr(1 == gu.quicksum(self.lmbda[i, r] for r in self.rosterinitial), name = "lmb("+str(i)+")")
        for t in self.days:
            for s in self.shifts:
                self.cons_demand[t, s] = self.model.addConstr(
                    gu.quicksum(self.motivation_i[i, t, s, r]*self.lmbda[i, r] for i in self.nurses for r in self.rosterinitial) +
                    self.slack[t, s] >= self.demand[t, s], "demand("+str(t)+","+str(s)+")")
        return self.cons_lmbda, self.cons_demand

    def generateObjective(self):
        self.model.setObjective(gu.quicksum(self.slack[t, s] for t in self.days for s in self.shifts),
                                sense=gu.GRB.MINIMIZE)

    def solveRelaxModel(self):
        try:
            self.model.Params.QCPDual = 1
            for v in self.model.getVars():
                v.setAttr('vtype', 'C')
            self.model.optimize()
        except gu.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))



    def getDuals_i(self):
        Pi_cons_lmbda = self.model.getAttr("Pi", self.cons_lmbda)
        return Pi_cons_lmbda

    def getDuals_ts(self):
        Pi_cons_demand = self.model.getAttr("QCPi", self.cons_demand)
        return Pi_cons_demand

    def updateModel(self):
        self.model.update()

    def setStartSolution(self):
        for i in self.nurses:
            for t in self.days:
                for s in self.shifts:
                    self.model.addConstr(0 == self.motivation_i[i ,t, s, 1])

    def solveModel(self, timeLimit):
        try:
            self.model.setParam('TimeLimit', timeLimit)
            self.model.Params.QCPDual = 1
            self.model.Params.OutputFlag = 0
            self.model.Params.IntegralityFocus = 1
            self.model.Params.FeasibilityTol = 1e-9
            self.model.Params.BarConvTol = 0.0
            self.model.Params.MIPGap = 1e-2
            self.model.optimize()
        except gu.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))



    def File2Log(self):
        self.model.Params.LogToConsole = 1
        self.model.Params.LogFile = "./log_file_cg.log"

    def addColumn(self, index, itr, schedule):
        self.nurseIndex = index
        self.rosterIndex = itr + 1
        for t in self.days:
            for s in self.shifts:
                qexpr = self.model.getQCRow(self.cons_demand[t, s])
                qexpr.add(schedule[self.nurseIndex, t, s, self.rosterIndex] * self.lmbda[self.nurseIndex, self.rosterIndex], 1)
                rhs = self.cons_demand[t, s].getAttr('QCRHS')
                sense = self.cons_demand[t, s].getAttr('QCSense')
                name = self.cons_demand[t, s].getAttr('QCName')
                newcon = self.model.addQConstr(qexpr, sense, rhs, name)
                self.model.remove(self.cons_demand[t, s])
                self.cons_demand[t, s] = newcon
        self.model.update()

    def addLambda(self, index, itr):
        self.nurseIndex = index
        self.rosterIndex = itr + 1
        self.newlmbcoef = 1.0
        current_lmb_cons = self.cons_lmbda[self.nurseIndex]
        expr = self.model.getRow(current_lmb_cons)
        new_lmbcoef = self.newlmbcoef
        expr.add(self.lmbda[self.nurseIndex, self.rosterIndex], new_lmbcoef)
        rhs_lmb = current_lmb_cons.getAttr('RHS')
        sense_lmb = current_lmb_cons.getAttr('Sense')
        name_lmb = current_lmb_cons.getAttr('ConstrName')
        newconlmb = self.model.addLConstr(expr, sense_lmb, rhs_lmb, name_lmb)
        self.model.remove(current_lmb_cons)
        self.cons_lmbda[self.nurseIndex] = newconlmb

    def checkForQuadraticCons(self):
        self.qconstrs = self.model.getQConstrs()
        print("*{:^{output_len}}*".format(f"Check for quadratic constraints {self.qconstrs}", output_len=output_len))
    def finalObj(self):
        obj = self.model.objval
        return obj

    def printLambdas(self):
        return self.model.getAttr("X", self.lmbda)

    def finalSolve(self, timeLimit):
        try:
            self.model.setParam('TimeLimit', timeLimit)
            self.model.Params.IntegralityFocus = 1
            self.model.Params.FeasibilityTol = 1e-9
            self.model.Params.BarConvTol = 0.0
            self.model.Params.MIPGap = 1e-2
            self.model.Params.OutputFlag = 1
            self.model.setAttr("vType", self.lmbda, gu.GRB.BINARY)
            self.model.update()
            self.model.optimize()
            self.model.write("Final.lp")
            self.model.write("Final.sol")
            if self.model.status == GRB.OPTIMAL:
                print("*" * (output_len + 2))
                print("*{:^{output_len}}*".format("***** Optimal solution found *****", output_len=output_len))
                print("*{:^{output_len}}*".format("", output_len=output_len))
            else:
                print("*" * (output_len + 2))
                print("*{:^{output_len}}*".format("***** No optimal solution found *****", output_len=output_len))
                print("*{:^{output_len}}*".format("", output_len=output_len))
        except gu.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))




class Subproblem:
    def __init__(self, duals_i, duals_ts, dfData, i, M, iteration, alpha):
        itr = iteration + 1
        self.days = dfData['T'].dropna().astype(int).unique().tolist()
        self.shifts = dfData['K'].dropna().astype(int).unique().tolist()
        self.duals_i = duals_i
        self.duals_ts = duals_ts
        self.Max = 5
        self.Min = 2
        self.M = M
        self.alpha = alpha
        self.model = gu.Model("Subproblem")
        self.index = i
        self.itr = itr

    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.generateObjective()
        self.model.update()

    def generateVariables(self):
        self.x = self.model.addVars([self.index], self.days, self.shifts, vtype=gu.GRB.BINARY, name='x')
        self.y = self.model.addVars([self.index], self.days, vtype=gu.GRB.BINARY, name='y')
        self.mood = self.model.addVars([self.index], self.days, vtype=gu.GRB.CONTINUOUS, lb=0, name='mood')
        self.motivation = self.model.addVars([self.index], self.days, self.shifts, [self.itr], vtype=gu.GRB.CONTINUOUS, lb=0, name='motivation')

    def generateConstraints(self):
        for i in [self.index]:
            for t in self.days:
                self.model.addLConstr(1 - self.alpha[i, t] == self.mood[i, t])
                self.model.addLConstr(self.y[i, t] == quicksum(self.x[i, t, s] for s in self.shifts))
                self.model.addLConstr(gu.quicksum(self.x[i, t, s] for s in self.shifts) <= 1)
                for s in self.shifts:
                    self.model.addLConstr(
                        self.motivation[i, t, s, self.itr] >= self.mood[i, t] - self.M * (1 - self.x[i, t, s]))
                    self.model.addLConstr(
                        self.motivation[i, t, s, self.itr] <= self.mood[i, t] + self.M * (1 - self.x[i, t, s]))
                    self.model.addLConstr(self.motivation[i, t, s, self.itr] <= self.x[i, t, s])
            for t in range(1, len(self.days) - self.Max + 1):
                self.model.addLConstr(gu.quicksum(self.y[i, u] for u in range(t, t + 1 + self.Max)) <= self.Max)
            self.model.addLConstr(self.Min <= quicksum(self.y[i, t] for t in self.days))


    def generateObjective(self):
        self.model.setObjective(
            0 - gu.quicksum(self.motivation[i, t, s, self.itr] * self.duals_ts[t, s] for i in [self.index] for t in self.days for s in self.shifts) -
            self.duals_i[self.index], sense=gu.GRB.MINIMIZE)

    def getNewSchedule(self):
        return self.model.getAttr("X", self.motivation)

    def getOptX(self):
        vals_opt = self.model.getAttr("X", self.x)
        vals_list = []
        for vals in vals_opt.values():
            vals_list.append(vals)
        return vals_list

    def getStatus(self):
        return self.model.status

    def solveModel(self, timeLimit):
        try:
            self.model.setParam('TimeLimit', timeLimit)
            self.model.Params.OutputFlag = 0
            self.model.Params.IntegralityFocus = 1
            self.model.Params.FeasibilityTol = 1e-9
            self.model.Params.BarConvTol = 0.0
            self.model.Params.MIPGap = 1e-2
            self.model.optimize()
        except gu.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))


#### Normal Solving
class Problem:
    def __init__(self, dfData, DemandDF, alpha):
        self.I = dfData['I'].dropna().astype(int).unique().tolist()
        self.T = dfData['T'].dropna().astype(int).unique().tolist()
        self.K = dfData['K'].dropna().astype(int).unique().tolist()
        self.demand = DemandDF
        self.Max = 5
        self.Min = 2
        self.M = 1e6
        self.alpha = alpha
        self.model = gu.Model("Problems")

    def buildModel(self):
        self.t0 = time.time()
        self.generateVariables()
        self.generateConstraints()
        self.generateObjective()
        self.model.update()

    def generateVariables(self):
        self.slack = self.model.addVars(self.T, self.K, vtype=gu.GRB.CONTINUOUS, lb=0, name='slack')
        self.motivation = self.model.addVars(self.I, self.T, self.K, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name='motivation')
        self.x = self.model.addVars(self.I, self.T, self.K, vtype=gu.GRB.BINARY, name='x')
        self.y = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name='y')
        self.mood = self.model.addVars(self.I, self.T, vtype=gu.GRB.CONTINUOUS, lb=0, name='mood')

    def generateConstraints(self):
        for t in self.T:
            for s in self.K:
                self.model.addConstr(
                    gu.quicksum(self.motivation[i, t, s,] for i in self.I) + self.slack[t, s] >= self.demand[t, s])
        for i in self.I:
            for t in self.T:
                self.model.addLConstr(self.mood[i, t] == 1 - self.alpha[i, t])
                self.model.addLConstr(quicksum(self.x[i, t, s] for s in self.K) == self.y[i, t])
                self.model.addLConstr(gu.quicksum(self.x[i, t, s] for s in self.K) <= 1)
                for s in self.K:
                    self.model.addLConstr(self.motivation[i, t, s] >= self.mood[i, t] - self.M * (1 - self.x[i, t, s]))
                    self.model.addLConstr(self.motivation[i, t, s] <= self.mood[i, t] + self.M * (1 - self.x[i, t, s]))
                    self.model.addLConstr(self.motivation[i, t, s] <= self.x[i, t, s])
            for t in range(1, len(self.T) - self.Max + 1):
                self.model.addLConstr(gu.quicksum(self.y[i, u] for u in range(t, t + 1 + self.Max)) <= self.Max)
            self.model.addLConstr(gu.quicksum(self.y[i, t] for t in self.T) >= self.Min)

    def generateObjective(self):
        self.model.setObjective(gu.quicksum(self.slack[t, s] for t in self.T for s in self.K), sense=gu.GRB.MINIMIZE)

    def solveModel(self, timeLimit):
        try:
            self.model.setParam('TimeLimit', timeLimit)
            self.model.Params.IntegralityFocus = 1
            self.model.Params.FeasibilityTol = 1e-9
            self.model.Params.BarConvTol = 0.0
            self.model.Params.MIPGap = 1e-2
            self.model.optimize()
            self.model.Params.LogFile = "./log_file_compact.log"
            self.t1 = time.time()
        except gu.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))



    def getTime(self):
        self.time_total = self.t1 - self.t0
        return self.time_total

    def get_final_values(self):
        dict = self.model.getAttr("X", self.x)
        liste = list(dict.values())
        final = [0.0 if x == -0.0 else x for x in liste]
        return final

    def get_final_values_dict(self):
        return self.model.getAttr("X", self.x)

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

# Lists
objValHistSP = []
objValHistRMP = []
avg_rc_hist = []

# Build & Solve MP
master = MasterProblem(DataDF, Demand_Dict, max_itr, itr)
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
master.model.write("Initial.lp")
master.model.write(f"Sol-{itr}.sol")

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

    # Solve SPs
    modelImprovable = False
    for index in I_list:
        subproblem = Subproblem(duals_i, duals_ts, DataDF, index, 1e6, itr, gen_alpha(seed))
        subproblem.buildModel()
        subproblem.solveModel(time_Limit)

        optx_values = subproblem.getOptX()
        Iter_schedules[f"Nurse_{index}"].append(optx_values)
        print("*{:^{output_len}}*".format(f"Optimal Values Iteration {itr} for SP {index}: {subproblem.getOptX()}", output_len=output_len))

        status = subproblem.getStatus()
        if status != 2:
            raise Exception("*{:^{output_len}}*".format("Pricing-Problem can not reach optimality!", output_len=output_len))

        reducedCost = subproblem.model.objval
        objValHistSP.append(reducedCost)
        print("*{:^{output_len}}*".format(f"Reduced cost in Iteration {itr}: {reducedCost}", output_len=output_len))
        if reducedCost < -1e-6:
            Schedules = subproblem.getNewSchedule()
            master.addColumn(index, itr, Schedules)
            master.addLambda(index, itr)
            master.updateModel()
            modelImprovable = True
            print("*{:^{output_len}}*".format(f"Reduced-cost < 0 columns found...", output_len=output_len))
    master.updateModel()
    master.model.write(f"LP-Iteration-{itr}.lp")


    avg_rc = sum(objValHistSP) / len(objValHistSP)
    avg_rc_hist.append(avg_rc)
    objValHistSP.clear()
    print("*{:^{output_len}}*".format("", output_len=output_len))
    print("*{:^{output_len}}*".format(f"End CG iteration {itr}", output_len=output_len))
    print("*{:^{output_len}}*".format("", output_len=output_len))
    print("*" * (output_len + 2))

    if not modelImprovable:
        print("*{:^{output_len}}*".format("", output_len=output_len))
        print("*{:^{output_len}}*".format("No more improvable columns found.", output_len=output_len))
        print("*{:^{output_len}}*".format("", output_len=output_len))
        print("*" * (output_len + 2))


# Solve MP
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
plot_obj_val(objValHistRMP)
plot_avg_rc(avg_rc_hist)
plot_together(objValHistRMP, avg_rc_hist)

# Results
printResults(itr, total_time_cg, time_problem, obj_val_problem, final_obj_cg, output_len)

# Roster Check
ListComp(get_nurse_schedules(Iter_schedules, master.printLambdas(), I_list), problem.get_final_values(), output_len)

# Optimality check
is_Opt(seed, final_obj_cg, obj_val_problem, output_len)


print(problem.get_final_values_dict())