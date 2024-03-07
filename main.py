from gurobipy import *
import gurobipy as gu
import pandas as pd
import os
import time
from plots import plot_obj_val, plot_avg_rc, plot_together
from utilitiy import ListComp
import textwrap
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

# General Parameter
time_Limit = 3600
max_itr = 10
seed = 123

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
        self.model.Params.QCPDual = 1
        for v in self.model.getVars():
            v.setAttr('vtype', 'C')
        self.model.optimize()

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
                    self.model.addConstr(self.motivation_i[i ,t, s, 1] == 0)

    def solveModel(self, timeLimit):
        self.model.setParam('TimeLimit', timeLimit)
        self.model.Params.QCPDual = 1
        self.model.Params.OutputFlag = 0
        self.model.optimize()

    def File2Log(self):
        self.model.Params.LogToConsole = 1
        self.model.Params.LogFile = "./log.txt"

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
        print("*{:^88}*".format(f"Check for quadratic constraints {self.qconstrs}"))

    def finalObj(self):
        obj = self.model.objval
        return obj

    def printLambdas(self):
        return self.model.getAttr("X", self.lmbda)

    def finalSolve(self, timeLimit):
        self.model.setParam('TimeLimit', timeLimit)
        self.model.setAttr("vType", self.lmbda, gu.GRB.INTEGER)
        self.model.update()
        self.model.optimize()
        self.model.write("Final.lp")
        self.model.write("Final.sol")
        if self.model.status == GRB.OPTIMAL:
            print("*" * 90)
            print("*{:^88}*".format(""))
            print("*{:^88}*".format("***** Optimal solution found *****"))
            print("*{:^88}*".format(""))
        else:
            print("*" * 90)
            print("*{:^88}*".format(""))
            print("*{:^88}*".format("***** No optimal solution found *****"))
            print("*{:^88}*".format(""))


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
                self.model.addLConstr(self.mood[i, t] == 1- self.alpha[i, t])
                self.model.addLConstr(quicksum(self.x[i, t, s] for s in self.shifts) == self.y[i, t])
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

    def getStatus(self):
        return self.model.status

    def solveModel(self, timeLimit):
        self.model.setParam('TimeLimit', timeLimit)
        self.model.Params.OutputFlag = 0
        self.model.optimize()

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
        self.model.setParam('TimeLimit', timeLimit)
        self.model.Params.OutputFlag = 0
        self.model.optimize()
        self.t1 = time.time()

    def getTime(self):
        self.time_total = self.t1 - self.t0
        return self.time_total

    def get_final_values(self):
        return {(i, j, k): round(value, 3) for (i, j, k), value in self.model.getAttr("X", self.motivation).items()}

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
print("*" * 90)
print("*{:^88}*".format(""))
print("*{:^88}*".format("Restricted Master Problem successfully built!"))
print("*{:^88}*".format(""))
print("*" * 90)
master.setStartSolution()
master.File2Log()
master.updateModel()
master.solveRelaxModel()
master.model.write("Initial.lp")
master.model.write(f"Sol-{itr}.sol")

# Get Duals from MP
duals_i = master.getDuals_i()
duals_ts = master.getDuals_ts()

print("*{:^88}*".format(""))
print("*{:^88}*".format("***** Column Generation Iteration *****"))
print("*{:^88}*".format(""))
print("*" * 90)
print("*{:^88}*".format(""))

t0 = time.time()
while (modelImprovable) and itr < max_itr:
    # Start
    itr += 1
    print('* Current CG iteration: ', itr)

    # Solve RMP
    master.current_iteration = itr + 1
    print("*{:^88}*".format(f"Current Roster: {master.roster}"))
    master.solveRelaxModel()
    objValHistRMP.append(master.model.objval)
    print("*{:^88}*".format(f"Current RMP ObjVal: {objValHistRMP}"))

    # Get Duals
    duals_i = master.getDuals_i()
    print("*{:^88}*".format(f"Duals in Iteration {itr}: {duals_i}"))
    duals_ts = master.getDuals_ts()

    # Solve SPs
    modelImprovable = False
    for index in I_list:
        subproblem = Subproblem(duals_i, duals_ts, DataDF, index, 1e6, itr, gen_alpha(seed))
        subproblem.buildModel()
        subproblem.solveModel(time_Limit)
        opt_val = subproblem.getNewSchedule()
        opt_val_rounded = {key: round(value, 3) for key, value in opt_val.items()}
        print("*{:^88}*".format(f"Optimal Values Iteration {itr} for SP {index}: {opt_val_rounded}"))

        status = subproblem.getStatus()
        if status != 2:
            raise Exception("*{:^88}*".format("Pricing-Problem can not reach optimality!"))

        reducedCost = subproblem.model.objval
        objValHistSP.append(reducedCost)
        print("*{:^88}*".format(f"Reduced cost in Iteration {itr}: {reducedCost}"))
        if reducedCost < -1e-6:
            Schedules = subproblem.getNewSchedule()
            master.addColumn(index, itr, Schedules)
            master.addLambda(index, itr)
            master.updateModel()
            modelImprovable = True
            print("*{:^88}*".format(f"Reduced-cost < 0 columns found..."))
    master.updateModel()
    master.model.write(f"LP-Iteration-{itr}.lp")


    avg_rc = sum(objValHistSP) / len(objValHistSP)
    avg_rc_hist.append(avg_rc)
    objValHistSP.clear()
    print("*{:^88}*".format(""))
    print("*{:^88}*".format(f"End CG iteration {itr}"))
    print("*{:^88}*".format(""))
    print("*" * 90)

    if not modelImprovable:
        print("*{:^88}*".format(""))
        print("*{:^88}*".format("No more improvable columns found."))
        print("*{:^88}*".format(""))
        print("*" * 90)


# Solve MP
master.finalSolve(time_Limit)
final_obj = master.model.objval

# Print Plots
plot_obj_val(objValHistRMP)
plot_avg_rc(avg_rc_hist)
plot_together(objValHistRMP, avg_rc_hist)

# Results
print("*" * 90)
print("*{:^88}*".format("***** Results *****"))
print("*{:^88}*".format(""))
print("*{:^88}*".format("Total iterations: " + str(itr)))
print("*{:^88}*".format("Total elapsed time: " + str(round((time.time() - t0), 4)) + " seconds"))
print("*{:^88}*".format("Final solution: " + str(round(master.model.objval, 3))))
print("*{:^88}*".format(""))
print("*{:^88}*".format("The optimal solution found by normal solver is: " + str(round(final_obj, 1))))
print("*{:^88}*".format("The optimal solution found by the CG solver is: " + str(round(obj_val_problem, 1))))
if round(final_obj, 1)-round(obj_val_problem, 1) != 0:
    print("*{:^88}*".format("The Optimality-GAP is ",round(final_obj, 1)/(round(final_obj, 1)-round(obj_val_problem, 1)) + "%"))
else:
    print("*{:^88}*".format(f"The Optimality-GAP is {round(final_obj, 1)-round(obj_val_problem, 1)}%: CG provides the global optimal solution!"))
print("*{:^88}*".format(""))
print("*{:^88}*".format(""))
if round((time.time() - t0), 4) < time_problem:
    print("*{:^88}*".format("CG is faster by " + str(time_problem - round((time.time() - t0), 4)) + " seconds"))
elif round((time.time() - t0), 4) > time_problem:
    print("*{:^88}*".format("Normal solver is faster by " + str(round((time.time() - t0), 4) - time_problem) + " seconds"))
else:
    print("*{:^88}*".format("CG and normal solver are equally fast: " + str(time_problem) + " seconds"))
print("*" * 90)

# Check for roster similarity
ListComp()