from gurobipy import *
import gurobipy as gu
import pandas as pd
import os
import time
import seaborn
import random
import matplotlib.pyplot as plt

clear = lambda: os.system('cls')
clear()

# General Prerequisites
for file in os.listdir():
    if file.endswith('.lp') or file.endswith('.sol') or file.endswith('.txt'):
        os.remove(file)

# Create DF out of Sets
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


random.seed(123)
alpha = {(i, t): round(random.random(), 3) for i in I_list for t in T_list}
print(alpha)


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
            v.setAttr('vtype', 'CONTINOUS')
        self.model.optimize()

    def getDuals_i(self):
        Pi_cons_lmbda = self.model.getAttr("Pi", self.cons_lmbda)
        print(f"Duals_Pi:{Pi_cons_lmbda}")
        return Pi_cons_lmbda

    def getDuals_ts(self):
        Pi_cons_demand = self.model.getAttr("QCPi", self.cons_demand)
        return Pi_cons_demand

    def getObjValues(self):
        obj = self.model.objVal
        return obj

    def updateModel(self):
        self.model.update()

    def addColumn(self, newSchedule):
        self.newvar["motivation_i"] = {}
        for i, t, s, r in newSchedule:
            self.newvar["motivation_i"][(i, t, s, r)] = newSchedule[i, t, s, r]
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

    def getObjVal(self):
        obj = self.model.getObjective()
        value = obj.getValue()
        return value

    def modifyConstraint(self, index, itr):
        self.nurseIndex = index
        self.rosterIndex = itr + 1
        for t in self.days:
            for s in self.shifts:
                qexpr = self.model.getQCRow(self.cons_demand[t, s])
                for i, j, k, l in self.newvar["motivation_i"]:
                    if i != self.nurseIndex:
                        new_coef = self.newvar["motivation_i"][(i, j, k, l)] * self.newvar["motivation_i"][
                            (self.nurseIndex, t, s, self.rosterIndex)]
                        qexpr.add(self.motivation_i[i, j, k, l] * self.lmbda[self.nurseIndex, self.rosterIndex],
                                  new_coef)
                rhs = self.cons_demand[t, s].getAttr('QCRHS')
                sense = self.cons_demand[t, s].getAttr('QCSense')
                name = self.cons_demand[t, s].getAttr('QCName')
                newcon = self.model.addQConstr(qexpr, sense, rhs, name)
                self.model.remove(self.cons_demand[t, s])
                self.cons_demand[t, s] = newcon

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
        print(f"Check for quadratic constraintrs {self.qconstrs}")

    def finalObj(self):
        obj = self.model.getObjective()
        value = obj.getValue()
        return value

    def printLambdas(self):
        return self.model.getAttr("X", self.lmbda)

    def finalSolve(self, timeLimit, EPS):
        self.model.setParam('TimeLimit', timeLimit)
        self.model.setAttr("vType", self.lmbda, gu.GRB.INTEGER)
        self.model.update()
        self.model.optimize()
        self.model.write("Final.lp")
        self.model.write("Final.sol")
        if self.model.status == GRB.OPTIMAL:
            print("Optimal solution found")
            for i in self.nurses:
                for r in self.roster:
                    if self.lmbda[i, r].x == 1:
                        for s in self.shifts:
                            for t in self.days:
                                print(f"Nurse {i}: Motivation {self.motivation_i[i, t, s, r].x} in Shift {s} on day {t}")
        else:
            print("No optimal solution found.")

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

    def getOptValues(self):
        d = self.model.getAttr("X", self.motivation)
        return d

    def getObjVal(self):
        obj = self.model.getObjective()
        value = obj.getValue()
        return value

    def getStatus(self):
        return self.model.status

    def solveModel(self, timeLimit, EPS):
        self.model.setParam('TimeLimit', timeLimit)
        self.model.Params.OutputFlag = 0
        self.model.optimize()
        if self.model.status == gu.GRB.OPTIMAL:
            print("Optimal solution found")
            for i in [self.index]:
                for t in self.days:
                    for s in self.shifts:
                        print(f"Nurse {self.index}: Motivation {self.x[i, t, s].x} in Shift {s} on day {t}")
        else:
            print("No optimal solution found.")


#### Column Generation
# CG Prerequisites
modelImprovable = True
t0 = time.time()
max_itr = 20
itr = 0

# Lists
objValHistSP = []
objValHistRMP = []
avg_rc_hist = []

# Build & Solve MP
master = MasterProblem(DataDF, Demand_Dict, max_itr, itr)
master.buildModel()
print("Restricted master problem successfully built!")
master.setStartSolution()
master.File2Log()
master.updateModel()
master.solveRelaxModel()
master.model.write("Initial.lp")
master.model.write(f"Sol-{itr}.sol")

# Get Duals from MP
duals_i = master.getDuals_i()
duals_ts = master.getDuals_ts()

print('*         *****Column Generation Iteration*****          \n*')
t0 = time.time()
while (modelImprovable) and itr < max_itr:
    # Start
    itr += 1
    print('*Current CG iteration: ', itr)

    # Solve RMP
    master.current_iteration = itr + 1
    print(f"Current Roster: {master.roster}")
    master.solveRelaxModel()
    lambdas = master.printLambdas()
    print(f"Lambdas: {lambdas}")
    objValHistRMP.append(master.getObjValues())
    print('*Current RMP ObjVal: ', objValHistRMP)

    # Get Duals
    duals_i = master.getDuals_i()
    print(f"Duals in Iteration {itr}: {duals_i}")
    duals_ts = master.getDuals_ts()

    # Solve SPs
    modelImprovable = False
    for index in I_list:
        subproblem = Subproblem(duals_i, duals_ts, DataDF, index, 1e6, itr, alpha)
        subproblem.buildModel()
        subproblem.solveModel(3600)
        val = subproblem.getOptValues()
        print(f" Optimal Values {val}")
        status = subproblem.getStatus()
        if status != 2:
            raise Exception("Pricing-Problem can not reach optimality!")
        reducedCost = subproblem.getObjVal()
        objValHistSP.append(reducedCost)
        print('*Reduced cost', reducedCost)
        if reducedCost < -1e-6:
            ScheduleCuts = subproblem.getNewSchedule()
            master.addColumn(ScheduleCuts)
            master.modifyConstraint(index, itr)
            master.addLambda(index, itr)
            master.updateModel()
            modelImprovable = True
            print(f"Reduced-cost < 0 columns found...")
    master.updateModel()
    master.model.write(f"LP-Iteration-{itr}.lp")


    avg_rc = sum(objValHistSP) / len(objValHistSP)
    avg_rc_hist.append(avg_rc)
    objValHistSP.clear()
    print('*End CG iteration: ', itr)

# Solve MP
master.finalSolve(3600)
final_obj = master.finalObj()

# Results
print('*                 *****Results*****                  \n*')
print('*Total iteration: ', itr)
t1 = time.time()
print('*Total elapsed time: ', t1 - t0)
print('*Exact solution:', master.getObjValues())

# Plot
seaborn.set(style='darkgrid')
seaborn.scatterplot(x=list(range(len(avg_rc_hist))), y=avg_rc_hist)
plt.xlabel('Iterations')
plt.xticks(range(0,len(avg_rc_hist)))
plt.ylabel('Objective function value')
title = 'Solution: ' + str(avg_rc_hist[-1])
plt.title(title)
plt.show()
print(objValHistRMP)

print('The relaxed feasible solution found by solver is:', objValHistRMP[-1])
print('The integer feasible solution found by solver is:', final_obj)
print('The GAP is:',final_obj/(final_obj-objValHistRMP[-1]))