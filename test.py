from gurobipy import *
import gurobipy as gu
import pandas as pd
import itertools
import time
import matplotlib.pyplot as plt

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


class MasterProblem:
    def __init__(self, dfData, DemandDF, iteration):
        self.iteration = iteration
        self.physicians = dfData['I'].dropna().astype(int).unique().tolist()
        self.days = dfData['T'].dropna().astype(int).unique().tolist()
        self.shifts = dfData['K'].dropna().astype(int).unique().tolist()
        self.roster = list(range(1, self.iteration + 2))
        self.demand = DemandDF
        self.model = gu.Model("MasterProblem")
        self.cons_demand = {}
        self.cons_lmbda = {}

    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.generateObjective()
        self.setStartSolution()
        self.model.update()

    def generateVariables(self):
        self.slack = self.model.addVars(self.days, self.shifts, vtype=gu.GRB.CONTINUOUS, lb=0, name='slack')
        self.motivation_i = self.model.addVars(self.physicians, self.days, self.shifts, self.roster,
                                               vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name='motivation_i')
        self.lmbda = self.model.addVars(self.physicians, self.roster, vtype=gu.GRB.BINARY, lb=0, name='lmbda')

    def generateConstraints(self):
        for i in self.physicians:
            self.cons_lmbda[i] = self.model.addLConstr(gu.quicksum(self.lmbda[i, r] for r in self.roster) == 1)
        for t in self.days:
            for s in self.shifts:
                self.cons_demand[t, s] = self.model.addConstr(
                    gu.quicksum(self.motivation_i[i, t, s, r] for i in self.physicians for r in self.roster) +
                    self.slack[t, s] >= self.demand[t, s])
        return self.cons_lmbda, self.cons_demand

    def generateObjective(self):
        self.model.setObjective(gu.quicksum(self.slack[t, s] for t in self.days for s in self.shifts),
                                sense=gu.GRB.MINIMIZE)

    def solveRelaxModel(self):
        for v in self.model.getVars():
            v.setAttr('vtype', 'C')
        self.model.optimize()

    def getDuals_i(self):
        Pi_cons_lmbda = self.model.getAttr("Pi", self.cons_lmbda)
        return Pi_cons_lmbda

    def getDuals_ts(self):
        Pi_cons_demand = self.model.getAttr("Pi", self.cons_demand)
        return Pi_cons_demand

    def getObjValues(self):
        obj = self.model.objVal
        return obj

    def updateModel(self):
        self.model.update()

    def addColumn(self, newSchedule, iter, index):
        colName = f"ScheduleUsed[{index},{iter}]"
        newScheduleList = []
        cons_demandList = []
        for item in newSchedule:
            newScheduleList.append(newSchedule[item])
            cons_demandList.append(self.cons_demand[item])
        rounded_ScheduleList = ['%.2f' % elem for elem in newScheduleList]
        Column = gu.Column(rounded_ScheduleList, cons_demandList)
        self.model.addVar(vtype=gu.GRB.CONTINUOUS, lb=0, column=Column, name=colName)
        self.model.update()

    def setStartSolution(self):
        startValues = {}
        for i, t, s, r in itertools.product(self.physicians, self.days, self.shifts, self.roster):
            startValues[(i, t, s, r)] = 0
        for i, t, s, r in startValues:
            self.motivation_i[i, t, s, r].Start = startValues[i, t, s, r]

    def solveModel(self, timeLimit, EPS):
        self.model.setParam('TimeLimit', timeLimit)
        self.model.setParam('MIPGap', EPS)
        self.model.Params.OutputFlag = 0
        self.model.optimize()

    def writeModel(self):
        self.model.write("master2.lp")

    def File2Log(self):
        self.model.Params.LogToConsole = 1
        self.model.Params.LogFile = "./log.txt"

    def getObjVal(self):
        obj = self.model.getObjective()
        value = obj.getValue()
        return value

    def finalSolve(self, timeLimit, EPS):
        self.model.setParam('TimeLimit', timeLimit)
        self.model.setParam('MIPGap', EPS)
        self.model.setAttr("vType", self.lmbda, gu.GRB.INTEGER)
        self.model.update()
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            print("Optimal solution found")
            for i in self.physicians:
                for t in self.days:
                    for s in self.shifts:
                        for r in self.roster:
                            print(f"Physician {i}: Motivation {self.motivation_i[i, t, s, r].x} in Shift {s} on day {t}")
        else:
            print("No optimal solution found.")

class Subproblem:
    def __init__(self, duals_i, duals_ts, dfData, i, M):
        self.days = dfData['T'].dropna().astype(int).unique().tolist()
        self.shifts = dfData['K'].dropna().astype(int).unique().tolist()
        self.duals_i = duals_i
        self.duals_ts = duals_ts
        self.Max = 6
        self.M = M
        self.Min = 4
        self.alpha = 0.5
        self.model = gu.Model("Subproblem")
        self.i = i

    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.generateObjective()
        self.model.update()

    def generateVariables(self):
        self.x = self.model.addVars(self.days, self.shifts, vtype=GRB.BINARY, name='x')
        self.mood = self.model.addVars(self.days, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='mood')
        self.motivation = self.model.addVars(self.days, self.shifts, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='motivation')

    def generateConstraints(self):
        for t in self.days:
            self.model.addConstr(self.mood[t] == 1 - self.alpha * quicksum(self.x[t, s] for s in self.shifts))
        for t in self.days:
            self.model.addConstr(gu.quicksum(self.x[t, s] for s in self.shifts) <= 1)
        for t in range(1, len(self.days) - self.Max + 1):
            self.model.addConstr(
                gu.quicksum(self.x[ u, s] for s in self.shifts for u in range(t, t + 1 + self.Max)) <= self.Max)
        self.model.addLConstr(quicksum(self.x[t, s] for t in self.days for s in self.shifts) >= self.Min)

        for t in self.days:
            for s in self.shifts:
                self.model.addLConstr(self.motivation[t, s] >= self.mood[t] - self.M * (1 - self.x[t, s]))
                self.model.addLConstr(self.motivation[t, s] <= self.mood[t] + self.M * (1 - self.x[t, s]))
                self.model.addLConstr(self.motivation[t, s] <= self.x[t, s])

    def generateObjective(self):
        self.model.setObjective(
            0 - gu.quicksum(self.motivation[t, s] * self.duals_ts[t, s] for t in self.days for s in self.shifts) -
            self.duals_i[self.i], sense=gu.GRB.MINIMIZE)

    def getNewSchedule(self):
        return self.model.getAttr("X", self.motivation)

    def printValues(self):
        values_opt = self.model.getAttr("X", self.x)
        return values_opt
    def printValues2(self):
        values_opt = self.model.getAttr("X", self.motivation)
        return values_opt

    def getObjVal(self):
        obj = self.model.getObjective()
        value = obj.getValue()
        return value

    def getStatus(self):
        return self.model.status

    def solveModel(self, timeLimit, EPS):
        self.model.setParam('TimeLimit', timeLimit)
        self.model.setParam('MIPGap', EPS)
        self.model.Params.OutputFlag = 0
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            print("Optimal solution found")
            for t in self.days:
                for s in self.shifts:
                    print(f"Physician {self.i}: Motivation {self.motivation[t, s].x} in Shift {s} on day {t}")
        else:
            print("No optimal solution found.")


#### Column Generation
# CG Prerequisites
modelImprovable = True
t0 = time.time()
max_itr = 1
itr = 0

# Lists
objValHistSP = []
objValHistRMP = []

# Build & Solve MP
master = MasterProblem(DataDF, Demand_Dict, itr)
master.buildModel()
master.File2Log()
master.updateModel()
master.solveRelaxModel()

# Get Duals from MP
duals_i = master.getDuals_i()
duals_ts = master.getDuals_ts()

print('*         *****Column Generation Iteration*****          \n*')
while (modelImprovable) and itr < max_itr:
    # Start
    itr += 1
    print('*Current iteration: ', itr)

    # Solve RMP
    master.updateModel()
    master.solveRelaxModel()
    objValHistRMP.append(master.getObjValues())
    print('*Current RMP ObjVal: ', objValHistRMP)

    # Get Duals
    duals_i = master.getDuals_i()
    duals_ts = master.getDuals_ts()

    # Solve SPs
    modelImprovable = False
    for index in I_list:
        subproblem = Subproblem(duals_i, duals_ts, DataDF, index, 1e6)
        subproblem.buildModel()
        subproblem.solveModel(3600, 1e-6)
        status = subproblem.getStatus()
        if status != 2:
            raise Exception("Pricing-Problem can not reach optimality!")
        reducedCost = subproblem.getObjVal()
        objValHistSP.append(reducedCost)
        print('*Reduced cost', reducedCost)
        if reducedCost < 1e-6:
            values = subproblem.printValues()
            values2 = subproblem.printValues2()
            print(f" Physician {index}: {values}")
            print(f" Physician {index}: {values2}")
            ScheduleCuts = subproblem.getNewSchedule()
            master.addColumn(ScheduleCuts, itr, index)
            master.updateModel()
            modelImprovable = True

# Solve MP
master.finalSolve(3600, 0.01)

# Results
master.writeModel()
print('*                 *****Results*****                  \n*')
print('*Total iteration: ', itr)
t1 = time.time()
print('*Total elapsed time: ', t1 - t0)
print('*Exact solution:', master.getObjValues())

# Plot
plt.scatter(list(range(len(objValHistRMP))), objValHistRMP, c='r')
plt.xlabel('History')
plt.ylabel('Objective function value')
title = 'Solution: ' + str(objValHistRMP[-1])
plt.title(title)
plt.show()
print(objValHistSP)
print(objValHistRMP)