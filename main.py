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
        self.modelFlags()
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

    def addColumn(self, newSchedule, iter, i):
        colName = f"ScheduleUsed[{i},{iter}]"
        Column = gu.Column(newSchedule, self.cons_demand)
        self.model.addVar(vtype=gu.GBR.CONTINOUS, lb=0, obj=1.0, column=Column, name=colName)
        # Set all previous lambda variables to zero
        for prev_iter in range(1, iter):
            for j in self.physicians:
                self.lmbda[j, prev_iter].Start = 0
        # Set the lambda variable for the current iteration to 1
        for j in self.physicians:
            self.lmbda[j, iter].Start = 1
        self.model.update()

    def setStartSolution(self):
        startValues = {}
        for i, t, s, r in itertools.product(self.physicians, self.days, self.shifts, self.roster):
            startValues[(i, t, s, r)] = 0
        for i, t, s, r in startValues:
            self.motivation_i[i, t, s, r].Start = startValues[i, t, s, r]

    def modelFlags(self):
        self.model.Params.OutputFlag = 0

    def solveModel(self, timeLimit, EPS):
        self.model.setParam('TimeLimit', timeLimit)
        self.model.setParam('MIPGap', EPS)
        self.model.optimize()

    def writeModel(self):
        self.model.write("master.lp")


class Subproblem:
    def __init__(self, duals_i, duals_ts, dfData, i):
        self.days = dfData['T'].dropna().astype(int).unique().tolist()
        self.shifts = dfData['K'].dropna().astype(int).unique().tolist()
        self.duals_i = duals_i
        self.duals_ts = duals_ts
        self.Max = 5
        self.M = 100
        self.alpha = 0.5
        self.model = gu.Model("Subproblem")
        self.i = i

    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.generateObjective()
        self.modelFlags()
        self.model.update()

    def generateVariables(self):
        self.x = self.model.addVars(self.days, self.shifts, vtype=GRB.BINARY, name='x')
        self.mood = self.model.addVars(self.days, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='mood')
        self.motivation = self.model.addVars(self.days, self.shifts, vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                             name='motivation')

    def generateConstraints(self):
        for t in self.days:
            self.model.addConstr(self.mood[t] == 1 - self.alpha * quicksum(self.x[t, s] for s in self.shifts))
        for t in self.days:
            self.model.addConstr(gu.quicksum(self.x[t, s] for s in self.shifts) <= 1)
        for t in range(1, len(self.days) - self.Max + 1):
            self.model.addConstr(
                gu.quicksum(self.x[u, s] for s in self.shifts for u in range(t, t + 1 + self.Max)) <= self.Max)
        for t in self.days:
            for s in self.shifts:
                self.model.addConstr(self.mood[t] + self.M * (1 - self.x[t, s]) >= self.motivation[t, s])
                self.model.addConstr(self.motivation[t, s] >= self.mood[t] - self.M * (1 - self.x[t, s]))
                self.model.addConstr(self.motivation[t, s] <= self.x[t, s])

    def generateObjective(self):
        self.model.setObjective(
            0 - gu.quicksum(self.motivation[t, s] * self.duals_ts[t, s] for t in self.days for s in self.shifts) -
            self.duals_i[self.i], sense=gu.GRB.MINIMIZE)

    def getNewSchedule(self):
        return self.model.getAttr("X", self.motivation)

    def getObjVal(self):
        obj = self.model.getObjective()
        value = obj.getValue()
        return value

    def getStatus(self):
        return self.model.status

    def solveModel(self, timeLimit, EPS):
        self.model.setParam('TimeLimit', timeLimit)
        self.model.setParam('MIPGap', EPS)
        self.model.optimize()

    def modelFlags(self):
        self.model.Params.OutputFlag = 0


#### Column Generation
# Prerequisites
modelImprovable = True
objValHist = []
t0 = time.time()
max_itr = 3000
itr = 0

# Build MP
master = MasterProblem(DataDF, Demand_Dict, itr)
master.buildModel()

master.updateModel()
master.solveRelaxModel()

# Get Duals
duals_i = master.getDuals_i()
duals_ts = master.getDuals_ts()

## Start
print('         *****Column Generation Iteration*****          \n')
while (modelImprovable) and itr < max_itr:
    # Start
    itr += 1
    print('current iteration time: ', itr)

    # Solve RMP
    master.updateModel()
    master.solveRelaxModel()
    objValHist.append(master.getObjValues)
    print('Current RMP ObjVal: ', objValHist)

    # Get Duals
    duals_i = master.getDuals_i()
    duals_ts = master.getDuals_ts()

    # Solve SPs
    for i in I_list:
        subproblem = Subproblem(duals_i, duals_ts, DataDF, i)
        subproblem.buildModel()
        subproblem.solveModel(3600, 1e-6)
        status = subproblem.getStatus()
        if status != 2:
            raise Exception("Pricing-Problem can not reach optimality!")
        reducedCost = subproblem.getObjVal()
        print('reduced cost', reducedCost)
        if reducedCost < -1e-6:
            ScheduleCuts = subproblem.getNewSchedule()
            master.addColumn(ScheduleCuts, itr, i)
        else:
            modelImprovable = False

# Solve MP
master.solveModel(3600, 0.01)

# Results
master.writeModel()
print('*** Results ***')
print('Total iteration: ', itr)
t1 = time.time()
print('Total elapsed time: ', t1 - t0)
print('Exact solution cost:', master.getAttr("ObjVal"))

# Plot
plt.scatter(list(range(len(rmp_objvals))), rmp_objvals, c='r')
plt.xlabel('History')
plt.ylabel('Objective function value')
title = 'Solution: ' + str(rmp_objvals[-1])
plt.title(title)
plt.show()
