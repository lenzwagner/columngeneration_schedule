from gurobipy import *
import gurobipy as gu
import pandas as pd
import itertools
from time import process_time


# Create DF out of Sets
I_list = [1,2,3]
T_list = [1,2,3,4,5,6,7]
K_list = [1,2,3]
I_list1 = pd.DataFrame(I_list, columns=['I'])
T_list1 = pd.DataFrame(T_list, columns=['T'])
K_list1 = pd.DataFrame(K_list, columns=['K'])
DataDF = pd.concat([I_list1, T_list1, K_list1], axis=1)
Demand_Dict = {(1, 1): 2, (1, 2): 1, (1, 3): 0, (2, 1): 1, (2, 2): 2, (2, 3): 0, (3, 1): 1, (3, 2): 1, (3, 3): 1,
          (4, 1): 1, (4, 2): 2, (4, 3): 0, (5, 1): 2, (5, 2): 0, (5, 3): 1, (6, 1): 1, (6, 2): 1, (6, 3): 1,
          (7, 1): 0, (7, 2): 3, (7, 3): 0}


class MasterProblem:
    def __init__(self, dfData, DemandDF):
        self.physicians = dfData['I'].dropna().astype(int).unique().tolist()
        self.days = dfData['T'].dropna().astype(int).unique().tolist()
        self.shifts = dfData['K'].dropna().astype(int).unique().tolist()
        self.roster = {i: list(itertools.product(self.days, self.shifts)) for i in self.physicians}
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
        self.motivation_i = self.model.addVars(self.physicians, self.days, self.shifts, self.roster, vtype=gu.GRB.CONTINUOUS, lb= 0, ub= 1, name = 'motivation_i')
        self.lmbda = self.model.addVars(self.physicians, self.roster, vtype=gu.GRB.INTEGER, lb = 0, name = 'lmbda')
    def generateConstraints(self):
        for i in self.physicians:
            self.cons_lmbda[i] = self.model.addLConstr(gu.quicksum(self.lmbda[i, r] for r in self.roster) == 1)
        return self.cons_lmbda
        for t in self.days:
            for s in self.shifts:
                self.cons_demand[t,s] = self.model.addLConstr(gu.quicksum(self.motivation_i[i, t, s, r] for r in self.roster for i in self.physicians) + self.slack[t, s] >= self.demand[t, s])
        return self.cons_demand

    def generateObjective(self):
        self.model.setObjective(gu.quicksum(self.slack[t, s] for t in self.days for s in self.shifts), sense = gu.GRB.MINIMIZE)

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

    def addColumn(self, objective, newSchedule):
        ctName = ('ScheduleUseVar[%s]' %len(self.model.getVars()))
        newColumn = gu.column(newSchedule, self.model.getConstrs())
        self.model.addVar(vtype = gu.GBR.INTEGER, lb = 0, obj = objective, column = newColumn, name = ctName)
        self.model.update()

    def setStartSolution(self):
        startValues = {}
        for i, t, s, r in itertools.product(self.physicians, self.days, self.shifts, self.roster):
            startValues[(i, t, s, r)] = 0
        for i, t, s, r in startValues:
            self.motivation_i[i, t, s, r].Start = startValues[i, t, s, r]

    def modelFlags(self):
        self.model.Params.OutputFlag = 0

class Subproblem:
    def __init__(self, duals_i, duals_ts, dfData):
        self.days = dfData['T'].dropna().astype(int).unique().tolist()
        self.shifts = dfData['K'].dropna().astype(int).unique().tolist()
        self.duals_i = duals_i
        self.duals_ts = duals_ts
        self.Max = 5
        self.M = 100
        self.alpha = 0.5
        self.model = gu.Model("Subproblem")

    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.generateObjective()
        self.model.update()

    def generateVariables(self):
        self.x = self.model.addVars(self.days, self.shifts, vtype=GRB.BINARY, name='x')
        self.mood = self.model.addVars(self.days, vtype=GRB.CONTINUOUS, lb = 0, ub = 1, name='mood')
        self.motivation = self.model.addVars(self.days, self.shifts, vtype=GRB.CONTINUOUS, lb = 0, ub = 1, name='motivation')

    def generateConstraints(self):
        for t in self.days:
            self.model.addConstr(self.mood[t] == 1 - self.alpha * quicksum(self.x[t, s] for s in self.shifts))
        for t in self.days:
            self.model.addConstr(gu.quicksum(self.x[t, s] for s in self.shifts) <= 1)
        for t in range(1, len(self.days) - self.Max + 1):
            self.model.addConstr(gu.quicksum(self.x[u, s] for s in self.shifts for u in range(t, t + 1 + self.Max)) <= self.Max)
        for t in self.days:
            for s in self.shifts:
                self.model.addConstr(self.mood[t] + self.M*(1-self.x[t, s]) >= self.motivation[t, s])
                self.model.addConstr(self.motivation[t, s] >= self.mood[t] - self.M * (1 - self.x[t, s]))
                self.model.addConstr(self.motivation[t, s] <= self.x[t, s])

    def generateObjective(self):
        self.model.setObjective(-gu.quicksum(self.motivation[t,s]*self.duals_ts[t,s] for t in self.days for s in self.shifts)-self.duals_i[i], sense = gu.GRB.MINIMIZE)

    def getNewSchedule(self):
        return self.model.getAttr("X", self.model.getVars())

def solveModel(self, timeLimit, EPS):
    self.model.setParam('TimeLimit', timeLimit)
    self.model.setParam('MIPGap', EPS)
    self.model.optimize()

#Build MP
master = MasterProblem(DataDF, Demand_Dict)
master.buildModel()

modelImprovable = True
objValHist = []

start = process_time()
while (modelImprovable):
    master.solveRelaxModel()
    duals_i = master.getDuals_i()
    duals_ts = master.getDuals_ts()
    objValHist.append(master.getObjValues)
    for i in I_list:
        subproblem = Subproblem(duals_i, duals_ts, DataDF)
        subproblem.buildModel()
        subproblem.solveModel(3600, 1e-6)
        modelImprovable = (subproblem.getObjectiveValue) < -1e-6
        newScheduleCost = subproblem.objVal
        newScheduleCuts = subproblem.getNewSchedule()
        master.addColumn(newScheduleCost, newScheduleCuts)
        # Print partial time
        sc = int(process_time() - start)
        mn = int(sc / 60)
        sc %= 60
        print("Partial time:", mn, "min", sc, "s")

master.solveModel(3600, 0.01)
end = process_time()
print("*** Results ***")
sec = int(end-start)
min = int(sec / 60)
sec %=  60
print("Time Elapsed:", min, "min", sec, "s")
print("Impact solution cost:", impactCost)
print("Exact solution cost:", masterModel.getAttr("ObjVal"))

#return objValHist
#objValHist.append(master.objVal)
#import matplotlib.pyplot as plt
#plt.plot(list(range(len(history))), history,c='r')
#plt.scatter(list(range(len(history))), history, c='r')
#plt.xlabel('history')
#plt.ylabel('objective function value')
#title = 'solution: ' + str(history[-1])
#plt.title(title)
#plt.show()