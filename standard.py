from gurobipy import *
import gurobipy as gu
import time

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
            self.model.Params.LogToConsole = 0
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