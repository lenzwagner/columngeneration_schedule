import gurobipy as gu
import math

class Subproblem:
    def __init__(self, duals_i, duals_ts, df, i, iteration, eps):
        itr = iteration + 1
        self.days = df['T'].dropna().astype(int).unique().tolist()
        self.shifts = df['K'].dropna().astype(int).unique().tolist()
        self.duals_i = duals_i
        self.duals_ts = duals_ts
        self.model = gu.Model("Subproblem")
        self.index = i
        self.itr = itr
        self.End = len(self.days)
        self.mu = 0.1
        self.epsilon = eps
        self.mue = 0.1
        self.chi = 5
        self.omega = math.floor(1 / 1e-6)
        self.M = len(self.days) + self.omega
        self.xi = 1 - self.epsilon * self.omega
        self.Days_Off = 2
        self.Min_WD = 2
        self.Max_WD = 5
        self.F_S = [(3, 1), (3, 2), (2, 1)]
        self.Days = len(self.days)

    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.generateObjective()
        self.model.update()

    def generateVariables(self):
        self.x = self.model.addVars([self.index], self.days, self.shifts, vtype=gu.GRB.BINARY, name="x")
        self.y = self.model.addVars([self.index], self.days, vtype=gu.GRB.BINARY, name="y")
        self.o = self.model.addVars(self.days, self.shifts, vtype=gu.GRB.CONTINUOUS, name="o")
        self.u = self.model.addVars(self.days, self.shifts, vtype=gu.GRB.CONTINUOUS, name="u")
        self.sc = self.model.addVars([self.index], self.days, vtype=gu.GRB.BINARY, name="sc")
        self.v = self.model.addVars([self.index], self.days, vtype=gu.GRB.BINARY, name="v")
        self.q = self.model.addVars([self.index], self.days, self.shifts, vtype=gu.GRB.BINARY, name="q")
        self.rho = self.model.addVars([self.index], self.days, self.shifts, vtype=gu.GRB.BINARY, name="rho")
        self.z = self.model.addVars([self.index], self.days, self.shifts, vtype=gu.GRB.BINARY, name="z")
        self.performance = self.model.addVars([self.index], self.days, self.shifts, [self.itr], vtype=gu.GRB.CONTINUOUS,
                                              lb=0, ub=1, name="performance")
        self.p = self.model.addVars([self.index], self.days, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="p")
        self.n = self.model.addVars([self.index], self.days, vtype=gu.GRB.INTEGER, ub=self.End, lb=0, name="n")
        self.n_h = self.model.addVars([self.index], self.days, vtype=gu.GRB.INTEGER, lb=0, ub=self.End, name="n_h")
        self.h = self.model.addVars([self.index], self.days, vtype=gu.GRB.BINARY, name="h")
        self.e = self.model.addVars([self.index], self.days, vtype=gu.GRB.BINARY, name="e")
        self.kappa = self.model.addVars([self.index], self.days, vtype=gu.GRB.BINARY, name="kappa")
        self.b = self.model.addVars([self.index], self.days, vtype=gu.GRB.BINARY, name="b")
        self.phi = self.model.addVars([self.index], self.days, vtype=gu.GRB.BINARY, name="phi")
        self.r = self.model.addVars([self.index], self.days, vtype=gu.GRB.BINARY, name="r")
        self.f = self.model.addVars([self.index], self.days, vtype=gu.GRB.BINARY, name="f")

    def generateConstraints(self):
        for i in [self.index]:
            for t in self.days:
                self.model.addLConstr(gu.quicksum(self.x[i, t, k] for k in self.shifts) <= 1)
                self.model.addLConstr(gu.quicksum(self.x[i, t, k] for k in self.shifts) == self.y[i, t])
        for i in [self.index]:
            for t in self.days:
                for k in self.shifts:
                    self.model.addLConstr(
                        self.performance[i, t, k, self.itr] >= self.p[i, t] - self.M * (1 - self.x[i, t, k]))
                    self.model.addLConstr(
                        self.performance[i, t, k, self.itr] <= self.p[i, t] + self.M * (1 - self.x[i, t, k]))
                    self.model.addLConstr(self.performance[i, t, k, self.itr] <= self.x[i, t, k])
        for i in [self.index]:
            for k in self.shifts:
                for t in self.days:
                    self.model.addLConstr(self.rho[i, t, k] <= 1 - self.q[i, t, k])
                    self.model.addLConstr(self.rho[i, t, k] <= self.x[i, t, k])
                    self.model.addLConstr(self.rho[i, t, k] >= (1 - self.q[i, t, k]) + self.x[i, t, k] - 1)
                    self.model.addLConstr(self.z[i, t, k] <= self.q[i, t, k])
                    self.model.addLConstr(self.z[i, t, k] <= (1 - self.y[i, t]))
                    self.model.addLConstr(self.z[i, t, k] >= self.q[i, t, k] + (1 - self.y[i, t]) - 1)
                for t in range(1, len(self.days)):
                    self.model.addLConstr(self.q[i, t + 1, k] == self.x[i, t, k] + self.z[i, t, k])
            for t in self.days:
                self.model.addLConstr(1 == gu.quicksum(self.x[i, t, k] for k in self.shifts) + (1 - self.y[i, t]))
                self.model.addLConstr(gu.quicksum(self.rho[i, t, k] for k in self.shifts) == self.sc[i, t])
        for i in [self.index]:
            for t in range(1, len(self.days) - self.Max_WD + 1):
                self.model.addLConstr(
                    gu.quicksum(self.y[i, u] for u in range(t, t + 1 + self.Max_WD)) <= self.Max_WD)
            for t in range(2, len(self.days) - self.Min_WD + 1):
                self.model.addLConstr(
                    gu.quicksum(self.y[i, u] for u in range(t + 1, t + self.Min_WD + 1)) >= self.Min_WD * (
                            self.y[i, t + 1] - self.y[i, t]))
        for i in [self.index]:
            for t in range(2, len(self.days) - self.Days_Off + 2):
                for s in range(t + 1, t + self.Days_Off):
                    self.model.addLConstr(1 + self.y[i, t] >= self.y[i, t - 1] + self.y[i, s])
        for i in [self.index]:
            for k1, k2 in self.F_S:
                for t in range(1, len(self.days)):
                    self.model.addLConstr(self.x[i, t, k1] + self.x[i, t + 1, k2] <= 1)
        for i in [self.index]:
            for t in range(1 + self.chi, len(self.days) + 1):
                self.model.addLConstr(
                    (1 - self.r[i, t]) <= (1 - self.f[i, t - 1]) + gu.quicksum(
                        self.sc[i, j] for j in range(t - self.chi, t)))
                self.model.addLConstr(self.M * (1 - self.r[i, t]) >= (1 - self.f[i, t - 1]) + gu.quicksum(
                    self.sc[i, j] for j in range(t - self.chi, t)))
            for t in range(1, 1 + self.chi):
                self.model.addLConstr(0 == self.r[i, t])
            for t in self.days:
                for tau in range(1, t + 1):
                    self.model.addLConstr(self.f[i, t] >= self.sc[i, tau])
                self.model.addLConstr(self.f[i, t] <= gu.quicksum(self.sc[i, tau] for tau in range(1, t + 1)))
        for i in [self.index]:
            self.model.addLConstr(0 == self.n[i, 1])
            self.model.addLConstr(0 == self.sc[i, 1])
            self.model.addLConstr(1 == self.p[i, 1])
            self.model.addLConstr(0 == self.h[i, 1])
            for t in self.days:
                self.model.addLConstr(
                    gu.quicksum(self.sc[i, j] for j in range(1, t + 1)) <= self.omega + self.M * self.kappa[i, t])
                self.model.addLConstr(self.omega + self.mu <= gu.quicksum(self.sc[i, j] for j in range(1, t + 1)) + (
                        1 - self.kappa[i, t]) * self.M)
            for t in range(2, len(self.days) + 1):
                self.model.addLConstr(self.n[i, t] == self.n_h[i, t] - self.e[i, t] + self.b[i, t])
                self.model.addLConstr(self.n_h[i, t] <= self.n[i, t - 1] + self.sc[i, t])
                self.model.addLConstr(self.n_h[i, t] >= (self.n[i, t - 1] + self.sc[i, t]) - self.M * self.r[i, t])
                self.model.addLConstr(self.n_h[i, t] <= self.M * (1 - self.r[i, t]))
                self.model.addLConstr(self.p[i, t] == 1 - self.epsilon * self.n[i, t] - self.xi * self.kappa[i, t])
                self.model.addLConstr(self.omega * self.h[i, t] <= self.n[i, t])
                self.model.addLConstr(self.n[i, t] <= ((self.omega - 1) + self.h[i, t]))
                self.model.addLConstr(self.e[i, t] <= self.sc[i, t])
                self.model.addLConstr(self.e[i, t] <= self.h[i, t - 1])
                self.model.addLConstr(self.e[i, t] >= self.sc[i, t] + self.h[i, t - 1] - 1)
                self.model.addLConstr(self.b[i, t] <= self.e[i, t])
                self.model.addLConstr(self.b[i, t] <= self.r[i, t])
                self.model.addLConstr(self.b[i, t] >= self.e[i, t] + self.r[i, t] - 1)

    def generateObjective(self):
        self.model.setObjective(
            0 - gu.quicksum(
                self.performance[i, t, s, self.itr] * self.duals_ts[t, s] for i in [self.index] for t in self.days for s
                in self.shifts) -
            self.duals_i[self.index], sense=gu.GRB.MINIMIZE)

    def getNewSchedule(self):
        return self.model.getAttr("X", self.performance)

    def getOptX(self):
        vals_opt = self.model.getAttr("X", self.x)
        vals_list = []
        for vals in vals_opt.values():
            vals_list.append(vals)
        return vals_list

    def getOptP(self):
        return self.model.getAttr("X", self.p)

    def getOptPerf(self):
        return self.model.getAttr("X", self.performance)
    def getStatus(self):
        return self.model.status

    def solveModel(self, timeLimit):
        try:
            self.model.setParam('TimeLimit', timeLimit)
            self.model.Params.OutputFlag = 0
            self.model.Params.IntegralityFocus = 1
            self.model.Params.FeasibilityTol = 1e-9
            self.model.Params.BarConvTol = 0.0
            self.model.Params.MIPGap = 1e-4
            self.model.optimize()
        except gu.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))