import openpyxl
import gurobipy as gu
from gurobipy import *
import pandas as pd
import os
import math

# Path
os.chdir(r'/Users/lorenzwagner/Library/CloudStorage/GoogleDrive-lorenz.wagner99@gmail.com/Meine Ablage/Doktor/Dissertation/Paper 1/Input')
#os.chdir(r'G:\Meine Ablage\Doktor\Dissertation\Paper 1\Input')

# Sets
work = pd.read_excel('Arzt.xlsx',sheet_name='Arzt')
df = pd.read_excel('NF.xlsx',sheet_name='NF')
df1 = pd.read_excel('NF.xlsx',sheet_name='Shift')
I = work['Id'].tolist()
W_I = work['Weekend'].tolist()
T = df['Day'].tolist()
K = df1['Shift'].tolist()
S_T = df1['Hours'].tolist()
I_T = work['WT'].tolist()
S_T = {a:c for a,c in zip(K, S_T)}
I_T = {a:d for a,d in zip(I, I_T)}
W_I = {a:e for a,e in zip(I, W_I)}
I_list1 = pd.DataFrame(I, columns=['I'])
T_list1 = pd.DataFrame(T, columns=['T'])
K_list1 = pd.DataFrame(K, columns=['K'])
DataDF = pd.concat([I_list1, T_list1, K_list1], axis=1)


Demand_Dict = {}
workbook = openpyxl.load_workbook('NF.xlsx')
worksheet = workbook['NF']
for row in worksheet.iter_rows(min_row=2, values_only=True):
    for i in range(1, len(row)):
        cell_value = row[i]
        if isinstance(cell_value, str) and cell_value.startswith('='):
            cell = worksheet.cell(row=row[0], column=i)
            cell_value = cell.value
        if isinstance(cell_value, int):
            Demand_Dict[(int(row[0]), i)] = cell_value
workbook.close()
print(Demand_Dict)


class Problem:
    def __init__(self, dfData, DemandDF):
        self.I = dfData['I'].dropna().astype(int).unique().tolist()
        self.T= dfData['T'].dropna().astype(int).unique().tolist()
        self.K = dfData['K'].dropna().astype(int).unique().tolist()
        self.End = len(self.T)
        self.Week = int(len(self.T) / 7)
        self.Weeks = range(1, self.Week + 1)
        self.demand = DemandDF
        self.model = gu.Model("MasterProblem")
        self.mu = 0.1
        self.epsilon = 1e-6
        self.mue = 0.1
        self.chi = 5
        self.omega = math.floor(1 / 1e-6)
        self.M = len(self.T) + self.omega
        self.xi = 1 - self.epsilon * self.omega
        self.Min_Week = 2
        self.Max_Week = 6
        self.Days_Off = 2
        self.Min_WD = 2
        self.Max_WD = 5
        self.LL = range(1, self.Max_WD + 1)
        self.F_S = [(3, 1), (3, 2), (2, 1)]

    def buildLinModel(self):
        self.generateVariables()
        self.genGenCons()
        self.genChangesCons()
        self.genRegCons()
        self.linRecovery()
        self.linPerformance()
        self.generateObjective()
        self.ModelParams()
        self.updateModel()

    def generateVariables(self):
        self.x = self.model.addVars(self.I, self.T, self.K, vtype=gu.GRB.BINARY, name="x")
        self.kk = self.model.addVars(self.I, self.Weeks, vtype=gu.GRB.BINARY, name="k")
        self.y = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="y")
        self.o = self.model.addVars(self.T, self.K, vtype=gu.GRB.CONTINUOUS, name="o")
        self.u = self.model.addVars(self.T, self.K, vtype=gu.GRB.CONTINUOUS, name="u")
        self.sc = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="sc")
        self.v = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="v")
        self.q = self.model.addVars(self.I, self.T, self.K, vtype=gu.GRB.BINARY, name="q")
        self.rho = self.model.addVars(self.I, self.T, self.K, vtype=gu.GRB.BINARY, name="rho")
        self.z = self.model.addVars(self.I, self.T, self.K, vtype=gu.GRB.BINARY, name="z")
        self.perf = self.model.addVars(self.I, self.T, self.K, vtype=gu.GRB.CONTINUOUS, name="perf")
        self.p = self.model.addVars(self.I, self.T, vtype=gu.GRB.CONTINUOUS, name="p")
        self.n = self.model.addVars(self.I, self.T, vtype=gu.GRB.INTEGER, lb=0, name="n")
        self.n_h = self.model.addVars(self.I, self.T, vtype=gu.GRB.INTEGER, lb=0, name="n_h")
        self.h = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="h")
        self.e = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="e")
        self.kappa = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="kappa")
        self.b = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="b")
        self.phi = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="phi")
        self.r = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="r")
        self.f = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="f")
        self.g = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="g")
        self.gg = self.model.addVars(self.I, self.T, name="gg")

    def genGenCons(self):
        for t in T:
            for k in K:
                self.model.addLConstr(
                    gu.quicksum(self.perf[i, t, k] for i in self.I) + self.u[t, k] >= self.demand[t, k])
        for i in self.I:
            for t in self.T:
                self.model.addLConstr(gu.quicksum(self.x[i, t, k] for k in self.K) <= 1)
                self.model.addLConstr(gu.quicksum(self.x[i, t, k] for k in self.K) == self.y[i, t])
                for k in self.K:
                    self.model.addLConstr(self.perf[i, t, k] >= self.p[i, t] - self.M * (1 - self.x[i, t, k]))
                    self.model.addLConstr(self.perf[i, t, k] <= self.p[i, t] + self.M * (1 - self.x[i, t, k]))
                    self.model.addLConstr(self.perf[i, t, k] <= self.x[i, t, k])

    def genChangesCons(self):
        for i in self.I:
            for k in self.K:
                for t in self.T:
                    self.model.addLConstr(self.rho[i, t, k] <= 1 - self.q[i, t, k])
                    self.model.addLConstr(self.rho[i, t, k] <= self.x[i, t, k])
                    self.model.addLConstr(self.rho[i, t, k] >= (1 - self.q[i, t, k]) + self.x[i, t, k] - 1)
                    self.model.addLConstr(self.z[i, t, k] <= self.q[i, t, k])
                    self.model.addLConstr(self.z[i, t, k] <= (1 - self.y[i, t]))
                    self.model.addLConstr(self.z[i, t, k] >= self.q[i, t, k] + (1 - self.y[i, t]) - 1)
                for t in range(1, len(self.T)):
                    self.model.addLConstr(self.q[i, t + 1, k] == self.x[i, t, k] + self.z[i, t, k])
            for t in self.T:
                self.model.addLConstr(gu.quicksum(self.rho[i, t, k] for k in self.K) == self.sc[i, t])

    def genRegCons(self):
        for i in self.I:
            for t in range(1, len(self.T) + 1 - self.Max_WD):
                self.model.addLConstr(gu.quicksum(self.y[i, u] for u in range(t, t + 1 + self.Max_WD)) <= self.Max_WD)
            for t in range(1, len(self.T) + 1 - self.Min_WD):
                self.model.addLConstr(gu.quicksum(self.y[i, u] for u in range(t + 1, t + self.Min_WD + 1)) >= self.Min_WD * (
                            self.y[i, t + 1] - self.y[i, t]))
            for k1, k2 in self.F_S:
                for t in range(1, len(self.T)):
                    self.model.addLConstr(self.x[i, t, k1] + self.x[i, t + 1, k2] <= 1)
        for t in range(2, len(self.T) - self.Days_Off + 2):
            for i in self.I:
                for s in range(t + 1, t + self.Days_Off):
                    self.model.addLConstr(1 + self.y[i, t] >= self.y[i, t - 1] + self.y[i, s])

    def linRecovery(self):
        for i in self.I:
            for t in range(1 + self.chi, len(self.T) + 1):
                self.model.addLConstr((1 - self.r[i, t]) <= (1 - self.f[i, t - 1]) + gu.quicksum(
                    self.sc[i, j] for j in range(t - self.chi, t)))
                self.model.addLConstr(self.M * (1 - self.r[i, t]) >= (1 - self.f[i, t - 1]) + gu.quicksum(
                    self.sc[i, j] for j in range(t - self.chi, t)))
            for t in range(1, 1 + self.chi):
                self.model.addLConstr(self.r[i, t] == 0)
            for t in self.T:
                for tau in range(1, t + 1):
                    self.model.addLConstr(self.f[i, t] >= self.sc[i, tau])
                self.model.addLConstr(self.f[i, t] <= gu.quicksum(self.sc[i, tau] for tau in range(1, t + 1)))

    def linPerformance(self):
        for i in self.I:
            self.model.addLConstr(self.n[i, 1] == 0)
            self.model.addLConstr(self.p[i, 1] == 1)
            self.model.addLConstr(self.b[i, 1] == 0)
            self.model.addLConstr(self.e[i, 1] == 0)
            for t in range(2, len(self.T) + 1):
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

    def expPerformance(self):
        for i in self.I:
            for t in range(2, len(self.T) + 1):
                self.model.addConstr(self.gg[i, t] == -0.5 * self.n_h[i, t])
                self.model.addGenConstrExp(self.gg[i, t], self.p[i, t], options="FuncNonlinear=1")

    def SPerformance(self):
        self.breakpoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.slopes = [0] + [-math.exp(-(5 - x)) for x in self.breakpoints[1:]]
        for i in self.I:
            for t in range(2, len(self.T) + 1):
                self.model.addGenConstrPWL(self.gg[i, t], self.n_h[i, t], self.breakpoints, self.slopes)
                self.model.addConstr(self.p[i, t] * (1 + self.gg[i, t]) == 1)

    def generateObjective(self):
        self.model.setObjective(gu.quicksum(self.u[t, k] for k in self.K for t in self.T), sense=gu.GRB.MINIMIZE)

    def getObjValues(self):
        obj = self.model.objVal
        return obj

    def updateModel(self):
        self.model.update()

    def ModelParams(self):
        self.model.Params.OutputFlag = 1

    def writeModel(self):
        self.model.write("model.lp")

    def File2Log(self):
        self.model.Params.LogToConsole = 1
        self.model.Params.LogFile = "./log.txt"

    def getObjVal(self):
        obj = self.model.getObjective()
        value = obj.getValue()
        return value

    def optValues(self):
        values = self.model.getAttr("X", self.x)
        print(values)

    def checkForQuadraticCons(self):
        self.qconstrs = self.model.getQConstrs()
        print(f"Check for quadratic constraintrs {self.qconstrs}")

    def solveModel(self, timeLimit, EPS):
        self.model.setParam('TimeLimit', timeLimit)
        self.model.setParam('MIPGap', EPS)
        self.model.optimize()
        self.model.write("final.lp")
        if self.model.status == GRB.OPTIMAL:
            print("Optimal solution found")
            for i in self.I:
                for t in self.T:
                    for k in self.K:
                        if self.x[i, t, k].x > 0.5:
                            print(f"Physician {i}: Shift {k} on day {t}")
        else:
            print("No optimal solution found.")

    def calc_perf(self):
        self.x_values = []
        for i in self.I:
            self.x_i_values = []
            for t in self.T:
                for k in self.K:
                    self.x_var = self.x[i, t, k]
                    self.x_value = self.model.getVarByName(self.x_var.varName).X
                    self.x_i_values.append(round(self.x_value, 3))
            self.x_values.append(self.x_i_values)
        self.perf_values = []
        for i in self.I:
            self.perf_i_values = []
            for t in self.T:
                for k in self.K:
                    self.perf_var = self.perf[i, t, k]
                    self.perf_value = self.model.getVarByName(self.perf_var.varName).X
                    self.perf_i_values.append(round(self.perf_value, 3))
            self.perf_values.append(self.perf_i_values)

        self.u_results = round(sum(self.u[t, k].x for t in self.T for k in self.K), 2)
        self.values2 = [self.demand[key] for key in self.demand.keys()]
        self.sum_values = sum(self.values2)
        print(self.values2)

        self.sum_xWerte = []
        for i in range(len(self.x_values[0])):
            self.sum_value = sum(row[i] for row in self.x_values)
            self.sum_xWerte.append(self.sum_value)
        print(self.sum_xWerte)
        self.sum_pWerte = []
        for i in range(len(self.perf_values[0])):
            self.sum_value = sum(row[i] for row in self.perf_values)
            self.sum_pWerte.append(self.sum_value)
        print(self.sum_pWerte)
        self.comparison_result = []
        for i in range(len(self.values2)):
            if self.values2[i] > self.sum_xWerte[i]:
                self.comparison_result.append(self.values2[i] - self.sum_xWerte[i])
            else:
                self.comparison_result.append(0)

        self.understaffing = round(sum(self.comparison_result), 3)
        self.perf_loss = round(self.u_results - self.understaffing, 3)

        print("")
        print(f"Undercoverage: {self.u_results}")
        print(f"Understaffing: {self.understaffing}")
        print(f"Performance Loss: {self.perf_loss}")
        print("")

        return self.u_results, self.understaffing, self.perf_loss

    def calc_understaffing(self):
        self.sum_all_doctors = 0
        self.x_values = []
        for i in self.I:
            self.x_i_values = []
            for t in self.T:
                for k in self.K:
                    self.x_var = self.x[i, t, k]
                    self.x_value = self.model.getVarByName(self.x_var.varName).X
                    self.x_i_values.append(round(self.x_value, 3))
            self.x_values.append(self.x_i_values)

        self.u_results = round(sum(self.u[t, k].x for t in self.T for k in self.K), 2)
        self.values2 = [self.demand[key] for key in self.demand.keys()]
        self.sum_xWerte2 = []
        for i in range(len(self.x_values[0])):
            self.sum_value = sum(row[i] for row in self.x_values)
            self.sum_xWerte2.append(self.sum_value)

        self.comparison1_result = []
        for i in range(len(self.values2)):
            if self.values2[i] < self.sum_xWerte2[i]:
                self.comparison1_result.append(0)
            else:
                self.comparison1_result.append(1)

        self.doctors_cumulative_multiplied = []
        for i in self.I:
            self.doctor_values = [int(self.sc[i, t].X) for t in self.T]
            self.y_values = []
            for t in self.T:
                if self.y[i, t].x > 0.5:
                    self.y_values.append(1)
                else:
                    self.y_values.append(0)
            self.r_values = []
            for t in self.T:
                if self.r[i, t].x > 0.5:
                    self.r_values.append(1)
                else:
                    self.r_values.append(0)
            self.cumulative_sum = [0]
            self.result = []
            for i in range(1, len(self.doctor_values)):
                if self.r_values[i] == 1:
                    self.cumulative_sum.append(0)
                else:
                    self.cumulative_sum.append(self.cumulative_sum[-1] + self.doctor_values[i])
            self.cumulative_values = [x * self.mue for x in self.cumulative_sum]
            self.multiplied_values = [self.cumulative_values[j] * self.y_values[j] for j in
                                      range(len(self.cumulative_values))]
            self.multiplied_values1 = [self.multiplied_values[j] * self.comparison1_result[j] for j in
                                       range(len(self.multiplied_values))]
            self.total_sum = sum(self.multiplied_values1)
            self.doctors_cumulative_multiplied.append(self.total_sum)
            self.sum_all_doctors += self.total_sum

        self.understaffing1 = self.u_results + self.sum_all_doctors
        print(f"Undercoverage: {self.understaffing1}")
        print(f"Understaffing: {self.u_results}")
        print(f"Performance Loss: {self.sum_all_doctors}")

# Lists
objValHistSP = []
objValHistRMP = []
avg_rc_hist = []

# Build & Solve MP
problem = Problem(DataDF, Demand_Dict)
problem.buildLinModel()
problem.updateModel()
problem.checkForQuadraticCons()
problem.model.optimize()
problem.optValues()

problem.calc_understaffing()
