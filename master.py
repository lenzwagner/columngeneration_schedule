from gurobipy import *
import gurobipy as gu

class MasterProblem:
    def __init__(self, df, Demand, max_iteration, current_iteration, last, nr, start, timeLim):
        self.iteration = current_iteration
        self.max_iteration = max_iteration
        self.nurses = df['I'].dropna().astype(int).unique().tolist()
        self.days = df['T'].dropna().astype(int).unique().tolist()
        self.shifts = df['K'].dropna().astype(int).unique().tolist()
        self._current_iteration = current_iteration
        self.roster = [i for i in range(1, self.max_iteration + 2)]
        self.rosterinitial = [i for i in range(1, 2)]
        self.demand = Demand
        self.model = gu.Model("MasterProblem")
        self.cons_demand = {}
        self.newvar = {}
        self.last_itr = last
        self.max_itr = max_iteration
        self.cons_lmbda = {}
        self.output_len = nr
        self.start = start
        self.timeLim = timeLim
    
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
        self.model.update()


    def generateConstraints(self):
        for i in self.nurses:
            self.cons_lmbda[i] = self.model.addLConstr(1 == gu.quicksum(self.lmbda[i, r] for r in self.rosterinitial), name = "lmb("+str(i)+")")
        for t in self.days:
            for s in self.shifts:
                self.cons_demand[t, s] = self.model.addConstr(
                    gu.quicksum(self.motivation_i[i, t, s, r]*self.lmbda[i, r] for i in self.nurses for r in self.rosterinitial) +
                    self.slack[t, s] >= self.demand[t, s], "demand("+str(t)+","+str(s)+")")
        self.model.update()
        return self.cons_lmbda, self.cons_demand

    def generateObjective(self):
        self.model.setObjective(gu.quicksum(self.slack[t, s] for t in self.days for s in self.shifts),
                                sense=gu.GRB.MINIMIZE)

    def solveRelaxModel(self):
        try:
            self.model.Params.OutputFlag = 0
            self.model.Params.QCPDual = 1
            self.model.Params.TimeLimit = self.timeLim
            self.model.Params.Method = 2
            self.model.Params.LogToConsole = 0
            self.model.Params.Crossover = 0
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
                    if (i, t, s) in self.start:
                        self.model.addLConstr(self.motivation_i[i, t, s, 1] == self.start[i, t, s])

    def solveModel(self):
        try:
            self.model.Params.QCPDual = 1
            self.model.Params.TimeLimit = self.timeLim
            self.model.Params.IntegralityFocus = 1
            self.model.Params.FeasibilityTol = 1e-9
            self.model.Params.BarConvTol = 0.0
            self.model.Params.MIPGap = 1e-4
            self.model.Params.OutputFlag = 0
            self.model.Params.LogToConsole = 0
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
        print("*{:^{output_len}}*".format(f"Check for quadratic constraints {self.qconstrs}", output_len=self.output_len))

    def finalObj(self):
        obj = self.model.objval
        return obj

    def printLambdas(self):
        return self.model.getAttr("X", self.lmbda)

    def finalSolve(self):
        try:
            self.model.Params.TimeLimit = self.timeLim
            self.model.Params.IntegralityFocus = 1
            self.model.Params.FeasibilityTol = 1e-9
            self.model.Params.BarConvTol = 0.0
            self.model.Params.MIPGap = 1e-4
            self.model.Params.OutputFlag = 0
            self.model.Params.LogToConsole = 0
            self.model.setAttr("vType", self.lmbda, gu.GRB.BINARY)
            self.model.update()
            self.model.optimize()
            if self.model.status == gu.GRB.OPTIMAL:
                print("*" * (self.output_len + 2))
                print("*{:^{output_len}}*".format("***** Optimal solution found *****", output_len=self.output_len))
                print("*" * (self.output_len + 2))
            else:
                print("*" * (self.output_len + 2))
                print("*{:^{output_len}}*".format("***** No optimal solution found *****", output_len=self.output_len))
                print("*" * (self.output_len + 2))
        except gu.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))
