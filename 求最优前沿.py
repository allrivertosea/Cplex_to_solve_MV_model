import cplex
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matlab.engine

cov_matrix =pd.read_csv(r'data\股票收盘价集合\100支股票收益率矩阵.csv')
expect_return = pd.read_csv(r'data\股票收盘价集合\100支股票期望收益率.csv')
print(expect_return)
StockNames = range(0,98)
cov_matrix = cov_matrix.to_numpy()
expect_return = expect_return['0'].tolist()

def sensitivity(exp_return):
    p = cplex.Cplex()
    p.objective.set_sense(p.objective.sense.minimize)
    names = ['w' + str(i) for i in range(98)]
    p.variables.add(obj=[0] * 98, ub=[1] * 98, lb=[0] * 98, names=names)
    my_senses = 'EG'
    my_rhs = [1, exp_return]
    rows = [[names, [1] * 98],[names, list(expect_return[:98])]]
    p.linear_constraints.add(lin_expr=rows, senses=my_senses, rhs=my_rhs)
    qmat = []
    for j in range(98):
        qmat.append([[i for i in range(98)], [2 * cov_matrix[j, m] for m in range(98)]])
    p.objective.set_quadratic(qmat)
    p.solve()
    return (p.solution.get_objective_value(), np.array(p.solution.get_values()).round(3))

return_values = np.arange(0.0000635, 0.0095, 0.00009436)
risks = []
for i in return_values:
    risks.append(sensitivity(i)[0])
return_risk = {'return':return_values, 'risk':risks}
return_risk = pd.DataFrame(return_risk)
return_values = return_values.tolist()
eng = matlab.engine.start_matlab()
PF_return = return_values
PF_risk = risks

x0 = PF_return
y0 = PF_risk

eng.two_d_plotx_y(x0,y0)  # 风险——收益二维图

