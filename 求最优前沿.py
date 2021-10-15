import cplex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#读取收益协方差矩阵和期望收益数据
cov_matrix =pd.read_csv(r'F:\mygithub\Cplex_to_solve_MV_model\data\股票收盘价集合\100支股票收益率矩阵.csv')
expect_return = pd.read_csv(r'F:\mygithub\Cplex_to_solve_MV_model\data\股票收盘价集合\100支股票期望收益率.csv')
cov_matrix = cov_matrix.to_numpy()
expect_return = expect_return['0'].tolist()
return_max = max(expect_return)
return_min = min(expect_return)
StockNames = range(0,98)
#构建Cplex模型
def setproblemdata(p):
    #目标函数最小化
    p.objective.set_sense(p.objective.sense.minimize)    # 在保持收益不变的情况下，目标最小化风险
    #设置变量并添加到Cplex中
    names = ['w' + str(i) for i in range(98)]    # 添加98个变量：也就是98只股票对应的权重（投资比例），上限为1，下限为0
    p.variables.add(obj=[0] * 98, ub=[1] * 98, lb=[0] * 98, names=names)
    #设置约束属性，小于等于
    my_senses = 'EG'
    #约束右侧目标值
    my_rhs = [1, 0.005]#预算约束右侧为1，组合收益约束右侧为0.005
    #设置约束系数
    rows = [[names, [1] * 98],[names, list(expect_return[:98])]]#预算约束系数为1，组合收益约束系数为各股票期望收益值
    #线性约束部分构造
    p.linear_constraints.add(lin_expr=rows, senses=my_senses, rhs=my_rhs )
    #矩阵qmat矩阵实现了目标函数的二次项部分
    qmat = []
    for j in range(98):
        qmat.append([[i for i in range(98)], [2 * cov_matrix[j, m] for m in range(98)]])
    p.objective.set_quadratic(qmat)

p = cplex.Cplex()
setproblemdata(p)
p.solve()
print(sum(p.solution.get_values()))
print(p.solution.get_objective_value())#最低风险值
print(np.array(p.solution.get_values()).round(3))#对应的投资组合

portfolio_names = []
portfolio_weights = []
for i in range(98):

    proportion = np.round(p.solution.get_values()[i], 3)  # keep results with 3 decimals and keep weights larger than 0
    if proportion != 0:
        portfolio_names.append(StockNames[i])
        portfolio_weights.append(proportion)

df = pd.DataFrame(columns = portfolio_names)
df.loc['Weight'] = portfolio_weights
print('Selected stocks and their weights:')
print(df)
print('\nPortfolio risk:')
print(round(p.solution.get_objective_value(),7))


# #敏感性测试
def sensitivity(exp_return):
    p = cplex.Cplex()
    p.objective.set_sense(p.objective.sense.minimize)
    names = ['w' + str(i) for i in range(98)]
    p.variables.add(obj=[0] * 98, ub=[1] * 98, lb=[0] * 98, names=names)
    my_senses = 'EG'
    my_rhs = [1, exp_return]
    rows = [[names, [1] * 98], [names, list(expect_return[:98])]]
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
print('收益',return_values)
print('风险',risks)
plt.figure(figsize=(15,8))
plt.plot(risks, return_values, '.--')
plt.xlabel('Portfolio Risk', fontsize = 18)
plt.ylabel('Portfolio Return', fontsize = 18)
plt.title('Efficient Frontier for our stocks pool\n', fontsize = 25)
plt.grid()
plt.show()

print(np.sum(sensitivity(0.005)[1]))

