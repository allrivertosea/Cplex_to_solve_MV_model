#得到100个投资组合，随意分配权重，上下限为[0,1]，然后进行权重归一化，计算各组合收益与方差
import pandas as pd
import numpy as np
import random
cov_matrix =pd.read_csv(r'data\模型2训练数据\100支股票预测期望收益率协方差矩阵.csv')
expect_return = pd.read_csv(r'data\模型2训练数据\100支股票预测期望收益率.csv')
cov_matrix = cov_matrix.to_numpy()
expect_return = expect_return['0'].tolist()
#该过程进行100次：首先在100以内随机选择一个数出来，作为投资组合资产数n，然后[0,...,99]之间随机选择出来n个资产组成投资组合
#组成投资组合后，计算投资组合的期望收益和方差，通过改进下面的评估函数来计算。
total_assets_list=[i for i in range(97)]


def evaluate(expect_return ,cov_matrix, normal_portfolios,normal_weights):
    sum_return = []
    sum_risk = []
    for i in range(len(normal_weights)):
        normal_return = [expect_return[f] for f in normal_portfolios[i]]
        normal_cov_matrix = []
        for h in normal_portfolios[i]:
            normal_cov_matrix.append([cov_matrix[h][e] for e in normal_portfolios[i]])

        a = np.mat(normal_weights[i]) * np.mat(normal_return).T
        b = a.tolist()
        f1 = b[0][0]  # 组合期望收益率
        sum_return.append(f1)
        c = np.multiply(np.mat(normal_weights[i]).T * np.mat(normal_weights[i]),
                        np.mat(normal_cov_matrix)).tolist()  # 协方差与各个权重成绩组成的矩阵，下面会求和
        c_total = 0  # 计算组合方差
        for j in range(len(normal_weights[i])):
            for k in range(len(normal_weights[i])):
                c_total += c[j][k]
        f2 = pow(c_total, 0.5)  # 收益率的标准偏差（注意不是方差）
        sum_risk.append(f2)
    return sum_return, sum_risk



def port_weight():
    normal_portfolios = []
    normal_weights = []
    for i in range(100):
        rand_num = random.randint(0,97)
        normal_portfolios.append(sorted(random.sample(total_assets_list, rand_num)))
        temp_weights = [random.random() for j in normal_portfolios[i]]
        weight_sum = 0
        for f in range(len(temp_weights)):
            weight_sum += temp_weights[f]
        for k in range(len(temp_weights)):  # 权重归一化满足预算约束
            temp_weights[k] = temp_weights[k] / weight_sum
        normal_weights.append(temp_weights)
    return normal_portfolios,normal_weights
total_return=[]
total_risk=[]
for i in range(20):
    normal_portfolios,normal_weights = port_weight()
    sum_return, sum_risk = evaluate(expect_return ,cov_matrix,normal_portfolios, normal_weights)
    total_return.append(sum_return)
    total_risk.append(sum_risk)
mean_return = np.mean([np.mean(i) for i in total_return])
std_return = np.std([np.std(i,ddof=1) for i in total_return])
mean_risk = np.mean([np.mean(i) for i in total_risk])
std_risk = np.std([np.std(i,ddof=1) for i in total_risk])
print(mean_return)
print(std_return)
print(mean_risk)
print(std_risk)


