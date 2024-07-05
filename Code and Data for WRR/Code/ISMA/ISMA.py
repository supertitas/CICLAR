import math  # 导入math库，提供基本的数学运算函数
import numpy as np  # 导入numpy库，用于数值计算
import random  # 导入random库，用于生成随机数
import copy  # 导入copy库，用于对象复制

from MA import boundary,CaculateFitness,SortFitness,SortPosition
def Levy(dim):  # 莱维飞行
    beta = 3 / 2
    sigma = (math.gamma(1 + beta) * np.sin(math.pi * beta / 2)) / (
                math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)) ** (1 / beta)
    u = np.random.randn(1, dim) * sigma
    v = np.random.randn(1, dim)
    step = u / np.abs(v) ** (1 / beta)
    L = 0.05 * step
    return L

def tent_map(x, a):  # tent混沌映射
    if x < a:
        return x / a
    else:
        return (1 - x) / (1 - a)


def generate_tent_map(x0, a, pop, dim):
    tent_map_values = []
    x = x0
    for i in range(pop * dim):
        x = tent_map(x, a)
        tent_map_values.append(x)
    return tent_map_values


def initial(pop, dim, ub, lb):
    x0 = 0.2
    a = 0.7
    tent_map_values = generate_tent_map(x0, a, pop, dim)
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = (ub[j] - lb[j]) * tent_map_values[i * dim + j] + lb[j]  # 利用了tent混沌映射初始化
    return X, lb, ub
    # 返回初始化后的种群及边界。
'''黏菌算法'''

def ISMA(P, T, Pt, Tt, fun):
    pop = 2  # 种群数量，这里设置为2
    MaxIter = 2  # 最大迭代次数，这里设置为2
    dim = 2  # 搜索维度，这里是2维，通常代表优化的参数数量
    z = 0.03 # 位置更新参数
    w_max = 1  # 正余弦改进因子max
    w_min = 0.4  # 正余弦改进因子min

    # 下面两行定义了每个参数的搜索范围
    lb = np.array([0.001, 10]).reshape(-1, 1)  # 搜索范围的下限，学习率和神经元个数的最小值
    ub = np.array([0.01, 100]).reshape(-1, 1)  # 搜索范围的上限，学习率和神经元个数的最大值

    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    for i in range(pop):
        X[i, :] = boundary(X[i, :], lb, ub)  # 应用边界条件
    fitness = CaculateFitness(X, fun, P, T, Pt, Tt)  # 计算种群的适应度
    fitness, sortIndex = SortFitness(fitness)  # 对适应度进行排序
    X = SortPosition(X, sortIndex)  # 根据适应度排序种群
    GbestScore = copy.copy(fitness[0])  # 记录最好的适应度值
    GbestPositon = np.zeros([1, dim])  # 初始化最优位置
    GbestPositon[0, :] = copy.copy(X[0, :])  # 记录最优位置
    Curve = np.zeros([MaxIter, 1])  # 初始化收敛曲线
    result = np.zeros([MaxIter, dim])  # 初始化结果
    W = np.zeros([pop,dim])

    # 开始迭代
    for t in range(MaxIter):
        worstFitness = fitness[-1]
        bestFitness = fitness[0]
        w = w_min + (w_max - w_min) * np.sin(t * math.pi / MaxIter)
        S = bestFitness - worstFitness + 1e-8

        for i in range(pop):
            if i < pop/2:
                W[i, :] = 1 + np.random.random([1, dim]) * np.log10((bestFitness - fitness[i]) / (S) + 1)
            else:
                W[i, :] = 1 - np.random.random([1, dim]) * np.log10((bestFitness - fitness[i]) / (S) + 1)
        tt = -(t/MaxIter)+1
        if tt!=-1 and tt!=1:
            a=np.math.atanh(tt)
        else:
            a = 1
        b = 1-t/MaxIter

        for i in range(pop):
            if np.random.random()<z:
                L = Levy(dim)  # 莱维飞行
                X[i, :] = (ub.T - lb.T) * np.random.random([1, dim]) + lb.T + (ub.T - lb.T) * L
            else:
                p= np.tanh(abs(fitness[i]-GbestScore))
                vb=2*a*np.random.random([1,dim])-a
                vc = 2 * b * np.random.random([1, dim]) - b
                for j in range(dim):
                    r=np.random.random()
                    A=np.random.randint(pop)
                    B = np.random.randint(pop)
                    if r<p:
                        X[i,j]=GbestPositon[0,j]+vb[0,j]*(W[i,j]*X[A,j]-X[B,j])
                    else:
                        X[i,j]=vc[0,j]*X[i,j]

        # 应用边界条件并重新计算适应度
        for ixx in range(pop):
            X[ixx, :] = boundary(X[ixx, :], lb, ub)
        fitness = CaculateFitness(X, fun, P, T, Pt, Tt)  # 计算适应度
        fitness, sortIndex = SortFitness(fitness)  # 对适应度进行排序
        X = SortPosition(X, sortIndex)  # 根据适应度排序种群

        lambda1 = 1 - t ** 2 / MaxIter ** 2  # 高斯变异扰动
        lambda2 = t ** 2 / MaxIter ** 2
        Cauchy = np.tan((random.random() - 0.5) * np.pi)  # 柯西随机数
        Temp = np.zeros([1, dim])
        Temp[0, :] = X[0, :] * (1 + lambda1 * Cauchy + lambda2 * np.random.randn())
        for j in range(dim):
            if Temp[0, j] > ub[j]:
                Temp[0, j] = ub[j]
            if Temp[0, j] < lb[j]:
                Temp[0, j] = lb[j]

        fitTemp = CaculateFitness(Temp,fun, P, T, Pt, Tt)
        if fitTemp < fitness[0]:
            X[0, :] = copy.copy(Temp)
            fitness[0] = fitTemp
        # 更新全局最优
        if fitness[0] <= GbestScore:
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0, :] = copy.copy(X[0, :])

        Curve[t] = GbestScore  # 更新收敛曲线
        result[t, :] = GbestPositon  # 更新结果
        print(t + 1, GbestScore, [int(GbestPositon[0, i]) if i > 0 else GbestPositon[0, i] for i in range(dim)])
    return GbestPositon, Curve, result