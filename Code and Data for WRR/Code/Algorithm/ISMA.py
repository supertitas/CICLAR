import numpy as np
import copy
import matplotlib.pyplot as plt
import math
import random
from Performance.method.function import fun1,fun2,fun3,fun4,fun5,fun6,fun7,fun8,fun9

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


def initialization(pop, ub, lb, dim):
    x0 = 0.2
    a = 0.7
    tent_map_values = generate_tent_map(x0, a, pop, dim)
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = (ub[j] - lb[j]) * tent_map_values[i * dim + j] + lb[j]  # 利用了tent混沌映射初始化
    return X


def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = 2 * ub[j] - X[i, j]  # 边界处理
            elif X[i, j] < lb[j]:
                X[i, j] = 2 * lb[j] - X[i, j]  # 边界处理
    return X


def CaculateFitness(X, F):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    if F == 1:
        for i in range(pop):
            fitness[i] = fun1(X[i, :])
    elif F == 2:
        for i in range(pop):
            fitness[i] = fun2(X[i, :])
    elif F == 3:
        for i in range(pop):
            fitness[i] = fun3(X[i, :])
    elif F == 4:
        for i in range(pop):
            fitness[i] = fun4(X[i, :])
    elif F == 5:
        for i in range(pop):
            fitness[i] = fun5(X[i, :])
    elif F == 6:
        for i in range(pop):
            fitness[i] = fun6(X[i, :])
    elif F == 7:
        for i in range(pop):
            fitness[i] = fun7(X[i, :])
    elif F == 8:
        for i in range(pop):
            fitness[i] = fun8(X[i, :])
    elif F == 9:
        for i in range(pop):
            fitness[i] = fun9(X[i, :])
    return fitness


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return np.round(Xnew, 2)


# 加一个自适应的权值策略
# 可以搞一个分段的策略 第一段来个布朗运动 第二段来一个莱维飞行
# 立方混沌映射初始化  反向学习策略  边界异常处理
# 高斯扰动


def SMA(pop, dim, lb, ub, maxIter, fun):
    w_max = 1  # 正余弦改进因子max
    w_min = 0.4  # 正余弦改进因子min
    z = 0.03
    X = initialization(pop, ub, lb, dim)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度的值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = copy.copy(fitness[0])
    GbestPosition = copy.copy(X[0, :])
    Curve = np.zeros([maxIter, 1])
    W = np.zeros([pop, dim])
    for t in range(maxIter):
        worstFitness = fitness[-1]
        bestFitness = fitness[0]
        w = w_min + (w_max - w_min) * np.sin(t * math.pi / maxIter)
        S = bestFitness - worstFitness + 1e-8
        for i in range(pop):
            if i < pop / 2:
                W[i, :] = 1 + np.random.random([1, dim]) * np.log10((bestFitness - fitness[i]) / S + 1)
            else:
                W[i, :] = 1 - np.random.random([1, dim]) * np.log10((bestFitness - fitness[i]) / S + 1)
        tt = -(t / maxIter) + 1
        if tt != -1 and tt != 1:
            a = np.math.atanh(tt)
        else:
            a = 1
        b = 1 - t / maxIter
        for i in range(pop):
            if np.random.random() < z:
                L = Levy(dim)  # 莱维飞行
                X[i, :] = (ub.T - lb.T) * np.random.random([1, dim]) + lb.T + (ub.T - lb.T) * L
            else:
                p = np.tanh(abs(fitness[i] - GbestScore))
                vb = 2 * a * np.random.random([1, dim]) - a
                vc = 2 * b * np.random.random([1, dim]) - b  # 自适应因子变化 优化线性变化
                for j in range(dim):
                    r = np.random.random()
                    A = np.random.randint(pop)
                    B = np.random.randint(pop)
                    if r < p:
                        X[i, j] = GbestPosition[j] + vb[0, j] * (W[i, j] * X[A, j] - X[B, j])
                    else:
                        X[i, j] = vc[0, j] * X[i, j]
        X = BorderCheck(X, ub, lb, pop, dim)
        fitness = CaculateFitness(X, fun)
        fitness, sortIndex = SortFitness(fitness)
        X = SortPosition(X, sortIndex)

        lambda1 = 1 - t ** 2 / maxIter ** 2  # 高斯变异扰动
        lambda2 = t ** 2 / maxIter ** 2
        Cauchy = np.tan((random.random() - 0.5) * np.pi)  # 柯西随机数
        Temp = np.zeros([1, dim])
        Temp[0, :] = X[0, :] * (1 + lambda1 * Cauchy + lambda2 * np.random.randn())
        for j in range(dim):
            if Temp[0, j] > ub[j]:
                Temp[0, j] = ub[j]
            if Temp[0, j] < lb[j]:
                Temp[0, j] = lb[j]

        fitTemp = CaculateFitness(Temp,fun)
        if fitTemp < fitness[0]:
            X[0, :] = copy.copy(Temp)
            fitness[0] = fitTemp

        if fitness[0] <= GbestScore:
            GbestScore = copy.copy(fitness[0])
            GbestPosition = copy.copy(X[0, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPosition, Curve

