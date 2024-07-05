import numpy as np
import copy
import matplotlib.pyplot as plt
from Performance.method.function import fun1,fun2,fun3,fun4,fun5,fun6,fun7,fun8,fun9

def initialization(pop, ub, lb, dim):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = (ub[j] - lb[j]) * np.random.random() + lb[j]
    return X

def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
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
    return Xnew

def SMA(pop, dim, lb, ub, maxIter, fun):
    z = 0.5
    X = initialization(pop, ub, lb, dim)
    fitness = CaculateFitness(X, fun)
    fitness, sortIndex = SortFitness(fitness)
    X = SortPosition(X, sortIndex)
    GbestScore = copy.copy(fitness[0])
    GbestPosition = copy.copy(X[0, :])
    Curve = np.zeros([maxIter, 1])
    W = np.zeros([pop, dim])
    for t in range(maxIter):
        worstFitness = fitness[-1]
        bestFitness = fitness[0]
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
                X[i, :] = (ub.T - lb.T) * np.random.random([1, dim]) + lb.T
            else:
                p = np.tanh(abs(fitness[i] - GbestScore))
                vb = 2 * a * np.random.random([1, dim]) - a
                vc = 2 * b * np.random.random([1, dim]) - b
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
        if fitness[0] <= GbestScore:
            GbestScore = copy.copy(fitness[0])
            GbestPosition = copy.copy(X[0, :])
        Curve[t] = GbestScore
    return GbestScore, GbestPosition, Curve

