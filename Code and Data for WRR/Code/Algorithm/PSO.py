import numpy as np
import copy
import matplotlib.pyplot as plt
from Performance.method.function import fun1,fun2,fun3,fun4,fun5,fun6,fun7,fun8,fun9

def initialization(pop, ub, lb, dim):
    X = np.zeros([pop, dim])
    V = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = (ub[j] - lb[j]) * np.random.random() + lb[j]
            V[i, j] = (ub[j] - lb[j]) * np.random.random() + lb[j]
    return X, V

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

def PSO(pop, dim, lb, ub, maxIter, fun, w=0.5, c1=1.5, c2=1.5):
    X, V = initialization(pop, ub, lb, dim)
    pBestScore = CaculateFitness(X, fun)
    pBestPosition = copy.copy(X)
    fitness = CaculateFitness(X, fun)
    fitness, sortIndex = SortFitness(fitness)
    X = SortPosition(X, sortIndex)
    GbestScore = copy.copy(fitness[0])
    GbestPosition = copy.copy(X[0, :])
    Curve = np.zeros([maxIter, 1])

    for t in range(maxIter):
        for i in range(pop):
            if fitness[i] < pBestScore[i]:
                pBestScore[i] = copy.copy(fitness[i])
                pBestPosition[i, :] = copy.copy(X[i, :])

            if fitness[i] < GbestScore:
                GbestScore = copy.copy(fitness[i])
                GbestPosition = copy.copy(X[i, :])

        for i in range(pop):
            for j in range(dim):
                r1, r2 = np.random.random(), np.random.random()
                V[i, j] = w * V[i, j] + c1 * r1 * (pBestPosition[i, j] - X[i, j]) + c2 * r2 * (GbestPosition[j] - X[i, j])
                X[i, j] = X[i, j] + V[i, j]

        X = BorderCheck(X, ub, lb, pop, dim)
        fitness = CaculateFitness(X, fun)
        fitness, sortIndex = SortFitness(fitness)
        X = SortPosition(X, sortIndex)
        Curve[t] = GbestScore

    return GbestScore, GbestPosition, Curve
