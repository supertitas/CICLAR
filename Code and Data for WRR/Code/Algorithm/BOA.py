import numpy as np
import random
import copy
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

def BOA(pop, dim, lb, ub, maxIter, fun):
    p = 0.8
    power_exponent = 0.1
    sensory_modality = 0.1
    X = initialization(pop, ub, lb, dim)
    fitness = CaculateFitness(X, fun)
    indexBest = np.argmin(fitness)
    GbestScore = fitness[indexBest]
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = X[indexBest, :]
    X_new = copy.copy(X)
    Curve = np.zeros([maxIter, 1])
    for t in range(maxIter):
        for i in range(pop):
            FP = sensory_modality * (fitness[i] ** power_exponent)
            if random.random() < p:
                dis = (random.random() * random.random()) * (GbestPositon - X[i, :])
                X_new[i, :] = X[i, :] + dis
            else:
                Temp = range(pop)
                JK = random.sample(Temp, pop)
                dis = (random.random() * random.random()) * (X[JK[0], :] - X[JK[1], :])
                X_new[i, :] = X[i, :] + dis

            for j in range(dim):
                if X_new[i, j] > ub[j]:
                    X_new[i, j] = ub[j]
                if X_new[i, j] < lb[j]:
                    X_new[i, j] = lb[j]

            X_newforfitness=np.zeros([1,dim])
            X_newforfitness[0,:]=X_new[i,:]
            if CaculateFitness(X_newforfitness, fun) < fitness[i]:
                X[i, :] = copy.copy(X_new[i, :])
                fitness[i] = copy.copy(CaculateFitness(X_newforfitness, fun))

        X = BorderCheck(X, ub, lb, pop, dim)
        fitness = CaculateFitness(X, fun)
        indexBest = np.argmin(fitness)
        if fitness[indexBest] <= GbestScore:
            GbestScore = copy.copy(fitness[indexBest])
            GbestPositon[0, :] = copy.copy(X[indexBest, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve
