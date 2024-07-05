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

def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index

def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew

def SSA(pop,dim,lb,ub,max_iter,fun):
    st=0.8
    pd=0.2
    sd=0.1
    pdnumber=int(pop*pd)
    sdnumber=int(pop*sd)
    X=initialization(pop,ub,lb,dim)
    fitness=CaculateFitness(X,fun)
    fitness,sortIndex=SortFitness(fitness)
    X=SortPosition(X,sortIndex)
    GbestScore=copy.copy(fitness[0])
    Gbestpositon=np.zeros([1,dim])
    Gbestpositon[0,:]=copy.copy(X[0,:])
    Curve=np.zeros([max_iter])
    for t in range(max_iter):
        BestF=copy.copy(fitness[0])
        Xworst=copy.copy(X[-1,:])
        Xbest=copy.copy(X[0,:])
        R2=np.random.random()
        for i in range(pdnumber):
            if R2<st:
                X[i,:]=X[i,:]*np.exp(-i/(np.random.random()*max_iter))
            else:
                X[i,:]=X[i,:]+np.random.randn()*np.ones([1,dim])
        X=BorderCheck(X,ub,lb,pop,dim)
        fitness=CaculateFitness(X,fun)
        bestII=np.argmin(fitness)
        Xbest=copy.copy(X[bestII,:])
        for i in range(pdnumber+1,pop):
            if i>(pop-pdnumber)/2+pdnumber:
                X[i,:]=np.random.randn()*np.exp((Xworst-X[i,:])/i**2)
            else:
                A=np.ones([dim,1])
                for a in range(dim):
                    if(np.random.random()>0.5):
                        A[a]=-1
                    AA=np.dot(A,np.linalg.inv(np.dot(A.T,A)))
                    X[i,:]=X[0,:]+np.abs(X[i,:]-Gbestpositon)*AA.T

        X=BorderCheck(X,ub,lb,pop,dim)
        fitness=CaculateFitness(X,fun)
        Temp=range(pop)
        RandIndex=random.sample(Temp,pop)
        sdchooseIndex=RandIndex[0:sdnumber]

        for i in range(sdnumber):
            if fitness[sdchooseIndex[i]]>BestF:
                X[sdchooseIndex[i],:]=Xbest+np.random.randn()*np.abs(X[sdchooseIndex[i],:]-Xbest)
            elif fitness[sdchooseIndex[i]]==BestF:
                k=2*np.random.random()-1
                X[sdchooseIndex[i],:]=X[sdchooseIndex[i],:]+k*(np.abs(X[sdchooseIndex[i],:]-X[-1,:])/(fitness[sdchooseIndex[i]]-fitness[-1]+10E-8))

        X=BorderCheck(X,ub,lb,pop,dim)
        fitness=CaculateFitness(X,fun)
        fitness,sortIndex=SortFitness(fitness)
        X=SortPosition(X,sortIndex)
        if(fitness[0]<GbestScore):
            GbestScore=copy.copy(fitness[0])
            Gbestpositon[0,:]=copy.copy(X[0,:])
        Curve[t]=GbestScore

    return GbestScore,Gbestpositon,Curve