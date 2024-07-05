
import numpy as np  # 导入numpy库，用于数值计算

'''边界检查函数'''

def boundary(pop, lb, ub):
    # 定义一个边界检查函数，确保种群中的个体不超出预定义的边界。
    pop = pop.flatten()
    lb = lb.flatten()
    ub = ub.flatten()
    # 将输入参数扁平化，以便进行元素级操作。

    # 防止跳出范围,除学习率之外 其他的都是整数
    pop = [int(pop[i]) if i > 0 else pop[i] for i in range(lb.shape[0])]
    # 将除了学习率以外的参数转换为整数。

    for i in range(len(lb)):
        if pop[i] > ub[i] or pop[i] < lb[i]:
            # 检查个体是否超出边界。
            if i == 0:
                pop[i] = (ub[i] - lb[i]) * np.random.rand() + lb[i]
                # 如果是学习率，则在边界内随机选择一个值。
            else:
                pop[i] = np.random.randint(lb[i], ub[i])
                # 对于整数参数，随机选择一个边界内的整数值。

    return pop
    # 返回修正后的个体。


''' 种群初始化函数 '''


def initial(pop, dim, ub, lb):
    # 定义一个初始化种群的函数。
    X = np.zeros([pop, dim])
    # 创建一个形状为[种群大小, 维度]的零矩阵。

    for i in range(pop):
        for j in range(dim):
            X[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
            # 在边界内随机初始化每个个体的每个参数。

    return X, lb, ub
    # 返回初始化后的种群及边界。


'''计算适应度函数'''


def CaculateFitness(X, fun, P, T, Pt, Tt):
    # 定义一个计算适应度的函数。
    pop = X.shape[0]
    # 获取种群的大小。
    fitness = np.zeros([pop, 1])
    # 创建一个形状为[种群大小, 1]的零矩阵来存储适应度。

    for i in range(pop):
        fitness[i] = fun(X[i, :], P, T, Pt, Tt)
        # 对每个个体调用适应度函数进行计算。

    return fitness
    # 返回计算得到的适应度。


'''适应度排序'''


def SortFitness(Fit):
    # 定义一个对适应度进行排序的函数。
    fitness = np.sort(Fit, axis=0)
    # 按适应度大小进行排序。
    index = np.argsort(Fit, axis=0)
    # 获取排序后的索引。

    return fitness, index
    # 返回排序后的适应度和索引。


'''根据适应度对位置进行排序'''


def SortPosition(X, index):
    # 定义一个根据适应度排序位置的函数。
    Xnew = np.zeros(X.shape)
    # 创建一个与X形状相同的零矩阵。

    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
        # 根据适应度的排序结果重新排列位置。

    return Xnew
    # 返回排序后的位置。

