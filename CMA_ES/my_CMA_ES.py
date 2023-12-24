import numpy as np
from math import log, sqrt, exp
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from cec2013lsgo.cec2013 import Benchmark

#fes 用于记录全局适应度评价次数
fes = 0

#Vec类，包含了向量的值和位置，用于记忆sub在global中的位置
class Vec:
    def __init__(self, elements, positions):
        # elements是向量的值
        self.elements = elements
        # positions是元素对应的位置列表
        self.positions = positions

    def __str__(self):
        # 用于打印实例时的输出
        return f"Vec(elements={self.elements}, positions={self.positions})"
    
    def __getitem__(self, key):
        # 用于支持下标访问
        return self.data[key]

#三种分组函数的定义
def MiVD(D,m,C,best):
    diag = np.diag(C) #协方差矩阵C对角线上的元素
    s = D/m #子群的维数
    subInfo = [] #子群信息
    sortedIndex = np.argsort(diag)
    group_best = Vec([],[])
    for i in range(1,m+1):
        Si = sortedIndex[int((i - 1) * s) : int(i * s)]
        group_best.positions.append(Si)
        subInfo.append(Si)

    # 对 best 进行按位编号，并按 subInfo 中的编号分组
    group_vec = [best[S] for S in group_best.positions]
    group_best.elements = group_vec
    return group_best

def MaVD(D,m,C,best):
    diag = np.diag(C) #协方差矩阵C对角线上的元素
    s = D/m #子群的维数
    subInfo = [] #子群信息
    sortedIndex = np.argsort(diag)
    group_best = Vec([],[])

    for i in range(0,m):
        Si = sortedIndex[int(i):int(D):int(s)+1]
        group_best.positions.append(Si)
        subInfo.append(Si)
    
    # 对 best 进行按位编号，并按 subInfo 中的编号分组
    group_vec = [best[S] for S in group_best.positions]
    group_best.elements = group_vec

    return group_best

def RD(D,m,C,best):
    diag = np.diag(C) #协方差矩阵C对角线上的元素
    s = D/m #子群的维数
    subInfo = [] #子群信息
    sortedIndex = np.argsort(diag)
    np.random.shuffle(sortedIndex)
    group_best = Vec([],[])

    for i in range(1,m+1):
        Si = sortedIndex[int((i - 1) * s) : int(i * s)]
        group_best.positions.append(Si)
        subInfo.append(Si)
    
    # 对 best 进行按位编号，并按 subInfo 中的编号分组
    group_vec = [best[S] for S in group_best.positions]
    group_best.elements = group_vec

    return group_best

#CMA-ES算法的实现 
class CMA_ES(object): 
    r"""Covariance Matrix Adaptation Evolution Strategy (CMA-ES)"""
    def __init__(self, centroid, sigma, global_dimension,**kargs):
        self.global_dimension = global_dimension
        self.params = kargs
        # Create a centroid as a numpy array
        self.centroid = np.array(centroid) # 个体的中心点

        self.dim = len(self.centroid) # 个体的维度
        self.sigma = sigma # 个体的标准差
        # 个体的历史最佳位置（pc）通常用于跟踪个体在搜索空间中的最佳位置，而全局历史最佳位置（ps）用于跟踪整个种群的最佳位置
        self.pc = np.zeros(self.dim) #数组表示了个体历史最佳位置
        self.ps = np.zeros(self.dim) #数组表示了全局历史最佳位置
        #计算PSO算法中的一个参数 chiN，通常用于调整粒子群的速度更新公式
        self.chiN = sqrt(self.dim) * (1 - 1. / (4. * self.dim)
                                      + 1. / (21. * self.dim ** 2)) 

        self.C = self.params.get("cmatrix", np.identity(self.dim)) # 个体的协方差矩阵
        self.diagD, self.B = np.linalg.eigh(self.C) # 个体的特征值和特征向量

        indx = np.argsort(self.diagD) # 对特征值进行排序
        self.diagD = self.diagD[indx] ** 0.5 # 对特征值进行开方
        self.B = self.B[:, indx] # 对特征向量进行排序
        self.BD = self.B * self.diagD  # 个体的特征向量乘以特征值

        self.cond = self.diagD[indx[-1]] / self.diagD[indx[0]] # 个体的条件数

        self.lambda_ = self.params.get("lambda_", int(4 + 3 * log(self.dim))) # 个体的子代数量 
        self.update_count = 0 # 个体的更新次数
        self.computeParams(self.params) 
        #使用给定的参数字典 self.params 来计算一些参数，并将结果用于进一步的操作
        self.fes = 0 # 个体的适应度评价次数

    def generate(self, ind_init):
        arz = np.random.standard_normal((self.lambda_, self.dim)) # 个体的随机向量
        arz = self.centroid + self.sigma * np.dot(arz, self.BD.T) # 个体的随机向量乘以特征向量乘以特征值
        return [ind_init(a) for a in arz] # 生成个体

    def fitness(self,ind):
        init_individual = np.zeros(self.global_dimension)
        for i, _ in zip(ind.elements,ind.positions):
            init_individual[_] = i
        return problem(init_individual)

    def sort(self,population): #注意update函数只有一个接口
        min_fitness = float('inf')
        for ind in population:
            ind.fitness = self.fitness(ind)
            min_fitness = min(min_fitness, ind.fitness)
        
        #同时返回最小的min_fitness，减少评估消耗次数
        return sorted(population, key=lambda ind: ind.fitness, reverse=False), min_fitness#求最大or最小值，reverse调整，降序是max，升序是min
    
    def update(self, population):
        self.sort(population) # 对个体进行排序
        old_centroid = self.centroid # 个体的历史最佳位置
        #由于population是一个类，因此需要对其进行一些切片操作
        sub_slice = population[0:self.mu]
        self.centroid = np.dot(self.weights, [ind.elements for ind in sub_slice]) # 个体的中心点

        c_diff = self.centroid - old_centroid # 个体的中心点与历史最佳位置的差值

        # Cumulation : update evolution path # 个体的演化路径
        self.ps = (1 - self.cs) * self.ps \
            + sqrt(self.cs * (2 - self.cs) * self.mueff) / self.sigma \
            * np.dot(self.B, (1. / self.diagD)
                        * np.dot(self.B.T, c_diff)) # 个体的全局历史最佳位置

        hsig = float((np.linalg.norm(self.ps)
                      / sqrt(1. - (1. - self.cs) ** (2. * (self.update_count + 1.))) / self.chiN
                      < (1.4 + 2. / (self.dim + 1.))))  #更新触发条件

        self.update_count += 1 # 个体的更新次数

        self.pc = (1 - self.cc) * self.pc + hsig \
            * sqrt(self.cc * (2 - self.cc) * self.mueff) / self.sigma \
            * c_diff # 个体的历史最佳位置

        # Update covariance matrix
        artmp = [ind.elements for ind in sub_slice] - old_centroid
        self.C = (1 - self.ccov1 - self.ccovmu + (1 - hsig)
                  * self.ccov1 * self.cc * (2 - self.cc)) * self.C \
            + self.ccov1 * np.outer(self.pc, self.pc) \
            + self.ccovmu * np.dot((self.weights * artmp.T), artmp) \
            / self.sigma ** 2

        self.sigma *= np.exp((np.linalg.norm(self.ps) / self.chiN - 1.)
                                * self.cs / self.damps)

        self.diagD, self.B = np.linalg.eigh(self.C)
        indx = np.argsort(self.diagD)

        self.cond = self.diagD[indx[-1]] / self.diagD[indx[0]]

        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD

    def computeParams(self, params):
        self.mu = params.get("mu", int(self.lambda_ / 2))
        rweights = "equal"  #params.get("weights", "superlinear")
        if rweights == "superlinear":
            self.weights = log(self.mu + 0.5) - \
                np.log(np.arange(1, self.mu + 1))
        elif rweights == "linear":
            self.weights = self.mu + 0.5 - np.arange(1, self.mu + 1)
        elif rweights == "equal":
            self.weights = np.ones(self.mu)
        else:
            raise RuntimeError("Unknown weights : %s" % rweights)

        self.weights /= sum(self.weights)
        self.mueff = 1. / sum(self.weights ** 2)

        self.cc = params.get("ccum", 4. / (self.dim + 4.))
        self.cs = params.get("cs", (self.mueff + 2.)
                             / (self.dim + self.mueff + 3.))
        self.ccov1 = params.get("ccov1", 2. / ((self.dim + 1.3) ** 2
                                               + self.mueff))
        self.ccovmu = params.get("ccovmu", 2. * (self.mueff - 2.
                                                 + 1. / self.mueff)
                                 / ((self.dim + 2.) ** 2 + self.mueff))
        self.ccovmu = min(1 - self.ccov1, self.ccovmu)
        self.damps = 1. + 2. * max(0, sqrt((self.mueff - 1.)
                                           / (self.dim + 1.)) - 1.) + self.cs
        self.damps = params.get("damps", self.damps)

#Record类，用于记录各次迭代的最优值和CC层分配的问题
class Record:
    def __init__(self, cycle_num): #per_record是一个3*cycle_num的矩阵，用于记录每种分类方式的优化程度 
        #best只记录两次，一次是前面一次是现在的best
        self.best = [] 
        self.per_record = [[1] * 5 for _ in range(3)]
        

        self.best_centroid = [] #记录三种方法的centroid
        self.best_C = [] #记录三种方法的最优C
        self.fitness_record = [] #记录每次迭代的最优值

    def __str__(self):
        # 用于打印实例时的输出
        return f"Vec(elements={self.elements}, positions={self.positions})"
    
    def __getitem__(self, key):
        # 用于支持下标访问
        return self.data[key]


#------------------测试------------------


#超参数
D = 50

def ackley(x):
    term1 = -20 * np.exp(-0.2 * np.sqrt(1/D * np.sum(x**2)))
    term2 = -np.exp(1/D * np.sum(np.cos(2 * np.pi * x)))
    result = term1 + term2 + 20 + np.exp(1)
    return result

x = np.arange(D)
print("x",x)
print("x**2",x**2)

def problem(x): #此处X为一行向量
    global fes
    fes += 1
    return ackley(x)

#strategy初始化
strategy = CMA_ES(centroid=[0]*D, sigma=0.5, lambda_=50, global_dimension=D) #此处参数是查验论文之后得到的结果 初始化

#全局初始化
def ind_init(position):
    # 假设每个个体是一个实数向量
    return position


#初始化全局种群
original_offspring = strategy.generate(ind_init) 
population = []
for _ in original_offspring:
    ind  = Vec(elements=[], positions=[]) 
    ind.elements = _
    ind.positions = np.arange(D)
    population.append(ind)

best_fitness = 0
best = []
for _ in range(1000):
    strategy.update(population)
    strategy.computeParams(strategy.params)
    sort_offspring,fitness = strategy.sort(population)
    if fitness < best_fitness:
        best_fitness = fitness
        best = sort_offspring[0].elements

best = sort_offspring[0].elements #初始化全局最优
print(fitness)
print(fes)



