import cma
import numpy as np
import random
import sys
from tqdm import tqdm
from cec2013lsgo.cec2013 import Benchmark
from opfunu.cec_based.cec2013 import F82013


D=1000
func = F82013(D)
bench = Benchmark()
question = 3
fun_fitness = bench.get_function(question)

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

def problem(x):
    x = np.ascontiguousarray(x)
    return fun_fitness(x) 

def shifted_rastrigin_function(x):
    # 设置问题维度
    dim = len(x)
    
    # 生成随机偏移向量
    shift_vector = np.random.uniform(-5, 5, dim)
    
    # 计算 Shifted Rastrigin's Function 的值
    sum_term = np.sum((x - shift_vector) ** 2 - 10 * np.cos(2 * np.pi * (x - shift_vector)))
    return sum_term + 10 * dim

def ackley(x):
    term1 = -20 * np.exp(-0.2 * np.sqrt(1/D * np.sum(x**2)))
    term2 = -np.exp(1/D * np.sum(np.cos(2 * np.pi * x)))
    result = term1 + term2 + 20 + np.exp(1)
    return result

es = cma.CMAEvolutionStrategy([0]*D, 16, inopts={'popsize': 50})
solutions = es.ask()
es.tell(solutions, [problem(x) for x in solutions])
best = es.result[0]

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

# group_best = RD(D,m,es.C,best)
# print(group_best.elements[0])


# sub_es = cma.CMAEvolutionStrategy([0]*len(best), 0.5, inopts={'popsize': 50})

while not es.stop():
    solutions = es.ask() #获取新的种群

    # #将子代的offspring转化为全局的offspring
    # sub_offspring_to_global = [] #由sub_ind_to_global组成
    # sub_ind_to_global = [] #将子代放回至best用以评估fitness

    # for ind in solutions: #从solution中取出每一个个体
    #     sub_ind_to_global = best.copy()
    #     for ele,pos in zip(ind,group_best.positions[0]):
    #         sub_ind_to_global[pos] = ele  
    #     sub_offspring_to_global.append(sub_ind_to_global)

    es.tell(solutions, [problem(x) for x in solutions])#更新种群
    es.logger.add()  #添加日志
    es.disp()

es.result_pretty()

# sub_es.result_pretty()
# print(sub_es.result[0])
# print(len(sub_es.result[0]))
# print(problem(sub_es.result[0]))
# print(sub_es.result[1])

# es.result_pretty()
# print(es.result[0])