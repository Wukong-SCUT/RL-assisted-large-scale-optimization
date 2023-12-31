import cma
import numpy as np
import random
import sys
from tqdm import tqdm
from math import exp
import time
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim

from cec2013lsgo.cec2013 import Benchmark
from opfunu.cec_based.cec2013 import F82013
import problem.bbob_torch as MetaBox_problem

#设置numpy全局随机种子
np.random.seed(2023)

#设置全局torch类型为float64
torch.set_default_dtype(torch.float64)


#Vec类，包含了向量的值和位置，用于记忆sub在global中的位置
class Vec:
    def __init__(self, elements, positions):
        # elements是向量的值
        self.elements = torch.tensor(elements)
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
    diag = torch.diag(C)  # 协方差矩阵 C 对角线上的元素
    s = D / m  # 子群的维数

    sorted_index = torch.argsort(diag)
    group_best = Vec([], [])

    for i in range(1, m + 1):
        Si = sorted_index[int((i - 1) * s):int(i * s)]
        group_best.positions.append(Si)

    # 对 best 进行按位编号，并按 subInfo 中的编号分组
    group_vec = [best[S] for S in group_best.positions]
    group_best.elements = group_vec

    return group_best

def MaVD(D,m,C,best):
    diag = torch.diag(C)  # 协方差矩阵 C 对角线上的元素
    s = D / m  # 子群的维数

    sorted_index = torch.argsort(diag)
    group_best = Vec([], [])

    for i in range(0, m):
        Si = sorted_index[int(i):int(D):int(s)]
        group_best.positions.append(Si)

    # 对 best 进行按位编号，并按 subInfo 中的编号分组
    group_vec = [best[S] for S in group_best.positions]
    group_best.elements = group_vec

    return group_best

def RD(D,m,C,best):
    diag = torch.diag(C)  # 协方差矩阵 C 对角线上的元素
    s = D / m  # 子群的维数

    sorted_index = torch.argsort(diag)
    torch.randperm(D, out=sorted_index)  # 替代 np.random.shuffle

    group_best = Vec([], [])

    for i in range(1, m + 1):
        Si = sorted_index[int((i - 1) * s):int(i * s)]
        group_best.positions.append(Si)

    # 对 best 进行按位编号，并按 subInfo 中的编号分组
    group_vec = [best[S] for S in group_best.positions]
    group_best.elements = group_vec

    return group_best

#Record类，用于记录各次迭代的最优值和CC层分配的问题
class Record:
    def __init__(self,cycle_num): #per_record是一个3*cycle_num的矩阵，用于记录每种分类方式的优化程度 

        #best只记录两次，一次是前面一次是现在的best
        self.best = []  #此处记录的是需要传入per_record的值
        self.per_record = [[1] * cycle_num for _ in range(3)]
        
        self.best_centroid = [] #记录三种方法的centroid
        self.best_C = [] #记录三种方法的最优C
        self.fitness_record = [] #记录每次迭代的最优值

        self.method = []#记录每次迭代的方法
        self.cycle_record = []#记录对应迭代的次数

    def __str__(self):
        # 用于打印实例时的输出
        return f"Vec(elements={self.elements}, positions={self.positions})"
    
    def __getitem__(self, key):
        # 用于支持下标访问
        return self.data[key]

#两个映射
method_mapping = {
    "MiVD": MiVD,
    "MaVD": MaVD,
    "RD": RD
    }

method_mapping_num = {
        "MiVD": 0,
        "MaVD": 1,
        "RD": 2
    }


def problem(x):
    global fes
    fes += 50
    #x = np.ascontiguousarray(x)
    return F3.func(x) #fun_fitness(x) 

# cec接口
# func = F82013(D)

time0 = time.time()

#cec2013lsgo接口
bench = Benchmark()
question = 2
fun_fitness = bench.get_function(question)
info = bench.get_info(question)
search_scope = info['upper']-info['lower']
sigma = search_scope*0.5

#参数初始化
D = info["dimension"]
m = 20
fes = 0
cycle_num = 5
cycle = 0

# MetaBox接口
dim=D

shift=torch.from_numpy(np.random.uniform(info['lower'], info['upper'], size=D))
rotate=torch.eye(D)

bias=0

lb=torch.tensor(info['lower'])
ub=torch.tensor(info['upper'])

F3= MetaBox_problem.F3(dim,shift,rotate,bias,lb,ub)

#CC层初始化
CC_method = ["MiVD", "MaVD", "RD"]

#cma-es初始化
#random_vectors = np.random.uniform(info['lower'], info['upper'], size=(200, D))

# 将 NumPy 数组转换为 PyTorch 张量
random_vectors = torch.from_numpy(np.random.uniform(info['lower'], info['upper'], size=(200, D)))

global_C = torch.from_numpy(np.eye(D))
global_Xw = torch.from_numpy(np.random.uniform(info['lower'], info['upper'], size=D))

# 计算每个向量对应的函数值
time1 = time.time()
function_values = problem(random_vectors)

time2 = time.time()

print(time2-time1)

# 找到最小值和对应的索引
min_index = np.argmin(function_values)

#记录器初始化
record = Record(cycle_num)   

best= random_vectors[min_index]
best_fitness = function_values[min_index]

record.best.insert(0,best_fitness) #记录第一次的best

cycle += 1
record.cycle_record.append(cycle)
record.fitness_record.append(best_fitness)


# 保存原始的 sys.stdout
original_stdout = sys.stdout

fes = 0

while fes < 3e6:
    #time2 = time.time()
    sys.stdout = original_stdout
    print("fes:",f"{fes:e}")
    #CC层选择：分解方式选择
    column_sum = np.sum(record.per_record, axis=1) # 按行求和
    exp_column_sum = np.exp(column_sum)  # 对每个元素取指数
    all_column_sum = np.sum(exp_column_sum)  # 求和

    # 计算 Softmax 概率
    choose_probability = exp_column_sum / all_column_sum
    
    selected_option = random.choices(CC_method, choose_probability, k=1)[0] #选择一个
    record.method.append(selected_option)#记录选择的方式
    
    #此处init_vector为vec类
    init_vector = method_mapping[selected_option](D, m, global_C, best) 

    #进入子空间 
    for i in tqdm(range(m),desc="sub_cycle"):

        sub_dimosions = len(init_vector.elements[i]) # 子代的维度

        # #从全局中提取子代的centroid
        pos = init_vector.positions[i]
        sub_centroid = global_Xw[pos]


        # 从全局中提取子代的 C
        sub_indices = init_vector.positions[i]
        sub_C = global_C[sub_indices][:, sub_indices]
        
        # 重定向 sys.stdout 到一个空的文件对象，即禁止输出
        sys.stdout = open('/dev/null', 'w')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sub_es = cma.CMAEvolutionStrategy(sub_centroid, sigma, {'popsize': 50,'bounds': [info['lower'], info['upper']]}) 
            sub_es.boundary_handler('BoundPenalty')
        sub_es.C = sub_C #设置初始协方差矩阵

        sub_cycle = 0
        sub_cycle_max = 10 #最多给5000fes
        while not sub_es.stop() and sub_cycle < sub_cycle_max:
            sub_cycle +=1

            offspring = sub_es.ask() #获取子代

            # 对子代进行封装，进入 Vec 类
            sub_population = []

            for elements in offspring:
                ind = Vec(elements=np.array(elements), positions=init_vector.positions[i])
                sub_population.append(ind)

            # 将子代的 offspring 转化为全局的 offspring
            sub_offspring_to_global = []

            for ind in sub_population:
                sub_ind_to_global = best.clone()
                sub_ind_to_global[ind.positions] = ind.elements
                sub_offspring_to_global.append(sub_ind_to_global)

            # 转换为 PyTorch 张量
            sub_offspring_to_global = torch.stack(sub_offspring_to_global)

            # 如果需要将结果转换为 NumPy 数组
            #global_offspring = np.array(sub_offspring_to_global)
            
            
            #计算子代的fitness
            sub_es.tell(offspring, problem(sub_offspring_to_global).detach().cpu().numpy()) #更新子代

            #从子代中提取最优解
            sub_best = sub_es.result[0]
            sub_best_fitness = sub_es.result[1]

            if sub_best_fitness < best_fitness:
                best_fitness = sub_best_fitness
                best[init_vector.positions[i]] = torch.tensor(sub_best)
            
            # # 将子代的最优解放入 global_Xw
            # global_Xw[init_vector.positions[i]] = sub_es.result[5]

            # # 将子代的 C 放入 global_C
            # global_C[np.ix_(init_vector.positions[i], init_vector.positions[i])] = sub_es.C

            # 将子代的最优解放入 global_Xw
            global_Xw[init_vector.positions[i]] = torch.tensor(sub_es.result[5])

            # 将子代的 C 放入 global_C
            global_C[np.ix_(init_vector.positions[i], init_vector.positions[i])] = torch.tensor(sub_es.C)

        sigma = sub_es.sigma

    cycle += 1
    record.cycle_record.append(cycle)
    record.fitness_record.append(best_fitness)

sys.stdout = original_stdout
print(best_fitness)

time15 = time.time()
print('all_time:',time15-time0)

            




        
