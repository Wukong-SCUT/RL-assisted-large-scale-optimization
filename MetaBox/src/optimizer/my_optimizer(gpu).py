import cma
import numpy as np
import torch
import random
import sys
from tqdm import tqdm
from math import exp
import time
import warnings

from optimizer.learnable_optimizer import Learnable_Optimizer

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

    sortedIndex = np.argsort(diag)
    group_best = Vec([],[])
    for i in range(1,m+1):
        Si = sortedIndex[int((i - 1) * s) : int(i * s)]
        group_best.positions.append(Si)

    # 对 best 进行按位编号，并按 subInfo 中的编号分组
    group_vec = [best[S] for S in group_best.positions]
    group_best.elements = group_vec

    return group_best

def MaVD(D,m,C,best):
    diag = np.diag(C) #协方差矩阵C对角线上的元素
    s = D/m #子群的维数

    sortedIndex = np.argsort(diag)
    group_best = Vec([],[])

    for i in range(0,m):
        Si = sortedIndex[int(i):int(D):int(s)]
        group_best.positions.append(Si)

    # 对 best 进行按位编号，并按 subInfo 中的编号分组
    group_vec = [best[S] for S in group_best.positions]
    group_best.elements = group_vec

    return group_best

def RD(D,m,C,best):
    diag = np.diag(C) #协方差矩阵C对角线上的元素
    s = D/m #子群的维数

    sortedIndex = np.argsort(diag)
    np.random.shuffle(sortedIndex)
    group_best = Vec([],[])

    for i in range(1,m+1):
        Si = sortedIndex[int((i - 1) * s) : int(i * s)]
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

#CC层方法目录
CC_method = ["MiVD", "MaVD", "RD"]

# 保存原始的 sys.stdout
original_stdout = sys.stdout 


class MyOptimizer(Learnable_Optimizer):
    def __init__(self, config):
        """
        Parameter
        ----------
        config: An argparse.Namespace object for passing some core configurations such as maxFEs.
        """
        super().__init__(config)
        self.config = config

        self.D = config.D #维度
        self.m = config.m #子群个数

        #注意,lb和ub需要设置为torch.tensor
        self.ub = config.ub #搜索上界
        self.lb = config.lb #搜索下界
        self.maxfes = config.maxfes #fes计数器

        self.search_scope = self.ub - self.lb
        self.sigma = 0.5*(self.search_scope) #初始步长

        self.lambda_ = 50 #种群大小
        self.cycle_num = 5 #per_record迭代次数
        
  
    def init_population(self, problem):
        
        #cma-es初始化
        random_vectors = torch.from_numpy(np.random.uniform(self.lb, self.ub, size=(200, self.D)))
        self.global_C = np.eye(self.D)
        self.global_Xw = np.random.uniform(self.lb, self.ub, size=self.D)

        # 计算每个向量对应的函数值
        function_values = problem(random_vectors)

        # 找到最小值和对应的索引
        min_index = np.argmin(function_values)

        #记录器初始化
        self.record = Record(self.cycle_num) 

        self.best= random_vectors[min_index].numpy() #将best返还至numpy(因为后面的cma-es只接受numpy)
        self.best_fitness = function_values[min_index]

        self.record.best.insert(0,self.best_fitness) #记录第一次的best

        self.fes = 0  # fes计数器初始化
        self.cost = [self.best_fitness]     # record the best cost of  first generation
        self.cur_logpoint = 1            # record the current logpoint

        #留给RL的接口
        """
        calculate the state
        """
        #return state
  
    def update(self, action, problem):
        """ update the population using action and problem.
            Used in Environment's step
  
        Parameter
        ----------
        action: the action inferenced by agent.
        problem: a problem instance.
  
        Must To Do
        ----------
        1. Update the counter "fes".
        2. Update the list "cost" if logpoint arrives.
  
        Return
        ----------
        state: represents the observation of current population.
        reward: the reward obtained for taking the given action.
        is_done: whether the termination conditions are met.
        """
  
        """
        update population using the given action and update self.fes
        """
        # append the best cost if logpoint arrives

        while self.fes < self.maxfes :

            #CC层选择：分解方式选择
            column_sum = np.sum(self.record.per_record, axis=1)  # 按行求和
            exp_column_sum = np.exp(column_sum)  # 对每个元素取指数
            all_column_sum = np.sum(exp_column_sum)  # 求和

            # 计算概率
            choose_probability = exp_column_sum / all_column_sum
            
            selected_option = random.choices(CC_method, choose_probability, k=1)[0] #选择一个
            self.record.method.append(selected_option)#记录选择的方式
            
            #此处init_vector为vec类
            init_vector = method_mapping[selected_option](self.D, self.m, self.global_C, self.best) 

            #进入子空间 
            for i in range(self.m):

                # 从全局中提取子代的 centroid
                pos = init_vector.positions[i]
                sub_centroid = self.global_Xw[pos]

                # 从全局中提取子代的 C
                sub_indices = init_vector.positions[i]
                sub_C = self.global_C[sub_indices][:, sub_indices]
                
                # 重定向 sys.stdout 到一个空的文件对象，即禁止输出
                sys.stdout = open('/dev/null', 'w')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sub_es = cma.CMAEvolutionStrategy(sub_centroid, self.sigma, {'popsize': self.sub_popsize,'bounds': [self.lb, self.ub]}) 
                    sub_es.boundary_handler('BoundPenalty')
                sub_es.C = sub_C #设置初始协方差矩阵

                #子空间优化次数计数器
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
                        sub_ind_to_global = torch.from_numpy(self.best.copy())
                        sub_ind_to_global[ind.positions] = torch.tensor(ind.elements)
                        sub_offspring_to_global.append(sub_ind_to_global)

                    # 转换为tensor
                    global_offspring = torch.stack(sub_offspring_to_global)
                    
                    #计算子代的fitness
                    sub_es.tell(offspring, problem(global_offspring).detach().cpu().numpy()) #更新子代

                    #从子代中提取最优解
                    sub_best = sub_es.result[0]
                    sub_best_fitness = sub_es.result[1]

                    if sub_best_fitness < self.best_fitness:
                        self.best_fitness = sub_best_fitness
                        self.best[init_vector.positions[i]] = sub_best
                    
                    # 将子代的最优解放入 global_Xw
                    self.global_Xw[init_vector.positions[i]] = sub_es.result[5]

                    # 将子代的 C 放入 global_C
                    self.global_C[np.ix_(init_vector.positions[i], init_vector.positions[i])] = sub_es.C

                self.sigma = sub_es.sigma

        if self.fes >= self.cur_logpoint * self.config.log_interval:
            self.cur_logpoint += 1
            self.cost.append(self.best_fitness)
        
        """
        get state, reward and check if it is done
        """
        
        if self.fes >= self.maxfes:
            is_done = 1
        else :
            is_done = 0

        if is_done:
            if len(self.cost) >= self.config.n_logpoint + 1:
                self.cost[-1] = self.best_fitness
            else:
                self.cost.append(self.best_fitness)

        return is_done #state, reward 留给RL的接口