import gym
from gym import spaces
import sys
import numpy as np
import cma
import warnings

from cec2013lsgo.cec2013 import Benchmark

# 保存原始的 sys.stdout
original_stdout = sys.stdout

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

class cmaes(gym.Env):
    def __init__(self, m, sub_popsize,question):
        super(cmaes, self).__init__() 
        bench = Benchmark()

        self.m = m
        self.sub_popsize = sub_popsize
        self.info = bench.get_info(question)
        self.D = self.info["dimension"]
        self.ub = self.info['upper']
        self.lb = self.info['lower']
        self.sigma = (self.ub - self.lb) * 0.5
        self.fun_fitness = bench.get_function(question)

        # 定义状态和动作空间
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32) #此处还没有设置好

        self.action_space = spaces.Discrete(3)

        # 初始化环境的内部状态等
        self.state = np.zeros(3)
        self.done = False

    def problem(self,x):
        global fes
        fes += self.sub_popsize
        x = np.ascontiguousarray(x)
        return self.fun_fitness(x)

    def reset(self):

        #cma-es初始化
        random_vectors = np.random.uniform(self.lb, self.ub, size=(200, self.D))
        self.global_C = np.eye(self.D)
        self.global_Xw = np.random.uniform(self.lb, self.ub, size=self.D)

        # 计算每个向量对应的函数值
        function_values = np.array([self.problem(random_vectors[i]) for i in np.arange(200)])

        # 找到最小值和对应的索引
        min_index = np.argmin(function_values)
   
        self.best= random_vectors[min_index]
        self.best_fitness = function_values[min_index]

        self.fes = 0

        self.done = False

        self.state = np.zeros(3) #此处还没有设置好

        return self.state

    def step(self, action):

        # 执行动作，更新环境状态，并返回新的状态、奖励、是否终止以及额外信息
        if action == 0:
            init_vector = MiVD(self.D,self.m,self.global_C,self.best)
        elif action == 1:
            init_vector = MaVD(self.D,self.m,self.global_C,self.best)
        elif action == 2:
            init_vector = RD(self.D,self.m,self.global_C,self.best)
        

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
                sub_es = cma.CMAEvolutionStrategy(sub_centroid, sigma, {'popsize': self.sub_popsize,'bounds': [self.info['lower'], self.info['upper']]}) 
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
                    sub_ind_to_global = self.best.copy()
                    sub_ind_to_global[ind.positions] = ind.elements
                    sub_offspring_to_global.append(sub_ind_to_global)

                #计算子代的fitness
                sub_es.tell(offspring, np.array([self.problem(sub_offspring_to_global[i]) for i in range(self.sub_popsize)])) #更新子代

                #从子代中提取最优解
                sub_best = sub_es.result[0]
                sub_best_fitness = sub_es.result[1]

                if sub_best_fitness < best_fitness:
                    best_fitness = sub_best_fitness
                    self.best[init_vector.positions[i]] = sub_best
                
                # 将子代的最优解放入 global_Xw
                self.global_Xw[init_vector.positions[i]] = sub_es.result[5]

                # 将子代的 C 放入 global_C
                self.global_C[np.ix_(init_vector.positions[i], init_vector.positions[i])] = sub_es.C

            sigma = sub_es.sigma
        
        # 例如，这里定义一个简单的奖励函数
        reward = self.best_fitness

        self.state = np.zeros(3) #此处还没有设置好

        self.done = False #此处还没有设置好

        return self.state, reward, self.done, {}

