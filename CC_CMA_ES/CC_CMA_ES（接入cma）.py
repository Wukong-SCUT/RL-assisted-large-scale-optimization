import cma
import numpy as np
import random
import sys
from tqdm import tqdm
from math import exp
import time
import matplotlib.pyplot as plt

from cec2013lsgo.cec2013 import Benchmark

from opfunu.cec_based.cec2013 import F82013

#best去掉
#200pop
#检查越界

#np 提高速度
#problem时间 

#metabox里面的问题测试，-> cmaes效果 时间 shift rotate 固定种子


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
        group_best.positions.append(Si) #记录每个子群的位置
        #subInfo.append(Si)

    # 对 best 进行按位编号，并按 subInfo 中的编号分组
    group_vec = [best[S] for S in group_best.positions] #记录每个子群的值
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


def ackley(x):
    term1 = -20 * np.exp(-0.2 * np.sqrt(1/D * np.sum(x**2)))
    term2 = -np.exp(1/D * np.sum(np.cos(2 * np.pi * x)))
    result = term1 + term2 + 20 + np.exp(1)
    return result

def problem(x):
    global fes
    fes += 1
    x = np.ascontiguousarray(x)
    return fun_fitness(x) 

# #cec接口
# func = F82013(D)

#cec2013lsgo接口
bench = Benchmark()
question = 1
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

#CC层初始化
CC_method = ["MiVD", "MaVD", "RD"]

#cma-es初始化 
es = cma.CMAEvolutionStrategy([random.random()*search_scope+info['lower'] for _ in range(D)], sigma, inopts={'popsize': 200}) #np 
es.C = np.eye(D) #设置初始协方差矩阵  
solutions = es.ask() 

es.tell(solutions, [problem(x) for x in solutions])#此处消耗了200次fes 

es.logger.add()

#记录器初始化
record = Record(cycle_num)

best = es.result[0] 
best_fitness = es.result[1]
record.best.insert(0,best_fitness) #记录第一次的best

cycle += 1
record.cycle_record.append(cycle)
record.fitness_record.append(best_fitness)


# 保存原始的 sys.stdout
original_stdout = sys.stdout


while fes < 3e6:

    #CC层选择：分解方式选择 sofmax np方便控制种子
    column_sum = np.sum(record.per_record, axis=1) #按行求和
    all_column_sum = exp(column_sum[0])+exp(column_sum[1])+exp(column_sum[2]) #求和
    choose_probability = [exp(column_sum[0])/all_column_sum,exp(column_sum[1])/all_column_sum,exp(column_sum[2])/all_column_sum]
    selected_option = random.choices(CC_method, choose_probability, k=1)[0] #选择一个
    record.method.append(selected_option)#记录选择的方式


    sys.stdout = original_stdout
    print("selected_option",selected_option)

    sys.stdout = original_stdout
    print(best_fitness)
    
    #此处init_vector为vec类
    init_vector = method_mapping[selected_option](D, m, es.C, best) 

    #进入子空间 
    for i in tqdm(range(m),desc="sub_cycle"):

        sys.stdout = original_stdout
        print(f"{best_fitness:e}")

        sub_dimosions = len(init_vector.elements[i]) # 子代的维度

        #从全局中提取子代的centroid xw
        sub_centroid = []
        for pos in init_vector.positions[i]:
            sub_centroid.append(es.mean[pos])
        

        #从全局中提取子代的C
        sub_C = np.zeros((sub_dimosions,sub_dimosions))
        for p,pos_1 in zip(range(sub_dimosions),init_vector.positions[i]):
            for q,pos_2 in zip(range(sub_dimosions),init_vector.positions[i]): #np提高速度
                sub_C[p][q]=es.C[pos_1][pos_2]
        

        # 重定向 sys.stdout 到一个空的文件对象，即禁止输出
        sys.stdout = open('/dev/null', 'w')
        sub_es = cma.CMAEvolutionStrategy(sub_centroid, sigma,inopts={'popsize': 50}) #,inopts={'popsize': 50}
        sub_es.C = sub_C #设置初始协方差矩阵

        sub_cycle = 0
        sub_cycle_max = 10 #最多给5000fes
        #sub_cycle_difference = 0
        # sub_cycle_flag = sub_cycle
        while not sub_es.stop() and sub_cycle < sub_cycle_max: 
            sub_cycle +=1
            offspring = sub_es.ask() #获取子代 边间检查 周期 or clip 试试

            #对子代进行封装，进入Vec类
            sub_population = []
            for _ in offspring:
                ind  = Vec(elements=[], positions=[]) 
                ind.elements = _
                ind.positions = init_vector.positions[i] 
                sub_population.append(ind) 

            #将子代的offspring转化为全局的offspring
            sub_offspring_to_global = [] #由sub_ind_to_global组成
            sub_ind_to_global = [] #将子代放回至best用以评估fitness

            for ind in sub_population:
                sub_ind_to_global = best.copy()
                for ele,pos in zip(ind.elements,ind.positions): #np提高速度
                    sub_ind_to_global[pos] = ele  
                sub_offspring_to_global.append(sub_ind_to_global)
            
            #sub_offspring_to_global[-1]=best #offspring：best 变换 ；best 

            #计算子代的fitness
            sub_es.tell(offspring, [problem(x) for x in sub_offspring_to_global]) #更新子代 测试problem需要多久 bbob 时间time

            #从子代中提取最优解
            sub_best = sub_es.result[0]
            sub_best_fitness = sub_es.result[1]

            if sub_best_fitness < best_fitness:
                best_fitness = sub_best_fitness
                #尽量避免for，np直接赋值
                for ele,pos in zip(sub_best,init_vector.positions[i]):
                    best[pos] = ele 
            
            #将子代的最优解放入sub_best_solution中 np
            for ele,pos in zip(sub_es.mean,init_vector.positions[i]):
                es.mean[pos] = ele

            #将sub_strategy的C放入global中 np
            for ele,pos in zip(sub_es.C,init_vector.positions[i]):
                for e,p in zip(ele,init_vector.positions[i]):
                    es.C[pos][p] = e

            #sub_cycle_difference = sub_cycle - sub_cycle_flag
            sub_es.logger.add()
            sys.stdout = original_stdout
            #sub_es.disp()

            sigma = sub_es.sigma 
    
    es.logger.add()  #添加日志
    # es.disp()


    cycle += 1
    record.cycle_record.append(cycle)
    record.fitness_record.append(best_fitness)


es.result_pretty()
sys.stdout = original_stdout
print(best_fitness)


# 绘制点线图
plt.plot(record.cycle_record,record.fitness_record, marker='o', linestyle='--', color='b', label='Data')

# 添加标题和标签
plt.title('Line Plot of Data')
plt.xlabel('cycle')
plt.ylabel('fitness')

# 显示图例
plt.legend()

# 显示图形
plt.savefig('line_plot_3.png')


            




        
