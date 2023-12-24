import numpy as np
from math import log, sqrt, exp
from tqdm import tqdm
import random
import pandas as pd
import matplotlib.pyplot as plt
from cec2013lsgo.cec2013 import Benchmark

'''
------------------参数目录------------------
fes:全局适应度评价次数
D:全局维度
m:子群数量
cycle_num:record记录的次数
----------------CMA-ES参数目录--------------
global_dimension:全局维度(用于与dim区分)
centroid:个体的中心点（均值）
sigma:步长
C:协方差矩阵
pc、ps:C和sigma的演化路径选择参数
'''

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
        Si = sortedIndex[int(i):int(D):int(s)]
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
        self.sigma = sigma # 步长

        self.best = np.arange(global_dimension,dtype=np.double) #best初始化设置为0

        # 个体的历史最佳位置（pc）通常用于跟踪个体在搜索空间中的最佳位置，而全局历史最佳位置（ps）用于跟踪整个种群的最佳位置
        self.pc = np.zeros(self.dim) #数组表示了个体历史最佳位置
        self.ps = np.zeros(self.dim) #数组表示了全局历史最佳位置
        #计算PSO算法中的一个参数 chiN，通常用于调整粒子群的速度更新公式
        self.chiN = sqrt(self.dim) * (1 - 1. / (4. * self.dim)
                                      + 1. / (21. * self.dim ** 2)) 

        self.C = self.params.get("cmatrix", np.identity(self.dim)) # 个体的协方差矩阵
        self.diagD, self.B = np.linalg.eigh(self.C) # 个体的特征值和特征向量

        indx = np.argsort(self.diagD) # 对特征值进行排序
        #self.diagD = self.diagD[indx] ** 0.5 # 对特征值进行开方
        self.diagD = np.sqrt(np.maximum(self.diagD[indx], 0.00001))
        self.B = self.B[:, indx] # 对特征向量进行排序
        self.BD = self.B * self.diagD  # 个体的特征向量乘以特征值


        #self.cond = self.diagD[indx[-1]] / self.diagD[indx[0]] # 个体的条件数
        # 添加除以零的检查
        if self.diagD[indx[0]] != 0:
            self.cond = self.diagD[indx[-1]] / self.diagD[indx[0]]
        else:
            # 处理分母为零的情况（避免除以零）
            self.cond = float(1000)  # 或者设置为其他适当的值

        self.lambda_ = self.params.get("lambda_", int(4 + 3 * log(self.dim))) # 个体的子代数量 
        self.update_count = 0 # 个体的更新次数
        self.computeParams(self.params) 
        #使用给定的参数字典 self.params 来计算一些参数，并将结果用于进一步的操作

    def generate(self, ind_init):
        arz = np.random.standard_normal((self.lambda_, self.dim)) # 个体的随机向量
        arz = self.centroid + self.sigma * np.dot(arz, self.BD.T) # 个体的随机向量乘以特征向量乘以特征值
        return [ind_init(a) for a in arz] # 生成个体

    def fitness(self,ind): #此处best为全局best
        init_individual = self.best
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
        population,_ = self.sort(population) # 对个体进行排序

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
        
        #要避免出现0
        self.diagD = np.sqrt(np.maximum(self.diagD[indx], 0.00001))
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

#fitness计算函数
def problem(x): 
    global fes
    fes += 1
    return fun_fitness(x)

#设置格式
def ind_init(position):
    # 假设每个个体是一个实数向量
    return position


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

#------------------测试------------------

#问题初始化
bench = Benchmark()
question = 3
fun_fitness = bench.get_function(question)
info = bench.get_info(question)
search_scope = info['upper']-info['lower']
sigma = search_scope


#超参数
D = info["dimension"] #全局问题维度
m = 20
cycle_num = 5

#记录器初始化
record = Record(cycle_num)
#fes 用于记录全局适应度评价次数
fes = 0
#循环计数器
cycle = 0
#记录sub最后的fitness
sub_fitness = 0 

#CC层初始化
CC_method = ["MiVD", "MaVD", "RD"]

#CMA-ES：strategy初始化
strategy = CMA_ES(centroid=[random.random()*search_scope+info["lower"] for _ in range(D)], sigma=sigma, lambda_=50,global_dimension=D) 

#初始化全局种群
original_offspring = strategy.generate(ind_init) 
population = []
for _ in original_offspring:
    ind  = Vec(elements=[], positions=[]) 
    ind.elements = _
    ind.positions = np.arange(D)
    population.append(ind)


sort_offspring,original_fitness = strategy.sort(population)
best = sort_offspring[0].elements #初始化全局最优
strategy.best = best

record.best.insert(0,original_fitness) #记录第一次的best
record.fitness_record.append(original_fitness)  #记录第一次 fitness
cycle += 1
record.cycle_record.append(cycle)


while fes < 3e6:
    #CC层选择：分解方式选择
    column_sum = np.sum(record.per_record, axis=1) #按行求和
    all_column_sum = exp(column_sum[0])+exp(column_sum[1])+exp(column_sum[2]) #求和
    choose_probability = [exp(column_sum[0])/all_column_sum,exp(column_sum[1])/all_column_sum,exp(column_sum[2])/all_column_sum]
    selected_option = random.choices(CC_method, choose_probability, k=1)[0] #选择一个
    record.method.append(selected_option)#记录选择的方式
    print("selected_option",selected_option)

    init_vector = method_mapping[selected_option](D, m, strategy.C, best) #best继承

#----------------子空间-----------------------
    for i in range(m): #这是对sub空间的循环
        sub_dimosions = len(init_vector.elements[i]) # 子代的维度

        #从全局中提取子代的centroid
        sub_centroid = []
        for pos in init_vector.positions[i]:
            sub_centroid.append(strategy.centroid[pos])

        #从全局中提取子代的C
        sub_C = np.zeros((sub_dimosions,sub_dimosions))
        for p,pos_1 in zip(range(sub_dimosions),init_vector.positions[i]):
            for q,pos_2 in zip(range(sub_dimosions),init_vector.positions[i]):
                sub_C[p][q]=strategy.C[pos_1][pos_2]
        
        sub_strategy = CMA_ES(centroid=sub_centroid, sigma=sigma, lambda_=50,global_dimension=D,cmatrix=sub_C)
        sub_strategy.best = best

        # 生成子代
        offspring = sub_strategy.generate(ind_init)
        offspring[-1] = init_vector.elements[i]

        #对子代进行封装，进入Vec类
        sub_population = []
        for _ in offspring:
            ind  = Vec(elements=[], positions=[]) 
            ind.elements = _
            ind.positions = init_vector.positions[i]
            sub_population.append(ind)
        
        sub_best_solution = init_vector.elements[i]

        #更新子代
        for j in range(0,5):
            sub_strategy.update(sub_population)
            sub_strategy.computeParams(sub_strategy.params)
            offspring = sub_strategy.generate(ind_init)
            offspring[-1] = sub_best_solution
            
            #对子代进行封装，进入Vec类
            sub_population = []
            for _ in offspring:
                ind  = Vec(elements=[], positions=[]) 
                ind.elements = _
                ind.positions = init_vector.positions[i]
                sub_population.append(ind)
            
            #对子代进行排序，生成初始化最优子代
            sort_solution,sub_original_fitness = sub_strategy.sort(sub_population)
            sub_best_solution = sort_solution[0].elements

            #更新最优
            if sub_original_fitness < record.fitness_record[-1]:
                #将sub_best_solution放入best中
                for ele,pos in zip(sub_best_solution,init_vector.positions[i]): #best已经被sub更改
                    best[pos] = ele
                sub_strategy.best = best

                for ele,pos in zip(sub_strategy.centroid,init_vector.positions[i]):
                    strategy.centroid[pos] = ele

                #将sub_strategy的C放入global中
                for ele,pos in zip(sub_strategy.C,init_vector.positions[i]):
                    for e,p in zip(ele,init_vector.positions[i]):
                        strategy.C[pos][p] = e
                
                sub_fitness = sub_original_fitness
    #------------------子空间结束------------------
    strategy.best = best
    record.fitness_record.append(sub_fitness)

    cycle += 1
    best_fitness = record.fitness_record[-1]
    print(f"cycle_num:{cycle}",f"best_fitness:{best_fitness}")

    record.cycle_record.append(cycle)
    record.best.insert(0,best_fitness)
    record.per_record[method_mapping_num[selected_option]].pop()
    record.per_record[method_mapping_num[selected_option]].insert(0,abs((record.best[0]-record.best[1])/record.best[1]))

    record.best.pop() #只保留new_best


last_fitness = record.fitness_record[-1]
print("last_fitness:",last_fitness)


#Excel表格记录

#记录方法
vector_series_method = pd.Series(record.method)
vector_series_fitness = pd.Series(record.fitness_record)

# 创建一个 DataFrame，将 Series 放入其中
df = pd.DataFrame({'Method': vector_series_method,'Fitness': vector_series_fitness})

# 将 DataFrame 写入 Excel 文件
excel_filename = 'Method_output_3.xlsx'  # 输出的 Excel 文件名
df.to_excel(excel_filename, index=False)

#图像记录
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




