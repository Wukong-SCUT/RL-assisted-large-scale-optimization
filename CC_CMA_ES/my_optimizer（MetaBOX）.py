import numpy as np
from math import sqrt, log
from optimizer.learnable_optimizer import Learnable_Optimizer

'''
变量名的意义
+-----------------+-----------------+-----------------+-----------------+
|global_dimension |问题维度          |ind_init         |指定数据类型,默认实向量|
+-----------------+-----------------+-----------------+-----------------+
|population       |群体              |ind_init         |指定数据类型,默认实向量|

'''

#一些必要的Class和函数
def ind_init(position):
    # 假设每个个体是一个实数向量
    return position

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

#Record类，用于记录各次迭代的最优值和CC层分配的问题
class Record:
    def __init__(self, cycle_num): #per_record是一个3*cycle_num的矩阵，用于记录每种分类方式的优化程度 
        #best只记录两次，一次是前面一次是现在的best
        self.best = [] 

        #per_record是一个3*cycle_num的矩阵，用于记录每种分类方式的优化程度
        '''
        +----------------+------------+--------------+--------------+------------------+
        | Group strategy | cycle0     | cycle1       | cycle2       | cycle3     ....  |
        +================+============+==============+==============+==================+
        | MiVD           | sigma_00   | sigma_01     | sigma_02     | sigma_03         |
        +----------------+------------+--------------+--------------+------------------+
        | MaVD           | sigma_10   | sigma_11     | sigma_12     | sigma_13         |
        +----------------+------------+--------------+--------------+------------------+
        | RD             | sigma_20   | sigma_21     | sigma_22     | sigma_23         |
        +----------------+------------+--------------+--------------+------------------+
        '''
        self.per_record = np.ones((3,cycle_num))

        self.best_centroid = [] #记录三种方法的centroid
        self.best_C = [] #记录三种方法的最优C

#CC层的分配策略
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

#两个必要映射
#映射到每个分解策略
method_mapping = {
    "MiVD": MiVD,
    "MaVD": MaVD,
    "RD": RD
}

max_index_mapping_method = {
    0: "MiVD",
    1: "MaVD",
    2: "RD"
}

#主策略
class MyOptimizer(Learnable_Optimizer):
    # 个体的协方差矩阵适应进化策略（CMA-ES）
    class CMA_ES(object): 
        r"""Covariance Matrix Adaptation Evolution Strategy (CMA-ES)"""
        def __init__(self, config):
            super().__init__(config)
            self.config = config

            #CMA-ES需要在config中定义的参数
            self.global_dimension = config.global_dimension # 全局维度
            self.sub_dimension = config.sub_dimension # 子空间维度
            self.params = config.kargs # 参数字典 注意此处接口

            #下面是CMA-ES的附属参数
            self.sigma = 5.0 # 个体的标准差
            self.centroid = [5.0]*self.sub_dimension # 个体的中心点
            self.dim = len(self.centroid) # 个体的维度
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


        def generate(self, ind_init):
            r"""Generate a population of :math:`\lambda` individuals of type
            *ind_init* from the current strategy.

            :param ind_init: A function object that is able to initialize an
                            individual from a list.
            :returns: A list of individuals.
            """
            arz = np.random.standard_normal((self.lambda_, self.dim)) # 个体的随机向量
            arz = self.centroid + self.sigma * np.dot(arz, self.BD.T) # 个体的随机向量乘以特征向量乘以特征值
            return [ind_init(a) for a in arz] # 生成个体

        def fitness(self,ind,problem):
            init_individual = np.zeros(self.global_dimension)
            for i, _ in zip(ind.elements,ind.positions):
                init_individual[_] = i
            self.fes += 1
            return problem.eval(init_individual)

        def sort(self,population): #注意update函数只有一个接口
            for ind in population:
                ind.fitness = self.fitness(ind)
            return sorted(population, key=lambda ind: ind.fitness, reverse=True) #求最大or最小值，reverse调整，降序是max，升序是min
        
        def update(self, population):
            """Update the current covariance matrix strategy from the
            *population*.

            :param population: A list of individuals from which to update the
                            parameters.
            """
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

        #这里是可以优化的
        def computeParams(self, params):
        #在算法的演化过程中根据用户提供的参数来动态调整算法的一些控制参数，以提高算法的性能和适应性。
            r"""Computes the parameters depending on :math:`\lambda`. It needs to
            be called again if :math:`\lambda` changes during evolution.

            :param params: A dictionary of the manually set parameters.
            """
            self.mu = params.get("mu", int(self.lambda_ / 2))
            rweights = params.get("weights", "superlinear")
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




    def __init__(self, config):
        """
        Parameter
        ----------
        config: An argparse.Namespace object for passing some core configurations such as maxFEs.
        """
        super().__init__(config)
        self.config = config
        self.global_dimension = config.global_dimension # 全局维度
        self.m = config.m
        
        self.cycle_num = config.cycle_num # 循环次数
        self.sub_cycle_num = config.sub_cycle_num # 子空间循环次数

        self.strategy = self.CMA_ES(config) # 选择的策略
        self.C = np.dot(self.strategy.BD, self.strategy.BD.T)

        #fes是一个计数器，用于记录函数评估的次数
        self.fes = 0

        """
        Do whatever other setup is needed
        """
  
    def init_population(self, problem):
        """ Called by method PBOEnv.reset.
            Init the population for optimization.
  
        Parameter
        ----------
        problem: a problem instance, you can call `problem.eval` to evaluate one solution.
  
        Must To Do
        ----------
        1. Initialize a counter named "fes" to record the number of function evaluations used.
        2. Initialize a list named "cost" to record the best cost at logpoints.
        3. Initialize a counter to record the current logpoint.
  
        Return
        ----------
        state: state features defined by developer.
        """
  
        """
        Initialize the population, calculate the cost using method problem.eval and renew everything (such as some records) that related to the current population.
        """
        best = self.strategy.generate(ind_init)[0] # 生成个体
        record = Record(self.cycle_num) 
        record.best.insert(0,problem.eval(best)) #记录第一次的best



        #self.fes = self.population_size  # record the number of function evaluations used
        self.cost = [self.best_cost]     # record the best cost of first generation
        self.cur_logpoint = 1            # record the current logpoint
        """
        calculate the state
        """
        return best,record #state RL接口
  
    def update(self, action, problem): #实际上是CC层的更新 
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

        best,record = self.init_population(self, problem)

        for CC_method in ["MiVD", "MaVD", "RD"]:
            init_vector = method_mapping[CC_method](self.global_dimension, self.m, self.C, best)
            # print(init_vector.positions)
            #进行一些赋值，为了后续记录per_record方便
            if CC_method == "MiVD":
                CC_method_num = 0
            elif CC_method == "MaVD":
                CC_method_num = 1
            else:
                CC_method_num = 2
            for cycle_num in range(self.sub_cycle_num):
                for i in range(self.m): #这是对sub空间的循环
                    sub_dimosions = len(init_vector.elements[i]) # 子代的维度
                    class sub_config:
                        global_dimension = self.global_dimension
                        sub_dimension = sub_dimosions
                        self.params = {} # 参数字典 注意此处接口
                    
                    sub_strategy = self.CMA_ES(sub_config) # 选择的策略
                    # 生成子代
                    offspring = sub_strategy.generate(ind_init)

                    #对子代进行封装，进入Vec类
                    sub_population = []
                    for _ in offspring:
                        ind  = Vec(elements=[], positions=[]) 
                        ind.elements = _
                        ind.positions = init_vector.positions[i]
                        sub_population.append(ind)

                    #更新子代
                    for j in range(0,5):
                        sub_strategy.update(sub_population)
                        sub_strategy.computeParams(sub_strategy.params)
                        offspring = sub_strategy.generate(ind_init)

                        sub_population = []
                        for _ in offspring:
                            ind  = Vec(elements=[], positions=[]) 
                            ind.elements = _
                            ind.positions = init_vector.positions[i]
                            sub_population.append(ind)

                    sub_best_solution = sub_strategy.sort(sub_population)[0]

                    for ele,pos in zip(sub_best_solution.elements,sub_best_solution.positions):
                        best[pos] = ele #此处是将各sub放入global中

                    #记录未改变strategy的centroid和C，不影响后续的计算
                    original_centroid = self.strategy.centroid
                    original_C = self.strategy.C


                    if cycle_num == self.sub_cycle_num-1:#记录最后一次的best 后续再调整
                        #将sub_strategy的centroid放入global中
                        for ele,pos in zip(sub_strategy.centroid,init_vector.positions[i]):
                            self.strategy.centroid[pos] = ele
                        
                        #将sub_strategy的C放入global中
                        for ele,pos in zip(sub_strategy.C,init_vector.positions[i]):
                            for e,p in zip(ele,init_vector.positions[i]):
                                self.strategy.C[pos][p] = e
                        
                        record.best_centroid.append(self.strategy.centroid)
                        record.best_C.append(self.strategy.C)

                        #将strategy的centroid和C还原
                        self.strategy.centroid = original_centroid
                        self.strategy.C = original_C
                
                best_fitness = problem.eval(best)
                record.best.insert(0,best_fitness)
                record.per_record[CC_method_num][cycle_num] = record.best[0]/record.best[1]
                record.best.pop() #只保留new_best

        #分解方法选择
        avg_column_sum = np.sum(record.per_record, axis=1)/cycle_num #按行求和
        max_index = np.argmax(avg_column_sum) #求最大值的索引

        #最优分解方法
        best_method = method_mapping[max_index_mapping_method[max_index]]
        print("best_method",max_index_mapping_method[max_index])

        #采取最优分解进行分解
        best_strategy = self.strategy(self.config)
        best_strategy.C = record.best_C[max_index]

        best_global_vector = best_strategy.generate(ind_init)[0]

        #此处有很大一部分继承了上面的操作，唯一的问题是此处的迭代可以增多
        init_vector = best_method(self.global_dimension, self.m, self.C, best_global_vector)
        for i in range(self.m): #这是对sub空间的循环
            sub_dimosions = len(init_vector.elements[i]) # 子代的维度
            class sub_config:
                        global_dimension = self.global_dimension
                        sub_dimension = sub_dimosions
                        self.params = {} # 参数字典 注意此处接口
                    
            sub_strategy = self.CMA_ES(sub_config) # 选择的策略
            # 生成子代
            offspring = sub_strategy.generate(ind_init)

            #对子代进行封装，进入Vec类
            sub_population = []
            for _ in offspring:
                ind  = Vec(elements=[], positions=[]) 
                ind.elements = _
                ind.positions = init_vector.positions[i]
                sub_population.append(ind)

            #更新子代
            for j in range(0,5):
                sub_strategy.update(sub_population)
                sub_strategy.computeParams(sub_strategy.params)
                offspring = sub_strategy.generate(ind_init)

                sub_population = []
                for _ in offspring:
                    ind  = Vec(elements=[], positions=[]) 
                    ind.elements = _
                    ind.positions = init_vector.positions[i]
                    sub_population.append(ind)

            sub_best_solution = sub_strategy.sort(sub_population)[0]

            for ele,pos in zip(sub_best_solution.elements,sub_best_solution.positions):
                best_global_vector[pos] = ele #此处是将各sub放入global中
        
        if self.fes >= self.cur_logpoint * self.config.log_interval:
            self.cur_logpoint += 1
            self.cost.append(self.best_cost)
        """
        get state, reward and check if it is done
        """
        if is_done:
            if len(self.cost) >= self.config.n_logpoint + 1:
                self.cost[-1] = self.best_cost
            else:
                self.cost.append(self.best_cost)
        return best_global_vector, #state, reward, is_done