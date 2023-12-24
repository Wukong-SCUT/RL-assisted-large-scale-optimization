#导入库
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap import cma #导入cma模块
from optimizer.basic_optimizer import Basic_Optimizer

#定义DEAP_CMAES类 继承"Basic_Optimizer"
class DEAP_CMAES(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        config.NP = 50 #种群大小
        self.__config = config #配置
        self.__toolbox = base.Toolbox() #工具箱
        self.__creator = creator #创建者
        self.__algorithm = algorithms #算法
        self.__creator.create("Fitnessmin", base.Fitness, weights=(-1.0,)) #创建适应度 设置为最小化
        self.__creator.create("Individual", list, fitness=creator.Fitnessmin) #创建个体
        self.log_interval = config.log_interval #日志间隔 这是避免日志过多

    def run_episode(self, problem): #运行一次episode 一次优化过程

        def problem_eval(x): #评估函数
            if problem.optimum is None:
                fitness = problem.eval(x)
            else:
                fitness = problem.eval(x) - problem.optimum
            return fitness,   # return a tuple

        self.__toolbox.register("evaluate", problem_eval)
        strategy = cma.Strategy(centroid=[problem.ub] * self.__config.dim, sigma=0.5, lambda_=self.__config.NP) #策略
        #centroid: 一个列表，表示优化过程中分布的中心。
        #在这里，使用了 problem.ub 的列表，长度为 self.__config.dim。
        #sigma: 控制分布的标准差。
        #lambda_: 控制每一代的个体数量，这里设为 self.__config.NP。
        self.__toolbox.register("generate", strategy.generate, creator.Individual)
        #generate: 一个函数，用于生成个体
        self.__toolbox.register("update", strategy.update)
        #update: 一个函数，用于更新策略

        hof = tools.HallOfFame(1)#记录最优个体
        stats = tools.Statistics(lambda ind: ind.fitness.values)#记录统计信息
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        fes = 0 #记录函数评估次数
        log_index = 0 #记录日志次数
        cost = []
        while True:
            _, logbook = self.__algorithm.eaGenerateUpdate(self.__toolbox, ngen=1, stats=stats, halloffame=hof, verbose=False)
            fes += len(logbook) * self.__config.NP #更新函数评估次数
            #在每达到一定的函数评估次数时记录当前的适应度值，以便后续分析和日志输出
            if fes >= log_index * self.log_interval: 
                log_index += 1
                cost.append(hof[0].fitness.values[0])
            #判断是否达到最大函数评估次数或者达到最优解
            if problem.optimum is None: #如果没有最优解
                done = fes >= self.__config.maxFEs #达到最大函数评估次数
            else:
                done = fes >= self.__config.maxFEs or hof[0].fitness.values[0] <= 1e-8 #达到最大函数评估次数或者达到最优解
            if done:
                if len(cost) >= self.__config.n_logpoint + 1: #如果记录的适应度值个数大于等于n_logpoint+1
                    cost[-1] = hof[0].fitness.values[0] #将最后一个适应度值设为最优解
                else:
                    cost.append(hof[0].fitness.values[0]) #否则将最优解添加到适应度值列表中
                break
        return {'cost': cost, 'fes': fes} #返回适应度值列表和函数评估次数
