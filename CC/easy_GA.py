import random

# 定义问题的目标函数（这里以简单的函数 x^2 为例）
def objective_function(x):
    return x**2

# 初始化种群
def initialize_population(population_size, lower_bound, upper_bound):
    population = []
    for _ in range(population_size):
        individual = random.uniform(lower_bound, upper_bound)
        population.append(individual)
    return population

# 评估种群中每个个体的适应度
def evaluate_population(population):
    fitness_scores = [objective_function(individual) for individual in population]
    return fitness_scores

# 选择操作，使用轮盘赌算法
def selection(population, fitness_scores):
    selected_population = []
    total_fitness = sum(fitness_scores)
    probabilities = [fitness / total_fitness for fitness in fitness_scores]

    for _ in range(len(population)):
        selected_individual = random.choices(population, probabilities)[0]
        selected_population.append(selected_individual)

    return selected_population

# 交叉操作
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# 变异操作
def mutation(individual, mutation_rate):
    mutated_individual = [bit if random.random() > mutation_rate else 1 - bit for bit in individual]
    return mutated_individual



# 遗传算法主循环
def genetic_algorithm(population_size, num_generations, crossover_rate, mutation_rate):
    lower_bound = -10
    upper_bound = 10

    # 初始化种群
    population = initialize_population(population_size, lower_bound, upper_bound)

    for generation in range(num_generations):
        # 评估种群中每个个体的适应度
        fitness_scores = evaluate_population(population)

        # 选择
        selected_population_1 = selection(population, fitness_scores)
        selected_population_2 = selection(population, fitness_scores)

        # 交叉
        new_population = []
        for i in range(0, population_size, 2):
            if random.random() < crossover_rate:
                parent1 = selected_population_1
                parent2 = selected_population_2
                child1, child2 = crossover(parent1, parent2)
                new_population.extend([child1, child2])
            else:
                new_population.extend([selected_population_1, selected_population_2])

        # 变异
        for i in range(population_size):
            if random.random() < mutation_rate:
                new_population[i] = mutation(new_population[i], mutation_rate)

        population = new_population

    # 返回最终种群中适应度最高的个体
    best_individual = max(population, key=objective_function)
    return best_individual

# 设置参数并运行遗传算法
population_size = 50
num_generations = 100
crossover_rate = 0.8
mutation_rate = 0.01

result = genetic_algorithm(population_size, num_generations, crossover_rate, mutation_rate)
print("最优解:", result)
print("最优解的适应度值:", objective_function(result))
