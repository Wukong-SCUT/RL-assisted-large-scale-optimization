from cec2013lsgo.cec2013 import Benchmark

class Make_dataset:
    def __init__(self, D, m, C, best):
        self.D = D
        self.m = m
        self.C = C
        self.best = best

        self.Benchmark = Benchmark()

    def train_problem_set(self):
        # 生成训练集
        train_problem_set = []
        for i in range(1, 6):
            train_problem_set.append(self.Benchmark.get_function(i))

        return train_problem_set
    
    def test_problem_set(self):
        # 生成测试集
        test_problem_set = []
        for i in range(6, 16):
            test_problem_set.append(self.Benchmark.get_function(i))

        return test_problem_set