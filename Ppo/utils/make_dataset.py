import random

class Make_dataset:
    def __init__(self, divide_method):
        self.divide_method = divide_method
        

    def problem_set(self, train_or_test):
        # 生成训练集

        train_problem_set = []
        if self.divide_method == "random_divide":
            all_numbers = list(range(1, 16))
            train_problem_set = random.sample(all_numbers, 6)
            test_problem_set = list(set(all_numbers) - set(train_problem_set))
        elif self.divide_method == "train_sep":
            train_problem_set = [1, 2, 3]
            test_problem_set = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        elif self.divide_method == "train_sep_parsep":
            train_problem_set = [1, 2, 3, 4, 5, 10, 11]
            test_problem_set = [6, 7, 8, 9, 12, 13, 14, 15]
        elif self.divide_method == "train_sep_parsep_2":
            train_problem_set = [1, 2, 3, 8, 9, 10, 11]
            test_problem_set = [4, 5, 6, 7, 12, 13, 14, 15]

        if train_or_test == "train":
            return train_problem_set
        elif train_or_test == "test":
            return test_problem_set
    
    
# #测试
# Make_dataset = Make_dataset("train_sep_parsep_2")
# print(Make_dataset.problem_set("train"))
# print(Make_dataset.problem_set("test"))

# for i in range(len(Make_dataset.train_problem_set())):
#     print(1)

