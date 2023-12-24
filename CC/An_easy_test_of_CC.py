import numpy

#定义相似度计算函数
def match_score(x,y,t): #t是子群的编码 x是匹配串 y是目标串
    x_length = len(x)
    score = 0
    for i in range(x_length):
        if x[i] == y[i+t]:
            score += 1
    return score

#match初始化
match_0 = [0,0,0]
match_1 = [0,0,0]
match_2 = [0,0,0]
#targe目标
targe = [0,1,0,1,0,1,0,1,0]


num = 0
while match_score_sum != 3:
    #match_0 match_1 match_2 从0和1里面随机选取
    match_0 = numpy.random.randint(0,2,1)#随机生成0或1 随机没有体现之间的相互关联性
    match_1 = numpy.random.randint(0,2,1)
    match_2 = numpy.random.randint(0,2,1)
    match_score_sum = match_score(match_0,targe,0)+match_score(match_1,targe,3)+match_score(match_2,targe,6)
    num += 1

print("匹配次数：",num)