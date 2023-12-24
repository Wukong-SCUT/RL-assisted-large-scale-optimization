import numpy as np
from math import log, sqrt, exp
from tqdm import tqdm
import random
import pandas as pd
import matplotlib.pyplot as plt
from cec2013lsgo.cec2013 import Benchmark


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def elliptic_function(x):
    D = len(x)
    return np.sum([10**(6*(i/(D-1))) * x[i-1]**2 for i in range(1, D+1)])

# 生成一些随机的三维点
x_values = np.linspace(-1000, 10, 1000)
y_values = np.linspace(-1000, 10, 1000)
X, Y = np.meshgrid(x_values, y_values)
Z = np.array([[elliptic_function(np.array([x, y])) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])

# 绘制三维图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.title('Elliptic Function in 3D')
plt.show()
plt.savefig('Elliptic Function in 3D.png')


