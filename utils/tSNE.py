# coding: utf-8

import ipdb
import numpy as np
from time import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets

# # Next line to silence pyflakes. This import is needed.
# Axes3D

n_points = 10
# X 是一个(1000, 3)的 2 维数据，color 是一个(1000,)的 1 维数据
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
# print(color)
# color = np.array([-0.97] * n_points)
color = ['b'] * n_points
n_neighbors = 10
n_components = 2

fig = plt.figure(figsize=(8, 8))
# 创建了一个 figure，标题为"Manifold Learning with 1000 points, 10 neighbors"
# plt.suptitle("Manifold Learning with %i points, %i neighbors"
#              % (1000, n_neighbors), fontsize=14)
#
# ax = fig.add_subplot(211, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
# ax.view_init(4, -72)  # 初始化视角
# plt.savefig('results/init.png')

'''t-SNE'''
t0 = time()
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
Y = tsne.fit_transform(X)  # 转换后的输出
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时
ax = fig.add_subplot(1, 1, 1)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("t-SNE (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
ax.yaxis.set_major_formatter(NullFormatter())

# plt.axis('tight')
# ipdb.set_trace()
# plt.show()
plt.savefig('results/test.png')
