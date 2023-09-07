import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 读取.txt文件，将其转化为numpy矩阵，并取前三列
data = np.loadtxt('/media/zyw/T7/VINS-radar/第六次/mmWave/xyz/xyz_854.txt')[:, :3]
data1 = np.load('/media/zyw/T7/VINS-radar/第六次/VINS/0831/newpoint/output_transformed_data10.npy')
# data2 = np.vstack((data, data1))
# # 读取.npy文件，这里是三乘四的变换矩阵
# transformation_matrix = np.load('/media/zyw/T7/VINS-radar/第六次/VINS/0831/pose/10.npy')
# transformed_coordinates = np.hstack((data2, np.ones((data2.shape[0], 1))))
# # 使用dot函数将坐标通过变换矩阵得到新的坐标位置
# # transformed_data = np.dot(data, transformation_matrix.T)  # 使用转置是为了匹配矩阵维度
# new_coordinates = np.dot(transformed_coordinates, transformation_matrix .T)[:, :3]
# # 保存新的坐标位置为.npy文件
# np.save('/media/zyw/T7/VINS-radar/第六次/VINS/0831/newpoint/output_transformed_data10.npy', new_coordinates)

# 创建一个三维绘图窗口
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 提取三维坐标的X、Y、Z分量
X = data1[:, 0]
Y = data1[:, 1]
Z = data1[:, 2]

# 绘制三维散点图
ax.scatter(X, Y, Z, c='b', marker='o')

# 设置坐标轴标签
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_zlabel('Z轴')

# 显示图形
plt.show()