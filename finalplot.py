import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
# 读取.txt文件并转化为numpy矩阵
file_path = '/media/zyw/T7/VINS-radar/第六次/mmWave/xyz/xyz_345.txt'  # 请替换为你的文件路径
data = np.loadtxt(file_path)
data = data[:, :3]  # 仅保留前三列表示点的三维坐标
# 定义要扩充到的点的总数
desired_points = 500
N = data.shape[0]

## 计算需要生成的额外点的数量
additional_points = desired_points - N

# 定义生成新点的采样范围（可以根据需求调整）
sampling_radius = 0.5  # 在已有点的周围生成新点的最大半径

# 初始化扩充后的点云数据
interpolated_data = data.copy()

# 使用随机采样在已有点的周围生成新点
for _ in range(additional_points):
    # 随机选择一个现有点
    existing_point = data[np.random.randint(N)]

    # 随机生成新点在球体内的坐标
    new_point = existing_point + np.random.uniform(-sampling_radius, sampling_radius, size=3)

    # 将新点添加到扩充后的点云数据中
    interpolated_data = np.vstack((interpolated_data, new_point))
# 计算点的距离
distances = np.linalg.norm(interpolated_data, axis=1)

# 创建一个归一化对象，用于颜色映射
normalize = plt.Normalize(distances.min(), distances.max())
colormap = cm.viridis

# 创建一个三维绘图窗口
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制三维散点图，使用颜色映射表示距离
sc = ax.scatter(interpolated_data[:, 0], interpolated_data[:, 1], interpolated_data[:, 2], c=distances, cmap=colormap, marker='o')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 添加颜色条
cbar = fig.colorbar(sc, ax=ax, pad=0.1, label='Distance')

# 显示图形
plt.show()
