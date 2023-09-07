import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from PIL.ImageColor import colormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# 从 .mat 文件中加载数据
data = scipy.io.loadmat('/media/zyw/T7/VINS-radar/Doppler_reshape_256_64_192/DopplerFFTOutresh_836.mat')
range_doppler_data = data['DopplerFFTOutresh']
file_path = '/home/zyw/桌面/RPDNet/results/out3.5/data_mask_500.npy'  # 替换为实际的文件路径
binary_data = np.load(file_path)
indices = np.argwhere(binary_data == 1)
extracted_data = range_doppler_data[indices[:, 0], indices[:, 1], :]
extracted_data = np.tile(extracted_data[np.newaxis, :, :], (1,1,1))
# 定义相关参数
num_channels = range_doppler_data.shape[2]
num_range_bins = range_doppler_data.shape[1]

max_range = 100  # 距离的最大值
max_doppler = 10  # 多普勒的最大值

azimuth_resolution = 360 / num_channels  # 方位角分辨率
elevation_resolution = 180 / num_range_bins  # 俯仰角分辨率

# 创建空数组来存储方位角、俯仰角和点云坐标
azimuth_angles = np.zeros((num_range_bins, num_channels))
elevation_angles = np.zeros((num_range_bins, num_channels))
xyz = np.zeros((num_range_bins, num_channels, 3))

# 遍历数据并计算方位角和俯仰角
for range_bin in range(num_range_bins):
    for channel in range(num_channels):
        # 获取当前 range-doppler 数据
        data = range_doppler_data[:, range_bin, channel]

        # 计算方位角
        max_doppler_bin = np.argmax(data)
        azimuth_angle = (channel - 1) * azimuth_resolution
        azimuth_angles[range_bin, channel] = azimuth_angle

        # 计算俯仰角
        max_range_bin = np.argmax(data)
        elevation_angle = (max_range_bin - 1) * elevation_resolution
        elevation_angles[range_bin, channel] = elevation_angle

# 根据方位角、俯仰角和距离计算点云的 XYZ 坐标
for range_bin in range(num_range_bins):
    for channel in range(num_channels):
        range_val = (range_bin - 1) * (max_range / num_range_bins)
        azimuth_angle = azimuth_angles[range_bin, channel]
        elevation_angle = elevation_angles[range_bin, channel]

        # 计算 XYZ 坐标
        x = range_val * np.sin(np.radians(azimuth_angle)) * np.cos(np.radians(elevation_angle))
        y = range_val * np.cos(np.radians(azimuth_angle)) * np.cos(np.radians(elevation_angle))
        z = range_val * np.sin(np.radians(elevation_angle))

        xyz[range_bin, channel, 0] = x
        xyz[range_bin, channel, 1] = y
        xyz[range_bin, channel, 2] = z

flattened_xyz = xyz.reshape(-1, 3)
# # 找到.npy文件中元素值为1的位置
# threshold = 50
# filtered_matrix = output_file.copy()
# filtered_matrix[:, :, 0][filtered_matrix[:, :, 0] < threshold] = 0
# 将数据展平成一维数组
distances = np.linalg.norm(xyz, axis=1)
normalize = plt.Normalize(distances.min(), distances.max())
colormap = cm.viridis
#
# accumulated_cloud = np.load(xyz)



#
# # 添加一个筛选条件，仅保留x值小于等于52的点
# filtered_indices = np.where((xyz[:, :, 0] <= 50))
# filtered_xyz = xyz[filtered_indices[0]]
# distances1 = np.linalg.norm(filtered_xyz, axis=1)# 从filtered_indices中获取第一个维度的索引
# filtered_distances = distances[filtered_indices[0]]  # 从distances中获取第一个维度的索引
#
# #print(filtered_xyz)
#
#
# normalize = plt.Normalize(distances.min(), distances.max())
# colormap = cm.viridis

# 创建一个三维绘图窗口
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
original_shape = (64, 192, 3)
# 计算展平后的数据点的数量
num_points = flattened_xyz.shape[0]

# 使用展平后的数据绘制点云
sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=distances, cmap=colormap, marker='o')
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_zlim(-15, 15)

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# # 过滤超出范围的点
# indices = np.where(
#     (xyz[:, 0] >= -50) & (xyz[:, 0] <= 60) &
#     (xyz[:, 1] >= -50) & (xyz[:, 1] <= 50) &
#     (xyz[:, 2] >= -50) & (xyz[:, 2] <= 50)
# )
# filtered_xyz = xyz[indices]
# # 创建 3D 坐标轴对象
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111, projection='3d')
#
# # 绘制散点图
# sc1 = ax1.scatter(filtered_xyz[:, 0], filtered_xyz[:, 1], filtered_xyz[:, 2])
#
# # 设置坐标轴范围
# ax1.set_xlim(-50, 60)
# ax1.set_ylim(-50, 50)
# ax1.set_zlim(-50, 50)
#
# # 显示散点图
# plt.show()
# # 更新散点图数据
# sc.set_offsets(filtered_xyz[:, :2])
# sc.set_3d_properties(filtered_xyz[:, 2], zdir='z')
# 添加颜色条
cbar = fig.colorbar(sc, ax=ax, pad=0.1, label='Distance')

# 显示图形
plt.show()



# # 创建一个三维绘图窗口
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制点云数据
# sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=distances, cmap=colormap, marker='o')
#
# # 设置坐标轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# # 添加颜色条
# cbar = fig.colorbar(sc, ax=ax, pad=0.1, label='Distance')
# # 显示图形
# plt.show()

# print(xyz)


# # 定义文件路径
# output_file_path = '/home/zyw/桌面/RPDNet/results/AugmentedPC/AugmentedPC1.5.txt'  # 替换为你想要保存文件的路径和名称
#
# # 打开文件以写入模式 ('w' 表示写入)
# with open(output_file_path, 'w') as output_file:
#     # 遍历每个点云坐标并将其写入文件
#     for range_bin in range(num_range_bins):
#         for channel in range(num_channels):
#             x, y, z = xyz[range_bin, channel]
#             # 将每个坐标组成的一行写入文件，用空格隔开
#             output_file.write(f"{x} {y} {z}\n")
