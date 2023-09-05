import numpy as np

# 读取两帧毫米波雷达点云
def read_point_cloud(file_path):
    # 假设点云数据保存在txt文件中，每行为一个点的坐标（x, y, z）
    point_cloud = np.loadtxt(file_path)
    return point_cloud

# 定义一个名为net的函数，用于计算位姿trans
def net(point_cloud1, point_cloud2):
    # 在这里实现你的位姿计算逻辑，返回位姿trans
    trans = np.zeros((4, 4))  # 假设返回一个4x4的变换矩阵
    return trans

# 计算两个点的欧氏距离
def compute_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# 读取第一帧点云
point_cloud1 = read_point_cloud('point_cloud1.txt')

# 读取第二帧点云
point_cloud2 = read_point_cloud('point_cloud2.txt')

# 计算位姿
trans = net(point_cloud1, point_cloud2)

# 对第一帧点云进行坐标变换
transformed_point_cloud1 = np.matmul(trans, np.concatenate((point_cloud1.T, np.ones((1, point_cloud1.shape[0])))))

# 存储新的第二帧点云
new_point_cloud2 = []

# 遍历计算第二帧点云中每个点与新点云位置的距离
for point in point_cloud2:
    distance = compute_distance(point, transformed_point_cloud1[:3])
    if distance <= 1.0:  # 如果距离在1m范围内，则保留该点
        new_point_cloud2.append(point)

# 将新的第二帧点云保存到文件中
np.savetxt('new_point_cloud2.txt', np.array(new_point_cloud2))
distance = o3d.registration.evaluate_registration(point_cloud1，new_point_cloud2)
distance_1_to_2 = directed_hausdorff(point_cloud1, new_point_cloud2)
distance_2_to_1 = directed_hausdorff(new_point_cloud2，point_cloud1)
# 取两个方向的最大值作为Modified
Hausdorff
Distancemodified_hausdorff_distance = maxdistance_1_to_2, distance_2_to_1