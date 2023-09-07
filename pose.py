import re
import numpy as np
from scipy.spatial.transform import Rotation as R

# 读取第一个txt文件
with open('/media/zyw/T7/VINS-radar/第六次/VINS/0831/vinsout/imu_propagate_1693365822870531321.txt', 'r') as file1:
    content1 = file1.read()

# 读取第二个txt文件
with open('/media/zyw/T7/VINS-radar/第六次/VINS/0831/vinsout/imu_propagate_1693365823225737095.txt', 'r') as file2:
    content2 = file2.read()
custom_path = '/media/zyw/T7/VINS-radar/第六次/VINS/0831/pose/'
# 使用正则表达式提取位置信息中的坐标数据
position_pattern = r'Position \(x, y, z\): \((-?\d+\.\d+), (-?\d+\.\d+), (-?\d+\.\d+)\)'
position_match1 = re.search(position_pattern, content1)
position_match2 = re.search(position_pattern, content2)

# 使用正则表达式提取方向信息中的四元数数据
orientation_pattern = r'Orientation \(x, y, z, w\): \((-?\d+\.\d+), (-?\d+\.\d+), (-?\d+\.\d+), (-?\d+\.\d+)\)'
orientation_match1 = re.search(orientation_pattern, content1)
orientation_match2 = re.search(orientation_pattern, content2)

if position_match1 and position_match2 and orientation_match1 and orientation_match2:
    # 提取位置信息中的坐标数据
    x1, y1, z1 = float(position_match1.group(1)), float(position_match1.group(2)), float(position_match1.group(3))
    x2, y2, z2 = float(position_match2.group(1)), float(position_match2.group(2)), float(position_match2.group(3))

    # 提取方向信息中的四元数数据
    q1 = np.array([float(orientation_match1.group(i)) for i in range(1, 5)])
    q2 = np.array([float(orientation_match2.group(i)) for i in range(1, 5)])

    # 计算平移向量
    translation = np.array([x2 - x1, y2 - y1, z2 - z1])

    # 将四元数转换为旋转矩阵
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)

    # 计算两个旋转矩阵之间的差异（旋转矩阵的乘积）
    rotation_matrix = (r2 * r1.inv()).as_matrix()

    # 合并为位姿矩阵
    pose_matrix = np.column_stack((rotation_matrix, translation))

    custom_filename = '10.npy'
    custom_full_path = custom_path + custom_filename

    # 保存为.npy文件
    np.save(custom_full_path, pose_matrix)
else:
    print("无法提取所需信息。")
