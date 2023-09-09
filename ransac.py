import cv2
import numpy as np
import random
from scipy.optimize import least_squares
import os
import torch
import glob
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
def extract_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def estimate_pose_ransac(triangulated_points, keypoints2, keypoints1, good_matches, num_iterations, reprojection_threshold):
    best_inliers = 0
    best_rvec = None
    best_tvec = None
    best_inlier_indices = None

    for _ in range(num_iterations):
        # 随机选择五组匹配点
        random_indices = random.sample(range(len(triangulated_points)), 5)
        zeros = np.zeros((5, 1))
        src_pts = np.array([triangulated_points[i] for i in random_indices]).reshape(-1, 3)
        dst_pts = np.array([keypoints2[i].pt for i in random_indices]).reshape(-1,  2)
        # src_pts = np.concatenate((src_pts, zeros), axis=1)
        # dst_pts = np.concatenate((dst_pts, zeros), axis=1)
        camera_matrix = np.float64([[1000.0, 0.0, 320.0],
                                  [0.0, 1000.0, 240.0],
                                  [0.0, 0.0, 1.0]])
        dist_coeffs = np.array([0.1, -0.2, 0.001, 0.002, 0])
        # Check if the selected points are collinear (this can cause issues)

        # 通过这五组点计算位姿
        _, rvec, tvec, inliers = cv2.solvePnPRansac(src_pts, dst_pts, camera_matrix , dist_coeffs)

        # 计算当前模型的内点数量
        num_inliers = np.sum(inliers)

        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_rvec = rvec
            best_tvec = tvec
            best_inlier_indices = inliers

    # 输出异常组和近似组的位姿计算结果
    outlier_indices = np.where(best_inlier_indices == 0)[0]
    similar_indices = np.where(best_inlier_indices == 1)[0]
    outlier_rvecs = [best_rvec for _ in outlier_indices]
    outlier_tvecs = [best_tvec for _ in outlier_indices]
    similar_rvecs = [best_rvec for _ in similar_indices]
    similar_tvecs = [best_tvec for _ in similar_indices]

    # 判断异常组的类型
    if best_inliers == 0:
        # 全为异常点
        outlier_points = [dst_pts[i].pt for i in outlier_indices]
        outlier_points1 = [keypoints1[good_matches[i].query].pt for i in outlier_indices]
    else:
        # 部分异常点和部分内点
        outlier_points = []
        outlier_points1 = []
        for i in outlier_indices:
            if best_inlier_indices[i] == 0:
                outlier_points = [dst_pts[i] for i in outlier_indices]
                outlier_points1.append(keypoints1[good_matches[i].queryIdx].pt)

    return best_rvec, best_tvec, outlier_rvecs, outlier_tvecs, similar_rvecs, similar_tvecs, outlier_points, outlier_points1

def triangulate_points(keypoints1, keypoints2, rvec, tvec):
    points_3d = []
    proj_matrix1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    proj_matrix2 = np.hstack((rvec, tvec))
    if len(keypoints1) < len(keypoints2):
        keypoints_to_iterate = keypoints1
    else:
        keypoints_to_iterate = keypoints2
    for i in range(len(keypoints_to_iterate)):
        point1 = keypoints1[i].pt
        point2 = keypoints2[i].pt
        point1_homogeneous = np.array([point1[0], point1[1]])
        point2_homogeneous = np.array([point2[0], point2[1]])
        point_4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, point1_homogeneous, point2_homogeneous)
        point_3d = point_4d[:3] / point_4d[3]
        points_3d.append(point_3d)
    return points_3d

def calculate_initial_pose(keypoints1, keypoints2, good_matches, camera_matrix):
    # Extract corresponding points for the selected matches
    src_pts = np.array([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.array([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # Calculate essential matrix using the matches
    essential_matrix, _ = cv2.findEssentialMat(src_pts, dst_pts, camera_matrix)

    # Recover the rotation and translation from the essential matrix
    _, rvec, tvec, _ = cv2.recoverPose(essential_matrix, src_pts, dst_pts, camera_matrix)

    return rvec, tvec

def residual(x, A, b):
    return (np.dot(A, x) - b).flatten()

def construct_overdetermined_equation(params, similar_rvecs, similar_tvecs, outlier_points, outlier_points1, k):
    p1 = params[:4]
    delta_d = params[4:]
    q1 = outlier_points
    q1 = np.array(q1).reshape(1, 2)
    q1new_matrix = np.hstack((q1, np.array([[1]])))
    q1new_matrix = np.dot(k, (p1 + delta_d)[:3])
    q2 = outlier_points1
    q2 = np.array(q2).reshape(1, 2)
    q1new_matrix = np.hstack((q2, np.array([[1]])))
    q1new_matrix = np.dot(k, p1[:3])
    rotation_vector = np.array(similar_rvecs).reshape(3, )
    # 计算旋转角度和旋转轴
    rotation_angle = np.linalg.norm(rotation_vector)
    rotation_axis = rotation_vector / rotation_angle
    # 计算旋转矩阵
    P = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
      ])
    rotation_matrix = np.identity(3) + np.sin(rotation_angle) * P + (1 - np.cos(rotation_angle)) * np.dot(P, P)
    transform_matrix = np.array(similar_tvecs).reshape(3, 1)
    new_row = np.array([[1]])
    transform = np.vstack((transform_matrix, new_row))
    RT = np.hstack((rotation_matrix, transform_matrix))
    t1 = RT[0]
    t1 = t1.reshape(-1, 1)
    t2 = RT[1]
    t2 = t2.reshape(-1, 1)
    t3 = RT[2]
    t3 = t3.reshape(-1, 1)
    q3 = np.array([t1.T @ (p1 + delta_d) / t3.T @ (p1 + delta_d), t2.T @ (p1 + delta_d) / t3.T @ (p1 + delta_d)])
    q4 = np.array([t1.T @ p1 / t3.T @ p1, t2.T @ p1 / t3.T @ p1])
    a = np.dot(p1 - transform[:, 0].T, p1 +delta_d - transform[:, 0])
    residual1 = 0.5 * np.linalg.norm(np.dot(p1, p1.T) + 0.5 *np.dot(p1 + delta_d, p1 + delta_d).T - 0.5 * 2 * np.dot(p1.T, p1 + delta_d) * (np.dot(q1, q3) / (np.linalg.norm(q1) * np.linalg.norm(q3))) -np.dot(delta_d, delta_d.T)) + np.linalg.norm(0.5 * np.dot(p1 - transform[:, 0], p1 - transform[:, 0]).T + 0.5 *np.dot(p1 + delta_d - transform[:, 0], p1 + delta_d - transform[:, 0]).T - 0.5 * 2 * np.dot(p1 - transform[:, 0].T, p1 +delta_d - transform[:, 0]) * (np.dot(q4, q2) / (np.linalg.norm(q4) * np.linalg.norm(q2)))-np.dot(delta_d, delta_d.T))
    residual = 0.5 * np.linalg.norm(
        np.dot(p1, p1.T) + 0.5 * np.dot(p1 + delta_d, p1 + delta_d).T - 0.5 * 2 * np.dot(p1.T, p1 + delta_d) * (
                    np.dot(q1, q3) / (np.linalg.norm(q1) * np.linalg.norm(q3))) - np.dot(delta_d,
                                                                                         delta_d.T) - 0.5 * np.dot(
            p1 - transform[:, 0], p1 - transform[:, 0]).T + 0.5 * np.dot(p1 + delta_d - transform[:, 0],
                                                                         p1 + delta_d - transform[:,
                                                                                        0]).T - 0.5 * 2 * np.dot(
            p1 - transform[:, 0].T, p1 + delta_d - transform[:, 0]) * (
                    np.dot(q4, q2) / (np.linalg.norm(q4) * np.linalg.norm(q2))))
    #outlier_points = np.array(outlier_points).reshape(2, 1)
    # 添加分量z，初始化为0，得到3x1的向量
    #matrix_3x1 = np.vstack([outlier_points, [0]])
    # 计算向量的模长
    #norm = np.linalg.norm(matrix_3x1)
    # 归一化分量
    #normalized_matrix_3x1 = matrix_3x1 / norm
    # 构造3x4的变换矩阵
    #A = np.dot(K, rotation_matrix)
    #b = normalized_matrix_3x1 - np.dot(K, transform_matrix)
    # 初始解
    #x0 = np.zeros(3)

    # 最小二乘求解
    #result = least_squares(residual, x0, args=(A, b))

    # 输出结果
    #delta_x = result.x
    #P1 = delta_x[:3]
    #delta_d = delta_x[3:]

    return residual1
# Specify the folder containing the PNG files
# image_folder = '/media/zyw/T7/VINS-radar/第六次/VINS/0831/infra1out'
#
# # Get a list of all PNG files in the folder
# png_files = sorted(glob.glob(os.path.join(image_folder, '*.png')))
# for i in range(len(png_files) - 1):
    # Read two consecutive images
image1 = cv2.imread('/media/zyw/T7/VINS-radar/第六次/VINS/0831/infra1out/1693365835985835791.png')
image2 = cv2.imread('/media/zyw/T7/VINS-radar/第六次/VINS/0831/infra1out/1693365836019191504.png')
# 读取两幅图像
transform = T.Compose([T.ToTensor()])
input1 = transform(image1).unsqueeze(0)
input2 = transform(image2).unsqueeze(0)
# Detect objects in both images
with torch.no_grad():
    output1 = model(input1)
    output2 = model(input2)
    boxes1 = output1[0]['boxes'].cpu().numpy()
    labels1 = output1[0]['labels'].cpu().numpy()
    scores1 = output1[0]['scores'].cpu().numpy()
    boxes2 = output2[0]['boxes'].cpu().numpy()
    labels2 = output2[0]['labels'].cpu().numpy()
    scores2 = output2[0]['scores'].cpu().numpy()

    # 设置置信度阈值
confidence_threshold = 0.95

# 过滤低置信度的检测结果
filtered_boxes1 = boxes1[scores1 > confidence_threshold]
filtered_labels1 = labels1[scores1 > confidence_threshold]
filtered_boxes2 = boxes2[scores2 > confidence_threshold]
filtered_labels2 = labels2[scores2 > confidence_threshold]

descriptors1_inside = []
descriptors2_inside = []
keypoint_coordinates1 = []
keypoint_coordinates2 = []
keypoints_outside1 = []
keypoints_outside2 = []
for box1 in filtered_boxes1:
    x1, y1, x2, y2 = map(int, box1)
    object_image1 = image1[y1:y2, x1:x2]
    keypoints_inside1, descriptors = extract_sift_features(object_image1)
    key_coordinates1 = [kp.pt for kp in keypoints_inside1]
    x1_outside, y1_outside, x2_outside, y2_outside = 0, 0, image1.shape[1], image1.shape[0]
    if x2_outside >= x1 and x1_outside <= x2 and y2_outside >= y1 and y1_outside <= y2:
        # Adjust the region to avoid overlap
        if x1_outside < x1:
            x1_outside = x1
        if x2_outside > x2:
            x2_outside = x2
        if y1_outside < y1:
            y1_outside = y1
        if y2_outside > y2:
            y2_outside = y2
            # Extract SIFT features for the region outside the box
    object_image1_outside = image1[y1_outside:y2_outside, x1_outside:x2_outside]
    keypoints_temp1, descriptors = extract_sift_features(object_image1_outside)
    descriptors1_inside.append(descriptors)
    keypoint_coordinates1.append(key_coordinates1)
    keypoints_outside1.append(keypoints_temp1)
for box2 in filtered_boxes2:
    x1, y1, x2, y2 = map(int, box2)
    object_image2 = image2[y1:y2, x1:x2]
    keypoints_inside2, descriptors = extract_sift_features(object_image2)
    key_coordinates2 = [kp.pt for kp in keypoints_inside2]
    x1_outside, y1_outside, x2_outside, y2_outside = 0, 0, image2.shape[1], image2.shape[0]
    if x2_outside >= x1 and x1_outside <= x2 and y2_outside >= y1 and y1_outside <= y2:
        # Adjust the region to avoid overlap
        if x1_outside < x1:
            x1_outside = x1
        if x2_outside > x2:
            x2_outside = x2
        if y1_outside < y1:
            y1_outside = y1
        if y2_outside > y2:
            y2_outside = y2
        # Extract SIFT features for the region outside the box
    object_image2_outside = image2[y1_outside:y2_outside, x1_outside:x2_outside]
    keypoints_temp2, descriptors = extract_sift_features(object_image2_outside)
    descriptors2_inside.append(descriptors)
    keypoint_coordinates2.append(key_coordinates2)
    keypoints_outside2.append(keypoints_temp2)
    # 提取SIFT特征点
keypoints1, descriptors1 = extract_sift_features(image1)
keypoints2, descriptors2 = extract_sift_features(image2)

# 特征匹配
good_matches = match_features(descriptors1, descriptors2)

# 使用RANSAC估计位姿
num_iterations = 100
reprojection_threshold = 5.0
camera_matrix = np.float64([[1000.0, 0.0, 320.0],
                                [0.0, 1000.0, 240.0],
                                [0.0, 0.0, 1.0]])

initial_rvec, initial_tvec = calculate_initial_pose(keypoints1, keypoints2, good_matches, camera_matrix)
# Triangulate recovered 3D points using initial pose
triangulated_points = triangulate_points(keypoints1, keypoints2, initial_rvec, initial_tvec)
# triangulated_points_out = triangulate_points(keypoints_temp1, keypoints_temp2, initial_rvec, initial_tvec)
rvec, tvec, outlier_rvecs, outlier_tvecs, similar_rvecs, similar_tvecs, outlier_points, outlier_points1 = estimate_pose_ransac(triangulated_points, keypoints2, keypoints1, good_matches, num_iterations, reprojection_threshold)

# P, X = construct_overdetermined_equation(camera_matrix, similar_rvecs, similar_tvecs, outlier_points)
#
# P_obj, delta_d = construct_overdetermined_equation(camera_matrix, similar_rvecs, similar_tvecs, outlier_points)

# 扁平化嵌套列表中的元组
flat_list = [item for sublist in keypoint_coordinates1 for item in sublist]

# 创建一个N乘2的矩阵，其中N是元素数量
matrix = np.array(flat_list)
flat_list1 = [item for sublist in keypoint_coordinates2 for item in sublist]

# 创建一个N乘2的矩阵，其中N是元素数量
matrix1 = np.array(flat_list1)
initial_params = np.ones(8)
num_samples = len(matrix1)  # Number of point pairs
all_solution_p1 = []  # Initialize a list to store all solution_p1
all_solution_delta_d = []
all_solution_p2 = []
for i in range(num_samples):
    point1 = matrix[i]
    point2 = matrix1[i]
    # 使用最小二乘法求解
    result = least_squares(construct_overdetermined_equation, initial_params, args=(similar_rvecs, similar_tvecs, point1, point2, camera_matrix))
    solution_p1 = result.x[:4]
    solution_p1 = solution_p1.reshape(-1, 1)
    fourth_element = solution_p1[3, 0]
    solution_p1 = solution_p1 / fourth_element
    solution_delta_d = result.x[4:]
    solution_delta_d = solution_delta_d.reshape(-1, 1)
    fourth_element1 = solution_delta_d[3, 0]
    solution_delta_d = solution_delta_d / fourth_element1
    solution_p1 = solution_p1[:3]
    solution_delta_d = solution_delta_d[:3]
    solution_p2 = solution_p1 + solution_delta_d
    # Store the result in the list
    all_solution_p1.append(solution_p1)
    all_solution_delta_d.append(solution_delta_d)
    all_solution_p2.append(solution_p2)
selected_p1 = random.sample(all_solution_p1, 5)
selected_p2 = random.sample(all_solution_p2, 5)
# 合并两个列表
combined_data = triangulated_points + selected_p1
combined_data1 = triangulated_points + selected_p2
combined_data2 = triangulated_points + all_solution_p1
combined_data3 = triangulated_points + all_solution_p2
# 将所有数组垂直堆叠在一起
stacked_array1 = np.vstack(combined_data2)

# 将堆叠后的数组重塑为N行3列的矩阵
matrix_combined_data2 = stacked_array1.reshape(-1, 3)
stacked_array2 = np.vstack(combined_data3)

# 将堆叠后的数组重塑为N行3列的矩阵
matrix_combined_data3 = stacked_array2.reshape(-1, 3)
stacked_array3 = np.vstack(all_solution_delta_d)

# 将堆叠后的数组重塑为N行3列的矩阵
matrix_all_solution_delta_d = stacked_array3.reshape(-1, 3)
# # Save the triangulated points and solutions to TXT files
# output_folder = '/home/zyw/桌面/RPDNet/adc_data (2)/visualfeature'
# output_file1 = os.path.join(output_folder, f'{i + 1}.txt')
# output_file2 = os.path.join(output_folder, f'{i + 2}.txt')
#
# with open(output_file1, 'w') as file:
#     for data in combined_data:
#         file.write(np.array_str(data))
#         file.write('\n')
#
# with open(output_file2, 'w') as file:
#     for data in combined_data1:
#         file.write(np.array_str(data))
#         file.write('\n')
#打印结果
print("Estimated rotation vector:", rvec)
print("Estimated translation vector:", tvec)
print("Triangulated 3D points:", triangulated_points )
print("Outlier rvecs:", outlier_rvecs)
print("Outlier tvecs:", outlier_tvecs)
print("Similar rvecs:", similar_rvecs)
print("Similar tvecs:", similar_tvecs)
print("Outlier points:", outlier_points)
# print("P_obj:", P_obj)
# print("delta_d:", delta_d)
# print("solution_p1:", solution_p1)
# print("solution_delta_d:", solution_delta_d)


# 指定要保存的文件路径
# file_path = "/home/zyw/桌面/RPDNet/adc_data (2)/visualfeature/4.npy"
# file_path1 = "/home/zyw/桌面/RPDNet/adc_data (2)/visualfeature/5.npy"
file_path2 = "/home/zyw/桌面/RPDNet/adc_data (2)/visualfeature/13.npy"
# 使用np.save保存矩阵为.npy文件
# np.save(file_path, matrix_combined_data2)
# np.save(file_path1, matrix_combined_data3)
np.save(file_path2, matrix_all_solution_delta_d)
print(f"Matrix saved to {file_path2}")


