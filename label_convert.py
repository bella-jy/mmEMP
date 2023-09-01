import os
import numpy as np
import scipy.io as sio
from scipy.fft import fft
import re
def process_mat_file(mat_file):
    # Load the .mat file using scipy
    mat_data = sio.loadmat(mat_file)
    matrix = mat_data['DopplerFFTOutinte']

    # Apply the threshold and create the binary matrix
    binary_matrix = np.where(matrix > 100, 1, 0)

    row_indices, col_indices = np.where(binary_matrix == 1)

    return binary_matrix, row_indices, col_indices

def process_3d_matrix(mat_file, row_indices, col_indices):
    # Load the 3D .mat file using scipy
    mat_data = sio.loadmat(mat_file)
    three_d_matrix = mat_data['DopplerFFTOutresh']

    angles = []
    positions = []
    for row, col in zip(row_indices, col_indices):
        # Extract the element from the 3D matrix
        element = three_d_matrix[row, col, :]

        # Apply FFT to the element to get angle information
        angle_info = np.fft.fft(element)
        angle = np.abs(angle_info)
        angle_mean = np.mean(angle)
        # 获取 three_d_matrix 这一行的距离信息
        row_fft_result = np.fft.fft(three_d_matrix[row, :, :])
        distance = np.abs(row_fft_result)
        distance_mean = np.mean(distance)
        # 获取 three_d_matrix 这一列的速度信息
        col_fft_result = np.fft.fft(three_d_matrix[:, col, :])
        velocity = np.abs(col_fft_result)
        velocity_mean = np.mean(velocity)
        angles.append(angle_info)
        x = distance_mean * np.cos(angle_mean) * 1e-5
        y = distance_mean * np.sin(angle_mean) * 1e-5
        z = velocity_mean * 1e-5
        position = (x, y, z)
        positions.append(position)
    return angles, positions

def save_binary_matrix_as_mat(binary_matrix, output_folder, idx):
    # Form the .mat filename with leading zeros
    mat_filename = os.path.join(output_folder, f'{idx:05d}.mat')

    # Save the binary matrix as a .mat file using scipy
    sio.savemat(mat_filename, {'binary_matrix': binary_matrix})

def extract_number(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        return -1

def process_txt_file(txt_file):
    # Load the .txt file and extract points into a matrix
    points_matrix = np.loadtxt(txt_file)
    return points_matrix

if __name__ == "__main__":
    mat_folder = '/media/zyw/T7/VINS-radar/Doppler_integrate_256_64'
    three_d_mat_folder = '/media/zyw/T7/VINS-radar/Doppler_reshape_256_64_192'
    txt_folder = '/media/zyw/T7/VINS-radar/vinspointcloudout/'
    output_folder = '/home/zyw/桌面/RPDNet/adc_data (2)/label6_thre3000'

    # Get a list of all .mat files in the folder, sorted by filename
    mat_files = [f for f in os.listdir(mat_folder) if f.endswith('.mat')]
    mat_files = sorted(mat_files, key=extract_number)

    three_d_mat_files = [f for f in os.listdir(three_d_mat_folder) if f.endswith('.mat')]
    three_d_mat_files = sorted(three_d_mat_files, key=extract_number)

    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    txt_files.sort()

    # Process each .mat file and save as binary .mat file
    for idx, mat_file in enumerate(mat_files):
        # Convert the .mat file to a binary matrix
        mat_file_path = os.path.join(mat_folder, mat_file)
        binary_matrix, row_indices, col_indices = process_mat_file(mat_file_path)
        #
        three_d_mat_file = os.path.join(three_d_mat_folder, three_d_mat_files[idx])
        angles, position = process_3d_matrix(three_d_mat_file, row_indices, col_indices)
        positions = np.array(position)

        txt_file = os.path.join(txt_folder, txt_files[idx // 2])  # 使用整除以匹配txt文件
        points_matrix = process_txt_file(txt_file)
        # Save the binary matrix as a .mat file

        for pos in positions:
            distances = np.linalg.norm(points_matrix - pos, axis=1)
            if np.any(distances > 3000):
                binary_matrix[row_indices, col_indices] = 0

        save_binary_matrix_as_mat(binary_matrix, output_folder, idx)

    print("Processing complete. Binary matrices saved as .mat files.")
