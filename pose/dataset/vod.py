#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import h5py
import numpy as np
import ujson
from sklearn.cluster import DBSCAN
from torch.utils.data import Dataset
import torchvision.models as models
import torch.nn as nn
import torch
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors


class vodDataset(Dataset):

    def __init__(self, args, root='/home/zyw/preprocess_res/flow_smp/', partition='train', textio=None):

        self.npoints_per_stride = 60
        self.num_strides = 10
        self.npoints = args.num_points
        self.textio = textio
        self.calib_path = 'dataset/vod_radar_calib.txt'
        self.res = {'r_res': 0.2, # m
                    'theta_res': 1.5 * np.pi/180, # radian
                    'phi_res': 1.5 *np.pi/180  # radian
                }
        self.read_calib_files()
        self.eval = args.eval
        self.partition = partition
        self.root = os.path.join(root, self.partition)
        self.interval = 0.10
        self.clips = sorted(os.listdir(self.root),key=lambda x:int(x.split("_")[1]))
        self.samples = []
        self.clips_info = []

        for clip in self.clips:
            clip_path = os.path.join(self.root, clip)
            samples = sorted(os.listdir(clip_path),key=lambda x:int(x.split("/")[-1].split("_")[0]))
            for idx in range(len(samples)):
                samples[idx] = os.path.join(clip_path, samples[idx])
            if self.eval:
                self.clips_info.append({'clip_name':clip,
                                    'index': [len(self.samples),len(self.samples)+len(samples)]
                                })
            if clip[:5] == 'delft':
                self.samples.extend(samples)

        self.textio.cprint(self.partition + ' : ' +  str(len(self.samples)))
    

    def __getitem__(self, index):
        
        sample = self.samples[index]
        with open(sample, 'rb') as fp:
            data = ujson.load(fp)

        data_1 = np.array(data["pc1"]).astype('float32')
        data_2 = np.array(data["pc2"]).astype('float32')

        interval = self.interval
        pos_1 = data_1[:, 0:3]
        pos_2 = data_2[:, 0:3]
        feature_1 = data_1[:, [4, 4, 4]]
        feature_2 = data_2[:, [4, 4, 4]]

        # GT labels and pseudo FG labels (from lidar)
        gt_labels = np.array(data["gt_labels"]).astype('float32')
        pse_labels = np.array(data["pse_labels"]).astype('float32')

        # GT mask or pseudo FG mask (from lidar)
        gt_mask = np.array(data["gt_mask"])
        pse_mask = np.array(data["pse_mask"])

        # use GT labels and motion seg. mask for evaluation on val and test set
        if self.partition in ['test','val', 'train_anno']:
            labels = gt_labels
            mask = gt_mask
            opt_flow =  np.zeros((pos_1.shape[0],2)).astype('float32')
            radar_u =  np.zeros(pos_1.shape[0]).astype('float32')
            radar_v =  np.zeros(pos_1.shape[0]).astype('float32')
        # use pseudo FG flow labels and FG mask as supervision signals for training 
        else:
            labels = pse_labels
            mask = pse_mask
            opt_info = data["opt_info"]
            opt_flow = np.array(opt_info["opt_flow"]).astype('float32')
            radar_u = np.array(opt_info["radar_u"]).astype('float32')
            radar_v = np.array(opt_info["radar_v"]).astype('float32')
        # static points transformation from frame 1 to frame 2  
        trans = np.linalg.inv(np.array(data["trans"])).astype('float32')
        # imu_1 = np.array(data["imu1"]).astype('float32')
        # imu_2 = np.array(data["imu2"]).astype('float32')
        # feature = np.array(data["feature"]).astype('float32')
        # feature = feature[:, [1, 0, 2]]
        ## downsample to npoints to enable fast batch processing (not in test)
        if not self.eval:

            npts_1 = pos_1.shape[0]
            npts_2 = pos_2.shape[0]

            # num_strides = self.num_strides
            # sample_idx1 = []
            # for stride_idx in range(num_strides):
            #     start_idx = int(stride_idx * (npts_1 // num_strides))
            #     end_idx = int((stride_idx + 1) * (npts_1 // num_strides)) if stride_idx != num_strides - 1 else npts_1
            #     stride_points = pos_1[start_idx:end_idx]
            #     stride_npts = stride_points.shape[0]
            #     if stride_npts < self.npoints_per_stride:
            #         stride_sample_idx = np.arange(start_idx, end_idx)
            #         stride_sample_idx = np.append(stride_sample_idx, np.random.choice(stride_sample_idx,
            #                                                                           self.npoints_per_stride - stride_npts,
            #                                                                           replace=True))
            #     else:
            #         stride_sample_idx = np.linspace(start_idx, end_idx - 1, self.npoints_per_stride).astype(np.int64)
            #     sample_idx1.append(stride_sample_idx)
            #
            # sample_idx1 = np.concatenate(sample_idx1)
            #
            # # 对于data_2也做相同的修改
            # sample_idx2 = []
            # for stride_idx in range(num_strides):
            #     start_idx = int(stride_idx * (npts_2 // num_strides))
            #     end_idx = int((stride_idx + 1) * (npts_2 // num_strides)) if stride_idx != num_strides - 1 else npts_2
            #     stride_points = pos_2[start_idx:end_idx]
            #     stride_npts = stride_points.shape[0]
            #     if stride_npts < self.npoints_per_stride:
            #         stride_sample_idx = np.arange(start_idx, end_idx)
            #         stride_sample_idx = np.append(stride_sample_idx, np.random.choice(stride_sample_idx,
            #                                                                           self.npoints_per_stride - stride_npts,
            #                                                                           replace=True))
            #     else:
            #         stride_sample_idx = np.linspace(start_idx, end_idx - 1, self.npoints_per_stride).astype(np.int64)
            #     sample_idx2.append(stride_sample_idx)
            #
            # sample_idx2 = np.concatenate(sample_idx2)
            if npts_1<self.npoints:
                sample_idx1 = np.arange(0,npts_1)
                sample_idx1 = np.append(sample_idx1, np.random.choice(npts_1,self.npoints-npts_1,replace=True))
            else:
                sample_idx1 = np.random.choice(npts_1, self.npoints, replace=False)
            if npts_2<self.npoints:
                sample_idx2 = np.arange(0,npts_2)
                sample_idx2 = np.append(sample_idx2, np.random.choice(npts_2,self.npoints-npts_2,replace=True))
            else:
                sample_idx2 = np.random.choice(npts_2, self.npoints, replace=False)
            pos_1 = pos_1[sample_idx1,:]
            pos_2 = pos_2[sample_idx2,:]
            feature_1 = feature_1[sample_idx1, :]
            feature_2 = feature_2[sample_idx2, :]
            radar_u = radar_u[sample_idx1]
            radar_v = radar_v[sample_idx1]
            opt_flow = opt_flow[sample_idx1,:]

            labels = labels[sample_idx1,:]
            # feature = feature[sample_idx1,:]
            mask = mask[sample_idx1]
        #
        def TLIO(imu_data1, imu_data2):

            extracted_columns1 = imu_data1[:, 1:7]
            extracted_columns2 = imu_data2[:, 1:7]
            merged_matrix = np.concatenate((extracted_columns1, extracted_columns2), axis=0)
            resnet = models.resnet18(pretrained=True)
            resnet.fc = nn.Linear(resnet.fc.in_features, 12)
            imu_matrix_net = merged_matrix.reshape(2, 6)
            # 转换为PyTorch张量并调整形状
            input_tensor = torch.Tensor(imu_matrix_net).unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.repeat(1, 3, 1, 1)
            # 调整输入尺寸
            input_tensor = nn.functional.interpolate(input_tensor, size=(224, 224))

            # 使用ResNet18进行特征提取和预测
            output_tensor = resnet(input_tensor)

            # 将Tensor变量转换为NumPy数组
            numpy_array = output_tensor.detach().numpy()

            # 将NumPy数组重新形状为4x4的矩阵
            matrix = numpy_array.reshape(3, 4)
            new_row = np.array([0, 0, 0, 1])
            matrix = np.vstack((matrix, new_row))
            def integrate_acceleration(acceleration, dt):
                velocity = np.cumsum(acceleration * dt, axis=0)
                return velocity

            def integrate_angular_velocity(angular_velocity, dt):
                rotation_matrix = np.eye(3)
                for i in range(len(angular_velocity)):
                    omega = angular_velocity[i]
                    omega_skew = np.array([[0, -omega[2], omega[1]],
                                           [omega[2], 0, -omega[0]],
                                           [-omega[1], omega[0], 0]])
                    delta_rotation = np.eye(3) + omega_skew * dt
                    rotation_matrix = np.dot(delta_rotation, rotation_matrix)
                return rotation_matrix

            # 读取2x7的矩阵，表示两帧IMU的时间和3轴加速度和角速度
            imu_matrix = np.concatenate((imu_data1, imu_data2), axis=0)

            # 提取时间、加速度和角速度数据
            time = imu_matrix[:, 0]
            acceleration = imu_matrix[:, 1:4]
            angular_velocity = imu_matrix[:, 4:7]

            # 预计分方法计算旋转和平移
            dt = time[1] - time[0]
            translation = integrate_acceleration(acceleration, dt)
            rotation = integrate_angular_velocity(angular_velocity, dt)

            # 构成4x4的旋转矩阵
            rotation_matrix = np.eye(4)
            rotation_matrix[:3, :3] = rotation
            rotation_matrix[:3, 3] = translation[-1]

            def state_update(state, measurement, covariance):
                # 预测步骤：更新状态和协方差
                predicted_state = state
                predicted_covariance = covariance

                # 测量步骤：计算卡尔曼增益
                innovation = measurement - predicted_state
                innovation_covariance = predicted_covariance + measurement_noise_covariance
                kalman_gain = np.dot(predicted_covariance, np.linalg.inv(innovation_covariance))

                # 更新状态和协方差
                updated_state = predicted_state + np.dot(kalman_gain, innovation)
                updated_covariance = np.dot((np.eye(4) - kalman_gain), predicted_covariance)

                return updated_state, updated_covariance

            # 通过传感器1得到的变换矩阵
            sensor1_transform = rotation_matrix

            # 通过传感器2得到的变换矩阵
            sensor2_transform = rotation_matrix

            # 定义初始状态和协方差
            initial_state = np.eye(4)
            initial_covariance = np.eye(4)

            # 定义测量噪声协方差
            measurement_noise_covariance = np.eye(4) * 0.00000001

            # 执行扩展卡尔曼滤波融合
            fused_state, fused_covariance = state_update(initial_state, sensor1_transform, initial_covariance)
            # fused_state, fused_covariance = state_update(fused_state, sensor2_transform, fused_covariance)

            return fused_state, rotation_matrix

        # tlio_trans, rotation_matrix= TLIO(imu_1, imu_2)
        # tlio_trans = np.eye(4)
        # rotation_matrix = np.eye(4)
        # return pos_1, pos_2, imu_1, imu_2, feature_1, feature_2, trans, labels, mask, interval, radar_u, radar_v, opt_flow, tlio_trans, rotation_matrix
        return pos_1, pos_2, feature_1, feature_2, trans, labels, mask, interval, radar_u, radar_v, opt_flow

    def read_calib_files(self):
        with open(self.calib_path, "r") as f:
            lines = f.readlines()
            intrinsic = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Intrinsics
            extrinsic = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Extrinsic
            extrinsic = np.concatenate([extrinsic, [[0, 0, 0, 1]]], axis=0)
            intrinsic = [566.8943529201453, 0.0, 322.10094802162763, 0.0, 0.0, 567.7699123433893, 242.8149724252196,
                        0.0, 0.0, 0.0, 1.0, 0.0]
            intrinsic = np.array(intrinsic).reshape((3, 4))
            extrinsic = np.array([[1.0, 0.0, 0.0, 0.128],
                                      [0.0, -1.0, 0.0, -0.044],
                                      [0.0, 0.0, -1.0, 0.106], [0.0, 0.0, 0.0, 1]])
        self.camera_projection_matrix = intrinsic
        self.t_camera_radar = extrinsic

    def __len__(self):
        return len(self.samples)
