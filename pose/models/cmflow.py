import torch.nn as nn
import torch
import numpy as np
import os
import torch.nn.functional as F
from utils.model_utils import *
from utils import *
import torchvision.models as models
import torch
import torch.nn as nn
class CMFlow(nn.Module):
    
    def __init__(self,args):
        
        super(CMFlow,self).__init__()
        
    
        self.npoints = args.num_points
        self.stat_thres = args.stat_thres
        
        ## multi-scale set feature abstraction 
        sa_radius = [2.0, 4.0, 8.0, 16.0]
        sa_nsamples = [4, 8, 16, 32]
        sa_mlps = [32, 32, 64]
        sa_mlp2s = [64, 64, 64]
        num_sas = len(sa_radius)
        self.mse_layer = MultiScaleEncoder(sa_radius, sa_nsamples, in_channel=3, \
                                         mlp = sa_mlps, mlp2 = sa_mlp2s)
            
        ## feature correlation layer (cost volumn)
        fc_inch = num_sas*sa_mlp2s[-1]*2  
        fc_mlps = [fc_inch,fc_inch,fc_inch]
        self.fc_layer = FeatureCorrelator(8, in_channel=fc_inch*2+3, mlp=fc_mlps)
        
        ## multi-scale set feature abstraction 
        ep_radius = [2.0, 4.0, 8.0, 16.0]
        ep_nsamples = [4, 8, 16, 32]
        ep_inch = fc_inch * 2 + 3
        ep_mlps = [fc_inch, int(fc_inch/2), int(fc_inch/8)]
        ep_mlp2s = [int(fc_inch/8), int(fc_inch/8), int(fc_inch/8)]
        num_eps = len(ep_radius)
        self.mse_layer2 = MultiScaleEncoder(ep_radius, ep_nsamples, in_channel=ep_inch, \
                                         mlp = ep_mlps, mlp2 = ep_mlp2s)
        
        ## heads
        sf_inch = num_eps * ep_mlp2s[-1]*2
        sf_mlps = [int(sf_inch/2), int(sf_inch/4), int(sf_inch/8)]
        self.fp = FlowHead(in_channel=sf_inch, mlp=sf_mlps)
        self.mp = MotionHead(in_channel=sf_inch, mlp=sf_mlps)
            
        
    def rigid_to_flow(self,pc,trans):
        
        h_pc = torch.cat((pc,torch.ones((pc.size()[0],1,pc.size()[2])).cuda()),dim=1)
        sf = torch.matmul(trans,h_pc)[:,:3] - pc
        return sf
    
    
    
    def Backbone(self,pc1,pc2,feature1,feature2):
        
        '''
        pc1: B 3 N
        pc2: B 3 N
        feature1: B 3 N
        feature2: B 3 N
        
        '''
        B = pc1.size()[0]
        N = pc1.size()[2]

        ## extract multi-scale local features for each point
        pc1_features = self.mse_layer(pc1,feature1)
        pc2_features = self.mse_layer(pc2,feature2)
        
        ## global features for each set
        gfeat_1 = torch.max(pc1_features,-1)[0].unsqueeze(2).expand(pc1_features.size()[0],pc1_features.size()[1],pc1.size()[2])
        gfeat_2 = torch.max(pc2_features,-1)[0].unsqueeze(2).expand(pc2_features.size()[0],pc2_features.size()[1],pc2.size()[2])
        
        ## concat local and global features
        pc1_features = torch.cat((pc1_features, gfeat_1),dim=1)
        pc2_features = torch.cat((pc2_features, gfeat_2),dim=1)
        
        ## associate data from two sets 
        cor_features = self.fc_layer(pc1, pc2, pc1_features, pc2_features)
        
        ## generate embeddings
        embeddings = torch.cat((feature1, pc1_features, cor_features),dim=1)
        prop_features = self.mse_layer2(pc1,embeddings)
        gfeat = torch.max(prop_features,-1)[0].unsqueeze(2).expand(prop_features.size()[0],prop_features.size()[1],pc1.size()[2])
         
        final_features = torch.cat((prop_features, gfeat),dim=1)
        
        return final_features


    def EgoMotionHead(self, flow, pc1, score):
        
        B = pc1.size()[0]
        N = pc1.size()[2]
        
        ## warped pc1 with scene flow estimation
        pc1_warp = pc1+flow
        
        # normalize score to weight
        score = score.squeeze(1) + 1e-4
        weight = (score) / (score.sum(dim=1).unsqueeze(1))
        ## estimate rigid transformation using initial scene flow
        trans = self.WeightedKabsch(pc1, pc1_warp, weight)
        
        return trans
    
    def refine_with_transform(self, flow, pc1, trans, mask):
        
        B = pc1.size()[0]
        # # from transformation to rigid scene flow
        sf_rg = self.rigid_to_flow(pc1,trans)
 
        # # use the rigid transformation and initial output in a weighted summation way
        # sf_agg = (1-scores) * flow + scores * sf_rg
        sf_agg = torch.zeros(sf_rg.size()).cuda()
        for b in range(B):
             sf_agg[b,:,mask[b]==1]= sf_rg[b,:,mask[b]==1]
             sf_agg[b,:,torch.logical_not(mask[b])]=flow[b,:,torch.logical_not(mask[b])]
       
        return sf_agg
    
    
    def WeightedKabsch(self, A, B, W):
    
        assert A.size() == B.size()
    
        batch_size, num_rows, num_cols = A.size()
       
        ## mask to 0/1 weights for motive/static points
        #W=W.type(torch.bool).unsqueeze(2)
        W = W.unsqueeze(2)
        # find mean column wise
        centroid_A = torch.sum(A.transpose(2,1).contiguous()*W, axis=1)
        centroid_B = torch.sum(B.transpose(2,1).contiguous()*W, axis=1)

    
        #print(W.sum(axis=1))
              
        # ensure centroids are 3x1
        centroid_A = centroid_A.reshape(batch_size,num_rows,1)
        centroid_B = centroid_B.reshape(batch_size,num_rows,1)
    
        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B

        H = torch.matmul(Am, Bm.transpose(2,1).contiguous()*W)

        # find rotation
        U, _, V = torch.svd(H)
        Z = torch.matmul(V,U.transpose(2,1).contiguous())
        # special reflection case
        d= (torch.linalg.det(Z) < 0).type(torch.int8)
        #d = torch.zeros(batch_size).type(torch.int8).cuda()
        # -1/1 
        d=d*2-1
        Vc = V.clone()
        Vc[:,2,:]*=-d.view(batch_size,1)
        R = torch.matmul(Vc,U.transpose(2,1).contiguous())
       
        t = torch.matmul(-R, centroid_A)+centroid_B
        
        Trans=torch.cat((torch.cat((R,t),axis=2), torch.tensor([0,0,0,1]).repeat(batch_size,1).cuda().view(batch_size,1,4)),axis=1)
    
        return Trans
                 
    def forward(self, pc1, pc2, feature1, feature2, label_m, mode):
        
        # extract backbone features 
        final_features = self.Backbone(pc1,pc2,feature1,feature2)
        
        # predict initial scene flow and classfication map
        output = self.fp(final_features)
        stat_cls = self.mp(final_features)
        
        # use pseudo mask label for ego-motion estimation during training
        if (mode=='train') and (label_m!=None):
            scores = label_m.unsqueeze(1)
        # use estimated motion mask for ego-motion estimation during inference
        else:
            scores = stat_cls
            
        # threshold the probabilities to obtain the binary mask output
        mask = (scores>self.stat_thres).squeeze(1)
        
        # non-parametric ego-motion estimation head
        pre_trans = self.EgoMotionHead(output, pc1, scores)
        
        # refine identified static points' scene flow with rigid transformation
        sf_agg = self.refine_with_transform(output,pc1,pre_trans,mask)
        
    
        return sf_agg, stat_cls, pre_trans, mask

    def TLIO(self, imu_data1, imu_data2):

        extracted_columns1 = imu_data1[:, 1:7]
        extracted_columns2 = imu_data2[:, 1:7]
        merged_matrix = np.concatenate((extracted_columns1, extracted_columns2), axis=0)
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Linear(resnet.fc.in_features, 16)
        imu_matrix_net = merged_matrix
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
        matrix = numpy_array.reshape(4, 4)

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
        sensor1_transform = matrix

        # 通过传感器2得到的变换矩阵
        sensor2_transform = rotation_matrix

        # 定义初始状态和协方差
        initial_state = np.eye(4)
        initial_covariance = np.eye(4)

        # 定义测量噪声协方差
        measurement_noise_covariance = np.eye(4) * 0.001

        # 执行扩展卡尔曼滤波融合
        fused_state, fused_covariance = state_update(initial_state, sensor1_transform, initial_covariance)
        fused_state, fused_covariance = state_update(fused_state, sensor2_transform, fused_covariance)

        return fused_state