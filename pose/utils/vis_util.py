import os
import argparse
import sys
import torch
from tqdm import tqdm
import cv2
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy.spatial.transform import Rotation as R
import matplotlib.ticker as ticker
from utils.vis_ops import flow_xy_to_colors


def transform_to_ego(pc, T):
    pos = (np.matmul(T[0:3, 0:3], pc) + T[0:3, 3:4])

    return pos


def get_matrix_from_ext(ext):
    N = np.size(ext, 0)
    if ext.ndim == 2:
        rot = R.from_euler('ZYX', ext[:, 3:], degrees=True)
        rot_m = rot.as_matrix()
        tr = np.zeros((N, 4, 4))
        tr[:, :3, :3] = rot_m
        tr[:, :3, 3] = ext[:, :3]
        tr[:, 3, 3] = 1
    if ext.ndim == 1:
        rot = R.from_euler('ZYX', ext[3:], degrees=True)
        rot_m = rot.as_matrix()
        tr = np.zeros((4, 4))
        tr[:3, :3] = rot_m
        tr[:3, 3] = ext[:3]
        tr[3, 3] = 1
    return tr


def visulize_result_2D(pc1, pc2, pc1_warp, gt, num_pcs, args):
    SIDE_RANGE = (-50, 50)
    FWD_RANGE = (0, 100)
    RES = 0.15625 / 2
    gt=gt[0].transpose(0,1).contiguous().cpu().detach().numpy()

    npcs1 = pc1.size()[2]
    npcs2 = pc2.size()[2]

    pc_1 = pc1[0].cpu().numpy()
    pc_2 = pc2[0].cpu().numpy()
    pc1_warp_gt=pc_1+gt
    wp_1 = pc1_warp[0].cpu().detach().numpy()
    # wp_1 = pc1_warp_gt
    radar_ext = np.array([0.06, -0.2, 0.7, -3.5, 2, 180])
    ego_to_radar = get_matrix_from_ext(radar_ext)
    pc_1 = transform_to_ego(pc_1, ego_to_radar)
    pc_2 = transform_to_ego(pc_2, ego_to_radar)
    wp_1 = transform_to_ego(wp_1, ego_to_radar)

    x_max = int((FWD_RANGE[1] - FWD_RANGE[0]) / RES)
    y_max = int((SIDE_RANGE[1] - SIDE_RANGE[0]) / RES)
    im = np.zeros([y_max, x_max, 3], dtype=np.uint8) + 255

    x_img_1 = np.floor((pc_1[0]) / RES).astype(int)
    y_img_1 = np.floor(-(pc_1[1] + SIDE_RANGE[0]) / RES).astype(int)
    for i in range(npcs1):
        im = cv2.circle(im, (x_img_1[i], y_img_1[i]), 2, (0, 0, 255), 2)

    x_img_2 = np.floor((pc_2[0]) / RES).astype(int)
    y_img_2 = np.floor(- (pc_2[1] + SIDE_RANGE[0]) / RES).astype(int)
    for j in range(npcs2):
        im = cv2.circle(im, (x_img_2[j], y_img_2[j]), 2, (255, 0, 0), 2)

    x_img_w = np.floor((wp_1[0]) / RES).astype(int)
    y_img_w = np.floor(-(wp_1[1] + SIDE_RANGE[0]) / RES).astype(int)
    x_img_w[x_img_w > (x_max - 1)] = x_max - 1
    y_img_w[y_img_w > (y_max - 1)] = y_max - 1
    x_img_w[x_img_w < 0] = 0
    y_img_w[y_img_w < 0] = 0

    for i in range(npcs1):
        # im=cv2.line(im, (x_img_1[i],y_img_1[i]), (x_img_w[i],y_img_w[i]), (34,139,34), 1)
        im = cv2.circle(im, (x_img_w[i], y_img_w[i]), 2, (0, 0, 255), 2)
    path_im = args.vis_path_2d + '/' + '{}.png'.format(num_pcs)
    im=cv2.putText(im, 'Frame1', (600,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    im=cv2.putText(im, 'Frame2', (600,75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    im=cv2.putText(im, 'Frame1_wrap', (600,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (34,139,34), 2, cv2.LINE_AA)

    cv2.imwrite(path_im, im)

#do not necessary
def visulize_result_2D_pre(pc1, pc2, pred_f, pc1_warp, gt, num_pcs, mask, args):

    pc_1=pc1[0].cpu().numpy()
    pc_2=pc2[0].cpu().numpy()
    pred_f=pred_f[0].cpu().detach().numpy()
    pc1_warp=pc1_warp[0].cpu().detach().numpy()
    gt=gt[0].transpose(0,1).contiguous().cpu().detach().numpy()
    pc1_warp_gt=pc_1+gt
    error = np.linalg.norm(pc1_warp - pc1_warp_gt, axis = 0)
    mask = mask[0].cpu().numpy()

    fig = plt.figure(figsize=(10, 6))

    x_locator = MultipleLocator(10)
    y_locator = MultipleLocator(10)

    x_flow, y_flow = pred_f[0], pred_f[1]
    rad = np.sqrt(np.square(x_flow) + np.square(y_flow))
    x_gt, y_gt = gt[0], gt[1]
    
    rad_max = np.max(rad)
    epsilon = 1e-5
    x_flow = x_flow / (rad_max + epsilon)
    y_flow = y_flow / (rad_max + epsilon)
 
    x_gt = x_gt / (rad_max + epsilon)
    y_gt = y_gt / (rad_max + epsilon)

    yy = np.linspace(-12.5, 12.5, 1000)
    yy1 = np.linspace(-10, 10, 1000)
    xx1 = np.sqrt(10**2-yy1**2)
    xx2 = np.sqrt(20**2-yy**2)
    xx3 = np.sqrt(30**2-yy**2)
    xx4 = np.sqrt(40**2-yy**2)
    xx5 = np.sqrt(50**2-yy**2)

    xx = np.linspace(0, 60, 1000)
    yy2 = np.zeros(xx.shape)
    yy3 = xx * np.tan(5*np.pi/180)
    yy4 = xx * np.tan(-5*np.pi/180)
    yy5 = xx * np.tan(10*np.pi/180)
    yy6 = xx * np.tan(-10*np.pi/180)
    yy7 = xx * np.tan(15*np.pi/180)
    yy8 = xx * np.tan(-15*np.pi/180)
 
    ax1 = plt.gca()
    
    colors = flow_xy_to_colors(x_flow, -y_flow)

    ax1.scatter(pc1_warp[0], pc1_warp[1], c = colors/255, marker='o', s=6)
    
    # ax1.plot(xx1, yy1, linewidth=0.5, color='white')
    # ax1.plot(xx2, yy, linewidth=0.5, color='white')
    # ax1.plot(xx3, yy, linewidth=0.5, color='white')
    # ax1.plot(xx4, yy, linewidth=0.5, color='white')
    # ax1.plot(xx5, yy, linewidth=0.5, color='white')
    # ax1.plot(xx, yy2, linewidth=0.5, color='white')
    # ax1.plot(xx, yy3, linewidth=0.5, color='white')
    # ax1.plot(xx, yy4, linewidth=0.5, color='white')
    # ax1.plot(xx, yy5, linewidth=0.5, color='white')
    # ax1.plot(xx, yy6, linewidth=0.5, color='white')
    # ax1.plot(xx, yy7, linewidth=0.5, color='white')
    # ax1.plot(xx, yy8, linewidth=0.5, color='white')
    #
    # ax1.text(10-0.55, -0.3, '10', fontsize=12, ma= 'center', color = 'white')
    # ax1.text(20-0.55, -0.3, '20', fontsize=12, ma = 'center', color = 'white')
    # ax1.text(30-0.55, -0.3, '30', fontsize=12, ma = 'center', color = 'white')
    # ax1.text(40-0.55, -0.3, '40', fontsize=12, ma = 'center', color = 'white')
    # ax1.text(50-0.55, -0.3, '50', fontsize=12, ma = 'center', color = 'white')

    ax1.set_xlim([0, 60])
    ax1.set_ylim([-15, 15])
    ax1.set_box_aspect(0.5)

    # ax1.patch.set_facecolor(np.array([80, 80, 80])/255)
    ax1.patch.set_facecolor('white')
    [ax1.spines[loc_axis].set_visible(False) for loc_axis in ['top','right','bottom','left']]
    ax1.set_xticks([])
    ax1.set_yticks([])

    fig.tight_layout()
    path_im=args.vis_path_flow+'/'+'{}.png'.format(num_pcs)
    fig.savefig(path_im, dpi=200)
    fig.clf
    plt.cla
    plt.close('all')



def visulize_result_2D_seg_pre(pc1, pc2, mask, pred_m, num_pcs, args):


    pc_1=pc1[0].cpu().numpy()
    pc_2=pc2[0].cpu().numpy()

    mask = mask[0].cpu().numpy()
    pred_m = pred_m[0].cpu().numpy()

    fig = plt.figure(figsize=(10, 6))

    x_locator = MultipleLocator(10)
    y_locator = MultipleLocator(10)

    yy = np.linspace(-12.5, 12.5, 1000)
    yy1 = np.linspace(-10, 10, 1000)
    xx1 = np.sqrt(10**2-yy1**2)
    xx2 = np.sqrt(20**2-yy**2)
    xx3 = np.sqrt(30**2-yy**2)
    xx4 = np.sqrt(40**2-yy**2)
    xx5 = np.sqrt(50**2-yy**2)

    xx = np.linspace(0, 60, 1000)
    yy2 = np.zeros(xx.shape)
    yy3 = xx * np.tan(5*np.pi/180)
    yy4 = xx * np.tan(-5*np.pi/180)
    yy5 = xx * np.tan(10*np.pi/180)
    yy6 = xx * np.tan(-10*np.pi/180)
    yy7 = xx * np.tan(15*np.pi/180)
    yy8 = xx * np.tan(-15*np.pi/180)
  
    ax1 = plt.gca()


    ax1.scatter(pc_1[0, pred_m==0],pc_1[1,pred_m==0], s=6, c=np.array([[255/255, 20/255, 147/255]]))
    ax1.scatter(pc_1[0, pred_m==1],pc_1[1,pred_m==1], s=6, c=np.array([[0/255, 128/255, 0/255]]))

    # ax1.plot(xx1, yy1, linewidth=0.5, color='white')
    # ax1.plot(xx2, yy, linewidth=0.5, color='white')
    # ax1.plot(xx3, yy, linewidth=0.5, color='white')
    # ax1.plot(xx4, yy, linewidth=0.5, color='white')
    # ax1.plot(xx5, yy, linewidth=0.5, color='white')
    # ax1.plot(xx, yy2, linewidth=0.5, color='white')
    # ax1.plot(xx, yy3, linewidth=0.5, color='white')
    # ax1.plot(xx, yy4, linewidth=0.5, color='white')
    # ax1.plot(xx, yy5, linewidth=0.5, color='white')
    # ax1.plot(xx, yy6, linewidth=0.5, color='white')
    # ax1.plot(xx, yy7, linewidth=0.5, color='white')
    # ax1.plot(xx, yy8, linewidth=0.5, color='white')
    #
    # ax1.text(10-0.55, -0.3, '10', fontsize=12, ma= 'center', color = 'white')
    # ax1.text(20-0.55, -0.3, '20', fontsize=12, ma = 'center', color = 'white')
    # ax1.text(30-0.55, -0.3, '30', fontsize=12, ma = 'center', color = 'white')
    # ax1.text(40-0.55, -0.3, '40', fontsize=12, ma = 'center', color = 'white')
    # ax1.text(50-0.55, -0.3, '50', fontsize=12, ma = 'center', color = 'white')

    ax1.set_xlim([0, 60])
    ax1.set_ylim([-15, 15])
    ax1.set_box_aspect(0.5)

    # ax1.patch.set_facecolor(np.array([80, 80, 80])/255)
    ax1.patch.set_facecolor('white')
    [ax1.spines[loc_axis].set_visible(False) for loc_axis in ['top','right','bottom','left']]
    ax1.set_xticks([])
    ax1.set_yticks([])

    fig.tight_layout()
    path_im=args.vis_path_seg+'/'+'{}.png'.format(num_pcs)
    fig.savefig(path_im, dpi=200)
    fig.clf
    plt.cla
    plt.close('all')

def visulize_odometry(trans, pre_trans, num_pcs, args):

    trans=trans[0].cpu().numpy()
    pre_trans=pre_trans[0].cpu().numpy()
    trans_reshape = trans.reshape(1, 16)
    pre_trans_reshape = pre_trans.reshape(1, 16)
    path_trans = args.vis_trans + '/' + '{}.npy'.format(num_pcs)
    path_pre_trans = args.vis_pre_trans + '/' + '{}.npy'.format(num_pcs)

    # 保存转换后的矩阵到指定路径
    np.save(path_trans, trans_reshape)
    np.save(path_pre_trans, pre_trans_reshape)