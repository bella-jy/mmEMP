import torch
import torchvision
import numpy as np
import cv2
a = np.load('/home/zyw/桌面/RPDNet/adc_data (2)/visualfeature/depth.npy')
# 加载预训练的Faster R-CNN模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 加载模型权重到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 定义目标类别标签
class_labels = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']

# 加载深度图像
depth_image1 = cv2.imread('/media/zyw/T7/VINS-radar/第六次/VINS/0831/depthout/1693365835985835791.png', cv2.IMREAD_UNCHANGED)
depth_image2 = cv2.imread('/media/zyw/T7/VINS-radar/第六次/VINS/0831/depthout/1693365836019191504.png', cv2.IMREAD_UNCHANGED)

# 转换深度图像为浮点数
depth_image1 = depth_image1.astype(np.float32) / 1000.0  # 假设深度图像以毫米为单位
depth_image2 = depth_image2.astype(np.float32) / 1000.0

# 将深度图像转换为PyTorch张量，并将其移动到GPU（如果可用）
depth_tensor1 = torch.from_numpy(depth_image1).unsqueeze(0).unsqueeze(0).to(device)
depth_tensor2 = torch.from_numpy(depth_image2).unsqueeze(0).unsqueeze(0).to(device)

# 运行Faster R-CNN模型进行目标检测
with torch.no_grad():
    # 检测图像1中的目标
    output1 = model(depth_tensor1)
    boxes1 = output1[0]['boxes'].cpu().numpy()
    labels1 = output1[0]['labels'].cpu().numpy()

    # 检测图像2中的目标
    output2 = model(depth_tensor2)
    boxes2 = output2[0]['boxes'].cpu().numpy()
    labels2 = output2[0]['labels'].cpu().numpy()

# 计算像素点的三维坐标和深度值
def compute_3d_coordinates(boxes, depth_image):
    coordinates = []
    for box in boxes:
        x1, y1, x2, y2 = box
        depth = depth_image[int(y1):int(y2), int(x1):int(x2)].mean()
        if not np.isnan(depth):
            z = depth
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            coordinates.append((x, y, z))
    return coordinates
# 计算两个图像上共同目标的三维坐标差值
def compute_coordinate_differences(coordinates1, coordinates2):
    coordinate_differences = []
    for coord1 in coordinates1:
        for coord2 in coordinates2:
            x_diff = coord2[0] - coord1[0]
            y_diff = coord2[1] - coord1[1]
            z_diff = coord2[2] - coord1[2]

            # 取绝对值并乘以0.1
            x_diff = abs(x_diff) * 0.1
            y_diff = abs(y_diff) * 0.1
            z_diff = abs(z_diff) * 0.1

            coordinate_differences.append((x_diff, y_diff, z_diff))

    return coordinate_differences
# 输出图像1中检测目标的三维坐标
coordinates1 = compute_3d_coordinates(boxes1, depth_image1)
for i, coordinate in enumerate(coordinates1):
    print(f'Object {i+1} in image 1: {coordinate}')

# 输出图像2中检测目标的三维坐标
coordinates2 = compute_3d_coordinates(boxes2, depth_image2)
for i, coordinate in enumerate(coordinates2):
    print(f'Object {i+1} in image 2: {coordinate}')

# 计算所有共同目标的坐标差值
coordinate_differences = compute_coordinate_differences(coordinates1, coordinates2)

if coordinate_differences:
    # 保存坐标差值到.npy文件
    np.save('/home/zyw/桌面/RPDNet/adc_data (2)/visualfeature/depth6.npy', np.array(coordinate_differences))
    print(f'Coordinate differences saved to /path/to/save/coordinate_differences.npy')
else:
    print("No common objects found in the images.")


