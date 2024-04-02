# mmEMP
This is the repository of the mmEMP. For technical details, please refer to our paper on ICRA 2024:

**Enhancing mmWave Radar Point Cloud via Visual-inertial Supervision**

Cong Fan, Shengkai Zhang, Kezhong Liu, Shuai Wang, Zheng Yang, Wei Wang
## Prerequisites
* Install Anaconda3
* Install Python3.7
  ```
  conda create -n $ENV_NAME$ python=3.7
  source activate $ENV_NAME$
  ```
* Install CUDA 11.0
* install cuDNN8.0
* Install PyTorch1.7.0
  ```
  conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
  ```
* Install PointNet++ library for basic point cloud operation
  ```
  cd pose lib
  python setup.py install
  cd ..
## Dataset
The dataset comprises raw data from millimeter-wave radar, vision, and IMU, along with the output results of PC-side automatic VI-SLAM execution. Due to variations in sampling frequencies among multimodal data, alignment procedures such as data synchronization are required.
## Getting started
* Using consecutive visual images as input, execute dynamic_points.py to obtain static and dynamic visual feature points.
* Using consecutive visual images as input, execute dynamic_points.py to obtain static and dynamic visual feature points.[链接文本]([链接URL](https://github.com/search?q=point+cloud+generation+lidar+language%3APython&type=repositories))

### Train
  ```
  python main.py train
  ```
### Evaluation
  ```
  python main.py eval
  ```
## Citation
