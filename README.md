# mmEMP
This is the repository of the mmEMP. For technical details, please refer to our paper on ICRA 2024:

**Enhancing mmWave Radar Point Cloud via Visual-inertial Supervision**

Cong Fan, Shengkai Zhang, Kezhong Liu, Shuai Wang, Zheng Yang, Wei Wang
<img width="920" alt="截屏2024-04-18 10 45 22" src="https://github.com/bella-jy/mmEMP/assets/74900308/b8608f57-1ea5-4135-89a3-c958b4267098">
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
Download the dataset [Dataset](https://pan.baidu.com/s/1XzCi2qMr9bAJm0nxFiIMLg?pwd=n6g7).The dataset comprises raw data from millimeter-wave radar, vision, and IMU, along with the output results of PC-side automatic VI-SLAM execution. Due to variations in sampling frequencies among multimodal data, alignment procedures such as data synchronization are required. Execute the assign_timestamp.py script to achieve data synchronization.
## Getting started
* Using consecutive visual images as input, execute dynamic_points.py to obtain static and dynamic visual feature points.
* After obtaining the three-dimensional coordinates of visual feature points, [RPDNet](https://github.com/thucyw/RPDNet) is run with these coordinates as supervisory signals to generate dense point clouds.
* To eliminate radar ghost points, we refer to the [CMFlow](https://github.com/Toytiny/CMFlow) approach and obtain motion estimation results using dense point clouds as input.
* Finally, run the ghost_eliminate.py file to compute the distance threshold and output the enhanced and filtered point clouds.
## Citation
If you find our work useful in your research, please consider citing:
  ```
@InProceedings{Fan_2024_ICRA,
    author    = {Fan, Cong and Zhang, Shengkai and Liu, Kezhong and Wang, Shuai and Yang, Zheng and Wang, Wei},
    title     = {Enhancing mmWave Radar Point Cloud via Visual-inertial Supervision},
    booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
    year      = {2024},
}
  ```
