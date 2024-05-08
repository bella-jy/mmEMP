# mmEMP
This is the repository of the mmEMP. For technical details, please refer to our paper on ICRA 2024:

**Enhancing mmWave Radar Point Cloud via Visual-inertial Supervision**

Cong Fan, Shengkai Zhang, Kezhong Liu, Shuai Wang, Zheng Yang, Wei Wang
<img width="920" alt="截屏2024-04-18 10 45 22" src="https://github.com/bella-jy/mmEMP/assets/74900308/b8608f57-1ea5-4135-89a3-c958b4267098">
## Prerequisites
* Install Python3.7 with Anaconda3
  ```
  conda create -n $ENV_NAME$ python=3.7
  source activate $ENV_NAME$
  ```
* Install [CUDA 11.0](https://developer.nvidia.com/cuda-11.0-download-archive)
* install [cuDNN8.0](https://developer.nvidia.com/cudnn)
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
Download the dataset [Dataset](https://pan.baidu.com/s/1XzCi2qMr9bAJm0nxFiIMLg?pwd=n6g7). The dataset comprises raw data from millimeter-wave radar, vision, and IMU, along with the output results of PC-side automatic VI-SLAM execution. Please make sure you have prepared the dataset by organizing the directory as: **data/your_dataset**. In the end, the dataset should be oragnized like this:
  ```
  Dataset
    ├── radar
    │   │── xyz
    │   │── doppler_reshape_256_64_192
    ├── rgb
    ├── pose
  ```
## Getting started
* Clone the repository to local
  ```
  git clone https://github.com/bella-jy/mmEMP.git
  ```
* Temporal alignment
  ```
  cd data/your_dataset
  python assign_timestamp.py
  ```
* Data preprocess
  ```
  python dynamic_points.py
  ```
  
* Generate dense point clouds.
  ```
  python main.py train
  ```
* Obtain motion estimation results using dense point clouds as input.
  ```
  cd pose
  python main.py --eval --vis --dataset_path ./demo_data/ --exp_name
  ```
* Enhance and filter point clouds.
  ```
  python ghost_eliminate.py
  ```
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
## Acknowledgments
This repository is based on the following codebases:
* [RPDNet](https://github.com/thucyw/RPDNet)
* [CMFlow](https://github.com/Toytiny/CMFlow)
