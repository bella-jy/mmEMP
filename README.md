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
Download the dataset [Dataset](https://pan.baidu.com/s/1XzCi2qMr9bAJm0nxFiIMLg?pwd=n6g7). The dataset comprises raw data from millimeter-wave radar, vision, and IMU, along with the output results of PC-side automatic VI-SLAM execution. Please make sure you have prepared the dataset by organizing the directory as: `data/dataset/your_dataset`. In the end, the dataset should be oragnized like this:
  ```
  Dataset
    ├── radar
    │   │── xyz
    │   │── doppler_reshape_256_64_192
    ├── rgb
    │   │── infra1out
    ├── pose
    │   │── vinsout
    │   │── vinspointcloudout
  ```
## Getting started
### 1. Clone the repository to local
  ```
  git clone https://github.com/bella-jy/mmEMP.git
  ```
### 2. Temporal alignment
  ```
  cd data/dataset/your_dataset
  python temporal_allignment.py
  ```
### 3. Dynamic visual-inertial 3D reconstruction
* Data preprocess
  ```
  python dynamic_points.py
  ```
  Place doppler_reshape_256_64_192 and the resulting label, respectively, under the paths `data/your_dataset/data` and `data/your_dataset/label`. E.g. `data/GTVINS2/data` and `data/GTVINS2/label`.
* Train
  
  Train a model by
  ```
  python main.py train
  ```
  In `config/base_confige.yml`, you might want to change the following settings: `data` root path of the dataset for training or testing, `batch_size` for traning
* Eval
  
  Evaluate the trained model by
  ```
  python main.py eval
  ```
### 4. Point cloud refinement
* Place dense point clouds, vinsout, and infra1out into the `pose/preprocess` folder and perform data preprocess
  ```
  cd pose
  python preprocess/preprocess_vod.py --root_dir $ROOT_DIR$ --save_dir $SAVE_DIR$
  ```
* Train
  
  Train a model by
  ```
  cd pose
  python main.py --dataset_path $DATA_PATH$ --exp_name $EXP_NAME$ --model cmflow
  ```
* Eval
  
  Evaluate the trained model by
  ```
  cd pose
  python main.py --eval --dataset_path $DATA_PATH$ --exp_name cmflow_cvpr --model cmflow
  ```
* Enhance and filter point clouds
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
