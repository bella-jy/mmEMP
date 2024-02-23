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
  cd lib
  python setup.py install
  cd ..
  ```
## Getting started
### Train

### Evaluation
## Citation
