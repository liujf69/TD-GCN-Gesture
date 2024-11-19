# TD-GCN-Gesture
**This is the official repo of TD-GCN and our work is accepted by IEEE Transactions on Multimedia (TMM).** <br />
**[Jinfu Liu, Xinshun Wang, Can Wang, Yuan Gao, Mengyuan Liu. Temporal Decoupling Graph Convolutional Network for Skeleton-based Gesture Recognition. IEEE Transactions on Multimedia (TMM), 2023.](https://ieeexplore.ieee.org/document/10113233)**
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/temporal-decoupling-graph-convolutional/skeleton-based-action-recognition-on-shrec)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-shrec?p=temporal-decoupling-graph-convolutional) <br />
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/temporal-decoupling-graph-convolutional/hand-gesture-recognition-on-dhg-14)](https://paperswithcode.com/sota/hand-gesture-recognition-on-dhg-14?p=temporal-decoupling-graph-convolutional) <br />
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/temporal-decoupling-graph-convolutional/skeleton-based-action-recognition-on-uav)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-uav?p=temporal-decoupling-graph-convolutional) <br />
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/temporal-decoupling-graph-convolutional/hand-gesture-recognition-on-dhg-28)](https://paperswithcode.com/sota/hand-gesture-recognition-on-dhg-28?p=temporal-decoupling-graph-convolutional) <br />
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/temporal-decoupling-graph-convolutional/skeleton-based-action-recognition-on-n-ucla)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-n-ucla?p=temporal-decoupling-graph-convolutional) <br />
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/temporal-decoupling-graph-convolutional/skeleton-based-action-recognition-on-ntu-rgbd)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd?p=temporal-decoupling-graph-convolutional) <br />
<div align=center>
<img src="https://github.com/liujf69/TD-GCN-Gesture/blob/master/fig.png"/>
</div>

# Prerequisites
You can install all dependencies by running ```pip install -r requirements.txt```  <br />
Then, you need to install torchlight by running ```pip install -e torchlight```  <br />

# Data Preparation
## Download four datasets:
1. **SHREC’17 Track** dataset from [http://www-rech.telecom-lille.fr/shrec2017-hand/](http://www-rech.telecom-lille.fr/shrec2017-hand/) <br />
2. **DHG-14/28** dataset from [http://www-rech.telecom-lille.fr/DHGdataset/](http://www-rech.telecom-lille.fr/DHGdataset/) <br />
3. **NTU RGB+D 60** Skeleton dataset from [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/) <br />
4. **NW-UCLA** dataset from [Download NW-UCLA dataset](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0) <br />
5. Put downloaded data into the following directory structure:
```
- data/
  - shrec/
    - shrec17_dataset/
	  - HandGestureDataset_SHREC2017/
	    - gesture_1
	      ...
  - DHG14-28/
    - DHG14-28_dataset/
	  - gesture_1
	    ...
  - NW-UCLA/
    - all_sqe
      ...
  - ntu/
    - nturgbd_raw/
	  - nturgb+d_skeletons
            ...
```
## Download from cloud drive:
1. **SHREC’17 Track** dataset from [Baidu Drive](https://pan.baidu.com/s/1agHIw92pLAqLh6jPMpyZVw), Password is **TDGC**. Download from [Google Drive](https://drive.google.com/file/d/1lhbbR22QcJWGT4NpOvypqx-euQ6bkwVd/view?usp=sharing). <br />
2. **DHG-14/28** dataset from [Baidu Drive](https://pan.baidu.com/s/1FVV-KZZ6F0KRy3IAHPn4tA), Password is **TDGC**. Download from [Google Drive](https://drive.google.com/file/d/1GIM3qQRrfHzZbRusXpcrakWQR2n31t86/view?usp=sharing).<br />
3. **NTU RGB+D 60** dataset from [Baidu Drive](https://pan.baidu.com/s/16WmFFkGwZM6be93L376WUQ), Password is **TDGC**. <br />

## SHREC’17 Track dataset:
1. First, extract all files to ```/data/shrec/shrec17_dataset``` <br />
2. Then, run ```python gen_traindataset.py``` and ```python gen_testdataset.py``` <br />

## DHG-14/28 dataset:
1. First, extract all files to ```./data/DHG14-28/DHG14-28_dataset``` <br />
2. Then, run ```python python gen_dhgdataset.py```

## NTU RGB+D 60 dataset
1. First, extract all skeleton files to ```./data/ntu/nturgbd_raw``` <br />
2. Then, run ```python get_raw_skes_data.py```, ```python get_raw_denoised_data.py``` and ```python seq_transformation.py``` in sequence <br />

## NW-UCLA dataset
1. Move folder ```all_sqe``` to ```./data/NW-UCLA```

# Training
You can change the configuration in the yaml file and in the main function. We also provide four default yaml configuration files. <br />
## SHREC’17 Track dataset:
Run ```python main.py --device 0 --config ./config/shrec17/shrec17.yaml``` <br />
## DHG-14/28 dataset:
Run ```python main.py --device 0 --config ./config/dhg14-28/DHG14-28.yaml``` <br />
## NTU RGB+D 60 dataset:
On the benchmark of cross-view, run ```python main.py --device 0 --config ./config/nturgbd-cross-view/default.yaml``` <br />
On the benchmark of cross-subject, run ```python main.py --device 0 --config ./config/nturgbd-cross-subject/default.yaml``` <br />
## NW-UCLA dataset:
Run ```python main.py --device 0 --config ./config/ucla/nw-ucla.yaml``` <br />

# Testing
We provide several trained weight files and place them in the [checkpoints](https://github.com/liujf69/TD-GCN-Gesture/tree/master/checkpoints) folder.
```
python main.py --device 0 --config <config.yaml> --phase test --weights <work_dir>/<weight.pt>
```

# Ensemble
```
1. Set Rate
2. Run:
python gesture_ensemble.py \
--joint_Score <joint_path> \
--bone_Score <bone_path> \
--jointmotion_Score <jointmotion_path> \
--val_sample <val_path> \
--benchmark <benchmark>

# Example for Shrec_28
1. Download .pkl file from: https://drive.google.com/drive/folders/1ux87mUirBQjmA4b4fEWtb9tuj-8wSYHt
2. Set Rate [0.5, 0.5, 0.5] or [0.5, 0.3, 0.2]
3. Run:
python gesture_ensemble.py \
--joint_Score ./joint.pkl \
--bone_Score ./bone.pkl \
--jointmotion_Score ./jointmotion.pkl \
--val_sample ./shrec17_28.txt \
--benchmark Shrec_28
```

# Citation
```python
# Result about SHREC’17 Track, DHG-14/28, NTU RGB+D 60 and NW-UCLA datasets.
@ARTICLE{10113233,
  author={Liu, Jinfu and Wang, Xinshun and Wang, Can and Gao, Yuan and Liu, Mengyuan},
  title={Temporal Decoupling Graph Convolutional Network for Skeleton-based Gesture Recognition}, 
  journal={IEEE Transactions on Multimedia (TMM)}, 
  year={2024}
}

# Result about UAV-Human dataset.
@inproceedings{liu2024HDBN,
  author={Liu, Jinfu and Yin, Baiqiao and Lin, Jiaying and Wen, Jiajun and Li, Yue and Liu, Mengyuan},
  title={HDBN: A Novel Hybrid Dual-branch Network for Robust Skeleton-based Action Recognition}, 
  booktitle={Proceedings of the IEEE International Conference on Multimedia and Expo Workshop (ICMEW)}, 
  year={2024}
}
```
Our project is based on the [DSTA-Net](https://github.com/lshiwjx/DSTA-Net), [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN). <br />
# Contact
For any questions, feel free to contact: ```liujf69@mail2.sysu.edu.cn```
