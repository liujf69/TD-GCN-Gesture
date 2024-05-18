# TD-GCN-Gesture
**This is a *fork* of the official repo of TD-GCN and the work accepted by IEEE Transactions on Multimedia (TMM).** <br />
**[Jinfu Liu, Xinshun Wang, Can Wang, Yuan Gao, Mengyuan Liu. Temporal Decoupling Graph Convolutional Network for Skeleton-based Gesture Recognition. IEEE Transactions on Multimedia (TMM), 2023.](https://ieeexplore.ieee.org/document/10113233)**
![image](https://github.com/liujf69/TD-GCN-Gesture/blob/master/fig.png)
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

```
## SHREC’17 Track dataset:
1. First, extract all files to ```/data/shrec/shrec17_dataset``` <br />
2. Then, run ```python gen_traindataset.py``` and ```python gen_testdataset.py``` <br />

## DHG-14/28 dataset:
1. First, extract all files to ```./data/DHG14-28/DHG14-28_dataset``` <br />
2. Then, run ```python python gen_dhgdataset.py```

## V-LIBRASIL dataset
1. First, get the data from [V-LIBRASIL](https://libras.cin.ufpe.br) <br />
2. Put the data into ```./data/vlibrail/nturgbd_raw``` 
3. Then, run ```python get_raw_skes_data.py```, ```python get_raw_denoised_data.py``` and ```python seq_transformation.py``` in sequence <br />


# Training
You can change the configuration in the yaml file and in the main function. We also provide four default yaml configuration files. <br />
## SHREC’17 Track dataset:
Run ```python main.py --device 0 1 --config ./config/shrec17/shrec17.yaml``` <br />
## DHG-14/28 dataset:
Run ```python main.py --device 0 1 --config ./config/dhg14-28/DHG14-28.yaml``` <br />
## V-LIBRASIL dataset:
Run  <br />

# Testing
We provide several trained weight files and place them in the checkpoints folder.

# Citation
```
@ARTICLE{10113233,
  author={Liu, Jinfu and Wang, Xinshun and Wang, Can and Gao, Yuan and Liu, Mengyuan},
  journal={IEEE Transactions on Multimedia}, 
  title={Temporal Decoupling Graph Convolutional Network for Skeleton-based Gesture Recognition}, 
  year={2024}
}
```
# Contact
For any questions, feel free to contact: ```liujf69@mail2.sysu.edu.cn```
