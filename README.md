# TD-GCN-Gesture
# Prerequisites
you can install all dependencies by running ```pip install -r requirements.txt```  <br />
Then you need to install torchlight by running ```pip install -e torchlight```  <br />

# Data Preparation
## Download four datasets:
1. **SHREC’17 Track** dataset from <br />
2. **DHG-14/28** dataset from <br />
3. **NTU RGB+D 60** Skeleton dataset from <br />
4. **NW-UCLA8** dataset from <br />

## SHREC’17 Track dataset:
1. First, extract all files to **./data/DHG14-28/DHG14-28_dataset** <br />
2. Then, run **python gen_traindataset.py** and **python gen_testdataset.py** <br />
