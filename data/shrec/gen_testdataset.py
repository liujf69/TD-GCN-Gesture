import numpy as np
import json

root_database_path = './shrec17_dataset/HandGestureDataset_SHREC2017' # 数据集根目录
test_txt_path = './shrec17_dataset/HandGestureDataset_SHREC2017/train_gestures.txt' # 训练train_gestures.txt 的路径
test_txt = np.loadtxt(test_txt_path, dtype=int)


Samples_sum = test_txt.shape[0] # 样本数
# print(Samples_sum) # 1960

data_dict = []

for i in range(Samples_sum): # 遍历每一个样本
    idx_gesture = test_txt[i][0] # gesture信息
    idx_finger = test_txt[i][1] # finger信息
    idx_subject = test_txt[i][2] # subject信息
    idx_essai = test_txt[i][3] # essai信息
    label_14 = test_txt[i][4] # label_14标签
    label_28 = test_txt[i][5] # label_28标签
    T = test_txt[i][6] # 单个样本的帧数

    skeleton_path = root_database_path + '/gesture_' + str(idx_gesture) + '/finger_' \
                    + str(idx_finger) + '/subject_' + str(idx_subject) + '/essai_' + str(idx_essai)+'/skeletons_world.txt' # 骨骼txt路径


    skeleton_data = np.loadtxt(skeleton_path) # 读取骨骼txt文件
    # print(skeleton_data.shape) # T * 66

    # T = skeleton_data.shape[0] # 计算样本的帧数
    skeleton_data = skeleton_data.reshape([T, 22, 3]) # T*66 reshape to T*N*C(T*22*3) # 维度变换
    # print(skeleton_data) # T*22*3

    file_name = "g"+str(idx_gesture).zfill(2) + "f"+str(idx_finger).zfill(2) + "s"+str(idx_subject).zfill(2) + "e"+str(idx_essai).zfill(2) # 获取filename

    data_json = {"file_name": file_name, "skeletons": skeleton_data.tolist(), "label_14": label_14.tolist(), "label_28": label_28.tolist()} # 保存每个样本的信息为json文件
    with open("./shrec17_jsons/test_jsons/" + file_name + ".json", 'w') as f:
        json.dump(data_json, f)

    tmp_data_dict = {"file_name": file_name, "length": T.tolist(), "label_14": label_14.tolist(), "label_28": label_28.tolist()} # 用一个字典记录所有样本的信息
    data_dict.append(tmp_data_dict)

with open("./shrec17_jsons/" + "test_samples.json", 'w') as t: # 将所有样本的信息保存为一个json格式文件
    json.dump(data_dict, t)




