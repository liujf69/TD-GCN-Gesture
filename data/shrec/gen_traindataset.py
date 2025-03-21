import numpy as np
import json

root_database_path = './shrec17_dataset/HandGestureDataset_SHREC2017' 
train_txt_path = './shrec17_dataset/HandGestureDataset_SHREC2017/train_gestures.txt' # train_gestures.txt
train_txt = np.loadtxt(train_txt_path, dtype=int)


Samples_sum = train_txt.shape[0] 
# print(Samples_sum) # 1960

data_dict = []

for i in range(Samples_sum): 
    idx_gesture = train_txt[i][0] # gesture
    idx_finger = train_txt[i][1] # finger
    idx_subject = train_txt[i][2] # subject
    idx_essai = train_txt[i][3] # essai
    label_14 = train_txt[i][4] # label_14
    label_28 = train_txt[i][5] # label_28
    T = train_txt[i][6] # frames

    skeleton_path = root_database_path + '/gesture_' + str(idx_gesture) + '/finger_' \
                    + str(idx_finger) + '/subject_' + str(idx_subject) + '/essai_' + str(idx_essai)+'/skeletons_world.txt' 


    skeleton_data = np.loadtxt(skeleton_path)
    # print(skeleton_data.shape) # T * 66

    # T = skeleton_data.shape[0] 
    skeleton_data = skeleton_data.reshape([T, 22, 3]) # T*66 reshape to T*N*C(T*22*3) 
    # print(skeleton_data) # T*22*3

    file_name = "g"+str(idx_gesture).zfill(2) + "f"+str(idx_finger).zfill(2) + "s"+str(idx_subject).zfill(2) + "e"+str(idx_essai).zfill(2) # filename

    data_json = {"file_name": file_name, "skeletons": skeleton_data.tolist(), "label_14": label_14.tolist(), "label_28": label_28.tolist()} # json
    with open("./shrec17_jsons/train_jsons/" + file_name + ".json", 'w') as f:
        json.dump(data_json, f)

    tmp_data_dict = {"file_name": file_name, "length": T.tolist(), "label_14": label_14.tolist(), "label_28": label_28.tolist()} # dict
    data_dict.append(tmp_data_dict)

with open("./shrec17_jsons/" + "train_samples.json", 'w') as t: # json
    json.dump(data_dict, t)
