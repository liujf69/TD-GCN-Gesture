import torch
import pickle
import argparse
import numpy as np
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser(description = 'multi-stream ensemble') 
    parser.add_argument(
        '--joint_Score', 
        type = str,
        default = './joint.pkl')
    parser.add_argument(
        '--bone_Score', 
        type = str,
        default = './bone.pkl')
    parser.add_argument(
        '--jointmotion_Score', 
        type = str,
        default = './jointmotion.pkl')
    parser.add_argument(
        '--val_sample', 
        type = str,
        default = './shrec17_14.txt')
    parser.add_argument(
        '--benchmark', 
        type = str,
        default = 'Shrec_14')
    return parser

def Cal_Score(File, Rate, ntu60XS_num, Numclass):
    final_score = torch.zeros(ntu60XS_num, Numclass)
    for idx, file in enumerate(File):
        fr = open(file,'rb') 
        inf = pickle.load(fr)

        df = pd.DataFrame(inf)
        df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
        score = torch.tensor(data = df.values)
        final_score += Rate[idx] * score

    return final_score

def Cal_Acc(final_score, true_label):
    wrong_index = []
    _, predict_label = torch.max(final_score, 1)
    for index, p_label in enumerate(predict_label):
        if p_label != true_label[index]:
            wrong_index.append(index)
            
    wrong_num = np.array(wrong_index).shape[0]
    print('wrong_num: ', wrong_num)

    total_num = true_label.shape[0]
    print('total_num: ', total_num)
    Acc = (total_num - wrong_num) / total_num
    return Acc

def gen_label(val_txt_path):
    true_label = np.loadtxt(val_txt_path, dtype = str)
    true_label = torch.from_numpy(np.array(true_label, dtype = int))
    return true_label

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    j_file = args.joint_Score
    b_file = args.bone_Score
    jm_file = args.jointmotion_Score
    val_txt_file = args.val_sample

    File = [j_file, b_file, jm_file] 
    Rate = [0., 0., 0.]    
    if args.benchmark == 'Shrec_14':
        Numclass = 14
        Sample_Num = 840
        final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
    
    elif args.benchmark == 'Shrec_28':
        Numclass = 28
        Sample_Num = 840
        final_score = Cal_Score(File, Rate, Sample_Num, Numclass)

    elif args.benchmark == 'Dhg_14':
        Numclass = 14
        Sample_Num = 140
        final_score = Cal_Score(File, Rate, Sample_Num, Numclass)

    elif args.benchmark == 'Dhg_28':
        Numclass = 28
        Sample_Num = 140
        final_score = Cal_Score(File, Rate, Sample_Num, Numclass)

    true_label = gen_label(val_txt_file)

    Acc = Cal_Acc(final_score, true_label)

    print('acc:', Acc)
