import json

from torch.utils.data import Dataset
import numpy as np
import random


class Feeder(Dataset):
    def __init__(self, data_path, label_path, repeat=1, label_flag=28, idx=1, random_choose=True, random_shift=False,
                 random_move=False,
                 window_size=150, normalization=False, debug=False, use_mmap=True):
        self.nw_DHG14_28_root = 'data/DHG14-28/DHG14-28_sample_json/'
        self.idx = idx

        if 'val' in label_path:  # 导入test_samples.json 并赋给self.data_dict
            self.train_val = 'val'
            with open(self.nw_DHG14_28_root + str(self.idx) + '/' + str(self.idx) + 'val_samples.json', 'r') as f1:
                json_file = json.load(f1)
            self.data_dict = json_file
            self.flag = str(self.idx) + '/val/'  # 这个用于后面导入数据时寻址
        else:  # 导入train_samples.json 并赋给self.data_dict
            self.train_val = 'train'
            with open(self.nw_DHG14_28_root + str(self.idx) + '/' + str(self.idx) + 'train_samples.json', 'r') as f2:
                json_file = json.load(f2)
            self.data_dict = json_file
            self.flag = str(self.idx) + '/train/'  # 这个用于后面导入数据时寻址

        # 定义手掌的骨骼连接
        self.bone = [(1, 2), (3, 1), (4, 3), (5, 4), (6, 5), (7, 2), (8, 7), (9, 8), (10, 9), (11, 2), (12, 11),
                     (13, 12), (14, 13), (15, 2), (16, 15), (17, 16), (18, 17), (19, 2), (20, 19), (21, 20), (22, 21),
                     (2, 2)]  # 新增(2, 2)以构成22对骨骼

        self.load_data()  # 运行load_data的函数，导入骨架数据
        self.data_path = data_path  # 用于生成四种模态数据
        self.repeat = repeat  # 重复次数 作用暂时不明
        self.window_size = window_size
        self.label_flag = label_flag  # 选取14个label或者28个label
        

        # 生成标签 (这里要根据提示选择label_14或者label_28)
        self.label = []
        for index in range(len(self.data_dict)):
            info = self.data_dict[index]
            if self.label_flag == 14:  # 选取14个label
                self.label.append(int(info['label_14']) - 1)
            elif self.label_flag == 28:  # 选取28个label
                self.label.append(int(info['label_28']) - 1)

        self.debug = debug
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.normalization = normalization
        self.use_mmap = use_mmap

    # 定义load_data函数，用于导入骨架数据
    def load_data(self):
        self.data = []  # data: T N C
        for data in self.data_dict:  # 遍历data_dict
            file_name = data['file_name']
            with open(self.nw_DHG14_28_root + self.flag + file_name + '.json', 'r') as f:  # 读取骨架数据的json文件
                json_file = json.load(f)
            skeletons = json_file['skeletons']  # 读取json文件的skeletons数据
            value = np.array(skeletons)
            self.data.append(value)

    # 定义随机平移函数
    def random_translation(self, ske_data):
        translate = np.eye(3)  # 平移矩阵
        random.random()
        t_x = random.uniform(-0.01, 0.01)  # 从(-0.01, 0.01)随机生成一个随机数
        t_y = random.uniform(-0.01, 0.01)
        t_z = random.uniform(-0.01, 0.01)

        translate[0, 0] = translate[0, 0] + t_x
        translate[1, 1] = translate[1, 1] + t_y
        translate[2, 2] = translate[2, 2] + t_z

        data = np.dot(ske_data, translate)
        return data

    def __len__(self):
        return len(self.data_dict) * self.repeat

    def __iter__(self):
        return self

    def __getitem__(self, index):
        label = self.label[index % len(self.data_dict)]  # 根据index选择标签label
        value = self.data[index % len(self.data_dict)]  # 根据index选择骨骼数据skeletons # T N C

        # 预处理
        data = self.random_translation(value)  # 随机平移
        T, N, C = data.shape
        temp_data = np.zeros([self.window_size, N, C])
        temp_data[:T, :, :] = data
        data = temp_data  # T N C -> self.window_size N C

        # bone 模态
        if 'bone' in self.data_path:
            data_bone = np.zeros_like(data)  # T N C
            for bone_idx in range(20):
                data_bone[:, self.bone[bone_idx][0] - 1, :] = data[:, self.bone[bone_idx][0] - 1, :] - data[:,
                                                                                                       self.bone[
                                                                                                           bone_idx][
                                                                                                           1] - 1, :]
            data = data_bone
        # motion 模态
        if 'motion' in self.data_path:
            data_motion = np.zeros_like(data)
            data_motion[:-1, :, :] = data[1:, :, :] - data[:-1, :, :]
            data = data_motion

        data = np.transpose(data, (2, 0, 1))
        C, T, N = data.shape
        data = np.reshape(data, (C, T, N, 1))  # C T N 1

        return data, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()

        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod