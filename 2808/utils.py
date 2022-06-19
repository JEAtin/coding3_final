import string
import random
import os.path
import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    _stopwords = set(string.punctuation)

    def __init__(self, path, is_train=True):
        super(MyDataset, self).__init__()
        self.data, self.label = self.read_data(path, is_train)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def read_data(path, is_train=True):
        _type = 'train' if is_train else 'test'
        data_dir = os.path.join(path, _type)
        data_dir_neg = [os.path.join(data_dir + '/neg', x) for x in os.listdir(data_dir + '/neg')]
        data_dir_pos = [os.path.join(data_dir + '/pos', x) for x in os.listdir(data_dir + '/pos')]

        file_dir = data_dir_neg + data_dir_pos

        data = []
        label = []
        for _dir in file_dir:
            is_pos = 'pos' in _dir
            with open(_dir, 'r', encoding='utf-8') as f:
                line = MyDataset.preprocess(f.read().strip())
                data.append(line)
                label.append(int(is_pos))
        return data, label

    @staticmethod
    def preprocess(text: str):
        text_clean = []
        for t in text.split():
            if t not in MyDataset._stopwords:
                text_clean.append(t)
        text_clean = ' '.join(text_clean)
        text_clean = text_clean.replace("<br />", "")
        return text_clean


def set_seed(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    dataset = MyDataset('data/aclImdb')



