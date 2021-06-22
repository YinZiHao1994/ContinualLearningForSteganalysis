import os
import numpy as np
from glob import glob
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, dataset_dir, steganography_enum, transform=None):
        self.transform = transform

        self.cover_dir = os.path.join(dataset_dir, 'BOSSBase_256')
        # self.stego_dir = DATASET_DIR + '/stego_suniward04'
        self.stego_dir = os.path.join(dataset_dir, 'BOSSBase_256_' + steganography_enum.name + '04')

        self.cover_list = [x.split('\\')[-1] for x in glob(self.cover_dir + '/*')]
        assert len(self.cover_list) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, idx):
        file_index = int(idx)

        cover_path = os.path.join(self.cover_dir, self.cover_list[file_index])
        stego_path = os.path.join(self.stego_dir, self.cover_list[file_index])

        # cover_data = cv2.imread(cover_path, 0)
        # stego_data = cv2.imread(stego_path, 0)
        cover_data = Image.open(cover_path)  # .convert('RGB')
        stego_data = Image.open(stego_path)  # .convert('RGB')
        # cover_data = np.array(cover_data)
        # stego_data = np.array(stego_data)

        # data = np.stack([cover_data, stego_data], ).transpose((0, 3, 1, 2))
        # print(np.array(cover_data).shape)
        # data_ = Image.fromarray(data).convert('RGB')
        label = np.array([0, 1], dtype='uint8')
        label = torch.from_numpy(label).long()
        # print(type(data) )
        # print(type(label) )

        # sample = {'data': data, 'label': label}
        # print(type(Image.fromarray(data)) )

        if self.transform:
            cover_data = self.transform(cover_data)
            stego_data = self.transform(stego_data)
            # print('cover_data',cover_data.shape)
        # data = torch.cat((cover_data, stego_data), 0)
        data = torch.stack((cover_data, stego_data))
        # print('data', data.shape)

        sample = {'data': data, 'label': label}
        return sample