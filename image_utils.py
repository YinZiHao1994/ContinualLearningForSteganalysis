import os
import pickle

import numpy

import shutil
import sys
import logging

from PIL import Image
import glob
import common_utils


def batch_jpg_to_pgm(in_dir, out_dir):
    if not os.path.exists(out_dir):
        print(out_dir, 'is not existed.')
        os.mkdir(out_dir)

    if not os.path.exists(in_dir):
        print(in_dir, 'is not existed.')
        return -1
    count = 0
    files_in_dir = common_utils.get_all_files_in_dir(in_dir)
    for file in files_in_dir:
        pure_file_name = common_utils.get_pure_file_name_from_path(file)
        out_file = pure_file_name + '.jpg'
        # 如果是rgb图，要转为单通道的灰度图；如果是灰度图，那么去掉convert，保持灰度图
        # im = Image.open(files).convert('L')
        im = Image.open(file)
        new_path = os.path.join(out_dir, out_file)
        print(count, ',', new_path)
        count = count + 1
        im.save(os.path.join(out_dir, out_file))


if __name__ == '__main__':
    batch_jpg_to_pgm('D:\Work\dataset\steganalysis\BOSSbase_1.01_256',
                     'D:\Work\dataset\steganalysis\BOSSbase_1.01_256_jpg')
