import os
import time

import imageio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# from scipy import misc

import common_utils


def compare_stego_and_cover():
    cover_path = r'D:\Work\dataset\steganalysis\BOWS2_256'
    stego_path = r'D:\Work\dataset\steganalysis\BOWS2_256_HILL04'
    files_in_dir = common_utils.get_all_files_in_dir(cover_path, '.pgm')
    stego_files_in_dir = common_utils.get_all_files_in_dir(stego_path, '.pgm')
    file_length = len(files_in_dir)
    for index, img_path in enumerate(files_in_dir):

        cover = Image.open(img_path)
        stego = Image.open(stego_files_in_dir[index])
        if cover.mode == 'RGB':
            cover = cover.convert('L')
        cover = np.array(cover)
        stego = np.array(stego)

        residual = (stego.astype('float64') - cover.astype('float64') + 1) / 2
        print(residual)

        plt.subplot(121)
        plt.imshow(cover, cmap='gray')
        plt.subplot(122)
        plt.imshow(residual, cmap='gray')
        plt.show()


if __name__ == '__main__':
    compare_stego_and_cover()
