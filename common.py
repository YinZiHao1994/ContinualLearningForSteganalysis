from enum import Enum


class SteganographyEnum(Enum):
    WOW = 1
    HILL = 2
    SUNI = 3
    UTGAN = 4
    JUNI = 5


class DatasetEnum(Enum):
    BOSSBase_256 = r'D:\Work\dataset\steganalysis\BOSSBase'
    BOSSBase_JPG_256 = r'D:\Work\dataset\steganalysis\BOSSBase_JPG'
    BOWS2_256 = r'D:\Work\dataset\steganalysis\BOWS2'
