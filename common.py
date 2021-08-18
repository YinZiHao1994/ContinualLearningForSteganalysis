from enum import Enum


class SteganographyEnum(Enum):
    HILL = 1
    SUNI = 2
    UTGAN = 3


class DatasetEnum(Enum):
    BOSSBase_256 = r'D:\Work\dataset\steganalysis\BOSSBase'
    BOWS2OrigEp3 = r'D:\Work\dataset\steganalysis\BOWS2'
