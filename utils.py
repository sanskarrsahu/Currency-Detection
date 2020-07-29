import os
import cv2
from detector import getFeatures
from config import *


def getSampleData():
    cache = {}

    for currencyValue in os.listdir(BASE_DIR):
        data = []
        for currencySampleImageName in os.listdir(BASE_DIR + os.sep + currencyValue):
            currencyTrainImagePath = BASE_DIR + os.sep + currencyValue + os.sep + currencySampleImageName
            currencyTrainImage = cv2.imread(currencyTrainImagePath, cv2.IMREAD_GRAYSCALE)
            data.append([currencyTrainImage, tuple(getFeatures(currencyTrainImage))])
        cache[currencyValue] = data

    return cache