import os
import logging
import time

from PIL import Image

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from transforms import *


def filesInFolder(folder, extn='.bmp'):
    files = list()
    for (dirpath, dirnames, filenames) in os.walk(folder):
        files += [os.path.join(dirpath, file) for file in filenames]
    files = [file for file in files if file.endswith('.bmp')]
    return files

# TODO: this class currently does both MSRC specific stuff as well as augmentations and post-processing
#       It's better to decouple to these so that it's easier to work with other datasets
class MSRCDataset(Dataset):
    def __init__(self, basePath, filesToKeep, augment=True, fixedSize=True, mapNegToLast=True):
        self.augment = augment
        self.fixedSize = fixedSize
        self.mapNegToLast = mapNegToLast

        imageFiles = filesInFolder(basePath + '/Images', extn='.bmp')
        groundTruthFiles = filesInFolder(
            basePath + '/GroundTruth', extn='_GT.bmp')

        filterSet = set([file[:-4] for file in filesToKeep])

        def fileInFilter(file):
            file = os.path.basename(file)
            if file.endswith('_GT.bmp'):
                file = file[:-7]
            if file.endswith('.bmp'):
                file = file[:-4]
            return file in filterSet

        imageFiles = [
            file for file in imageFiles if fileInFilter(file)]
        groundTruthFiles = [
            file for file in groundTruthFiles if fileInFilter(file)]

        imageFiles.sort()
        groundTruthFiles.sort()

        self.data = []
        for imageFile, groundTruthFile in zip(imageFiles, groundTruthFiles):
            assert os.path.basename(imageFile)[
                :-4] == os.path.basename(groundTruthFile)[:-7]

            gtFileRoot = os.path.basename(groundTruthFile)[:-7]
            hqFile = basePath + 'SegmentationsGTHighQuality/' + gtFileRoot + '_HQGT.bmp'

            if os.path.exists(hqFile):
                groundTruthFile = hqFile

            self.data.append((imageFile, groundTruthFile))

        assert len(self.data) == len(filesToKeep)

        if self.augment:
            self.randomHorizontalFlip = RandomHorizontalFlip(p=0.5)
            self.randomGaussianBlur = RandomGaussianBlur(p=0.5)
            self.randomRotate = RandomRotate(p=0.3)
            self.randomScaledCrop = RandomScaledCrop(p=0.5)
            self.randomColorJitter = RandomColorJitter(p=0.5)
            self.randomNoise = RandomNoise(p=0.5)
            # self.scale = Scale()

        if self.fixedSize:
            self.squareCrop = SquareCrop()
            

    def __len__(self):
        if self.augment:
            return len(self.data) * 1000
        return len(self.data)

    def __getitem__(self, index):
        if self.augment:
            seed = int(time.time()*1e9) % 1000
            np.random.seed(seed)

        imageFile, groundTruthFile = self.data[(index % len(self.data))]

        image = Image.open(imageFile)
        groundTruth = Image.open(groundTruthFile)

        assert image.mode == 'RGB'
        assert groundTruth.mode == 'RGB'

        if self.augment:
            image, groundTruth = self.randomHorizontalFlip(image, groundTruth)
            image, groundTruth = self.randomGaussianBlur(image, groundTruth)
            if np.random.rand() < 0.5:
                image, groundTruth = self.randomRotate(image, groundTruth)
            else:
                image, groundTruth = self.randomScaledCrop(image, groundTruth)
            image, groundTruth = self.randomColorJitter(image, groundTruth)
            # image, groundTruth = self.randomNoise(image, groundTruth)
            # image, groundTruth = self.scale(image, groundTruth)

        if self.fixedSize:
            image, groundTruth = self.squareCrop(image, groundTruth)

        groundTruth = self.fixGroundTruth(groundTruth)

        image = PILtoTensor(image)
        image = Normalize(image)

        groundTruth = np.array(groundTruth, dtype=np.int32)
        groundTruth = torch.LongTensor(groundTruth)

        return Variable(image), Variable(groundTruth)

    def fixGroundTruth(self, groundTruth):
        groundTruth = np.array(groundTruth)
        groundTruthFixed = np.zeros(groundTruth.shape[:2], dtype=int)

        for mapping in MSRCDataset.colourMap:
            idx = np.all(groundTruth == mapping['rgb_values'], axis=2)
            
            mappingIdx = mapping['id']
            if self.mapNegToLast and mappingIdx == -1:
                mappingIdx = 21
            
            groundTruthFixed[idx] = mappingIdx

        return groundTruthFixed

    colourMap = [
        {"id": -1, "name": "void", "rgb_values": [0,   0,    0]},
        {"id": 0,  "name": "building", "rgb_values": [128, 0,    0]},
        {"id": 1,  "name": "grass", "rgb_values": [0,   128,  0]},
        {"id": 2,  "name": "tree", "rgb_values": [128, 128,  0]},
        {"id": 3,  "name": "cow", "rgb_values": [0,   0,    128]},
        {"id": 4,  "name": "sheep", "rgb_values": [0,   128,  128]},
        {"id": 5,  "name": "sky", "rgb_values": [128, 128,  128]},
        {"id": 6,  "name": "airplane", "rgb_values": [192, 0,    0]},
        {"id": 7,  "name": "water", "rgb_values": [64,  128,  0]},
        {"id": 8, "name": "face", "rgb_values": [192, 128,  0]},
        {"id": 9, "name": "car", "rgb_values": [64,  0,    128]},
        {"id": 10, "name": "bicycle", "rgb_values": [192, 0,    128]},
        {"id": 11, "name": "flower", "rgb_values": [64,  128,  128]},
        {"id": 12, "name": "sign", "rgb_values": [192, 128,  128]},
        {"id": 13, "name": "bird", "rgb_values": [0,   64,   0]},
        {"id": 14, "name": "book", "rgb_values": [128, 64,   0]},
        {"id": 15, "name": "chair", "rgb_values": [0,   192,  0]},
        {"id": 16, "name": "road", "rgb_values": [128, 64,   128]},
        {"id": 17, "name": "cat", "rgb_values": [0,   192,  128]},
        {"id": 18, "name": "dog", "rgb_values": [128, 192,  128]},
        {"id": 19, "name": "body", "rgb_values": [64,  64,   0]},
        {"id": 20, "name": "boat", "rgb_values": [192, 64,   0]}
    ]


def parseListfromFile(filePath):
    files = []
    with open(filePath, 'r') as f:
        files += [l.rstrip('\n') for l in f]
    return files


def makeMSRCDataLoader(basePath, splitPath, trainBatchSize):
    trainFiles = parseListfromFile(splitPath + '/Train.txt')
    valFiles = parseListfromFile(splitPath + '/Validation.txt')
    testFiles = parseListfromFile(splitPath + '/Test.txt')

    trainDataset = MSRCDataset(
        basePath, trainFiles, augment=True, fixedSize=True)
    valDataset = MSRCDataset(
        basePath, valFiles, augment=False, fixedSize=False)
    testDataset = MSRCDataset(
        basePath, testFiles, augment=False, fixedSize=False)

    trainDataLoader = DataLoader(
        trainDataset, batch_size=trainBatchSize, shuffle=True, num_workers=8)
    valDataLoader = DataLoader(
        valDataset, batch_size=1, shuffle=False, num_workers=1)
    testDataLoader = DataLoader(
        testDataset, batch_size=1, shuffle=False, num_workers=1)

    return trainDataLoader, valDataLoader, testDataLoader