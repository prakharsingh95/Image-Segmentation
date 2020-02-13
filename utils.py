import numpy as np

import os

from PIL import Image

from MSRCDataset import MSRCDataset

import config as C

import torch
import torch.nn as nn
import torch.nn.functional as F

import metrics

colourMap = MSRCDataset.colourMap
cmap = np.zeros((22,3)).astype(np.uint8)

for mapping in colourMap:
    _id = mapping['id']
    rgb = mapping['rgb_values']
    cmap[_id, :] = np.array(rgb)

def invertGroundTruth(groundTruth):
    return cmap[groundTruth] 

def saveTorchNormalizedImage(img, fileName):

    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    img = img.detach().cpu().numpy()
    img = np.moveaxis(img, 0, -1)
    img = img * STD + MEAN

    img = (img*255).astype(np.uint8)

    img = Image.fromarray(img)

    img.save(fileName)

def saveTorchGroundTruth(gt, fileName):
    gt = gt.detach().cpu().numpy()

    gt = invertGroundTruth(gt)

    gt = Image.fromarray(gt)

    gt.save(fileName)

smax = nn.LogSoftmax(dim=1).to(C.DEVICE)
def computeTorchStdGroundTruth(decoded, mapLastToNeg=False):
    pred = smax(decoded)
    pred = torch.argmax(pred, dim=1, keepdim=False)
    if mapLastToNeg:
        pred[pred == C.NUM_CLASSES-1] = -1
    return pred

def loadModelFromFile(path, model, opt=None):
    path = C.MODEL_PATH.format(path)
    if os.path.exists(path):
        print('Reading model from {}'.format(path), flush=True)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and opt is not None:
            opt.load_state_dict(checkpoint['optimizer'])
            print('Reading optimizer from {}'.format(path), flush=True)
        # optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print('Path {} does not exist'.format(path), flush=True)

def saveModelToFile(path, model, opt=None):
    path = C.MODEL_PATH.format(path)
    checkpoint = {}
    checkpoint['model'] = model.state_dict()
    if opt is not None:
        checkpoint['optimizer'] = opt.state_dict()
    torch.save(checkpoint, path)

def makeGenerator(iterable):
    while True:
        for item in iterable:
            yield item

def makePredict(resNetTrunk, asppModel, decoder):

    def predict(images, mode='train', detachTrunk=False):
        if mode == 'eval':
            resNetTrunk.eval()
            asppModel.eval()
            decoder.eval()

        images = images.to(C.DEVICE)

        _, _, inputWidth, inputHeight = images.shape

        newWidth = int(8 * np.ceil(inputWidth/8))
        newHeight = int(8 * np.ceil(inputHeight/8))
        newSize = (newWidth, newHeight)

        images = F.interpolate(images, (newWidth, newHeight), mode='bilinear', align_corners=True)

        images = images.to(C.DEVICE)
        # print(images.shape)

        featDense, featLowLevel = resNetTrunk(images)
        # print(featDense.shape, featLowLevel.shape)

        if detachTrunk:
            featDense = featDense.detach()
            featLowLevel = featLowLevel.detach()

        aspp = asppModel(featDense)
        # print(aspp.shape)

        decoded = decoder(aspp, featLowLevel)
        decoded = F.interpolate(
            decoded, (inputWidth, inputHeight), mode='bilinear', align_corners=True)
        # print(decoded.shape)

        resNetTrunk.train()
        asppModel.train()
        decoder.train()

        return decoded

    return predict

def calcMetrics(decoded, groundTruths):
    groundTruths = groundTruths.to(C.DEVICE)

    loss = metrics.CELoss(decoded, groundTruths)
    acc = metrics.pixelAccuracy(decoded, groundTruths)
    pred = computeTorchStdGroundTruth(decoded)

    return loss, acc, pred
