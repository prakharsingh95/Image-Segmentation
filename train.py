from PIL import Image

import numpy as np

from MSRCDataset import makeMSRCDataLoader

import config as C

from models import ResNetTrunk, ASPP, Decoder

import torch
import torch.nn as nn
import torch.optim as optim

import utils

import argparse


def main():
    parser = argparse.ArgumentParser(description='Image Segmentation')
    parser.add_argument('--loadCheckpoint', action='store_true', help='Whether to start training from previously saved weights')
    parser.add_argument('--multiGPU', action=f'store_true', help='Whether to use multiple GPUs (if available), else "{C.DEVICE}" will be used')
    parser.add_argument('--saveFreq', default=None, type=int, help='Frequency of saving model weights (via early-stopping)')
    parser.add_argument('--valImages', default=10, type=int, help='Size of a random subset of the valition set for estimating model performance')
    parser.add_argument('--logFreq', default=50, type=int, help='Logging frequency')
    parser.add_argument('--weightDecay', default=C.WEIGHT_DECAY, type=float, help='Whether to use weight decay regularization')
    parser.add_argument('--trainTrunk', action='store_true', help='Whether to train the trunk (Resnet) or not')
    args = parser.parse_args()

    trainDataLoader, valDataLoader, testDataLoader = makeMSRCDataLoader(
        C.MSRC_BASE_PATH, C.MSRC_SPLIT_PATH, C.TRAIN_BATCH_SIZE)
    testDataLoader = utils.makeGenerator(testDataLoader)

    resNetTrunk = ResNetTrunk().to(C.DEVICE)
    asppModel = ASPP().to(C.DEVICE)
    decoder = Decoder(C.NUM_CLASSES).to(C.DEVICE)

    if args.trainTrunk:
        resNetTrunkOptimizer = optim.Adam(
            resNetTrunk.parameters(), lr=C.LEARNING_RATE_RES_NET_TRUNK)
    asppModelOptimizer = optim.Adam(
        asppModel.parameters(), lr=C.LEARNING_RATE_ASPP_MODEL, weight_decay=args.weightDecay)
    decoderOptimizer = optim.Adam(
        decoder.parameters(), lr=C.LEARNING_RATE_DECODER, weight_decay=args.weightDecay)


    if args.loadCheckpoint:
        if args.trainTrunk:
            utils.loadModelFromFile(
                'resNetTrunk', resNetTrunk, resNetTrunkOptimizer)
        utils.loadModelFromFile('asppModel', asppModel, asppModelOptimizer)
        utils.loadModelFromFile('decoder', decoder, decoderOptimizer)

    if args.multiGPU:
        resNetTrunk = nn.DataParallel(resNetTrunk)
        asppModel = nn.DataParallel(asppModel)
        decoder = nn.DataParallel(decoder)

    predict = utils.makePredict(resNetTrunk, asppModel, decoder)

    trainLossTotal = 0.0
    trainAccTotal = 0.0
    bestValAccTotal = 0.0
    for itr, batch in enumerate(trainDataLoader):
        # print(itr, flush=True)
        images, groundTruths = batch

        if args.trainTrunk:
            decoded = predict(images, detachTrunk=False)
        else:
            decoded = predict(images, detachTrunk=True)

        trainLoss, trainAcc, trainPred = utils.calcMetrics(
            decoded, groundTruths)
        trainLossTotal += trainLoss.item()
        trainAccTotal += trainAcc.item()

        # print(torch.min(groundTruths), torch.max(groundTruths))
        # print(torch.min(trainPred), torch.max(trainPred))

        if args.trainTrunk:
            resNetTrunk.zero_grad()
        asppModel.zero_grad()
        decoder.zero_grad()

        trainLoss.backward()

        if args.trainTrunk:
            resNetTrunkOptimizer.step()
        asppModelOptimizer.step()
        decoderOptimizer.step()

        if (itr % args.logFreq) == args.logFreq-1:

            trainLossTotal /= args.logFreq
            trainAccTotal /= args.logFreq

            valLossTotal = 0.0
            valAccTotal = 0.0
            resNetTrunk.eval()
            asppModel.eval()
            decoder.eval()

            imgIdx = np.random.randint(0, args.valImages)
            for i in range(args.valImages):
                images, groundTruths = next(testDataLoader)

                decoded = predict(images, mode='eval')
                valLoss, valAcc, valPred = utils.calcMetrics(
                    decoded, groundTruths)

                valLossTotal += valLoss.item()
                valAccTotal += valAcc.item()

                if i == imgIdx:
                    utils.saveTorchNormalizedImage(
                        images[0], 'images/{}_img.jpg'.format(itr))
                    utils.saveTorchGroundTruth(
                        groundTruths[0], 'images/{}_gt.jpg'.format(itr))
                    utils.saveTorchGroundTruth(
                        valPred[0], 'images/{}_pred.jpg'.format(itr))

            valLossTotal /= args.valImages
            valAccTotal /= args.valImages

            print('Iteration: {}, TrainLoss: {}, TrainAcc: {}, ValLoss: {}, ValAcc: {}'.format(
                itr, trainLossTotal, trainAccTotal, valLossTotal, valAccTotal), flush=True)

            resNetTrunk.train()
            asppModel.train()
            decoder.train()

            if (itr % args.saveFreq) == args.saveFreq-1 and valAccTotal > bestValAccTotal:
                bestValAccTotal = valAccTotal

                if args.multiGPU:
                    resNetTrunkSave = resNetTrunk.module
                    asppModelSave = asppModel.module
                    decoderSave = decoder.module
                else:
                    resNetTrunkSave = resNetTrunk
                    asppModelSave = asppModel
                    decoderSave = decoder

                if args.trainTrunk:
                    utils.saveModelToFile(
                        'resNetTrunk', resNetTrunkSave, resNetTrunkOptimizer)
                utils.saveModelToFile(
                    'asppModel', asppModelSave, asppModelOptimizer)
                utils.saveModelToFile('decoder', decoderSave, decoderOptimizer)
                print('Saved models @iteration {} with accuracy {} and loss {}...'.format(
                    itr, valAccTotal, valLossTotal))
            else:
                print('Skipped saving @iteration {}...'.format(itr))

            trainLossTotal = 0.0
            trainAccTotal = 0.0


if __name__ == "__main__":
    main()
