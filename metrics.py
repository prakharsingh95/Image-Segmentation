import torch
import torch.nn as nn

import config as C
import utils

classWeights = torch.FloatTensor(C.CLASS_WEIGHTS)
celossw = nn.CrossEntropyLoss(weight=classWeights).to(C.DEVICE)
celossu = nn.CrossEntropyLoss().to(C.DEVICE)
def CELoss(decoded, target):
    # return (celossu(decoded, target) + celossw(decoded, target))/2
    return celossw(decoded, target)

def pixelAccuracy(decoded, target):
    pred = utils.computeTorchStdGroundTruth(decoded)
    corr = (pred == target).float()
    return torch.mean(corr)
