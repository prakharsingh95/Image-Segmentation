import os
import logging

from PIL import Image, ImageFilter
import numpy as np

from torchvision import transforms

Normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

PILtoTensor = transforms.ToTensor()


class Scale(object):
    def __init__(self, scale=8):
        self.scale = scale

    def __call__(self, image, groundTruth):
        origWidth, origHeight = image.size

        newWidth = int(self.scale * np.ceil(origWidth/self.scale))
        newHeight = int(self.scale * np.ceil(origHeight/self.scale))
        newSize = (newWidth, newHeight)

        image = image.resize(newSize, Image.BILINEAR)
        groundTruth = groundTruth.resize(newSize, Image.NEAREST)

        return image, groundTruth


class SquareCrop(object):
    def __init__(self, baseSize=192):
        self.baseSize = baseSize

    def __call__(self, image, groundTruth):
        origWidth, origHeight = image.size
        flip270 = False

        if(origWidth < origHeight):
            image = image.transpose(Image.ROTATE_90)
            groundTruth = groundTruth.transpose(Image.ROTATE_90)
            origWidth, origHeight = image.size
            flip270 = True

        if(origHeight < self.baseSize):
            scale = float(self.base) / origHeight
            newWidth = int(origWidth * scale)
            newHeight = self.baseSize
            newSize = (newWidth, newHeight)
            image = image.resize(newSize, Image.BILINEAR)
            groundTruth = groundTruth.resize(newSize, Image.NEAREST)

        newWidth, newHeight = image.size

        cropX = np.random.randint(0, newWidth-self.baseSize+1)
        cropY = np.random.randint(0, newHeight-self.baseSize+1)

        image = image.crop(
            (cropX, cropY, cropX + self.baseSize, cropY + self.baseSize))
        groundTruth = groundTruth.crop(
            (cropX, cropY, cropX + self.baseSize, cropY + self.baseSize))

        if flip270:
            image = image.transpose(Image.ROTATE_270)
            groundTruth = groundTruth.transpose(Image.ROTATE_270)

        return image, groundTruth


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, groundTruth):
        if np.random.rand() <= self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            groundTruth = groundTruth.transpose(Image.FLIP_LEFT_RIGHT)
        return image, groundTruth


class RandomNoise(object):
    def __init__(self, p=0.5, noiseLevel=5):
        self.p = p
        self.noiseLevel = noiseLevel

    def __call__(self, image, groundTruth):
        if np.random.rand() <= self.p:
            noiseArray = np.random.randint(-self.noiseLevel,
                                           self.noiseLevel+1, size=(3,))
            imageNp = np.array(image).astype(np.int32) + noiseArray
            imageNp = np.clip(imageNp, 0, 255).astype(np.uint8)
            image = Image.fromarray(imageNp)
        return image, groundTruth


class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, groundTruth):
        if np.random.rand() <= self.p:
            gaussianFilter = ImageFilter.GaussianBlur(
                radius=np.random.rand()*1)
            image = image.filter(gaussianFilter)
        return image, groundTruth


class RandomScaledCrop(object):
    def __init__(self, p=0.5, scale=2.5):
        self.p = p
        self.scale = scale

    def __call__(self, image, groundTruth):
        origWidth, origHeight = image.size

        scaleX = 1 + np.random.rand() * (self.scale-1)
        scaleY = 1 + np.random.rand() * (self.scale-1)

        newWidth, newHeight = int(origWidth*scaleX), int(origHeight*scaleY)
        newSize = (newWidth, newHeight)

        image = image.resize(newSize, Image.BILINEAR)
        groundTruth = groundTruth.resize(newSize, Image.NEAREST)

        cropX = np.random.randint(0, newWidth-origWidth+1)
        cropY = np.random.randint(0, newHeight-origHeight+1)

        image = image.crop(
            (cropX, cropY, cropX + origWidth, cropY + origHeight))
        groundTruth = groundTruth.crop(
            (cropX, cropY, cropX + origWidth, cropY + origHeight))

        assert image.size == (origWidth, origHeight)
        assert groundTruth.size == (origWidth, origHeight)

        return image, groundTruth


class RandomColorJitter(object):
    def __init__(self, p=0.5):
        self.p = p
        self.colorJitter = transforms.ColorJitter(
            brightness=0.08, contrast=0.08, saturation=0.08, hue=0.08)

    def __call__(self, image, groundTruth):
        if np.random.rand() <= self.p:
            image = self.colorJitter(image)
        return image, groundTruth


class RandomRotate(object):
    def __init__(self, p=0.5, maxDegree=5, scale=1.6):
        self.p = p
        self.maxDegree = maxDegree
        self.scale = scale

    def __call__(self, image, groundTruth):
        if np.random.rand() <= self.p:
            rnd = np.random.rand()*2-1
            deg = rnd * self.maxDegree

            origWidth, origHeight = image.size

            newWidth, newHeight = int(
                origWidth*self.scale), int(origHeight*self.scale)

            image = image.resize((newWidth, newHeight), Image.BILINEAR)
            groundTruth = groundTruth.resize(
                (newWidth, newHeight), Image.NEAREST)

            tx, ty = (newWidth-origWidth)/2, (newHeight-origHeight)/2
            bx, by = tx + origWidth, ty + origHeight

            image = image.rotate(deg, Image.BILINEAR)
            groundTruth = groundTruth.rotate(deg, Image.NEAREST)

            image = image.crop((tx, ty, bx, by))
            groundTruth = groundTruth.crop((tx, ty, bx, by))

        return image, groundTruth
