# Image Segmentation

The aim of this project is to semantically segment images, i.e., label each pixel of the image for the class of the object it displays. For this, I implement the DeepLabV3+ paper available [here](https://arxiv.org/abs/1802.02611).

For training and validation, I use the MSRC dataset available [here](https://jamie.shotton.org/work/data.html).

# Results

Below are the segmentation results for a few images from the validation set. The first column is the input image, the middle column is the ground truth while the last column is the model's prediction.

The 22-class output is visualized by giving a unique color to each class. The class-to-color mapping is available in the file `MSRCDataset.py`.

A cool result to notice is that while the ground truth does not fully label all the pixels in the input image, the model still learns to predict well for ALL pixels!

Input Image           |  Ground Truth | Prediction
:-------------------------:|:-------------------------: | :-------------------------:
<img src="results/A.img" height="200" width="200" style="display:inline;">  |  <img src="results/A_gt.img" height="200" width="200" style="display:inline;"> | <img src="results/A_pred.img" height="200" width="200" style="display:inline;">
<img src="results/B.img" height="200" width="200" style="display:inline;">  |  <img src="results/B_gt.img" height="200" width="200" style="display:inline;"> | <img src="results/B_pred.img" height="200" width="200" style="display:inline;">
<img src="results/C.img" height="200" width="200" style="display:inline;">  |  <img src="results/C_gt.img" height="200" width="200" style="display:inline;"> | <img src="results/C_pred.img" height="200" width="200" style="display:inline;">
<img src="results/D.img" height="200" width="200" style="display:inline;">  |  <img src="results/D_gt.img" height="200" width="200" style="display:inline;"> | <img src="results/D_pred.img" height="200" width="200" style="display:inline;">
<img src="results/E.img" height="200" width="200" style="display:inline;">  |  <img src="results/E_gt.img" height="200" width="200" style="display:inline;"> | <img src="results/E_pred.img" height="200" width="200" style="display:inline;">

# Usage

1. Download the dataset.
2. Edit in `config.py` with hyperparameters and paths to the dataset and weights.
3. Run `$ train.py` (you would need at least one NVIDA GPU).

You can also pass parameters to `train.py` for more control as below

```
usage: train.py [-h] [--loadCheckpoint] [--multiGPU] [--saveFreq SAVEFREQ]
                [--valImages VALIMAGES] [--logFreq LOGFREQ]
                [--weightDecay WEIGHTDECAY] [--trainTrunk]

Image Segmentation

optional arguments:
  -h, --help            show this help message and exit
  --loadCheckpoint      Whether to start training from previously saved
                        weights
  --multiGPU            Whether to use multiple GPUs (if available), else
                        "{C.DEVICE}" will be used
  --saveFreq SAVEFREQ   Frequency of saving model weights (via early-stopping)
  --valImages VALIMAGES
                        Size of a random subset of the valition set for
                        estimating model performance
  --logFreq LOGFREQ     Logging frequency
  --weightDecay WEIGHTDECAY
                        Whether to use weight decay regularization
  --trainTrunk          Whether to train the trunk (Resnet) or not
```
