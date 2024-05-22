from .unet import UNet
from .RevCol.revcol import *
import segmentation_models_pytorch as smp

def get_model(modelname, inchannels=3, pretrained=True):

    if modelname == "revcol":
        return FullNet(num_classes=1, channels=[128, 256, 512, 1024], layers = [1, 2, 6, 2], num_subnet = 8, drop_path=0.5)
    if modelname == "unet":
        # initialize model (random weights)
        return UNet(n_channels=inchannels,
                     n_classes=1,
                     bilinear=False)
    if modelname == "unet++":
        return smp.UnetPlusPlus(in_channels=inchannels, classes=1)
    if modelname == "manet":
        return smp.MAnet(in_channels=inchannels, classes=1)
    else:
        raise NotImplementedError()

