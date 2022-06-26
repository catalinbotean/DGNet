from torch.nn import SyncBatchNorm as Batch_Normalization_Layer, \
                     functional

from networks.deep_lab_v3plus.deep_lab_v3plus import GarmentClassifier


def get_network():
    return GarmentClassifier()


def Normalization_2D_Layer(in_channels):
    normalization_layer = Batch_Normalization_Layer(in_channels)
    return normalization_layer


def Upsample_Layer(x, size):
    return functional.interpolate(x, size=size, mode='bilinear', align_corners=True)
