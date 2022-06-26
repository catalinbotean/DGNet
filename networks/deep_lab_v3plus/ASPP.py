from torch import cat as Concatenation
from torch.nn import AdaptiveAvgPool2d as Average_Pooling_Layer, \
                     Conv2d as Convolution_2D_Layer, \
                     Module, \
                     ModuleList, \
                     ReLU, \
                     Sequential as Sequential_Layer

from networks import Normalization_2D_Layer, Upsample_Layer


class ASPP(Module):
    def __init__(self, input_size, output_size=256, output_stride=16, atrous_rates=(3, 6, 9)):
        super(ASPP, self).__init__()
        atrous_rates = [(output_stride // 32) * rate for rate in atrous_rates]

        self.features = []
        self.features.append(
            Sequential_Layer(
                Convolution_2D_Layer(input_size, output_size, kernel_size=1, bias=False),
                Normalization_2D_Layer(output_size),
                ReLU(inplace=True)
            )
        )

        for rate in atrous_rates:
            self.features.append(Sequential_Layer(
                Convolution_2D_Layer(input_size, output_size, kernel_size=3, dilation=rate, padding=rate, bias=False),
                Normalization_2D_Layer(output_size),
                ReLU(inplace=True)
            ))
        self.features = ModuleList(self.features)

        self.img_pooling = Average_Pooling_Layer(1)
        self.img_conv = Sequential_Layer(
            Convolution_2D_Layer(input_size, output_size, kernel_size=1, bias=False),
            Normalization_2D_Layer(output_size),
            ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()
        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample_Layer(img_features, x_size[2:])
        out = img_features

        for convolution in self.features:
            y = convolution(x)
            out = Concatenation((out, y), 1)
        return out
