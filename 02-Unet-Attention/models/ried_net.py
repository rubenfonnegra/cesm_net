from torch import nn
import torch

class RIEDNet(nn.Module):
    def __init__(self, in_channels, actOut):
        super().__init__()
        self.in_channels    = in_channels
        self.actOut         = actOut

        """ Encoder """
        self.inceptionBlock1    = inception( in_channels, 32)
        self.downSampling1      = downSamplingBlock(32, 32)
        self.inceptionBlock2    = inception( 32, 64)
        self.downSampling2      = downSamplingBlock(64, 64)
        self.inceptionBlock3    = inception( 64, 128)
        self.downSampling3      = downSamplingBlock(128, 128)
        self.inceptionBlock4    = inception( 128, 256)
        self.downSampling4      = downSamplingBlock(256, 256)
        self.inceptionBlock5    = inception( 256, 512)

        """ Decoder """
        self.upSamplingBlock1   = upSamplingBlock(512, 256)
        self.inceptionBlock6    = inception(512, 256)
        self.upSamplingBlock2   = upSamplingBlock(256, 128)
        self.inceptionBlock7    = inception(256, 128)
        self.upSamplingBlock3   = upSamplingBlock(128, 64)
        self.inceptionBlock8    = inception(128, 64)
        self.upSamplingBlock4   = upSamplingBlock(64, 32)
        self.inceptionBlock9    = inception(64, 32)

        self.conv_out       = nn.Conv2d(
            in_channels     = 32,
            out_channels    = 1,
            kernel_size     = 1,
            stride          = 1,
            padding     = 0
        )
    
    def forward(self, x):

        out, skip1  = self.inceptionBlock1(x)   # (32, 256, 256)
        out         = self.downSampling1(out)   # (32, 128, 128)
        out, skip2  = self.inceptionBlock2(out) # (64, 128, 128)
        out         = self.downSampling2(out)   # (64, 64, 64)
        out, skip3  = self.inceptionBlock3(out) # (128, 64, 64)
        out         = self.downSampling3(out)   # (128, 32, 32)
        out, skip4  = self.inceptionBlock4(out) # (256, 32, 32)
        out         = self.downSampling4(out)   # (256, 16, 16)
        out, _      = self.inceptionBlock5(out) # (512, 16, 16)

        out         = self.upSamplingBlock1(out, skip4)     # (512, 32, 32)
        out, _      = self.inceptionBlock6(out)             # (256, 32, 32)
        out         = self.upSamplingBlock2(out, skip3)     # (256, 64, 64)
        out, _      = self.inceptionBlock7(out)             # (128, 64, 64)
        out         = self.upSamplingBlock3(out, skip2)     # (128, 128, 128)
        out, _      = self.inceptionBlock8(out)             # (64, 128, 128)
        out         = self.upSamplingBlock4(out, skip1)     # (64, 256, 256)
        out, _      = self.inceptionBlock9(out)             # (32, 256, 256)

        out         = self.conv_out(out)

        if(self.actOut != None):
            out     =   self.actOut(out) 

        return out


class inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(inception, self).__init__()

        self.conv3x3_1 = nn.Conv2d(
            in_channels     = in_channels,
            out_channels    = out_channels,
            kernel_size     = (3,3),
            stride          = (1,1),
            padding         = 1
        )

        self.relu = nn.ReLU()

        self.conv3x3_2 = nn.Conv2d(
            in_channels     = out_channels,
            out_channels    = out_channels,
            kernel_size     = 3,
            stride          = 1,
            padding         = 1
        )

        self.conv1x1 = nn.Conv2d(
            in_channels     = in_channels,
            out_channels    = out_channels,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0
        )
    
    def forward(self, x):
        
        out_3x3 = self.conv3x3_1(x)
        out_3x3 = self.relu(out_3x3)
        out_3x3 = self.conv3x3_2(out_3x3)
        out_3x3 = self.relu(out_3x3)

        out_1x1 = self.conv1x1(x)
        out_1x1 = self.relu(out_1x1)

        out = torch.add(out_3x3, out_1x1)
        out = self.relu(out)
        return out, out_3x3

class downSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downSamplingBlock, self).__init__()

        self.conv3x3 = nn.Conv2d(
            in_channels     = in_channels,
            out_channels    = out_channels,
            kernel_size     = 3,
            stride          = 2,
            padding         = 1
        )

        self.relu = nn.ReLU()
    
    def forward(self, x):
        
        out = self.conv3x3(x)
        out = self.relu(out)
        return out

class upSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upSamplingBlock, self).__init__()

        self.convT3x3 = nn.ConvTranspose2d(
            in_channels     = in_channels,
            out_channels    = out_channels,
            kernel_size     = 3,
            stride          = 2,
            padding         = 1,
            output_padding  = 1
        )

        self.relu = nn.ReLU()
    
    def forward(self, x, skip):
        out = self.convT3x3(x)
        out = self.relu(out)
        out = torch.cat((out,skip), 1)
        return out