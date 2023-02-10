import torch
import torch.nn as nn
from layers import *


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_glorot (m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None: 
            m.bias.data.fill_(0.01)

class Residual_PA_UNet_Generator(nn.Module):
    
    def __init__(self, in_channels ):
        super(Residual_PA_UNet_Generator, self).__init__()

        """ Input Convolutional """
        self.convInput = nn.Conv2d(
            in_channels     = in_channels,
            out_channels    = 32,
            kernel_size     = 3,
            stride          = 1,
            padding         = 'same'
        )

        """ DownSampling Block """
        self.RPA1   = Residual_PA_block_1(32,32)
        self.DS1    = DS_block(32,64)
        self.RPA2   = Residual_PA_block_1(64,64)
        self.DS2    = DS_block(64,128)
        self.RPA3   = Residual_PA_block_1(128,128)
        self.DS3    = DS_block(128,256)
        self.RPA4   = Residual_PA_block_1(256, 256)
        self.DS4    = DS_block(256,256)

        """ Fusion Block """
        self.convFusion = nn.Conv2d(
            in_channels     = 256,
            out_channels    = 256,
            kernel_size     = 3,
            stride          = 1,
            padding         = 'same'
        )
        self.batchnormFusion = nn.BatchNorm2d(256, momentum=0.8)
        self.reluFusion = nn.ReLU()

        """ Upsampling Block"""
        self.US1 = US_block( 256, 256 )
        self.RPA5 = Residual_PA_block_1( 256, 256 )
        self.US2 = US_block( 256, 128 )
        self.RPA6 = Residual_PA_block_1( 128, 128 )
        self.US3 = US_block( 128, 64 )
        self.RPA7 = Residual_PA_block_1( 64, 64 )
        self.US4 = US_block( 64, 32 )
        self.RPA8 = Residual_PA_block_1( 32, 32 )
        
        """ Output Convolutional """
        self.convOut = nn.Conv2d(
            in_channels         = 32,
            out_channels        = 1,
            kernel_size         = 3,
            stride              = 1,
            padding             = 'same'
        )
        
        self.actOut = nn.Sigmoid()
    
    def forward(self, img_input):
        
        """ DownSampling Block Forward """
        outConvInit         = self.convInput(img_input)
        outRPA1, attn1      = self.RPA1(outConvInit)
        outDS               = self.DS1(outRPA1)
        outRPA2, attn2      = self.RPA2(outDS)
        outDS               = self.DS2(outRPA2)
        outRPA3, attn3      = self.RPA3(outDS)
        outDS               = self.DS3(outRPA3)
        outRPA4, attn4      = self.RPA4(outDS)
        outDS               = self.DS4(outRPA4)

        """ Fusion Block Forward """
        out = self.convFusion(outDS)
        out = self.batchnormFusion(out)
        out = self.reluFusion(out)

        """ Upsampling Block Forward """
        out         = self.US1( out, outRPA4 )
        out, attn5  = self.RPA5( out )
        out         = self.US2( out, outRPA3 )
        out, attn6  = self.RPA6( out )
        out         = self.US3(out, outRPA2 )
        out, attn7  = self.RPA7(out)
        out         = self.US4( out, outRPA1 )
        out, attn8  = self.RPA8(out)

        """ Output Convolution """
        out = self.convOut(out)
        out = self.actOut(out)

        outDictionary = {

            "image_input": img_input,
            "attn1": attn1,
            "attn2": attn2,
            "attn3": attn3,
            "attn4": attn4,
            "attn5": attn5,
            "attn6": attn6,
            "attn7": attn7,
            "output_image": out,
        }

        return out, outDictionary

class PatchGAN_Discriminator(nn.Module):
    def __init__(self, n_channels=3):
        super(PatchGAN_Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(n_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)