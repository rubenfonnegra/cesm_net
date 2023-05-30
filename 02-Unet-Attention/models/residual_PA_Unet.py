import torch
import torch.nn as nn
from models.layers import *


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

""" 
******************************************************************
***** Implementation Residual Pixel Attention Unet Generator *****
******************************************************************
"""
class RPA_Unet_Generator(nn.Module):
    
    def __init__(self, in_channels, actOut):
        super(RPA_Unet_Generator, self).__init__()

        self.attn_maps  = {}
        self.actOut     = actOut

        """ Input Convolutional """
        self.convInput = nn.Conv2d(
            in_channels     = in_channels,
            out_channels    = 32,
            kernel_size     = 3,
            stride          = 1,
            padding         = 'same'
        )

        """ DownSampling Block """
        self.RPA0   = Residual_PA_block( 32, 32)
        self.DS1    = DS_block(32,64)
        self.RPA1   = Residual_PA_block(64,64)
        self.DS2    = DS_block(64,128)
        self.RPA2   = Residual_PA_block(128,128)
        self.DS3    = DS_block(128,256)
        self.RPA3   = Residual_PA_block(256, 256)
        self.DS4    = DS_block(256,512)
        self.RPA4   = Residual_PA_block(512, 512)
        self.DS5    = DS_block(512,1024)

        """ Fusion Block """
        self.convFusion = nn.Conv2d(
            in_channels     = 1024,
            out_channels    = 1024,
            kernel_size     = 3,
            stride          = 1,
            padding         = 'same'
        )
        self.batchnormFusion = nn.BatchNorm2d(1024, momentum=0.8)
        self.actFusion = nn.LeakyReLU(0.2)

        """ Upsampling Block"""
        self.US1 = US_block(1024, 512)
        self.RB2 = Residual_PA_block( 512, 512 )
        self.US2 = US_block( 512, 256 )
        self.RB3 = Residual_PA_block( 256, 256 )
        self.US3 = US_block( 256, 128 )
        self.RB4 = Residual_PA_block( 128, 128 )
        self.US4 = US_block( 128, 64 )
        self.RB5 = Residual_PA_block( 64, 64 )
        self.US5 = US_block( 64, 32 )
        self.RB6 = Residual_PA_block( 32, 32 )
        
        """ Output Convolutional """
        self.convOut = nn.Conv2d(
            in_channels         = 32,
            out_channels        = 1,
            kernel_size         = 3,
            stride              = 1,
            padding             = 'same'
        )
            
    def forward(self, img_input):
        
        """ DownSampling Block Forward """
        outConvInit         = self.convInput(img_input)     # (B, 32, 256, 256)
        outRB1, attn0       = self.RPA0(outConvInit)         # (B, 32, 256, 256)
        outDS               = self.DS1(outRB1)              # (B, 64, 128, 128)
        outRPA1, attn1      = self.RPA1(outDS)              # (B, 64, 128, 128)
        outDS               = self.DS2(outRPA1)             # (B, 128, 64, 64)
        outRPA2, attn2      = self.RPA2(outDS)              # (B, 128, 64, 64)
        outDS               = self.DS3(outRPA2)             # (B, 256, 32, 32)
        outRPA3, attn3      = self.RPA3(outDS)              # (B, 256, 32, 32)
        outDS               = self.DS4(outRPA3)             # (B, 512, 16, 16)
        outRPA4, attn4      = self.RPA4(outDS)              # (B, 512, 16, 16)
        outDS               = self.DS5(outRPA4)             # (B, 1024, 8, 8)

        """ Fusion Block Forward """
        out = self.convFusion(outDS)                        # (B, 1024, 8, 8)
        out = self.batchnormFusion(out)                     # (B, 1024, 8, 8)
        out = self.actFusion(out)                          # (B, 1024, 8, 8)

        """ Upsampling Block Forward """
        out             = self.US1( out, outRPA4 )              # (B, 512, 16, 16)
        out, attn5      = self.RB2( out )                       # (B, 512, 16, 16)
        out             = self.US2( out, outRPA3 )              # (B, 256, 32, 32)
        out, attn6      = self.RB3( out )                       # (B, 256, 32, 32)
        out             = self.US3(out, outRPA2 )               # (B, 128, 64, 64)
        out, attn7      = self.RB4(out)                         # (B, 128, 64, 64)
        out             = self.US4(out, outRPA1 )               # (B, 64, 128, 128)
        out, attn8      = self.RB5(out)                         # (B, 64, 128, 128)
        out             = self.US5(out, outRB1 )                # (B, 32, 256, 256)
        out, attn9      = self.RB6(out)                         # (B, 32, 256, 256)

        """ Output Convolution """
        out = self.convOut(out)

        if (self.actOut != None):
            out = self.actOut(out)

        self.attn_maps = {

            "image_input": img_input,
            "attn0": attn0,
            "attn1": attn1,
            "attn2": attn2,
            "attn3": attn3,
            "attn4": attn4,
            "attn5": attn5,
            "attn6": attn6,
            "attn7": attn7,
            "attn8": attn8,
            "attn9": attn9,
            "output_image": out,
        }

        return out
