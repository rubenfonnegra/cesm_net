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


""" 
*********************************************************
************ Implementation Encoder VAE *****************
*********************************************************
"""
class Encoder_VAE(nn.Module):
    #
    def __init__(self, in_channels ):
        super(Encoder_VAE, self).__init__()

        """ Input Convolutional """
        self.convInput = nn.Conv2d(
            in_channels     = in_channels,
            out_channels    = 32,
            kernel_size     = 3,
            stride          = 1,
            padding         = 'same'
        )

        """ DownSampling Block """
        self.RB1 = R_block(32,32)       
        self.DS1 = DS_block(32,64)                
        self.RB2 = R_block(64,64)       
        self.DS2 = DS_block(64,128)     
        self.RB3 = R_block(128,128)     
        self.DS3 = DS_block(128,256)
        self.RB4 = R_block(256,256)
        self.DS4 = DS_block(256,512)
        self.RB5 = R_block(512,512)
        self.DS5 = DS_block(512,1024)
        
        """ Fusion Block """
        self.convFusion = nn.Conv2d(
            in_channels     = 1024,
            out_channels    = 1024,
            kernel_size     = 3,
            stride          = 1,
            padding         = 'same'
        )
        self.batchnormFusion = nn.BatchNorm2d(1024, momentum=0.8)
        self.reluFusion = nn.ReLU()
    
    def forward(self, img_input):
        
        skips = []

        """ DownSampling Block Forward """
        outConvInit = self.convInput(img_input) # (B, 32, 256,256)
        outRB1  = self.RB1(outConvInit)         # (B, 32, 256,256)
        outDS   = self.DS1(outRB1)              # (B, 64, 128,128)
        outRB2  = self.RB2(outDS)               # (B, 64, 128,128)
        outDS   = self.DS2(outRB2)              # (B, 128, 64, 64)
        outRB3  = self.RB3(outDS)               # (B, 128, 64, 64)
        outDS   = self.DS3(outRB3)              # (B, 256, 32, 32)
        outRB4  = self.RB4(outDS)               # (B, 256, 32, 32)
        outDS   = self.DS4(outRB4)              # (B, 512, 16, 16)
        outRB5  = self.RB5(outDS)               # (B, 512, 16, 16)
        outDS   = self.DS5(outRB5)              # (B, 1024, 8, 8)

        """ Fusion Block Forward """
        out = self.convFusion(outDS)            # (B, 1024, 8, 8)
        out = self.batchnormFusion(out)         # (B, 1024, 8, 8)
        out = self.reluFusion(out)              # (B, 1024, 8, 8)

        skips.append(outRB1.detach())
        skips.append(outRB2.detach())
        skips.append(outRB3.detach())
        skips.append(outRB4.detach())
        skips.append(outRB5.detach())

        return out, skips


""" 
*********************************************************
************ Implementation Decoder VAE *****************
*********************************************************
"""
class Decoder_VAE(nn.Module):
    #
    def __init__(self):
        super(Decoder_VAE, self).__init__()

        """ Upsampling Block""" 
        self.US1    = US_block(1024, 512)
        self.RB6    = R_block(512, 512)
        self.US2    = US_block(512, 256)
        self.RB7    = R_block(256, 256)          
        self.US3    = US_block(256, 128)        
        self.RB8    = R_block(128, 128)
        self.US4    = US_block(128, 64)        
        self.RB9    = R_block(64, 64)
        self.US5    = US_block(64, 32)        
        self.RB10   = R_block(32, 32)             
        
        """ Output Convolutional """
        self.convOut = nn.Conv2d(
            in_channels         = 32,
            out_channels        = 1,
            kernel_size         = 3,
            stride              = 1,
            padding             = 'same'
        )
        
        self.actOut = nn.Sigmoid()

    def forward( self, input, skips ):
        
        """ Upsampling Block Forward """
        out = self.US1( input, skips[-1] )      # (B, 512, 16, 16)
        out = self.RB6( out )                   # (B, 512, 16, 16)
        out = self.US2( out, skips[-2] )        # (B, 256, 32, 32)
        out = self.RB7( out )                   # (B, 256, 32, 32)
        out = self.US3(out, skips[-3] )         # (B, 128, 64, 64)
        out = self.RB8(out)                     # (B, 128, 64, 64)
        out = self.US4(out, skips[-4] )         # (B, 64, 128, 128)
        out = self.RB9(out)                     # (B, 64, 128, 128)
        out = self.US5(out, skips[-5] )         # (B, 32, 256, 256)
        out = self.RB10(out)                    # (B, 32, 256, 256)

        """ Output Convolution """
        out = self.convOut(out)                 # (B, 1, 256, 256)
        out = self.actOut(out)                  # (B, 1, 256, 256)
        
        return out