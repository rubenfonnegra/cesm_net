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
********************************************************************
***** Implementation Self-Attention Unet Generator version #1 ******
********************************************************************

Self Attention Unet with Gamma = 0.5 and learned.
"""
class SA_Unet_v1(nn.Module):

    def __init__(self, in_channels, gamma ):
        super(SA_Unet_v1, self).__init__()

        """ Input Convolutional """
        self.convInput = nn.Conv2d(
            in_channels     = in_channels,
            out_channels    = 32,
            kernel_size     = 3,
            stride          = 1,
            padding         = 'same'
        )

        """ DownSampling Block """
        self.RB1    = R_block(32,32)
        self.DS1    = DS_block(32,64)
        self.RB2    = R_block(64,64)
        self.DS2    = DS_block(64,128)
        self.attn1  = Self_Attention_v1(128, gamma=gamma)
        self.RB3    = R_block(128,128)
        self.DS3    = DS_block(128,256)
        self.RB4    = R_block(256,256)
        self.DS4    = DS_block(256,512)
        self.RB5    = R_block(512,512)
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
        self.reluFusion = nn.ReLU()

        """ Upsampling Block"""
        self.US1    = US_block( 1024, 512 )
        self.RB6    = R_block( 512, 512 )
        self.US2    = US_block( 512, 256 )
        self.RB7    = R_block( 256, 256 )
        self.US3    = US_block( 256, 128 )
        self.RB8    = R_block( 128, 128 )
        self.US4    = US_block( 128, 64 )
        self.RB9    = R_block( 64, 64 )
        self.US5    = US_block( 64, 32 )
        self.RB10   = R_block( 32, 32 )
        
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
        outConvInit                 = self.convInput(img_input)         # (B, 32, 256, 256)
        outRB1                      = self.RB1(outConvInit)             # (B, 32, 256, 256)
        outDS                       = self.DS1(outRB1)                  # (B, 64, 128, 128)
        outRB2                      = self.RB2(outDS)                   # (B, 64, 128, 128)
        outDS                       = self.DS2(outRB2)                  # (B, 128, 64, 64)
        outRB3                      = self.RB3(outDS)                   # (B, 128, 64, 64)
        outAttn1, attnMaps1, gamma  = self.attn1(outRB3)                # (B, 128, 64, 64)
        outDS                       = self.DS3(outAttn1)                # (B, 256, 32, 32)
        outRB4                      = self.RB4(outDS)                   # (B, 256, 32, 32)
        outDS                       = self.DS4(outRB4)                  # (B, 512, 16, 16)
        outRB5                      = self.RB5(outDS)                   # (B, 512, 16, 16)
        outDS                       = self.DS5(outRB5)                  # (B, 1024, 8, 8)

        """ Fusion Block Forward """
        out = self.convFusion(outDS)                                # (B, 1024, 8, 8)                            
        out = self.batchnormFusion(out)                             # (B, 1024, 8, 8)
        out = self.reluFusion(out)                                  # (B, 1024, 8, 8)

        """ Upsampling Block Forward """
        out             = self.US1( out, outRB5 )                   # (B, 512, 16, 16)
        out             = self.RB6( out )                           # (B, 512, 16, 16)
        out             = self.US2( out, outRB4 )                   # (B, 256, 32, 32)
        out             = self.RB7( out )                           # (B, 256, 32, 32)
        out             = self.US3(out, outAttn1 )                  # (B, 128, 64, 64)
        out             = self.RB8(out)                             # (B, 128, 64, 64)
        out             = self.US4( out, outRB2 )                   # (B, 64, 128, 128)
        out             = self.RB9(out)                             # (B, 64, 128, 128)
        out             = self.US5( out, outRB1 )                   # (B, 32, 256, 256)
        out             = self.RB10(out)                            # (B, 32, 256, 256)

        """ Output Convolution """
        out = self.convOut(out)
        out = self.actOut(out)

        outDictionary = {

            "image_input": img_input,
            "attn1": attnMaps1,
            "output_image": out,
        }

        return out, outDictionary, gamma


""" 
********************************************************************
***** Implementation Self-Attention Unet Generator version #2 ******
********************************************************************

Self Attention Unet with Gamma = 0.5 and learned after # epochs.
"""
class SA_Unet_v2(nn.Module):

    def __init__(self, in_channels ):
        super(SA_Unet_v2, self).__init__()

        """ Input Convolutional """
        self.convInput = nn.Conv2d(
            in_channels     = in_channels,
            out_channels    = 32,
            kernel_size     = 3,
            stride          = 1,
            padding         = 'same'
        )

        """ DownSampling Block """
        self.RB1    = R_block(32,32)
        self.DS1    = DS_block(32,64)
        self.RB2    = R_block(64,64)
        self.DS2    = DS_block(64,128)
        self.attn1  = Self_Attention_v2(128)
        self.RB3    = R_block(128,128)
        self.DS3    = DS_block(128,256)
        self.RB4    = R_block(256,256)
        self.DS4    = DS_block(256,512)
        self.RB5    = R_block(512,512)
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
        self.reluFusion = nn.ReLU()

        """ Upsampling Block"""
        self.US1    = US_block( 1024, 512 )
        self.RB6    = R_block( 512, 512 )
        self.US2    = US_block( 512, 256 )
        self.RB7    = R_block( 256, 256 )
        self.US3    = US_block( 256, 128 )
        self.RB8    = R_block( 128, 128 )
        self.US4    = US_block( 128, 64 )
        self.RB9    = R_block( 64, 64 )
        self.US5    = US_block( 64, 32 )
        self.RB10   = R_block( 32, 32 )
        
        """ Output Convolutional """
        self.convOut = nn.Conv2d(
            in_channels         = 32,
            out_channels        = 1,
            kernel_size         = 3,
            stride              = 1,
            padding             = 'same'
        )
        
        self.actOut = nn.Sigmoid()
    
    def forward(self, img_input, learned=False):
        
        """ DownSampling Block Forward """
        outConvInit                 = self.convInput(img_input)         # (B, 32, 256, 256)
        outRB1                      = self.RB1(outConvInit)             # (B, 32, 256, 256)
        outDS                       = self.DS1(outRB1)                  # (B, 64, 128, 128)
        outRB2                      = self.RB2(outDS)                   # (B, 64, 128, 128)
        outDS                       = self.DS2(outRB2)                  # (B, 128, 64, 64)
        outRB3                      = self.RB3(outDS)                   # (B, 128, 64, 64)
        outAttn1, attnMaps1, gamma  = self.attn1(outRB3, learned)       # (B, 128, 64, 64)
        outDS                       = self.DS3(outAttn1)                # (B, 256, 32, 32)
        outRB4                      = self.RB4(outDS)                   # (B, 256, 32, 32)
        outDS                       = self.DS4(outRB4)                  # (B, 512, 16, 16)
        outRB5                      = self.RB5(outDS)                   # (B, 512, 16, 16)
        outDS                       = self.DS5(outRB5)                  # (B, 1024, 8, 8)

        """ Fusion Block Forward """
        out = self.convFusion(outDS)                                # (B, 1024, 8, 8)                            
        out = self.batchnormFusion(out)                             # (B, 1024, 8, 8)
        out = self.reluFusion(out)                                  # (B, 1024, 8, 8)

        """ Upsampling Block Forward """
        out             = self.US1( out, outRB5 )                   # (B, 512, 16, 16)
        out             = self.RB6( out )                           # (B, 512, 16, 16)
        out             = self.US2( out, outRB4 )                   # (B, 256, 32, 32)
        out             = self.RB7( out )                           # (B, 256, 32, 32)
        out             = self.US3(out, outAttn1 )                  # (B, 128, 64, 64)
        out             = self.RB8(out)                             # (B, 128, 64, 64)
        out             = self.US4( out, outRB2 )                   # (B, 64, 128, 128)
        out             = self.RB9(out)                             # (B, 64, 128, 128)
        out             = self.US5( out, outRB1 )                   # (B, 32, 256, 256)
        out             = self.RB10(out)                            # (B, 32, 256, 256)

        """ Output Convolution """
        out = self.convOut(out)
        out = self.actOut(out)

        outDictionary = {

            "image_input": img_input,
            "attn1": attnMaps1,
            "output_image": out,
        }

        return out, outDictionary, gamma



""" 
********************************************************************
***** Implementation Self-Attention Unet Generator version #1 ******
********************************************************************

Self Attention Unet with Gamma = 0.5 and learned. The attention layer is in decoder
"""
class SA_Unet_v3(nn.Module):

    def __init__(self, in_channels, gamma ):
        super(SA_Unet_v1, self).__init__()

        """ Input Convolutional """
        self.convInput = nn.Conv2d(
            in_channels     = in_channels,
            out_channels    = 32,
            kernel_size     = 3,
            stride          = 1,
            padding         = 'same'
        )

        """ DownSampling Block """
        self.RB1    = R_block(32,32)
        self.DS1    = DS_block(32,64)
        self.RB2    = R_block(64,64)
        self.DS2    = DS_block(64,128)
        self.RB3    = R_block(128,128)
        self.DS3    = DS_block(128,256)
        self.RB4    = R_block(256,256)
        self.DS4    = DS_block(256,512)
        self.RB5    = R_block(512,512)
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
        self.reluFusion = nn.ReLU()

        """ Upsampling Block"""
        self.US1    = US_block( 1024, 512 )
        self.RB6    = R_block( 512, 512 )
        self.US2    = US_block( 512, 256 )
        self.RB7    = R_block( 256, 256 )
        self.US3    = US_block( 256, 128 )
        self.RB8    = R_block( 128, 128 )
        self.attn1  = Self_Attention_v1(128, gamma=gamma)
        self.US4    = US_block( 128, 64 )
        self.RB9    = R_block( 64, 64 )
        self.US5    = US_block( 64, 32 )
        self.RB10   = R_block( 32, 32 )
        
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
        outConvInit                 = self.convInput(img_input)         # (B, 32, 256, 256)
        outRB1                      = self.RB1(outConvInit)             # (B, 32, 256, 256)
        outDS                       = self.DS1(outRB1)                  # (B, 64, 128, 128)
        outRB2                      = self.RB2(outDS)                   # (B, 64, 128, 128)
        outDS                       = self.DS2(outRB2)                  # (B, 128, 64, 64)
        outRB3                      = self.RB3(outDS)                   # (B, 128, 64, 64)
        outDS                       = self.DS3(outRB3)                # (B, 256, 32, 32)
        outRB4                      = self.RB4(outDS)                   # (B, 256, 32, 32)
        outDS                       = self.DS4(outRB4)                  # (B, 512, 16, 16)
        outRB5                      = self.RB5(outDS)                   # (B, 512, 16, 16)
        outDS                       = self.DS5(outRB5)                  # (B, 1024, 8, 8)

        """ Fusion Block Forward """
        out = self.convFusion(outDS)                                # (B, 1024, 8, 8)                            
        out = self.batchnormFusion(out)                             # (B, 1024, 8, 8)
        out = self.reluFusion(out)                                  # (B, 1024, 8, 8)

        """ Upsampling Block Forward """
        out                         = self.US1( out, outRB5 )                   # (B, 512, 16, 16)
        out                         = self.RB6( out )                           # (B, 512, 16, 16)
        out                         = self.US2( out, outRB4 )                   # (B, 256, 32, 32)
        out                         = self.RB7( out )                           # (B, 256, 32, 32)
        out                         = self.US3(out, outRB3 )                    # (B, 128, 64, 64)
        out                         = self.RB8(out)                             # (B, 128, 64, 64)
        outAttn1, attnMaps1, gamma  = self.attn1(out)                           # (B, 128, 64, 64)
        out                         = self.US4( outAttn1, outRB2 )              # (B, 64, 128, 128)
        out                         = self.RB9(out)                             # (B, 64, 128, 128)
        out                         = self.US5( out, outRB1 )                   # (B, 32, 256, 256)
        out                         = self.RB10(out)                            # (B, 32, 256, 256)

        """ Output Convolution """
        out = self.convOut(out)
        out = self.actOut(out)

        outDictionary = {

            "image_input": img_input,
            "attn1": attnMaps1,
            "output_image": out,
        }

        return out, outDictionary, gamma
    
"""
---------- Implementation of Self-Attention Layer -----------
Taken from: https://github.com/heykeetae/Self-Attention-GAN/issues/54

Version Original Self-Attention
"""

class Self_Attention_v1(nn.Module):
    def __init__(self, inChannels, k=8, gamma=0.0):
        super(Self_Attention_v1, self).__init__()

        embedding_channels  = inChannels // k  # C_bar
        self.key            = nn.Conv2d(inChannels, embedding_channels, 1)
        self.query          = nn.Conv2d(inChannels, embedding_channels, 1)
        self.value          = nn.Conv2d(inChannels, embedding_channels, 1)
        self.self_att       = nn.Conv2d(embedding_channels, inChannels, 1)
        self.gamma          = nn.Parameter(torch.tensor(gamma))
        self.softmax        = nn.Softmax(dim=1)

    def forward(self,x):
        """
            inputs:
                x: input feature map [Batch, Channel, Height, Width]
            returns:
                out: self attention value + input feature
                attention: [Batch, Channel, Height, Width]
        """
        batchsize, C, H, W = x.size()
        N = H * W                                       # Number of features
        f_x = self.key(x).view(batchsize,   -1, N)      # Keys                  [B, C_bar, N]
        g_x = self.query(x).view(batchsize, -1, N)      # Queries               [B, C_bar, N]
        h_x = self.value(x).view(batchsize, -1, N)      # Values                [B, C_bar, N]

        s =  torch.bmm(f_x.permute(0,2,1), g_x)         # Scores                [B, N, N]
        beta = self.softmax(s)                          # Attention Map         [B, N, N]

        v = torch.bmm(h_x, beta)                        # Value x Softmax       [B, C_bar, N]
        v = v.view(batchsize, -1, H, W)                 # Recover input shape   [B, C_bar, H, W]
        o = self.self_att(v)                            # Self-Attention output [B, C, H, W]
        
        y = self.gamma * o + x                               # Learnable gamma + residual
        return y, o, self.gamma




"""
---------- Implementation of Self-Attention Layer -----------
Taken from: https://github.com/heykeetae/Self-Attention-GAN/issues/54

Version Original Self-Attention
"""

class Self_Attention_v2(nn.Module):
    def __init__(self, inChannels, k=8):
        super(Self_Attention_v2, self).__init__()

        embedding_channels  = inChannels // k  # C_bar
        self.key            = nn.Conv2d(inChannels, embedding_channels, 1)
        self.query          = nn.Conv2d(inChannels, embedding_channels, 1)
        self.value          = nn.Conv2d(inChannels, embedding_channels, 1)
        self.self_att       = nn.Conv2d(embedding_channels, inChannels, 1)
        self.gamma          = nn.Parameter(torch.tensor(0.5),requires_grad=False)
        self.gamma_         = nn.Parameter(torch.tensor(0.5),requires_grad=True)
        self.softmax        = nn.Softmax(dim=1)

    def forward(self,x, learned=False):
        """
            inputs:
                x: input feature map [Batch, Channel, Height, Width]
            returns:
                out: self attention value + input feature
                attention: [Batch, Channel, Height, Width]
        """

        if (learned):
            self.gamma = self.gamma_
        
        batchsize, C, H, W = x.size()
        N = H * W                                       # Number of features
        f_x = self.key(x).view(batchsize,   -1, N)      # Keys                  [B, C_bar, N]
        g_x = self.query(x).view(batchsize, -1, N)      # Queries               [B, C_bar, N]
        h_x = self.value(x).view(batchsize, -1, N)      # Values                [B, C_bar, N]

        s =  torch.bmm(f_x.permute(0,2,1), g_x)         # Scores                [B, N, N]
        beta = self.softmax(s)                          # Attention Map         [B, N, N]

        v = torch.bmm(h_x, beta)                        # Value x Softmax       [B, C_bar, N]
        v = v.view(batchsize, -1, H, W)                 # Recover input shape   [B, C_bar, H, W]
        o = self.self_att(v)                            # Self-Attention output [B, C, H, W]
        
        y = self.gamma * o + x                               # Learnable gamma + residual
        return y, o, self.gamma