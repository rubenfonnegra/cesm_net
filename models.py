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
***** Implementation Unet Generator Not Deep ************
*********************************************************
"""
class UNet_Generator_Not_Deep(nn.Module):
    #
    def __init__(self, in_channels ):
        super(UNet_Generator_Not_Deep, self).__init__()

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
        
        """ Fusion Block """
        self.convFusion = nn.Conv2d(
            in_channels     = 256,
            out_channels    = 256,
            kernel_size     = 3,
            stride          = 1,
            padding         = 'same'
        )
        self.batchnormFusion = nn.BatchNorm2d(256, momentum=0.8)
        #self.leakyReluFusion = nn.LeakyReLU(negative_slope=0.2)
        self.reluFusion = nn.ReLU()

        """ Upsampling Block"""
        self.US1 = US_block( 256, 128 )       # (B, 128, 64, 64)
        self.RB5 = R_block( 128, 128 )          # (B, 128, 64, 64)
        self.US2 = US_block( 128, 64 )        # (B, 64, 128, 128)
        self.RB6 = R_block( 64, 64 )            # (B, 64, 128, 128)
        self.US3 = US_block( 64, 32 )         # (B, 32, 256, 256)
        self.RB7 = R_block( 32, 32 )            # (B, 32, 256, 256)
        
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
        outConvInit = self.convInput(img_input) # (B, 32, 256,256)
        outRB1  = self.RB1(outConvInit)         # (B, 32, 256,256)
        outDS   = self.DS1(outRB1)              # (B, 64, 128,128)
        outRB2  = self.RB2(outDS)               # (B, 64, 128,128)
        outDS   = self.DS2(outRB2)              # (B, 128, 64, 64)
        outRB3  = self.RB3(outDS)               # (B, 128, 64, 64)
        outDS   = self.DS3(outRB3)              # (B, 256, 32, 32)
        outRB4  = self.RB4(outDS)               # (B, 256, 32, 32)

        """ Fusion Block Forward """
        out = self.convFusion(outRB4)            # (B, 256, 32, 32)
        out = self.batchnormFusion(out)         # (B, 256, 32, 32)
        out = self.reluFusion(out)              # (B, 256, 32, 32)

        """ Upsampling Block Forward """
        out = self.US1( out, outRB3 )           # (B, 128, 64, 64)
        out = self.RB5( out )                   # (B, 128, 64, 64)
        out = self.US2( out, outRB2 )           # (B, 64, 128, 128)
        out = self.RB6( out )                   # (B, 64, 128, 128)
        out = self.US3(out, outRB1 )            # (B, 32, 256, 256)
        out = self.RB7(out)                     # (B, 32, 256, 256)

        """ Output Convolution """
        out = self.convOut(out)                 # (B, 1, 256, 256)
        out = self.actOut(out)                  # (B, 1, 256, 256)
        
        return out

""" 
*********************************************************
***** Implementation Unet Generator Not Deep ************
*********************************************************
"""
class UNet_Generator_Deep(nn.Module):
    #
    def __init__(self, in_channels ):
        super(UNet_Generator_Deep, self).__init__()

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
        self.RB6 = R_block(1024,1024)       
        
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
        self.RB7    = R_block( 512, 512 )
        self.US2    = US_block( 512, 256 )
        self.RB8    = R_block( 256, 256 )          
        self.US3    = US_block( 256, 128 )        
        self.RB9    = R_block( 128, 128 )
        self.US4    = US_block( 128, 64 )        
        self.RB10   = R_block( 64, 64 )
        self.US5    = US_block( 64, 32 )        
        self.RB11   = R_block( 32, 32 )             
        
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
        outRB6  = self.RB6(outDS)               # (B, 1024, 8, 8)

        """ Fusion Block Forward """
        out = self.convFusion(outRB6)           # (B, 1024, 8, 8)
        out = self.batchnormFusion(out)         # (B, 1024, 8, 8)
        out = self.reluFusion(out)              # (B, 1024, 8, 8)

        """ Upsampling Block Forward """
        out = self.US1( out, outRB5 )           # (B, 512, 16, 16)
        out = self.RB7( out )                   # (B, 512, 16, 16)
        out = self.US2( out, outRB4 )           # (B, 256, 32, 32)
        out = self.RB8( out )                   # (B, 256, 32, 32)
        out = self.US3(out, outRB3 )            # (B, 128, 64, 64)
        out = self.RB9(out)                     # (B, 128, 64, 64)
        out = self.US4(out, outRB2 )            # (B, 64, 128, 128)
        out = self.RB10(out)                    # (B, 64, 128, 128)
        out = self.US5(out, outRB1 )            # (B, 32, 256, 256)
        out = self.RB11(out)                    # (B, 32, 256, 256)

        """ Output Convolution """
        out = self.convOut(out)                 # (B, 1, 256, 256)
        out = self.actOut(out)                  # (B, 1, 256, 256)
        
        return out


""" 
*********************************************************
***** Implementation Pixel-Attention Unet Generator *****
*********************************************************
"""
class PA_UNet_Generator(nn.Module):
    #
    def __init__(self, in_channels ):
        super(PA_UNet_Generator, self).__init__()

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
        self.PA1 = PixelAttention2D(256, 256)
        self.RB4 = R_block(256,256)
        self.DS4 = DS_block(256,256)

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
        self.RB5 = R_block( 256, 256 )
        self.US2 = US_block( 256, 128 )
        self.RB6 = R_block( 128, 128 )
        self.US3 = US_block( 128, 64 )
        self.RB7 = R_block( 64, 64 )
        self.US4 = US_block( 64, 32 )
        self.RB8 = R_block( 32, 32 )
        
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
        outConvInit = self.convInput(img_input)
        outRB1  = self.RB1(outConvInit)
        outDS   = self.DS1(outRB1)
        outRB2  = self.RB2(outDS)
        outDS   = self.DS2(outRB2)
        outRB3  = self.RB3(outDS)
        outDS   = self.DS3(outRB3)
        outPA   = self.PA1(outDS)
        outRB4  = self.RB4(outPA)
        outDS   = self.DS4(outRB4)

        """ Fusion Block Forward """
        out = self.convFusion(outDS)
        out = self.batchnormFusion(out)
        #out = self.leakyReluFusion(out)
        out = self.reluFusion(out)

        """ Upsampling Block Forward """
        out = self.US1( out, outRB4 )
        out = self.RB5( out )
        out = self.US2( out, outRB3 )
        out = self.RB6( out )
        out = self.US3(out, outRB2 )
        out = self.RB7(out)
        out = self.US4( out, outRB1 )
        out = self.RB8(out)

        """ Output Convolution """
        out = self.convOut(out)
        out = self.actOut(out)

        return out


""" 
******************************************************************
***** Implementation Residual Pixel Attention Unet Generator *****
******************************************************************
"""
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
        self.RPA1   = Residual_PA_block_2( 32, 32)
        self.DS1    = DS_block(32,64)
        self.RPA2   = Residual_PA_block_2(64,64)
        self.DS2    = DS_block(64,128)
        self.RPA3   = Residual_PA_block_2(128,128)
        self.DS3    = DS_block(128,256)
        self.RPA4   = Residual_PA_block_2(256, 256)
        self.DS4    = DS_block(256,512)

        """ Fusion Block """
        self.convFusion = nn.Conv2d(
            in_channels     = 512,
            out_channels    = 512,
            kernel_size     = 3,
            stride          = 1,
            padding         = 'same'
        )
        self.batchnormFusion = nn.BatchNorm2d(512, momentum=0.8)
        self.reluFusion = nn.ReLU()

        """ Upsampling Block"""
        self.US1 = US_block( 512, 256 )
        self.RPA5 = Residual_PA_block_2( 256, 256 )
        self.US2 = US_block( 256, 128 )
        self.RPA6 = Residual_PA_block_2( 128, 128 )
        self.US3 = US_block( 128, 64 )
        self.RPA7 = Residual_PA_block_2( 64, 64 )
        self.US4 = US_block( 64, 32 )
        self.RPA8 = Residual_PA_block_2( 32, 32 )
        
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
        outConvInit         = self.convInput(img_input)     # (B, 32, 256, 256)
        outRPA1, attn1      = self.RPA1(outConvInit)        # (B, 32, 256, 256)
        outDS               = self.DS1(outRPA1)             # (B, 64, 128, 128)
        outRPA2, attn2      = self.RPA2(outDS)              # (B, 64, 128, 128)
        outDS               = self.DS2(outRPA2)             # (B, 128, 64, 64)
        outRPA3, attn3      = self.RPA3(outDS)              # (B, 128, 64, 64)
        outDS               = self.DS3(outRPA3)             # (B, 256, 32, 32)
        outRPA4, attn4      = self.RPA4(outDS)              # (B, 256, 32, 32)
        outDS               = self.DS4(outRPA4)             # (B, 512, 16, 16)

        """ Fusion Block Forward """
        out = self.convFusion(outDS)                        # (B, 512, 16, 16)
        out = self.batchnormFusion(out)                     # (B, 512, 16, 16)
        out = self.reluFusion(out)                          # (B, 512, 16, 16)

        """ Upsampling Block Forward """
        out         = self.US1( out, outRPA4 )              # (B, 256, 32, 32)
        out, attn5  = self.RPA5( out )                      # (B, 256, 32, 32)
        out         = self.US2( out, outRPA3 )              # (B, 128, 64, 64)
        out, attn6  = self.RPA6( out )                      # (B, 128, 64, 64)
        out         = self.US3(out, outRPA2 )               # (B, 64, 128, 128)
        out, attn7  = self.RPA7(out)                        # (B, 64, 128, 128)
        out         = self.US4( out, outRPA1 )              # (B, 32, 256, 256)
        out, attn8  = self.RPA8(out)                        # (B, 32, 256, 256)

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
            "attn8": attn8,
            "output_image": out,
        }

        return out, outDictionary

""" 
*********************************************************
***** Implementation Self-Attention Unet Generator ******
*********************************************************
"""
class SA_UNet_Generator(nn.Module):

    def __init__(self, in_channels ):
        super(SA_UNet_Generator, self).__init__()

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
        self.attn1  = Self_Attention(128)
        self.RB3    = R_block(128,128)
        self.DS3    = DS_block(128,256)
        self.attn2  = Self_Attention(256)
        self.RB4    = R_block(256,256)
        self.DS4    =  DS_block(256,256)

        """ Fusion Block """
        self.convFusion = nn.Conv2d(
            in_channels     = 256,
            out_channels    = 256,
            kernel_size     = 3,
            stride          = 1,
            padding         = 'same'
        )
        self.batchnormFusion = nn.BatchNorm2d(256, momentum=0.8)
        #self.leakyReluFusion = nn.LeakyReLU(negative_slope=0.2)
        self.reluFusion = nn.ReLU()

        """ Upsampling Block"""
        self.US1    = US_block( 256, 256 )
        self.attn3  = Self_Attention(256)
        self.RB5    = R_block( 256, 256 )
        self.US2    = US_block( 256, 128 )
        self.attn4  = Self_Attention(128)
        self.RB6    = R_block( 128, 128 )
        self.US3    = US_block( 128, 64 )
        self.RB7    = R_block( 64, 64 )
        self.US4    = US_block( 64, 32 )
        self.RB8    = R_block( 32, 32 )
        
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
        outConvInit             = self.convInput(img_input)
        outRB1                  = self.RB1(outConvInit)
        outDS                   = self.DS1(outRB1)
        outRB2                  = self.RB2(outDS)
        outDS                   = self.DS2(outRB2)
        outAttn1, attnMaps1     = self.attn1(outDS)
        outRB3                  = self.RB3(outAttn1)
        outDS                   = self.DS3(outRB3)
        outAttn2, attnMaps2     = self.attn2(outDS)
        outRB4                  = self.RB4(outAttn2)
        outDS                   = self.DS4(outRB4)

        """ Fusion Block Forward """
        out = self.convFusion(outDS)
        out = self.batchnormFusion(out)
        #out = self.leakyReluFusion(out)
        out = self.reluFusion(out)

        """ Upsampling Block Forward """
        out             = self.US1( out, outRB4 )
        out, attnMaps3  = self.attn3( out )
        out             = self.RB5( out )
        out             = self.US2( out, outRB3 )
        out, attnMaps4  = self.attn4( out )
        out             = self.RB6( out )
        out             = self.US3(out, outRB2 )
        out             = self.RB7(out)
        out             = self.US4( out, outRB1 )
        out             = self.RB8(out)

        """ Output Convolution """
        out = self.convOut(out)
        out = self.actOut(out)

        outDictionary = {

            "image_input": img_input,
            "attn1": attnMaps1,
            "attn2": attnMaps2,
            "attn3": attnMaps3,
            "attn4": attnMaps4,
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
