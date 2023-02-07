import torch
import torch.nn as nn


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
---------- Implementation of Residual Block -----------

def R_block(layer_input, filters):

    d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
    d = BatchNormalization(momentum=0.8)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
    d = BatchNormalization(momentum=0.8)(d)
    d = tf.add(d, layer_input)

    return d
"""
class R_block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.R_block_layer = nn.Sequential(
            nn.Conv2d(
                in_channels     = in_channels,
                out_channels    = out_channels,
                kernel_size     = 3,
                stride          = 1,
                padding         = 'same'
            ),
            nn.BatchNorm2d(
                out_channels,
                momentum        =  0.8 
            ),
            
            # nn.LeakyReLU(
            #     negative_slope  = 0.2
            # ),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels     = in_channels,
                out_channels    = out_channels,
                kernel_size     = 3,
                stride          = 1,
                padding         = 'same'
            ),
            nn.BatchNorm2d(
                out_channels,
                momentum        =  0.8 
            )
        )
    
    def forward(self, input_layer ):

        out = self.R_block_layer(input_layer)
        out = torch.add(out, input_layer)
        return out

"""
---------- Implementation of DownSampling Block -----------

def DS_block(layer_input, filters):

    ds = Conv2D(filters, kernel_size=3, strides=2, padding='same')(layer_input)
    ds = BatchNormalization(momentum=0.8)(ds)
    ds = LeakyReLU(alpha=0.2)(ds)

    return ds
"""
class DS_block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.DS_block_layer = nn.Sequential(
            nn.Conv2d(
                in_channels     = in_channels,
                out_channels    = out_channels,
                kernel_size     = 3,
                stride          = (2, 2), #2
                padding         = 1, #'same'
            ),
            nn.BatchNorm2d(
                out_channels,
                momentum        =  0.8 
            ),
            # nn.LeakyReLU(
            #     negative_slope  = 0.2
            # )
            nn.ReLU()
        )
    
    def forward(self, input_layer ):

        return self.DS_block_layer(input_layer)

"""
---------- Implementation of UpSampling Block -----------

def US_block(layer_input, skip_input, filters):

    us = Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same')(layer_input)
    us = BatchNormalization(momentum=0.8)(us)
    us = LeakyReLU(alpha=0.2)(us)
    us = Concatenate()([us, skip_input])
    us = Conv2D(filters, kernel_size=3, strides=1, padding='same')(us)

    return us
"""
class US_block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # self.US_block_layer = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         in_channels     = in_channels,
        #         out_channels    = out_channels,
        #         kernel_size     = 3,
        #         stride          = (2, 2), #2
        #         padding         = 1 #'valid'
        #     ),
        #     nn.BatchNorm2d(
        #         out_channels,
        #         momentum        =  0.8 
        #     ),
        #     nn.LeakyReLU(
        #         negative_slope  = 0.2
        #     )
        # )

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convT1 = nn.ConvTranspose2d(
                in_channels     = in_channels,
                out_channels    = out_channels,
                kernel_size     = 3,
                stride          = (2, 2), #2
                padding         = 1, 
                #bias = True
            )
        
        self.bn1 = nn.BatchNorm2d(
                out_channels,
                momentum        =  0.8 
            )
        
        # self.lrelu = nn.LeakyReLU(
        #         negative_slope  = 0.2
        #     )
        self.relu = nn.ReLU()

        self.convOut = nn.Conv2d(
            in_channels         = out_channels * 2,
            out_channels        = out_channels,
            kernel_size         = 3,
            stride              = 1,
            padding             = 'same'
        )
    
    def forward(self, input_layer, skip_input ):
        
        # out = self.US_block_layer( input_layer )
        output_size = [input_layer.size()[2]*2, input_layer.size()[2]*2]
        out = self.convT1( input_layer, output_size = output_size)
        out = self.bn1  ( out )
        #out = self.lrelu( out )
        out = self.relu( out )
        out = torch.cat( (out, skip_input), dim = 1)
        out = self.convOut( out )
        return out

"""
---------- Implementation of PixelAttention2D Block -----------

Based in visual-attention-tf: https://github.com/vinayak19th/Visual_attention_tf/blob/main/build/lib/visual_attention/pixel_attention.py

class PixelAttention2D(tf.keras.layers.Layer):

    def __init__(self, nf, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf
        self.conv1 = Conv2D(filters=nf, kernel_size=1)

    @tf.function
    def call(self, x):
        y = self.conv1(x)
        self.sig = tf.keras.activations.sigmoid(y)
        out = tf.math.multiply(x, y)
        out = self.conv1(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"Att_filters": self.nf})
        return config

"""
class PixelAttention2D(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv2D = nn.Conv2d(
                        in_channels     = in_channels,
                        out_channels    = out_channels,
                        kernel_size     = 1,
                        stride          = 1,
                        padding         = 'valid'
                    )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_layer ):
        
        y = self.conv2D( input_layer )
        y = self.sigmoid( y )
        out = torch.mul( input_layer, y)
        out = self.conv2D( out )
        return out


# from turtle import forward
# from sklearn.multiclass import OutputCodeClassifier
# import torch.nn as nn
# from layers import *

class SA_UNet_Generator(nn.Module):
    #
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
        #self.leakyReluFusion = nn.LeakyReLU(negative_slope=0.2)
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
