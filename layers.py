import torch
import torch.nn as nn

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

"""
---------- Implementation of PixelAttention2D Block -----------

Based in PAN repository: https://github.com/zhaohengyuan1/PAN/blob/master/codes/models/archs/PAN_arch.py

class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out

"""
class PixelAttention(nn.Module):
    
    """
    Pixel Attention layer
    """

    def __init__(self, n_filters):
        super().__init__()

        self.conv2D = nn.Conv2d(
                        in_channels     = n_filters,
                        out_channels    = n_filters,
                        kernel_size     = 1,
                        stride          = 1,
                    )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_features_maps ):
        
        y = self.conv2D( input_features_maps )
        attn_maps = self.sigmoid( y )
        out = torch.mul( input_features_maps, attn_maps)
        return out, attn_maps

"""
---------- Implementation of Residual Block Option 1-----------
"""
class Residual_PA_block_1(nn.Module):

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
            
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels     = out_channels,
                out_channels    = out_channels,
                kernel_size     = 3,
                stride          = 1,
                padding         = 'same'
            )
        )
        
        self.pixelAttention = PixelAttention(out_channels)
    
    def forward(self, input_layer ):

        out = self.R_block_layer( input_layer )
        out, attn = self.pixelAttention(out)
        out = torch.add(out, input_layer)
        return out, attn

"""
---------- Implementation of Residual Block Option 2-----------
"""
class Residual_PA_block_2(nn.Module):

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
            
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels     = out_channels,
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

        self.pixelAttention = PixelAttention(out_channels)
    
    def forward(self, input_layer ):

        out             = self.R_block_layer(input_layer)
        out, attn       = self.pixelAttention( out )
        out             = torch.add(out, input_layer)
        return out, attn

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
        
        self.relu = nn.ReLU()

        self.convOut = nn.Conv2d(
            in_channels         = in_channels,
            out_channels        = out_channels,
            kernel_size         = 3,
            stride              = 1,
            padding             = 'same'
        )
    
    def forward(self, input_layer, skip_input ):
        
        output_size = [input_layer.size()[2]*2, input_layer.size()[2]*2]
        out = self.convT1( input_layer, output_size = output_size)
        out = self.bn1  ( out )
        out = self.relu( out )
        out = torch.cat( (out, skip_input), dim = 1)
        out = self.convOut( out )
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

            nn.ReLU()
        )
    
    def forward(self, input_layer ):

        return self.DS_block_layer(input_layer)


"""
---------- Implementation of Self-Attention Layer -----------
Taken from: https://github.com/heykeetae/Self-Attention-GAN/issues/54
"""

class Self_Attention(nn.Module):
    def __init__(self, inChannels, k=8):
        super(Self_Attention, self).__init__()

        embedding_channels  = inChannels // k  # C_bar
        self.key            = nn.Conv2d(inChannels, embedding_channels, 1)
        self.query          = nn.Conv2d(inChannels, embedding_channels, 1)
        self.value          = nn.Conv2d(inChannels, embedding_channels, 1)
        self.self_att       = nn.Conv2d(embedding_channels, inChannels, 1)
        self.gamma          = nn.Parameter(torch.tensor(0.0))
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
        
        y = self.gamma * o + x                          # Learnable gamma + residual
        return y, o