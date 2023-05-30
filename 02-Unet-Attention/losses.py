import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.filters import sobel



class SobelLoss(nn.Module):

    def __init__(self, mask=False, valueMask = 0.0, versionMask = "1"):
        super(SobelLoss, self).__init__()

        self.mask           = mask
        self.valueMask      = valueMask
        self.versionMask    = versionMask

        if(self.mask == True):
            self.loss       = nn.L1Loss(reduction="sum")
        else:
            self.loss       = nn.L1Loss()

    def forward(self, fake, real):

        fake_sobel = self.sobel_filter(fake)
        real_sobel = self.sobel_filter(real)

        if(self.mask == True):
            return self.calcularLossTotal(fake_sobel, real_sobel)
        else:
            return self.loss(fake_sobel, real_sobel)
        

    def calcularLossTotal(self, im_fake, im_real):

        if( self.versionMask == "1"):
            mask = (im_real > self.valueMask) * 1
        elif (self.versionMask == "2"):
            mask = ((im_real > self.valueMask)  or (im_fake > self.valueMask)) * 1. 

        im_real = im_real * mask
        im_fake = im_fake * mask

        loss = self.loss(im_fake, im_real)
        loss = loss / torch.count_nonzero(mask == 1.)
        loss = loss / im_real.shape[0]

        return loss
    
    def sobel_filter(self, imgs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        imgs = imgs.cpu().detach().numpy()
        
        if (len(imgs.shape) == 3):
            edges = sobel(imgs[0,...])
            edges = edges[np.newaxis, ...]
            
        elif(len(imgs.shape) == 4):
            
            edges = []
            for batch in range(imgs.shape[0]):
                sobel_ = sobel(imgs[batch, 0, ...])
                edges.append( sobel_[np.newaxis, ...] )
            
            edges = np.array(edges)
    
        edges = torch.from_numpy(edges).to(device)
        return edges



class WeightedSumLoss(nn.Module):

    def __init__(self, alpha_breast = 0.8, alpha_background = 0.2, gamma_loss = 1. ):
        super(WeightedSumLoss, self).__init__()

        self.loss               = nn.L1Loss(reduction="sum")
        self.alpha_breast       = alpha_breast
        self.alpha_background   = alpha_background
        self.gamma_loss         = gamma_loss

    def forward(self, fake_out, real_out):

        """ Create Mask of breast and background """
        mask_bg     = (real_out <=  -0.9) * 1.
        mask_breast = (real_out >   -0.9) * 1.

        """ Apply mask to real_out and fake_out images """
        real_out_bg         = real_out * mask_bg
        fake_out_bg         = fake_out * mask_bg
        real_out_breast     = real_out * mask_breast
        fake_out_breast     = fake_out * mask_breast

        """ Calculate losses """
        self.loss_pixel_breast      = self.loss(fake_out_breast, real_out_breast)
        self.loss_pixel_breast      = self.calcularLossTotal(self.loss_pixel_breast, mask_breast)

        self.loss_pixel_bg          = self.loss(fake_out_bg, real_out_bg)
        self.loss_pixel_bg          = self.calcularLossTotal(self.loss_pixel_bg, mask_bg)
        
        self.weightedSum            = (self.alpha_breast * self.loss_pixel_breast) + (self.alpha_background * self.loss_pixel_bg)
        
        self.pixel_loss             = self.loss_pixel_breast + self.loss_pixel_bg
        self.breast_weighted        = (self.alpha_breast * self.loss_pixel_breast)
        self.bg_weighted            = (self.alpha_background * self.loss_pixel_bg)

        self.weightedSumLossTotal   = self.gamma_loss * self.weightedSum
        
        return self.weightedSumLossTotal

    def calcularLossTotal(self, loss, mask):

        loss = loss / torch.count_nonzero(mask == 1.)
        loss = loss / mask.shape[0]
        return loss


class WeightedSumEdgeSobelLoss(nn.Module):

    def __init__(
        self, 
        alpha_breast = 0.8, alpha_background = 0.2, 
        lambda_pixel = 100, lambda_edge = 100, 
        mask = False, valueMask = 0.05, versionMask = "0"
        ):
        super(WeightedSumEdgeSobelLoss, self).__init__()

        self.loss_weighted_sum      = WeightedSumLoss( alpha_breast = alpha_breast, alpha_background = alpha_background)
        self.loss_edge              = SobelLoss( mask = mask, valueMask = valueMask, versionMask = versionMask)
        self.lambda_pixel           = lambda_pixel
        self.lambda_edge            = lambda_edge

    def forward(self, fake_out, real_out):

        self.weightedSum                    = self.loss_weighted_sum( fake_out, real_out)
        self.edgeSobel                      = self.loss_edge( fake_out, real_out)
        self.weightedSumEdgeSobelLossTotal  = (self.lambda_pixel * self.weightedSum) + (self.lambda_edge * self.edgeSobel)

        self.weightedSummLossTotal          = self.lambda_pixel * self.weightedSum
        self.edgeSobelLossTotal             = self.lambda_edge * self.edgeSobel
        
        return self.weightedSumEdgeSobelLossTotal



class MAEEdgeSobelLoss(nn.Module):

    def __init__(self, lambda_pixel = 100, lambda_edge = 100):
        super(MAEEdgeSobelLoss, self).__init__()

        self.loss_pixel             = nn.L1Loss()
        self.loss_edge              = SobelLoss()
        self.lambda_pixel           = lambda_pixel
        self.lambda_edge            = lambda_edge

    def forward(self, fake_out, real_out):

        self.maeLoss                     = self.loss_pixel( fake_out, real_out)
        self.edgeSobelLoss               = self.loss_edge( fake_out, real_out)
        self.maeEdgeSobelLoss            = (self.lambda_pixel * self.maeLoss) + (self.lambda_edge * self.edgeSobelLoss)
        
        return self.maeEdgeSobelLoss




class MAELoss(nn.Module):

    def __init__(self, gamma_loss = 100):
        super(MAELoss, self).__init__()

        self.loss_pixel     = nn.L1Loss()
        self.gamma_loss     = gamma_loss

    def forward(self, fake_out, real_out):

        self.pixel_loss = self.loss_pixel( fake_out, real_out)
        self.total_loss = self.gamma_loss * self.pixel_loss
        
        return self.total_loss