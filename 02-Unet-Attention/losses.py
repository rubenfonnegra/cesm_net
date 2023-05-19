import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelLoss(nn.Module):

    def __init__(self):
        super(SobelLoss, self).__init__()

        self.loss       = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, fake, real):

        fake_sobel = self.grad_layer(fake)
        real_sobel = self.grad_layer(real)
        return self.loss(fake_sobel, real_sobel)

class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x




class WeightedSumLoss(nn.Module):

    def __init__(self, alpha_breast = 0.8, alpha_background = 0.2, gamma_loss = 1. ):
        super(WeightedSumLoss, self).__init__()

        self.loss               = nn.L1Loss()
        self.alpha_breast       = alpha_breast
        self.alpha_background   = alpha_background
        self.gamma_loss         = gamma_loss

    def forward(self, fake_out, real_out):

        """ Create Mask of breast and background """
        mask_bg     = (real_out <=  0.0) * 1.
        mask_breast = (real_out >   0.0) * 1.

        """ Apply mask to real_out and fake_out images """
        real_out_bg         = real_out * mask_bg
        fake_out_bg         = fake_out * mask_bg
        real_out_breast     = real_out * mask_breast
        fake_out_breast     = fake_out * mask_breast

        """ Calculate losses """
        self.loss_pixel_breast      = self.loss(fake_out_breast, real_out_breast)
        self.loss_pixel_bg          = self.loss(fake_out_bg, real_out_bg)
        self.weightedSumLoss        = (self.alpha_breast * self.loss_pixel_breast) + (self.alpha_background * self.loss_pixel_bg)
        self.weightedSumLoss        = self.gamma_loss * self.weightedSumLoss
        
        return self.weightedSumLoss 






class WeightedSumEdgeSobelLoss(nn.Module):

    def __init__(self, alpha_breast = 0.8, alpha_background = 0.2, lambda_pixel = 100, lambda_edge = 100):
        super(WeightedSumEdgeSobelLoss, self).__init__()

        self.loss_pixel             = WeightedSumLoss( alpha_breast = alpha_breast, alpha_background = alpha_background)
        self.loss_edge              = SobelLoss()
        self.lambda_pixel           = lambda_pixel
        self.lambda_edge            = lambda_edge

    def forward(self, fake_out, real_out):

        self.weightedSummLoss            = self.loss_pixel( fake_out, real_out)
        self.edgeSobelLoss               = self.loss_edge( fake_out, real_out)
        self.weightedSumEdgeSobelLoss    = (self.lambda_pixel * self.weightedSummLoss) + (self.lambda_edge * self.edgeSobelLoss)
        
        return self.weightedSumEdgeSobelLoss



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

        self.maeLoss    = self.loss_pixel( fake_out, real_out)
        mae_pixel_loss  = self.gamma_loss * self.maeLoss
        
        return mae_pixel_loss