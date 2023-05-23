import lightning.pytorch as pl
from torch import nn, optim
from dataloader import *
from models.SA_Unet import SA_Unet_v1
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import mean_absolute_error as mae
from torchmetrics.functional import multiscale_structural_similarity_index_measure as mssim
from losses import *



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

class SA_Unet_v1_Module(pl.LightningModule):
    def __init__(self, config, actOut):
        super().__init__()

        self.model  = SA_Unet_v1( in_channels = config.channels, gamma = config.gamma, actOut = actOut)
        self.config = config
        self.save_hyperparameters()

        """ Get Weights Losses """
        self.lambda_pixel       = config.lambda_pixel
        self.lambda_edge        = config.lambda_edge
        self.alpha_breast       = config.alpha_breast
        self.alpha_background   = config.alpha_background
        self.gamma_loss         = config.gamma_loss

        """ Get Loss """
        self.loss               = self.get_loss(config.loss)

        self.init_weigths()
        self.testing_step_mae       = []
        self.testing_step_ssim      = []
        self.testing_step_psnr      = []
        self.testing_step_mssim     = []

    def training_step(self, batch):

        """ Get images and calculate loss """
        real_input, real_out    = batch
        fake_out                = self.model(real_input)
        loss_G                  = self.loss(fake_out, real_out)

        """ Logg the losses """
        if(self.config.loss == "MAE"):
            self.log(f"Total_Loss", loss_G, on_epoch=True, prog_bar=True, logger=True)

        elif(self.config.loss == "WeightSum"):
            self.log(f"Total_Loss", loss_G, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"Breast_Loss", self.loss.loss_pixel_breast, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"Background_Loss", self.loss.loss_pixel_bg, on_epoch=True, prog_bar=True, logger=True)
            
        elif(self.config.loss == "WeightSumEdgeSobel"):
            self.log(f"Total_Loss", loss_G, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"Pixel_Loss", self.loss.weightedSummLoss, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"Edge_Loss", self.loss.edgeSobelLoss, on_epoch=True, prog_bar=True, logger=True)

        elif(self.config.loss == "MAEEdgeSobel"):
            self.log(f"Total_Loss", loss_G, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"MAE_Loss", self.loss.maeLoss, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"Edge_Loss", self.loss.edgeSobelLoss, on_epoch=True, prog_bar=True, logger=True)
        
        self.log(f"Gamma Attention", self.model.gamma, logger=True)
            
        return loss_G

    def validation_step(self, batch, batch_idx):

        real_in, real_out           = batch
        fake_out                    = self.model(real_in)
        mae, mssim, psnr, ssim      = self.get_metrics(prediction= fake_out, target = real_out)

        self.log("val_mae", mae, on_epoch=True )
        self.log("val_mssim", mssim, on_epoch=True )
        self.log("val_psnr", psnr, on_epoch=True )
        self.log("val_ssim", ssim, on_epoch=True )
        metrics = [mae, mssim, psnr, ssim]
        return fake_out, metrics

    def test_step(self, batch, batch_idx):
        
        real_in, real_out           = batch
        fake_out                    = self.model(real_in)
        mae, mssim, psnr, ssim      = self.get_metrics(prediction = fake_out, target = real_out)

        self.log("test_mae", mae, prog_bar = True, on_epoch=True)
        self.log("test_mssim", mssim, prog_bar = True, on_epoch=True)
        self.log("test_psnr", psnr, prog_bar = True, on_epoch=True)
        self.log("test_ssim", ssim, prog_bar = True, on_epoch=True)

        self.testing_step_mae.append(mae)
        self.testing_step_ssim.append(ssim)
        self.testing_step_psnr.append(psnr)
        self.testing_step_mssim.append(mssim)

        metrics = [mae, mssim, psnr, ssim]

        return fake_out, metrics

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.config.lr, betas= (self.config.b1, self.config.b2) )
        return optimizer
        
    def init_weigths(self):

        if (self.config.weigth_init == "normal"):
            self.model.apply(weights_init_normal)
        elif (self.config.weigth_init == "glorot"):
            self.model.apply(weights_init_glorot)
    
    def get_metrics(self, prediction, target):

        ssim_   = ssim(preds = prediction, target= target)
        mssim_  = mssim(preds = prediction, target= target)
        psnr_   = psnr(preds = prediction, target= target)
        mae_    = mae(preds = prediction, target= target)

        return mae_, mssim_, psnr_, ssim_
    
    def get_loss(self, loss):

        if(loss == "MAE"):
            return MAELoss(gamma_loss = self.gamma_loss )
        elif(loss == "WeightSum"):
            return WeightedSumLoss(alpha_breast = self.alpha_breast, alpha_background = self.alpha_background, gamma_loss = self.gamma_loss )
        elif(loss == "WeightSumEdgeSobel"):
            return WeightedSumEdgeSobelLoss(
                alpha_breast = self.alpha_breast,
                alpha_background = self.alpha_background,
                lambda_pixel= self.lambda_pixel,
                lambda_edge = self.lambda_edge
            )
        elif(loss == "MAEEdgeSobel"):
            return MAEEdgeSobelLoss(lambda_pixel = self.lambda_pixel, lambda_edge = self.lambda_edge)