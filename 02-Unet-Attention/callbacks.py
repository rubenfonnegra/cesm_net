from asyncio.log import logger
from distutils.archive_util import make_archive
from email import utils
from pyexpat import model
from subprocess import call
from lightning.pytorch.callbacks import Callback
import os, torch
from lightning.pytorch.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import plot_images_with_metrics, fig2img, plot_test_images
from pytorch_lightning.loggers import WandbLogger
import pandas as pd

class Custom_Callbacks():
    def __init__(self, config):
        super().__init__()
        self.dir_checkpoint = os.path.join( config.result_dir, config.exp_name, "saved_models")
        os.makedirs(self.dir_checkpoint)
        self.config = config
        self.checkpoint     = ModelCheckpoint(dirpath= self.dir_checkpoint, filename = "G_chkp_{epoch:03d}", every_n_epochs = config.checkpoint_interval) 
        
    
    def get_callbacks(self):

        callbacks = [self.checkpoint, SaveValImages(), SaveTestImages(), SaveTestMetrics()]

        if(self.config.model == "SA-Unet-v1") or (self.config.model == "SA-Unet-v2"):
            callbacks.append(SaveParameters())

        return callbacks 


class IncreasingLambda(Callback):

    def on_train_epoch_end(self, trainer, pl_module):

        epoch = pl_module.current_epoch
        increasing = pl_module.lambda_increasing
        interval = pl_module.lambda_interval

        if (epoch % interval) == 0:
            pl_module.lambda_edge = pl_module.lambda_edge + increasing

class SaveParameters(Callback):

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        
        gamma = trainer.model.model.gamma
        epoch = trainer.current_epoch 
        torch.save(gamma, os.path.join(trainer.default_root_dir, "saved_models", f"gamma_{epoch:03d}.pth"))

class SaveValImages(Callback):
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the validation batch ends."""

        epoch = trainer.current_epoch
        sample_interval = trainer.model.config.checkpoint_interval

        if(epoch % sample_interval) == 0:

            dir_out = os.path.join( trainer.default_root_dir, "Val_Images", f"epoch_{epoch:03d}" )
            os.makedirs(dir_out, exist_ok=True)

            """ Extract Validation Step"""
            real_in, real_out   = batch
            real_in             = real_in.cpu().numpy()
            real_out            = real_out.cpu().numpy()
            fake_out            = outputs[0].cpu().numpy()

            """ Extract metrics of Test Step"""
            metrics                 = outputs[1]

            """ Init Logger """    
            wandb_logger    = trainer.logger
            diff_map        = abs(real_out - fake_out)

            captions = ["Low Energy", "Recombined", "Generated"]
            images      = [real_in, real_out, fake_out]

            plot_images_with_metrics(
                images      = images,
                captions    = captions,
                metrics     = metrics,
                diff_map    = diff_map,
                dir_out     = dir_out,
                id          = batch_idx,
                logger      = wandb_logger,
                setup       = "Val",
                epoch       = epoch
            )

            
class SaveTestImages(Callback):
    
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx):

            dir_out = os.path.join( trainer.default_root_dir, "Test_Images", "Image_Complete" )
            os.makedirs(dir_out, exist_ok=True)

            """ Extract Validation Step"""
            real_in, real_out   = batch
            real_in             = real_in.cpu().numpy()
            real_out            = real_out.cpu().numpy()
            fake_out            = outputs[0].cpu().numpy()

            """ Extract metrics of Validation Step"""
            metrics                 = outputs[1]

            """ Init Logger """    
            wandb_logger    = trainer.logger
            diff_map        = abs(real_out - fake_out)

            captions = ["Low Energy", "Recombined", "Generated"]
            images      = [real_in, real_out, fake_out]

            plot_images_with_metrics(
                images      = images,
                captions    = captions,
                metrics     = metrics,
                diff_map    = diff_map,
                dir_out     = dir_out,
                id          = batch_idx,
                logger      = wandb_logger,
                setup       = "Test",
            )

            dir_out = os.path.join( trainer.default_root_dir, "Test_Images", "Images" )
            os.makedirs(dir_out, exist_ok=True)

            plot_test_images(
                dir_out     = dir_out,
                batch_idx   = batch_idx,
                images      = images,
                captions    = captions,
                diff_map    = diff_map,
                logger      = wandb_logger 
             )

class SaveTestMetrics(Callback):
    
    def on_test_end(self, trainer, pl_module):

        dir_out = os.path.join( trainer.default_root_dir, "Test_Images")
        os.makedirs(dir_out, exist_ok=True)

        mae_mean    = torch.stack(pl_module.testing_step_mae).mean()
        ssim_mean   = torch.stack(pl_module.testing_step_ssim).mean()
        psnr_mean   = torch.stack(pl_module.testing_step_psnr).mean()
        mssim_mean  = torch.stack(pl_module.testing_step_mssim).mean()

        mae_std     = torch.stack(pl_module.testing_step_mae).std()
        ssim_std    = torch.stack(pl_module.testing_step_ssim).std()
        psnr_std    = torch.stack(pl_module.testing_step_psnr).std()
        mssim_std   = torch.stack(pl_module.testing_step_mssim).std()

        headers = ["Avg_MAE", "Avg_PSNR", "Avg_SSIM", "Avg_MSSIM"]
        content = [ 
            f"{mae_mean:03f} - {mae_std:03f}", f"{psnr_mean:03f} - {psnr_std:03f}",
            f"{ssim_mean:03f} - {ssim_std:03f}", f"{mssim_mean:03f} - {mssim_std:03f}",
        ]

        df = pd.DataFrame(content)
        df = df.transpose()
        df.to_csv(os.path.join(dir_out, "Test_Metrics.csv"), sep = ",", index=False, header = headers)



            

                
    
            
 
