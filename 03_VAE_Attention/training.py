 
from math import gamma
import os
import sys
import time
import argparse
import datetime
import itertools
from matplotlib.pyplot import step
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from torchinfo import summary
import torch
import torch.nn.functional as F
import shutil
from torch.autograd import Variable
import torchvision.transforms as transforms
from utils import *
from dataloader import *


def setup_configs(args):
    
    """ Create folder of experiment and code """
    path = os.path.join(args.result_dir, args.exp_name, "code")
    os.makedirs( path, exist_ok=True)

    args.result_dir = os.path.join(args.result_dir, args.exp_name)
    
    for name in glob.glob('*.py'):
        shutil.copyfile( os.path.join(".", name), os.path.join(args.result_dir, "code", name))
    for name in glob.glob('*.sh'):
        shutil.copyfile( os.path.join(".", name), os.path.join(args.result_dir, "code", name))


    save_configs(args)

    args.cuda = True if torch.cuda.is_available() else False

def run_model(args):
    
    # Loads Models Encoders and Decoder
    encoder_LE, encoder_RC, decoder = load_model(args)
    
    """ Print Summary Models """
    summary(encoder_LE, input_size=(5, 1, 256,256))
    summary(encoder_RC, input_size=(5, 1, 256,256))
    
    to_cuda = [encoder_LE, encoder_RC, decoder]
    
    """ Define the trainning loss """
    pixelwise_loss, lambda_pixel = torch.nn.L1Loss(), args.lambda_pixel
    jensen_distance = jensen_shannon_distance()
    to_cuda.append( pixelwise_loss )
    to_cuda.append( jensen_distance )

    # Move everything to gpu
    if args.cuda:
        for model in to_cuda:
            model.cuda()

    # Load model in case of training. Otherwise, random init
    if args.epoch != 0:

        encoder_LE.load_state_dict(torch.load("{0}/saved_models/Encoder_LC_chkp_{1}.pth".format(args.result_dir, args.epoch)))
        encoder_RC.load_state_dict(torch.load("{0}/saved_models/Encoder_RC_chkp_{1}.pth".format(args.result_dir, args.epoch)))
        decoder.load_state_dict(torch.load("{0}/saved_models/Decoder_chkp_{1}.pth".format(args.result_dir, args.epoch)))
        print ("Weights from checkpoint: {0}/saved_models/G_chkp_{1}.pth".format(args.result_dir, args.epoch))

    else:
               
        if   args.weigth_init == "normal":
            encoder_LE.apply(weights_init_normal)
            encoder_RC.apply(weights_init_normal)
            decoder.apply(weights_init_normal)

        elif args.weigth_init == "glorot":
            encoder_LE.apply(weights_init_glorot)
            encoder_RC.apply(weights_init_glorot)
            decoder.apply(weights_init_glorot)
    

    # Configure dataloaders
    transforms_ = [
        transforms.ToTensor(),
    ]
    
    # Initialize data loader
    data_loader = Loader ( data_path = args.data_dir, proj = args.projection, format = args.format, num_workers = args.workers,
                           batch_size = args.batch_size, img_res=(args.image_size, args.image_size), n_channels = args.channels,
                           transforms = transforms_, dataset_name = args.dataset_name, img_complete = args.img_complete)
    
    # Configuration Optimizers
    optimizer_Encoder_LE    = torch.optim.Adam(encoder_LE.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_Encoder_RC    = torch.optim.Adam(encoder_RC.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_Decoder       = torch.optim.Adam(decoder.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Initialize logger 
    monitor = Monitor(logs_path = args.result_dir + "/Logs")

    # Tensor type
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    # Setting up name of logs
    tb_names = [    "Batch/Total_Loss", "Bach/JSD_loss",
                    "Batch/Pixel_Loss", "Batch/Encoder_Loss",
                    "Decoder_Loss" ] 

    avg_names = [   "Avg_Ep/Total_Loss", "Avg_Ep/JSD_loss",
                    "Avg_Ep/Pixel_Loss", "Avg_Ep/Encoder_loss", 
                    "Avg_Ep/Decoder_loss", "Avg_Ep/Time/" ]

    # ------------------------------------
    #               Training
    # ------------------------------------

    print ("\n [*] -> Starting training....\n\n")
    prev_time = time.time()

    for epoch in range(args.epoch, args.n_epochs):
        
        epoch_stats, it = {}, 0
        for name in tb_names: epoch_stats[name] = []

        for i in range(0, len(data_loader), args.batch_size):

            it += 1
            batch = data_loader.train_generator[i: i + args.batch_size]

            # Model inputs
            real_in  = Variable(batch["in"].type(Tensor))
            real_out = Variable(batch["out"].type(Tensor))

            # ------------------------------------
            #       Extract Features Encoders
            # ------------------------------------

            optimizer_Encoder_LE.zero_grad()
            optimizer_Encoder_RC.zero_grad()
            optimizer_Decoder.zero_grad()

            out_LE, skips_LE    = encoder_LE(real_in)
            out_RC, _           = encoder_RC(real_in)
            fake_img            = decoder(out_LE.detach(), skips_LE)

            # Pixel-wise loss
            loss_pixel  = pixelwise_loss(fake_img, real_out)
            loss_D      = loss_pixel * lambda_pixel
            loss_D.backward()
            optimizer_Decoder.step()

            loss_jsd    = jensen_distance(out_LE, out_RC)
            
            loss_E      = loss_jsd + (loss_pixel.item() * lambda_pixel)
            loss_total  = loss_jsd + (loss_pixel * lambda_pixel)
            
            

            loss_E.backward()
            optimizer_Encoder_LE.step()
            optimizer_Encoder_RC.step()


            
            # ------------------------------------
            #             Log Progress
            # ------------------------------------

            if it % 1 == 0:
                # Determine approximate time left
                elapsed_time = time.time() - prev_time
                hours = elapsed_time // 3600; elapsed_time = elapsed_time - 3600 * hours
                minutes = elapsed_time // 60; seconds = elapsed_time - 60 * minutes

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d][Batch %d/%d][total_loss: %f][JSD: %f][pixel: %f][decoder_loss: %f][encoder_loss: %f][ETA: %s]" 
                    % (
                        epoch,
                        args.n_epochs,
                        i,
                        len(data_loader),
                        loss_total.item(),
                        loss_jsd.item(),
                        loss_pixel.item(),
                        loss_D.item(),
                        loss_E.item(),
                        'ETA: %d:%d:%d' %(hours,minutes,seconds),
                    )
                )

                # save batch stats to logger
                tb_logs = [
                            loss_total.item(),
                            loss_jsd.item(),
                            loss_pixel.item(),
                            loss_E.item(),
                            loss_D.item(),
                          ]

                for name, value in zip(tb_names, tb_logs): 
                    epoch_stats[name].append(value)
                
                monitor.write_logs(tb_names, tb_logs, (epoch*args.batch_size)+i)
                
                if args.use_wandb:
                    
                    wandb.log(
                        {
                        "Batch/Total_Loss":     loss_total.item(),
                        "Bach/JSD_loss":        loss_jsd.item(),
                        "Batch/Pixel_Loss":     loss_pixel.item(),
                        "Batch/Encoder_Loss":   loss_E.item(),
                        "Batch/Decoder_Loss":   loss_D.item(),
                        },
                        step=(epoch*args.batch_size)+i
                    )
        
        # save avg stats to logger
        avg_logs = []
        for name in tb_names: avg_logs.append(np.mean(epoch_stats[name]))
        avg_logs.append((time.time() - prev_time) // 3600)
        
        monitor.write_logs(avg_names, avg_logs, epoch)
        
        if args.use_wandb:

            wandb.log(
                        {
                        "Avg_Ep/Total_Loss":    avg_logs[0],
                        "Avg_Ep/JSD_loss":      avg_logs[1],
                        "Avg_Ep/Pixel_Loss":    avg_logs[2],
                        "Avg_Ep/Encoder_loss":  avg_logs[3],
                        "Avg_Ep/Decoder_loss":  avg_logs[4],
                        "Avg_Ep/Time":          avg_logs[5],
                        },
                    )

        # Shuffle train data everything
        data_loader.on_epoch_end(shuffle = "train")
        
        if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
            # Save model checkpoints
            os.makedirs("%s/saved_models/" % (args.result_dir), exist_ok = True)
            torch.save(encoder_LE.state_dict(), "{0}/saved_models/Encoder_LE_chkp_{1:03d}.pth".format(args.result_dir, epoch))
            torch.save(encoder_RC.state_dict(), "{0}/saved_models/Encoder_RC_chkp_{1:03d}.pth".format(args.result_dir, epoch))
            torch.save(decoder.state_dict(), "{0}/saved_models/Decoder_chkp_{1:03d}.pth".format(args.result_dir, epoch))

        #If at sample interval save image
        if epoch % args.sample_interval == 0:
            
            if(not(args.img_complete)):
                sample_images(args, data_loader, [encoder_LE, decoder], epoch, img_complete= False)
            else:
                sample_images(args, data_loader, [encoder_LE, decoder], epoch, img_complete = True)
                
                
        
    
    print ("\n [✓] -> Done! \n\n")

