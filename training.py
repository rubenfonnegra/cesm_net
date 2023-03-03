 
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
from torch.optim.lr_scheduler import StepLR

from utils import *
from models import *
from dataloader import *


def setup_configs(args):
    
    path = os.path.join(args.result_dir, args.exp_name, "code")
    os.makedirs( path, exist_ok=True)
    args.result_dir = args.result_dir + "/%s" % args.exp_name
    
    for name in glob.glob('*.py'):
        shutil.copyfile( os.path.join(".", name), os.path.join(args.result_dir, "code", name))

    if not args.model and not args.generate: 
        raise NotImplementedError ("Which model to use? Implemented: UNet, GAN ")

    if not args.generate:
        save_configs(args)
    
    args.cuda = True if torch.cuda.is_available() else False

def run_model(args):
    
    # Initialize generator and discriminator
    if(args.model == "UNet_Deep"):
        generator = UNet_Generator_Deep(in_channels = args.channels)
    elif(args.model == "Residual-PA-Unet"):
        generator = Residual_PA_UNet_Generator(in_channels= args.channels)
    elif(args.model == "PA-UNet"):
        generator = PA_UNet_Generator(in_channels= args.channels)
    elif(args.model == "SA-UNet"):
        generator = SA_UNet_Generator(in_channels= args.channels)
    
    summary(generator, input_size=(5, 1, 256,256))
    
    # Choose correct type of D
    if args.generate:
        discriminator = None
        to_cuda = [generator]
    
    elif args.type_model != "GAN":
        to_cuda = [generator]
        discriminator = None 
    
    elif args.type_model == "GAN":

        # Create D
        discriminator = PatchGAN_Discriminator(n_channels = args.channels)
        GAN_loss = nn.MSELoss()

        # Calculate output of image discriminator (PatchGAN)
        patch = (1, args.image_size // 2 ** 4, args.image_size // 2 ** 4)
        to_cuda = [generator, discriminator]
    
    else: 
        discriminator = None
        to_cuda = [generator]
    
    pixelwise_loss, lambda_pixel = torch.nn.L1Loss(), args.lambda_pixel
    to_cuda.append( pixelwise_loss )

    # Move everything to gpu
    if args.cuda:
        for model in to_cuda:
            model.cuda()

    # Load model in case of training. Otherwise, random init
    if args.epoch != 0:

        generator.load_state_dict(torch.load("{0}/saved_models/G_chkp_{1}.pth".format(args.result_dir, args.epoch)))
        print ("Weights from checkpoint: {0}/saved_models/G_chkp_{1}.pth".format(args.result_dir, args.epoch))

        if discriminator != None: 
            discriminator.load_state_dict(torch.load("{0}/saved_models/D_chkp_{1}.pth".format(args.result_dir, args.epoch)))
            print ("Weights from checkpoint: {0}/saved_models/D_chkp_{1}.pth".format(args.result_dir, args.epoch))

    else:
               
        if   args.weigth_init == "normal":
            generator.apply(weights_init_normal)
            if args.type_model == "GAN": 
                discriminator.apply(weights_init_normal)
        elif args.weigth_init == "glorot":
            generator.apply(weights_init_glorot)
            if args.type_model == "GAN": 
                discriminator.apply(weights_init_glorot)
    

    # Configure dataloaders
    transforms_ = [
        transforms.ToTensor(),
    ]
    
    # Initialize data loader
    data_loader = Loader ( data_path = args.data_dir, proj = args.projection, format = args.format, num_workers = args.workers,
                           batch_size = args.batch_size, img_res=(args.image_size, args.image_size), n_channels = args.channels,
                           transforms = transforms_, dataset_name = args.dataset_name, img_complete = args.img_complete)
    
    # In case of validation option. Otherwise, move to train
    if args.generate: 
        #
        """ Not used in image complete exp """
        if(not(args.img_complete)):
            print ("\n [*] -> Generating test patches.... \n")
            generate_images_with_stats(args, data_loader, generator, args.epoch, \
                                    shuffled = False, write_log = True, \
                                    output_dir = "{0}/generated_images/ep_{1}/".format(args.result_dir, args.epoch),
                                    img_complete=False)
            print ("\n [✓] -> Done! \n\n")
        
        print ("\n [*] -> Generating test images complete.... \n")
        generate_images_with_stats(args, data_loader, generator, args.epoch, \
                                   shuffled = False, write_log = True, \
                                   output_dir = "{0}/generated_images/ep_{1}/".format(args.result_dir, args.epoch),
                                   img_complete=True)
        print ("\n [✓] -> Done! \n\n")
        exit()
    

    # Optimizer and Sheduler
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    sheduler    = StepLR( optimizer= optimizer_G, step_size=200, gamma=0.1)
    
    if args.type_model == "GAN": 
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Initialize logger 
    monitor = Monitor(logs_path = args.result_dir + "/Logs")

    # Tensor type
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    # Setting up name of logs
    tb_names = [
                "Batch/D",  "Batch/G",
                "Batch/G/Adv", "Batch/G/Pixel_Loss"] #

    avg_names = [
                "Avg_Ep/D", "Avg_Ep/G",
                "Avg_Ep/G/Adv", "Avg_Ep/G/Pixel_Loss", #
                "Avg_Ep/Time/"]

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
            #           Train Generator
            # ------------------------------------

            optimizer_G.zero_grad()

            if (args.type_model == "Attention"):
                fake_out, _ = generator(real_in)
            else:
                fake_out    = generator(real_in)

            if args.type_model != "GAN":
                loss_GAN = torch.tensor(0)
            
            elif args.type_model == "GAN":
                # Adversarial ground truths
                valid = Variable(Tensor(np.ones ((real_in.size(0), *patch))), requires_grad=False)
                fake  = Variable(Tensor(np.zeros((real_in.size(0), *patch))), requires_grad=False)

                fake_pred = discriminator(fake_out, real_in)
                loss_GAN = GAN_loss(fake_pred, valid)
                        
            # Pixel-wise loss
            loss_pixel = pixelwise_loss(fake_out, real_out) 
            loss_G = (loss_pixel * lambda_pixel) + loss_GAN
            
            loss_G.backward()
            optimizer_G.step()
            sheduler.step()

            # ------------------------------------
            #          Train Discriminator
            # ------------------------------------

            if args.type_model != "GAN":
                loss_D = torch.tensor(0)
            
            if args.type_model == "GAN":
                optimizer_D.zero_grad()

                # Real lossreal_in
                real_pred = discriminator(real_out, real_in)
                loss_real = GAN_loss(real_pred, valid)

                # Fake loss
                fake_pred = discriminator(fake_out.detach(), real_in)
                loss_fake = GAN_loss(fake_pred, fake)

                # Total loss
                loss_D = 0.5 * (loss_real + loss_fake)

                loss_D.backward()
                optimizer_D.step()
                      
            
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
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, pixel: %f] ETA: %s" # 
                    % (
                        epoch,
                        args.n_epochs,
                        i,
                        len(data_loader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_GAN.item(),
                        loss_pixel.item(),
                        'ETA: %d:%d:%d' %(hours,minutes,seconds),
                    )
                )

                # save batch stats to logger
                tb_logs = [
                            loss_D.item(),
                            loss_G.item(),
                            loss_GAN.item(),
                            loss_pixel.item(),
                          ]

                for name, value in zip(tb_names, tb_logs): 
                    epoch_stats[name].append(value)
                
                monitor.write_logs(tb_names, tb_logs, (epoch*args.batch_size)+i)
                
                if args.use_wandb:
                    
                    wandb.log(
                        {
                        "Batch/G": loss_G.item(),
                        "Batch/G_Pixel_Loss": loss_pixel.item(),
                        },
                        step=(epoch*args.batch_size)+i
                    )
        
        # save avg stats to logger
        avg_logs = []
        for name in tb_names: avg_logs.append(np.mean(epoch_stats[name]))
        avg_logs.append((time.time() - prev_time) // 3600)
        
        monitor.write_logs(avg_names, avg_logs, epoch)
        
        if args.use_wandb:

            data = {
                    'Avg_Ep/G': avg_logs[1],
                    'Avg_Ep/G_Pixel_Loss': avg_logs[3]
                }
                    
            wandb.log(data)

        # Shuffle train data everything
        data_loader.on_epoch_end(shuffle = "train")
        
        if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
            # Save model checkpoints
            os.makedirs("%s/saved_models/" % (args.result_dir), exist_ok = True)
            torch.save(generator.state_dict(), "{0}/saved_models/G_chkp_{1:03d}.pth".format(args.result_dir, epoch))
            if args.type_model == "GAN": 
                torch.save(discriminator.state_dict(), "{0}/saved_models/D_chkp_{1:03d}.pth".format(args.result_dir, epoch))

        # If at sample interval save image
        if epoch % args.sample_interval == 0:
            
            if(args.img_complete):
                sample_images(args, data_loader, generator, epoch, img_complete= True)
            else:
                sample_images(args, data_loader, generator, epoch, img_complete= False)
                
                
        
    
    print ("\n [✓] -> Done! \n\n")

