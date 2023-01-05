 
import os
import sys
import time
import argparse
import datetime
import itertools
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from torchinfo import summary 

from utils import *
from models import *
from dataloader import *


def setup_configs(args):
    #
    os.makedirs(args.result_dir + "/%s" % args.exp_name, exist_ok=True)
    args.result_dir = args.result_dir + "/%s" % args.exp_name

    #if args.WGAN: args.PatchGAN = False
    if not args.model and not args.generate: 
        raise NotImplementedError ("Which model to use? Implemented: UNet, GAN ")

    #print(args)
    if not args.generate:
        save_configs(args)
    
    args.cuda = True if torch.cuda.is_available() else False
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


def run_model(args): 
    #
    import torch
    import torch.nn.functional as F
    
    from torch.autograd import Variable
    import torchvision.transforms as transforms
    
    # Initialize generator and discriminator
    generator = SA_UNet_Generator(in_channels = args.channels)
    summary(generator, input_size=(5, 1, 256,256))
    
    # Choose correct type of D
    if args.generate:
        #
        discriminator = None
        to_cuda = [generator]
    
    elif args.model == "UNet":
        to_cuda = [generator] 

    elif args.model == "GAN":
        # Create D
        discriminator = PatchGAN_Discriminator(n_channels = args.channels)
        GAN_loss = nn.MSELoss()
        to_cuda = [generator, discriminator]
    
    else: 
        discriminator = None
        to_cuda = [generator]
    

    pixelwise_loss, lambda_pixel = torch.nn.L1Loss(), 100
    to_cuda.append (pixelwise_loss)

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
        # # Initialize weights 
        # if os.path.isfile("{0}/saved_models/G_chkp_{1:03d}.pth".format(args.result_dir, args.epoch)):
        #     print("Model already exist. No retraining!")
        #     exit()
        
        if   args.weigth_init == "normal":
            generator.apply(weights_init_normal)
            if args.model != "UNet": 
                discriminator.apply(weights_init_normal)
        elif args.weigth_init == "glorot":
            generator.apply(weights_init_glorot)
            if args.model != "UNet": 
                discriminator.apply(weights_init_glorot)
    

    # Configure dataloaders
    transforms_ = [
        # transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        # min_max_scaling(range = [-1,1]),
        #transforms.RandomHorizontalFlip(p=0.5),  ##Data augmentation
    ]
    
    # Initialize data loader
    data_loader = Loader ( data_path = args.data_dir, proj = args.projection, format = args.format, num_workers = args.workers,
                           batch_size = args.batch_size, img_res=(args.image_size, args.image_size), n_channels = args.channels,
                           transforms = transforms_, dataset_name = args.dataset_name)
    
    # In case of validation option. Otherwise, move to train
    if args.generate: 
        #
        print ("\n [*] -> Generating images.... \n")
        generate_images_with_stats(args, data_loader, generator, args.epoch, \
                                   shuffled = False, write_log = True, \
                                   output_dir = "{0}/generated_images/ep_{1}/".format(args.result_dir, args.epoch))
        print ("\n [✓] -> Done! \n\n")
        exit()
    

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    if args.model != "UNet": 
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
        #
        epoch_stats, it = {}, 0
        for name in tb_names: epoch_stats[name] = []

        for i in range(0, len(data_loader), args.batch_size):
            #
            it += 1
            batch = data_loader.train_generator[i: i + args.batch_size]

            # Model inputs
            real_in  = Variable(batch["in"].type(Tensor))
            real_out = Variable(batch["out"].type(Tensor))

            # ------------------------------------
            #           Train Generator
            # ------------------------------------

            optimizer_G.zero_grad()

            # GAN loss 
            fake_out = generator(real_in)

            if args.model == "UNet":
                loss_GAN = torch.tensor(0)
            
            elif args.model == "GAN":
                # Adversarial ground truths
                valid = Variable(Tensor(np.ones ((real_in.size(0), *patch))), requires_grad=False)
                fake  = Variable(Tensor(np.zeros((real_in.size(0), *patch))), requires_grad=False)

                fake_pred = discriminator(fake_out, real_in)
                loss_GAN = GAN_loss(fake_pred, valid)

                # # Pixel-wise loss
                # loss_pixel = pixelwise_loss(fake_out, real_out) #if args.pixel_loss else torch.tensor(0)
            
            # Pixel-wise loss
            loss_pixel = pixelwise_loss(fake_out, real_out) 
            loss_G = (loss_pixel * lambda_pixel) + loss_GAN
            
            loss_G.backward()
            optimizer_G.step()

            # ------------------------------------
            #          Train Discriminator
            # ------------------------------------

            if args.model == "UNet":
                loss_D = torch.tensor(0)
            
            if args.model == "GAN":
                #
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
        
        # save avg stats to logger
        avg_logs = []
        for name in tb_names: avg_logs.append(np.mean(epoch_stats[name]))
        avg_logs.append((time.time() - prev_time) // 3600)
        
        monitor.write_logs(avg_names, avg_logs, epoch)

        # Shuffle train data everything
        data_loader.on_epoch_end(shuffle = "train")
        
        if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
            # Save model checkpoints
            os.makedirs("%s/saved_models/" % (args.result_dir), exist_ok = True)
            torch.save(generator.state_dict(), "{0}/saved_models/G_chkp_{1:03d}.pth".format(args.result_dir, epoch))
            if args.model != "UNet": 
                torch.save(discriminator.state_dict(), "{0}/saved_models/D_chkp_{1:03d}.pth".format(args.result_dir, epoch))

        # If at sample interval save image
        if epoch % args.sample_interval == 0:
            sample_images(args, data_loader, generator, epoch)
        
    
    print ("\n [✓] -> Done! \n\n")

