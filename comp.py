from importlib.resources import path
import os, csv
from tqdm import tqdm
from dataloader import *
import torchvision.transforms as transforms
from torch.autograd import Variable
from models import *
from metrics import *
import matplotlib.pyplot as plt
import argparse
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


parser = argparse.ArgumentParser(description= "Training GANs using CA loss")
    
# Configs  
parser.add_argument("--name_exp_fig", type=str, default="", help="name of the experiment in the fig")
parser.add_argument("--path_results", type=str, default="Results/", help="Results path")
parser.add_argument("--path_data", type=str, default="/media/mirplab/TB2/Experiments-Mammography/01_Data/data_img_complete/", help="Data path")
parser.add_argument("--projection", type=str, default="CC")
parser.add_argument("--format", type=str, default="tif")
parser.add_argument("--workers", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--channels", type=int, default=1)
parser.add_argument("--dataset_name", type=str, default="cesm")
parser.add_argument("--model", type=str, default="UNet")
parser.add_argument("--epoch", type=int, default=400)
args = parser.parse_args()


name_exp_fig    = args.name_exp_fig
path_results    = args.path_results
path_data       = args.path_data
projection      = args.projection
format          = args.format
workers         = args.workers
batch_size      = args.batch_size
image_size      = args.image_size
channels        = args.channels
dataset_name    = args.dataset_name
model           = args.model
epoch           = args.epoch

name_exp        = {

    "Exp1": {
        "name_exp": "unet_CC_bg_20_01",
        "path_exp": "Results/02-exp-bg-borde/unet_CC_bg_20_01/",
        "name_fig": "SA-Unet-bg-20",
        "model"   : "UNet"
    },

    "Exp2": {
        "name_exp": "residual-PA2-unet-data-image-complete",
        "path_exp": "Results/04-Exp-Image-Complete/residual-PA2-unet-data-image-complete/",
        "name_fig": "R-PA-Unet-Comp",
        "model"   : "Residual-PA-Unet"
    },

    "Exp3":{
        "name_exp": "SA-Unet-Generator-data-image-complete",
        "path_exp": "Results/04-Exp-Image-Complete/SA-Unet-Generator-data-image-complete/",
        "name_fig": "SA-Unet-Comp",
        "model"   : "UNet"
    },

    "Exp4":{
        "name_exp": "residual-PA2-unet-image-complete-crop-2",
        "path_exp": "Results/04-Exp-Image-Complete/residual-PA2-unet-image-complete-crop-2/",
        "name_fig": "R-PA-Unet-Comp-Crop",
        "model"   : "Residual-PA-Unet"
    },

    "Exp5":{
        "name_exp": "self-attention-unet-image-complete-crop",
        "path_exp": "Results/04-Exp-Image-Complete/self-attention-unet-image-complete-crop/",
        "name_fig": "SA-Unet-Comp-Crop",
        "model"   : "UNet"
    },
}

imgs_exps, img_real, names_exps = [], [], []
path_output = os.path.join( path_results, "comparation_exp")
os.makedirs( path_output, exist_ok=True )

transforms_ = [
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size))
    ]

data_loader = Loader ( data_path = path_data, proj = projection, format = format, num_workers = workers,
                           batch_size = batch_size, img_res=(image_size, image_size), n_channels = channels,
                           transforms = transforms_, dataset_name = dataset_name, img_complete = True)

data_loader = data_loader.test_img_complete_generator

for i, exp in enumerate(name_exp):
    
    path_exp    = name_exp[exp]["path_exp"]
    model       = name_exp[exp]["model"]
    name_fig    = name_exp[exp]["name_fig"]
    Tensor = torch.cuda.FloatTensor
    
    # Initialize generator and discriminator
    if(model == "UNet"):
        generator = SA_UNet_Generator(in_channels = channels)
    elif(model == "Residual-PA-Unet"):
        generator = Residual_PA_UNet_Generator(in_channels= channels)

    generator.load_state_dict(torch.load( os.path.join( path_exp, "saved_models", "G_chkp_400.pth") ))
    print (f"Weights from checkpoint: {os.path.join( path_exp, 'saved_models', 'G_chkp_400.pth')}")

    generator.cuda()
    lucky = np.arange(0, 20)
    imgs_exp =  []


    for k, l in tqdm(enumerate(lucky), ncols=100):
        
        img = data_loader[int(l)]
        real_in  = Variable(img["in" ].type(Tensor)); real_in = real_in[None, :]
        real_out = Variable(img["out"].type(Tensor)); real_out = real_out[None, :]

        if(model == "UNet"):
            fake_out    = generator(real_in)
        else:
            fake_out, _ = generator(real_in)

        """ Convert torchs tensor to numpy arrays """
        real_in     = real_in.data.cpu().numpy()[0,0,...]
        real_out    = real_out.data.cpu().numpy()[0,0,...]
        fake_out    = fake_out.data.cpu().numpy()[0,0,...]

        if(i == 0):
            img_real.append(real_out)
        
        imgs_exp.append(fake_out)
        
    
    imgs_exps.append(imgs_exp)
    names_exps.append(name_fig)

imgs_exps = np.asarray(imgs_exps)



for img in range(imgs_exps.shape[1]):

    fig, axes = plt.subplots(nrows= len(name_exp), ncols=3, figsize=(10,15))

    for e in range(imgs_exps.shape[0]):
        

        """ Create mask """
        mask        = (img_real[img] != 0.) * 1.
        fake_fig    = imgs_exps[e,img,...] * mask
        real_fig    = img_real[img] * mask

        diffmap = abs(real_fig - fake_fig)
        mae, ssim, psnr = pixel_metrics(real_fig, fake_fig)


        axes[e,0].imshow(real_fig, cmap="gray", vmin=0, vmax=1)
        axes[e,0].set_xlabel(f"MAE:{mae:.4f}")
        axes[e,0].set_ylabel(f"{names_exps[e]}")
        axes[e,0].set_yticklabels([]); axes[e,0].set_xticklabels([])

        axes[e,1].imshow(fake_fig, cmap="gray", vmin=0, vmax=1)
        axes[e,1].set_xlabel(f"SSIM:{ssim:.4f}")
        axes[e,1].set_yticklabels([]); axes[e,1].set_xticklabels([])

        im = axes[e,2].imshow(diffmap, cmap="hot", vmin=0, vmax=1)
        axes[e,2].set_xlabel(f"PSNR:{psnr:.4f}")
        axes[e,2].set_yticklabels([]); axes[e,2].set_xticklabels([])

        divider = make_axes_locatable(axes[e,2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        if(e == 0):
            axes[e,0].set_title("Real Substracted")
            axes[e,1].set_title("Fake Substracted")
            axes[e,2].set_title("Difference Map")
    
    plt.subplots_adjust(
                    wspace=0.01,
                    hspace=0.2
                )

    plt.tight_layout(pad=0.1)

    fig = plt.gcf()
    fig = fig2img(fig)
    fig.save( os.path.join( path_output, f"test_image_{img}.png") )
    plt.close('all')



print("Done")
