from importlib.resources import path
import os, csv
from tqdm import tqdm
from dataloader import *
import torchvision.transforms as transforms
from torch.autograd import Variable
from metrics import *
import matplotlib.pyplot as plt
import argparse
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from models.SA_Unet import *
from models.models import *

def crop_image_only_outside(im_input_, im_output_, tol=0):
    
    mask = im_output_ > 0

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top

    # Get the contents of the bounding box.
    im_input_cropped = im_input_[x0:x1, y0:y1]
    im_output_cropped = im_output_[x0:x1, y0:y1]
    
    return im_input_cropped, im_output_cropped


parser = argparse.ArgumentParser(description= "Training GANs using CA loss")
    
# Configs  
parser.add_argument("--name_exp_fig", type=str, default="", help="name of the experiment in the fig")
parser.add_argument("--path_results", type=str, default="Results/", help="Results path")
parser.add_argument("--path_data", type=str, default="Data/sura_full_images/", help="Data path")
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
        "name_exp": "SA_Unet_v1_lr_1e3_gamm_05",
        "path_exp": "Results/02-Exp-March17-March24/",
        "name_fig": "SA Unet v1",
        "model"   : "SA-Unet-v1",
        "type_model": "attention"
    },

    "Exp2": {
        "name_exp": "SA_Unet_v2_lr_1e3_gamm_05",
        "path_exp": "Results/02-Exp-March17-March24/",
        "name_fig": "SA Unet v2",
        "model"   : "SA-Unet-v2",
        "type_model": "attention"
    },

    "Exp3":{
        "name_exp": "SA_Unet_v1_lr_1e3_gamm_05_MSE",
        "path_exp": "Results/02-Exp-March17-March24/",
        "name_fig": "SA Unet v1 MSE",
        "model"   : "SA-Unet-v1",
        "type_model": "attention"
    },

    "Exp4":{
        "name_exp": "SA_Unet_v2_lr_1e3_gamm_05_MSE",
        "path_exp": "Results/02-Exp-March17-March24/",
        "name_fig": "SA Unet v2 ",
        "model"   : "SA-Unet-v2",
        "type_model": "attention"
    },
}

imgs_exps, img_real, names_exps = [], [], []
path_output = os.path.join( path_results, "comparation_exp", "Sura Full Image", "CC")
os.makedirs( path_output, exist_ok=True )

transforms_ = [
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size))
    ]

data_loader = Loader ( data_path = path_data, proj = projection, format = format, num_workers = workers,
                           batch_size = batch_size, img_res=(image_size, image_size), n_channels = channels,
                           transforms = transforms_, dataset_name = dataset_name, img_complete = True)

data_loader = data_loader.test_generator

for i, exp in enumerate(name_exp):
    
    path_exp    = name_exp[exp]["path_exp"]
    name        = name_exp[exp]["name_exp"]
    model       = name_exp[exp]["model"]
    name_fig    = name_exp[exp]["name_fig"]
    type_model  = name_exp[exp]["type_model"]
    Tensor = torch.cuda.FloatTensor
    
    # Initialize generator and discriminator
    if(model == "Unet"):
        generator = UNet_Generator(in_channels = args.channels)
    elif(model == "SA-Unet-v1"):
        generator = SA_Unet_v1(in_channels= args.channels)
    elif(model == "SA-Unet-v2"):
        generator = SA_Unet_v2(in_channels= args.channels)

    generator.load_state_dict(torch.load( os.path.join( path_exp, name, "saved_models", "G_chkp_400.pth") ))
    print (f"Weights from checkpoint: {os.path.join( path_exp, 'saved_models', 'G_chkp_400.pth')}")

    generator.cuda()
    lucky = np.arange(0, 20)
    imgs_exp =  []


    for k, l in tqdm(enumerate(lucky), ncols=100):
        
        img = data_loader[int(l)]
        real_in  = Variable(img["in" ].type(Tensor)); real_in = real_in[None, :]
        real_out = Variable(img["out"].type(Tensor)); real_out = real_out[None, :]

        if(type_model == "attention" and (model == "SA-Unet")):
            fake_out, _, _ = generator(real_in)
        elif(type_model == "attention" and (model != "SA-Unet")):
            fake_out, _, _ = generator(real_in)
        else:
            fake_out    = generator(real_in)

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
            axes[e,0].set_title("Real Recombined")
            axes[e,1].set_title("Fake Recombined")
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
