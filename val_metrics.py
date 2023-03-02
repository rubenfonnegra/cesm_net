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
parser.add_argument("--name_exp", type=str, default="Unet-Not-Deep-Image-Complete-CC-Step-lr", help="name of the experiment")
parser.add_argument("--name_exp_fig", type=str, default="Unet Deep Image Complete CC", help="name of the experiment in the fig")
parser.add_argument("--path_results", type=str, default="Results/07-Unet-Based/", help="Results path")
parser.add_argument("--path_data", type=str, default="Data/cesm_patches/data_img_complete/", help="Data path")
parser.add_argument("--projection", type=str, default="CC")
parser.add_argument("--format", type=str, default="tif")
parser.add_argument("--workers", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--channels", type=int, default=1)
parser.add_argument("--dataset_name", type=str, default="cesm")
parser.add_argument("--model", type=str, default="UNet_Not_Deep")
parser.add_argument("--type_model", type=str, default="UNet")
parser.add_argument("--epoch", type=int, default=600)
args = parser.parse_args()

name_exp        = args.name_exp
name_exp_fig    = args.name_exp_fig
path_results    = args.path_results
path_data       = args.path_data
projection      = args.projection
format          = args.format
workers         = args.workers
batch_size      = args.batch_size
image_size      = args.image_size
channels        = args.channels
model           = args.model
type_model      = args.type_model
epoch           = args.epoch

print(f"\n\n******************** Saving image metrics from {name_exp} experiment ********************")

path_exp    = os.path.join( path_results, name_exp)
path_res    = os.path.join( path_exp, "metrics", "images")
path_met    = os.path.join( path_exp, "metrics")
os.makedirs( path_res, exist_ok=True )

Tensor = torch.cuda.FloatTensor

transforms_ = [
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size))
    ]

data_loader = Loader ( data_path = path_data, proj = projection, format = format, num_workers = workers,
                           batch_size = batch_size, img_res=(image_size, image_size), n_channels = channels,
                           transforms = transforms_, img_complete = True)

data_loader = data_loader.test_generator

# Initialize generator and discriminator
if(args.model == "UNet_Deep"):
    generator = UNet_Generator_Deep(in_channels = args.channels)
elif(args.model == "UNet_Not_Deep"):
    generator = UNet_Generator_Not_Deep(in_channels= args.channels)
elif(args.model == "Residual-PA-Unet"):
    generator = Residual_PA_UNet_Generator(in_channels= args.channels)
elif(args.model == "PA-UNet"):
    generator = PA_UNet_Generator(in_channels= args.channels)
elif(args.model == "SA-UNet"):
    generator = SA_UNet_Generator(in_channels= args.channels)

generator.load_state_dict(torch.load( os.path.join( path_exp, "saved_models", f"G_chkp_{epoch}.pth") ))
print (f"Weights from checkpoint: {os.path.join( path_exp, 'saved_models', f'G_chkp_{epoch}.pth')}")

generator.cuda()
lucky = range(0, len(data_loader))

m_fi, s_fi, p_fi = [], [], []

for k, l in tqdm(enumerate(lucky), ncols=100):
            
    img = data_loader[int(l)]
    real_in  = Variable(img["in" ].type(Tensor)); real_in = real_in[None, :]
    real_out = Variable(img["out"].type(Tensor)); real_out = real_out[None, :]

    if(type_model == "Attention"):
        fake_out, _ = generator(real_in)
    else:
        fake_out    = generator(real_in)

    """ Convert torchs tensor to numpy arrays """
    real_in     = real_in.data.cpu().numpy()[0,0,...]
    real_out    = real_out.data.cpu().numpy()[0,0,...]
    fake_out    = fake_out.data.cpu().numpy()[0,0,...]

    """ Create mask """
    mask                = (real_out != 0.) * 1.
    fake_out_mask       = fake_out * mask
    real_out_mask       = real_out * mask


    diffmap_mask    = abs(real_out_mask - fake_out_mask)
    diffmap         = abs(real_out - fake_out)
    m_, s_, p_      = pixel_metrics(real_out_mask, fake_out_mask)
    m_fi.append(m_), s_fi.append(s_), p_fi.append(p_)

    fig, axes = plt.subplots(3, 2, figsize=(6,10))

    axes[0,0].imshow(real_out, cmap="gray", vmin=0, vmax=1)
    axes[0,0].set_title("Recombined")
    axes[0,1].imshow(fake_out, cmap="gray", vmin=0, vmax=1)
    axes[0,1].set_title("Recombined Generated")
    axes[1,0].imshow(real_out_mask, cmap="gray", vmin=0, vmax=1)
    axes[1,0].set_title("Recombined Real Mask")
    axes[1,1].imshow(fake_out_mask, cmap="gray", vmin=0, vmax=1)
    axes[1,1].set_title("Recombined Generated Mask")

    axes[2,0].set_title("Difference Map")
    axes[2,1].set_title("Difference Map Mask")

    im1 = axes[2,0].imshow(diffmap, cmap='hot', vmin=0, vmax=1)
    divider = make_axes_locatable(axes[2,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)

    im2 = axes[2,1].imshow(diffmap_mask, cmap='hot', vmin=0, vmax=1)
    divider = make_axes_locatable(axes[2,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)

    fig.suptitle(f"{name_exp_fig}", fontsize=14, fontweight='bold')
    plt.figtext(0.5, 0.01, f"MAE: {m_:.3f}, PSNR: {p_:.3f}, SSIM: {s_:.3f}", ha="center", fontsize=14)

    for ax in axes.ravel(): ax.set_axis_off()
    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(hspace=.001,wspace=0.2)

    fig = plt.gcf()
    fig = fig2img(fig)
    fig.save( os.path.join( path_res, f"test_image_{k}.png") )
    plt.close('all')

stats_fi = "{0},{1:.6f},{2:.6f},{3:.6f}".format(name_exp, \
                                    np.mean(m_fi),np.mean(s_fi),np.mean(p_fi))

dict = {name_exp + "avg_fim" : stats_fi}

w = csv.writer(open("{0}/{1}_Test_Mask_stats_img_complete.csv".format(path_met, name_exp), "a"))
    
for key, val in dict.items(): w.writerow([key, val]) #"""
print ("\n [!] -> Results saved in: Results/{0}_stats.csv \n".format(name_exp))









