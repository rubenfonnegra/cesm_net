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
parser.add_argument("--name_exp", type=str, default="", help="name of the experiment")
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
dataset_name    = "cdd-cesm"
model           = args.model
epoch           = args.epoch

print(f"******************** Saving image metrics from {name_exp} experiment ********************")

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
                           transforms = transforms_, dataset_name = dataset_name, img_complete = True)

data_loader = data_loader.test_img_complete_generator

# Initialize generator and discriminator
if(model == "UNet"):
    generator = SA_UNet_Generator(in_channels = channels)
elif(model == "Residual-PA-Unet"):
    generator = Residual_PA_UNet_Generator(in_channels= channels)

generator.load_state_dict(torch.load( os.path.join( path_exp, "saved_models", f"G_chkp_{epoch}.pth") ))
print (f"Weights from checkpoint: {os.path.join( path_exp, 'saved_models', f'G_chkp_{epoch}.pth')}")

generator.cuda()
lucky = np.arange(0, 20)

m_fi, s_fi, p_fi = [], [], []

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

    """ Create mask """
    mask        = (real_out != 0.) * 1.
    fake_out    = fake_out * mask
    real_out    = real_out * mask


    diffmap = abs(real_out - fake_out)
    m_, s_, p_ = pixel_metrics(real_out, fake_out)
    m_fi.append(m_), s_fi.append(s_), p_fi.append(p_)

    fig, axes = plt.subplots(1, 5, figsize=(24,6))

    axes[0].imshow(real_in, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Low Energy")
    axes[1].imshow(real_out, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Substacted")
    axes[2].imshow(fake_out, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Substacted Generate")
    axes[3].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Mask Metric")
    axes[4].set_title("Difference Map")

    im = axes[4].imshow(diffmap, cmap='hot', vmin=0, vmax=1)
    divider = make_axes_locatable(axes[4])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    #sns.heatmap( np.squeeze(diffmap), cmap = "hot", ax=axes[4], vmin=0, vmax=1)

    fig.suptitle(f"{name_exp_fig}", fontsize=14, fontweight='bold')
    plt.figtext(0.5, 0.01, f"MAE: {m_:.3f}, PSNR: {p_:.3f}, SSIM: {s_:.3f}", ha="center", fontsize=14)

    for ax in axes.ravel(): ax.set_axis_off()
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(hspace=.5)

    fig = plt.gcf()
    fig = fig2img(fig)
    fig.save( os.path.join( path_res, f"test_image_{k}.png") )
    plt.close('all')

stats_fi = "{0},{1:.6f},{2:.6f},{3:.6f}".format(name_exp, \
                                    np.mean(m_fi),np.mean(s_fi),np.mean(p_fi))

dict = {name_exp + "avg_fim" : stats_fi}

w = csv.writer(open("{0}/{1}_Test_stats_img_complete.csv".format(path_met, name_exp), "a"))
    
for key, val in dict.items(): w.writerow([key, val]) #"""
print ("\n [!] -> Results saved in: Results/{0}_stats.csv \n".format(name_exp))









