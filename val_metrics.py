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
parser.add_argument("--name_exp", type=str, default="SA-unet-image-complete-encoder-decoder-patches", help="name of the experiment")
parser.add_argument("--name_exp_fig", type=str, default="Prueba", help="name of the experiment in the fig")
parser.add_argument("--path_results", type=str, default="Results/06-Self-Attention", help="Results path")
parser.add_argument("--path_data", type=str, default="Data/cesm_patches/data_without_background/", help="Data path")
parser.add_argument("--projection", type=str, default="CC")
parser.add_argument("--format", type=str, default="tif")
parser.add_argument("--workers", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--channels", type=int, default=1)
parser.add_argument("--dataset_name", type=str, default="cesm")
parser.add_argument("--model", type=str, default="SA-Unet")
parser.add_argument("--type_model", type=str, default="attention")
parser.add_argument("--epoch", type=int, default=400)
parser.add_argument("--img_complete", default=True, action="store_true")
parser.add_argument("--sample_size", type=int, default=20)
args = parser.parse_args()

print(f"\n\n******************** Saving image metrics from {args.name_exp} experiment ********************")

path_exp    = os.path.join( args.path_results, args.name_exp)

if(not(args.img_complete)):
    path_res_p  = os.path.join(path_exp, "metrics", "patches")
    os.makedirs(path_res_p, exist_ok = True)
    
path_res    = os.path.join( path_exp, "metrics", "image_complete")
path_met    = os.path.join( path_exp, "metrics")
os.makedirs( path_res, exist_ok=True )

Tensor = torch.cuda.FloatTensor

transforms_ = [
        transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size))
    ]

data_loader = Loader ( data_path = args.path_data, proj = args.projection, format = args.format, num_workers = args.workers,
                        batch_size = args.batch_size, img_res=(args.image_size, args.image_size), n_channels = args.channels,
                        transforms = transforms_, img_complete = args.img_complete)

data_loader_c = data_loader.test_generator if(args.img_complete) else data_loader.val_generator

if( not( args.img_complete )):
    data_loader_p = data_loader.test_generator

# Initialize generator and discriminator
if(args.model == "Unet"):
    generator = UNet_Generator(in_channels = args.channels)
elif(args.model == "Residual-PA-Unet"):
    generator = Residual_PA_UNet_Generator(in_channels= args.channels)
elif(args.model == "PA-Unet"):
    generator = PA_UNet_Generator(in_channels= args.channels)
elif(args.model == "SA-Unet"):
    generator = SA_UNet_Generator(in_channels= args.channels)

generator.load_state_dict(torch.load( os.path.join( path_exp, "saved_models", f"G_chkp_{args.epoch}.pth") ))
print (f"Weights from checkpoint: {os.path.join( path_exp, 'saved_models', f'G_chkp_{args.epoch}.pth')}")

generator.cuda()

if( args.sample_size < len(data_loader_c)):
    lucky_c = np.random.randint(0, len(data_loader_c), args.sample_size)
else:
    lucky_c = np.arange(0, len(data_loader_c))
if(not(args.img_complete)): 
    if(args.sample_size < len(data_loader_p)):
        lucky_p = np.random.randint(0, len(data_loader_p), args.sample_size)
    else:
        lucky_p = np.arange(0, len(data_loader_p))

m_fi, s_fi, p_fi = [], [], []

""" Plotting Image Complete """
for k, l in tqdm(enumerate(lucky_c), ncols=100):
            
    img = data_loader_c[int(l)]
    real_in  = Variable(img["in" ].type(Tensor)); real_in = real_in[None, :]
    real_out = Variable(img["out"].type(Tensor)); real_out = real_out[None, :]

    if(args.type_model == "attention" and (args.model == "SA-Unet")):
        fake_out, _, _ = generator(real_in)
    elif(args.type_model == "attention" and (args.model != "SA-Unet")):
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

    fig.suptitle(f"{args.name_exp_fig}", fontsize=14, fontweight='bold')
    plt.figtext(0.5, 0.01, f"MAE: {m_:.3f}, PSNR: {p_:.3f}, SSIM: {s_:.3f}", ha="center", fontsize=14)

    for ax in axes.ravel(): ax.set_axis_off()
    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(hspace=.001,wspace=0.2)

    fig = plt.gcf()
    fig = fig2img(fig)
    fig.save( os.path.join( path_res, f"test_image_{k}.png") )
    plt.close('all')

stats_fi = "{0},{1:.6f},{2:.6f},{3:.6f}".format(args.name_exp, \
                                    np.mean(m_fi),np.mean(s_fi),np.mean(p_fi))

dict = {args.name_exp + "avg_fim" : stats_fi}

w = csv.writer(open("{0}/{1}_Test_Mask_stats_img_complete.csv".format(path_met, args.name_exp), "a"))
    
for key, val in dict.items(): w.writerow([key, val]) #"""
print ("\n [!] -> Results saved in: Results/{0}_stats.csv \n".format(args.name_exp))


if( not(args.img_complete)):

    """ Plotting Patches """
    for k, l in tqdm(enumerate(lucky_p), ncols=100):
                
        img = data_loader_p[int(l)]
        real_in  = Variable(img["in" ].type(Tensor)); real_in = real_in[None, :]
        real_out = Variable(img["out"].type(Tensor)); real_out = real_out[None, :]

        if(args.type_model == "Attention"):
            fake_out, _ = generator(real_in)
        else:
            fake_out    = generator(real_in)

        """ Convert torchs tensor to numpy arrays """
        real_in     = real_in.data.cpu().numpy()[0,0,...]
        real_out    = real_out.data.cpu().numpy()[0,0,...]
        fake_out    = fake_out.data.cpu().numpy()[0,0,...]
        
        diffmap         = abs(real_out - fake_out)
        m_, s_, p_      = pixel_metrics(real_out, fake_out)
        m_fi.append(m_), s_fi.append(s_), p_fi.append(p_)

        fig, axes = plt.subplots(1, 4, figsize=(12,4))

        axes[0].imshow(real_in, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Low Energy")
        axes[1].imshow(real_out, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Recombined")
        axes[2].imshow(fake_out, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Recombined Generate")
        axes[3].set_title("Difference Map")

        im = axes[3].imshow(diffmap, cmap='hot', vmin=0, vmax=1)
        divider = make_axes_locatable(axes[3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        fig.suptitle(f"{args.name_exp_fig}", fontsize=14, fontweight='bold')
        plt.figtext(0.5, 0.01, f"MAE: {m_:.3f}, PSNR: {p_:.3f}, SSIM: {s_:.3f}", ha="center", fontsize=14)

        for ax in axes.ravel(): ax.set_axis_off()
        plt.tight_layout(pad=0.05)
        plt.subplots_adjust(wspace=0.02, hspace=0.02)

        fig = plt.gcf()
        fig = fig2img(fig)
        fig.save( os.path.join( path_res_p, f"test_image_{k}.png") )
        plt.close('all')

    stats_fi = "{0},{1:.6f},{2:.6f},{3:.6f}".format(args.name_exp, \
                                        np.mean(m_fi),np.mean(s_fi),np.mean(p_fi))

    dict = {args.name_exp + "avg_fim" : stats_fi}

    w = csv.writer(open("{0}/{1}_Test_Mask_stats_patch.csv".format(path_met, args.name_exp), "a"))
        
    for key, val in dict.items(): w.writerow([key, val]) #"""
    print ("\n [!] -> Results saved in: Results/{0}_stats.csv \n".format(args.name_exp))



