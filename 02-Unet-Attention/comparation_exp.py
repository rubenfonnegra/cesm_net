from importlib.resources import path
import os, csv
from tabnanny import check
from tqdm import tqdm
from dataloader import *
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from models.SA_Unet import *
from models.residual_PA_Unet import RPA_Unet_Generator
from metrics import pixel_metrics, fig2img
from models.ried_net import RIEDNet
from modules.ried_net_module import riedNet
from modules.RPA_Unet_module import RPA_Unet_Module
from modules.SA_Unet_v1_Module import SA_Unet_v1_Module
from modules.SA_Unet_v2_Module import SA_Unet_v2_Module

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
parser.add_argument("--exp_name", type=str, default="pruebaWSES", help="name of the experiment")
parser.add_argument("--data_dir", type = str, default = "/home/mirplab/Documents/kevin/01-cesm_net/Data/cesm_images_complete_256_val/", help="Data dir path")
parser.add_argument("--result_dir", type = str, default = "/home/mirplab/Documents/kevin/01-cesm_net/Results/05-Exp-May15-May19-2/", help = "Results path. Default = %(default)s")
parser.add_argument("--img_complete", help="Image complete or patches exp?", default=True, action="store_true")
parser.add_argument("--tag_exp", type=str, default="Exp", help="Tag for wandb")

# Model configs
parser.add_argument("--model", help="Model to use.", default = "riedNet", choices=["riedNet", "RPA-Unet", "SA-Unet-v1", "SA-Unet-v2"])
parser.add_argument("--act_out", type=str, default="Linear", help="Activation out model", choices=["Linear", "ReLU", "Sigmoid"])
parser.add_argument("--lambda_pixel", type=float, default=100., help="The weight of pixel loss, default = 100")
parser.add_argument("--lambda_edge", type=float, default=10., help="The weight of pixel loss, default = 100")
parser.add_argument("--alpha_breast", type=float, default=0.8, help="The weight of breast in Weigthed loss, default = 0.8")
parser.add_argument("--alpha_background", type=float, default=0.2, help="The weight of background in Weigthed loss, default = 0.2")
parser.add_argument("--gamma_loss", type=float, default=100., help="The weight of losses, default = 100.")
parser.add_argument("--gamma", type=float, default=0.0, help="Gamma Self-Attention layer")

# Dataset params
parser.add_argument("--projection", type=str, default="CC", help="Projection in which mammograms were taken")
parser.add_argument("--image_size", type=int, default = 256, help = "Input image size")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--format", type=str, default="tif", help="Image format") # "png"

# Training params
parser.add_argument("--epoch", type=int, default=0, help="Epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=401, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--weigth_init", type=str, default="normal", choices = ['normal', 'glorot'], help="weights initializer")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints. Default = %(default)s (no save)")
parser.add_argument("--loss", type=str, default="MAE", help="Loss of model", choices=["MAE", "WeightSum", "WeightSumEdgeSobel", "MAEEdgeSobel"])
args = parser.parse_args()


path_results    = args.result_dir
path_data       = args.data_dir
projection      = args.projection
batch_size      = args.batch_size
image_size      = args.image_size
channels        = args.channels


name_exp        = {

    "Exp1": {
        "name_exp": "RiedNet_MAE_lr2e3_New_Data_256",
        "path_exp": "/home/mirplab/Documents/kevin/01-cesm_net/Results/05-Exp-May15-May19-2/",
        "name_fig": "Ried-Net",
        "model"   : "riedNet",
    },

    "Exp2": {
        "name_exp": "RPA_Unet_MAE_lr1e4_New_Data_256",
        "path_exp": "/home/mirplab/Documents/kevin/01-cesm_net/Results/05-Exp-May15-May19-2/",
        "name_fig": "RPA Unet",
        "model"   : "RPA-Unet",
    },

    "Exp3":{
        "name_exp": "SA_Unet_v1_MAE_lr1e4_New_Data_256_gamma_05",
        "path_exp": "/home/mirplab/Documents/kevin/01-cesm_net/Results/05-Exp-May15-May19-2/",
        "name_fig": "SA Unet v1, gamma=0.5",
        "model"   : "SA-Unet-v1",
    },

    "Exp4":{
        "name_exp": "SA_Unet_v2_MAE_lr1e4_New_Data_256",
        "path_exp": "/home/mirplab/Documents/kevin/01-cesm_net/Results/05-Exp-May15-May19-2/",
        "name_fig": "SA_Unet_v2",
        "model"   : "SA-Unet-v2",
    },
}

imgs_exps, img_real, names_exps, img_in = [], [], [], []
path_output = os.path.join( path_results, "comparation_exp_0")
os.makedirs( path_output, exist_ok=True )

transforms_ = [
        transforms.ToTensor(),
    ]

data_module = Loader ( 
        data_path = args.data_dir, proj = args.projection, format = args.format, batch_size = args.batch_size, 
        img_res=(args.image_size, args.image_size), n_channels = args.channels
    )

data_module.prepare_data()
data_module.setup(stage="test")
data_loader = data_module.test_dataloader()

for i, exp in enumerate(name_exp):
    
    path_exp    = name_exp[exp]["path_exp"]
    name        = name_exp[exp]["name_exp"]
    model       = name_exp[exp]["model"]
    name_fig    = name_exp[exp]["name_fig"]
    Tensor = torch.cuda.FloatTensor
    
    if(model == "SA-Unet-v1") or (model == "SA-Unet-v2"):
        gamma = torch.load( os.path.join( path_exp, name, "saved_models", f"gamma_399.pth"))

    # Initialize generator and discriminator
    actOut = nn.ReLU()

    
    if(model == "riedNet"):
        generator =  RIEDNet( 1, actOut = actOut)
    elif(model == "RPA-Unet"):
        generator = RPA_Unet_Generator(1, actOut = actOut)
    elif(model == "SA-Unet-v1"):
        generator = SA_Unet_v1(1, gamma = gamma,  actOut = actOut)
    elif(model == "SA-Unet-v2"):
        generator = SA_Unet_v2(1, gamma = gamma, actOut = actOut)

    checkpoint = torch.load(os.path.join( path_exp, name, 'saved_models', 'G_chkp_epoch=399.ckpt'))['state_dict']
    for key in list(checkpoint.keys()):
        if 'model.' in key:
            checkpoint[key.replace('model.', '')] = checkpoint[key]
            del checkpoint[key]
    generator.load_state_dict(checkpoint)
    #generator = generator.load_state_dict(torch.load(), strict =False)
    generator.eval()

    # generator.load_state_dict(checkpoint['state_dict'])
    # print (f"Weights from checkpoint: {os.path.join( path_exp,  'saved_models', 'G_chkp_epoch=399.ckpt')}")

    generator.cuda()
    lucky = np.arange(0, 59)
    imgs_exp =  []


    for l, (real_in, real_out) in tqdm(enumerate(data_loader), ncols=100):
        
        real_in = real_in.cuda()
        real_out = real_out.cuda()
        fake_out    = generator(real_in)

        """ Convert torchs tensor to numpy arrays """
        real_in     = real_in.data.cpu().numpy()[0,0,...]
        real_out    = real_out.data.cpu().numpy()[0,0,...]
        fake_out    = fake_out.data.cpu().numpy()[0,0,...]

        if(i == 0):
            img_real.append(real_out)
            img_in.append(real_in)

        
        imgs_exp.append(fake_out)
        
    
    imgs_exps.append(imgs_exp)
    names_exps.append(name_fig)

imgs_exps = np.asarray(imgs_exps)



for img in range(imgs_exps.shape[1]):

    fig, axes = plt.subplots(nrows= len(name_exp), ncols=4, figsize=(18,20))

    for e in range(imgs_exps.shape[0]):
        

        """ Create mask """
        mask        = (img_real[img] != 0.) * 1.
        fake_fig    = imgs_exps[e,img,...] * mask
        real_fig    = img_real[img] * mask
        real_in     = img_in[img]

        diffmap = abs(real_fig - fake_fig)
        mae, ssim, psnr = pixel_metrics(real_fig, fake_fig)

        axes[e,0].imshow(real_in, cmap="gray", vmin=0, vmax=1)
        axes[e,0].set_ylabel(f"{names_exps[e]}")
        axes[e,0].set_yticklabels([]); axes[e,0].set_xticklabels([])

        axes[e,1].imshow(real_fig, cmap="gray", vmin=0, vmax=1)
        axes[e,1].set_xlabel(f"MAE:{mae:.4f}")
        axes[e,1].set_yticklabels([]); axes[e,1].set_xticklabels([])

        axes[e,2].imshow(fake_fig, cmap="gray", vmin=0, vmax=1)
        axes[e,2].set_xlabel(f"SSIM:{ssim:.4f}")
        axes[e,2].set_yticklabels([]); axes[e,2].set_xticklabels([])

        im = axes[e,3].imshow(diffmap, cmap="hot", vmin=0, vmax=1)
        axes[e,3].set_xlabel(f"PSNR:{psnr:.4f}")
        axes[e,3].set_yticklabels([]); axes[e,3].set_xticklabels([])

        divider = make_axes_locatable(axes[e,3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        if(e == 0):
            axes[e,0].set_title("Low Energy")
            axes[e,1].set_title("Real Recombined")
            axes[e,2].set_title("Fake Recombined")
            axes[e,3].set_title("Difference Map")
    
    plt.subplots_adjust(
                    wspace=0.01,
                    hspace=0.02
                )

    plt.tight_layout(pad=0.1)

    fig = plt.gcf()
    fig = fig2img(fig)
    fig.save( os.path.join( path_output, f"test_image_{img}.png") )
    plt.close('all')



print("Done")
