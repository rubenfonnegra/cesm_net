import os, json
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
from PIL import Image

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def plot_test_images(dir_out = None, batch_idx = None, images = [], captions = [], diff_map = None, logger = None ):

    dir_out = os.path.join(dir_out, f"Test_Image_{batch_idx}")
    os.makedirs(dir_out, exist_ok=True)

    fig, ax = plt.subplots(1,1)

    for image, caption in zip (images, captions):

        ax.imshow(image[0,0,...], cmap="gray", vmin=-1, vmax=1)
        ax.set_title(caption)
        ax.set_axis_off()
        plt.savefig( os.path.join( dir_out, f"{caption}_Image_{batch_idx}.png" ))
        plt.cla()
    
    sns.heatmap(np.squeeze(diff_map), cmap = "hot", ax=ax, vmin=0, vmax=1)
    ax.set_axis_off()
    ax.set_title("Map Difference")
    plt.savefig( os.path.join( dir_out, f"Map_Difference_Image_{batch_idx}.png" ))

    plt.cla()
    plt.clf()
    plt.close("all")


def plot_images_with_metrics(
    images = [], captions = [], metrics = [], diff_map=None,
    figsize = (14,3), dir_out = None, id = None, logger = None, 
    setup = "Validation", epoch = 0
    ):

    """ Extract metrics values """
    mae, mssim, psnr, ssim  = metrics[0], metrics[1], metrics[2], metrics[3]

    """ Init matplotlib figure """
    fig, axs                = plt.subplots(1, 4,  figsize = (14,3))
    axs                     = axs.ravel()

    """ Create image """
    for i, (image, caption) in enumerate(zip(images, captions)):
        axs[i].imshow(image[0,0,...], cmap = "gray", vmin=-1, vmax=1)
        axs[i].set_title(caption)
        axs[i].set_axis_off()

    sns.heatmap(np.squeeze(diff_map), cmap = "hot", ax=axs[-1], vmin=0, vmax=1)
    axs[-1].set_axis_off()
    axs[-1].set_title("Map Difference")

    plt.figtext(0.5, 0.01, f"MAE: {mae:.3f}, PSNR: {psnr:.3f}, SSIM: {ssim:.3f}, MS-SSIM: {mssim:.3f}", ha="center", fontsize=14)
    plt.savefig( os.path.join( dir_out, f"{setup}_Image_{id}.png" ))

    if setup == "Val":
        logger.log_image(
            key = f"Validation Images Epoch: {epoch}",
            images = [fig2img(fig)],
            step = id
        )
    elif setup == "Test":
        logger.log_image(
            key = f"Test Images",
            images = [fig2img(fig)],
            step = id
        )

    plt.close('all')

def save_configs (args):
    #
    with Path("%s/%s/%s" %(args.result_dir, args.exp_name, "config.txt")).open("a") as f:        
        f.write("\n##############\n   Settings\n##############\n\n")
        args_dict = vars(args)
        for key in args_dict.keys():
            f.write("{0}: {1}, \n".format(json.dumps(key), json.dumps(args_dict[key])))
        f.write("\n")