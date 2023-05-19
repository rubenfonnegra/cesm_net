from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from skimage import feature
from pathlib import Path




def convert_squared(im, name):


    im_width  = im.shape[1]
    im_height = im.shape[0]
    dim_faltante = im_height - im_width

    if ((name.find("L")) != -1):
        im = np.concatenate( ( im, np.zeros((im_height, dim_faltante))), axis= 1 )
    else:
        im = np.concatenate( ( np.zeros((im_height, dim_faltante)), im), axis=1 )
    return im

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def scaler(im, range_in = [], range_out = []):
    if range_in == []:  min_, max_ = im.min(), im.max()
    else: min_, max_ = np.min(range_in), np.max(range_in)
    range_out = np.asarray(range_out)
    return (range_out.max() - range_out.min()) * (im - min_) / (max_ - min_) + range_out.min()

def save_images(dir_save = None, list_data = [], dir_data = None, subset = None ):

    os.makedirs(os.path.join( dir_save, "LE", subset), exist_ok=True)
    os.makedirs(os.path.join( dir_save, "RC", subset), exist_ok=True)
    os.makedirs(os.path.join( dir_save, "vis", subset), exist_ok=True)
    
    
  


gamma = 1
dir_data    = "/home/mirplab/Documents/kevin/01-cesm_net/Data/cesm_images_complete"
dir_save    = f"/home/mirplab/Documents/kevin/01-cesm_net/Data/cesm_images_complete/Canny_edge_{gamma}"

os.makedirs( dir_save, exist_ok=True )


data = glob.glob( os.path.join( dir_data, 'RC/**/*.tif') )

for im in data:

    img = Image.open( im )
    img = img.resize((256,256), Image.BICUBIC)
    img = np.array(img)

    img_edge = feature.canny( img, gamma)

    fig, axes = plt.subplots(1,2)

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Recombined Image")
    axes[0].set_axis_off()

    axes[1].imshow(img_edge, cmap="gray")
    axes[1].set_title("Recombined Image Edge Canny")
    axes[1].set_axis_off()

    image_name = Path(im).stem
    plt.savefig(fname = os.path.join( dir_save, f"{image_name}.png"), dpi=250, format="png")
    print(f"Done: {image_name}")
    plt.close("all")

print("Done!")

