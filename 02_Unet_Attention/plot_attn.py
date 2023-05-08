import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from metrics import *

path_result = "/home/mirplab/Documents/kevin/01-cesm_net/Results/03-Residual-Pixel-Attention-Unet/residual-PA2-unet-data-30-porcent/"
epoch = 400

path = os.path.join( path_result, "generated_images", f"ep_{epoch}", "attn_maps")

dirs = os.listdir(path)

for dir in dirs:

    imgs = os.listdir( os.path.join( path, dir))

    for img in imgs:
    
        files =  os.listdir( os.path.join( path, dir, img))

        for file in files:

            if 'attn' in file:
                data = np.load(os.path.join( path, dir, img, file))
                name = os.path.splitext(file)[0]
                os.makedirs( os.path.join( path, dir, img, name), exist_ok=True )
                
                for i, attn in enumerate(data[0,...]):

                    fig, axes = plt.subplots(1,1, figsize=(6,6))
                    fig.suptitle(f"{name}_{i} image: {img}", fontsize=14, fontweight='bold')

                    im = axes.imshow(attn, cmap="hot", vmin= np.amin(attn), vmax=np.amax(attn))
                    divider = make_axes_locatable(axes)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)
                    axes.set_axis_off()

                    fig = plt.gcf()
                    fig = fig2img(fig)
                    fig.save( os.path.join( path, dir, img, name, f"attn_{i}_img_{img}.png") )
                    plt.close('all')

print("Done!")