import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.transform import resize

attn1 = np.load("Results/03-Residual-Pixel-Attention-Unet/residual-PA2-unet-data-30-porcent/generated_images/ep_400/attn_maps/imgs_completa/0/attn1.npy")
img = np.load("Results/03-Residual-Pixel-Attention-Unet/residual-PA2-unet-data-30-porcent/generated_images/ep_400/attn_maps/imgs_completa/0/image_input.npy")

img     = img[0,0,...]
attn    = attn1[0,15,...]
img_r   = resize(img, attn.shape)

mask = (img > 0) * 1.
mask = resize(mask, attn.shape)
attn = attn * mask

fig, axes = plt.subplots(1,1, figsize=(6,6))

axes.imshow(img_r, cmap="gray", vmin=0, vmax=1)
axes.imshow(attn, cmap="gray", vmin=np.amin(attn), vmax=np.amax(attn), alpha=0.1)
axes.imshow(attn, cmap="jet", vmin=np.amin(attn), vmax=np.amax(attn), alpha=0.3)
divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="5%", pad=0.05)

img_ax = im.AxesImage(axes, cmap="jet")
plt.colorbar(img_ax, cax=cax)
plt.show()
print("Done!")