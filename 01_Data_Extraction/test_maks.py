import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import cv2


def convert_squared(im, name):

    im_width  = im.shape[1]
    im_height = im.shape[0]
    dim_faltante = im_height - im_width

    if ((name.find("L")) != -1):
        im = np.concatenate( ( im, np.ones((im_height, dim_faltante))*-1.), axis= 1 )
    else:
        im = np.concatenate( ( np.ones((im_height, dim_faltante))*-1, im), axis=1 )
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
    
    for im in list_data:
        im = im.replace(" ", "")
        img = np.array(Image.open( os.path.join(dir_data, "RC", f"{im}.jpg") ).convert('L'))
        
        img = convert_squared(img, im)  
        img = Image.fromarray(img)
        img = np.array(img.resize((256,256), Image.BICUBIC))
        
        img = scaler(
            im = img,
            range_in= [np.amin(img), np.amax(img)],
            range_out=[-1,1]
        )
        
        """ Create Mask of breast and background """
        mask_bg     = (img <=  -0.90) * 1.
        mask_breast = (img >   -0.90) * 1.

        kernel      = np.ones((7,7))
        mask_bg     = cv2.morphologyEx(mask_bg, cv2.MORPH_OPEN, kernel)
        mask_breast = cv2.morphologyEx(mask_breast, cv2.MORPH_CLOSE, kernel)

        fig, axes = plt.subplots(1,3)

        axes[0].imshow(img, cmap="gray", vmin=-1., vmax=1.)
        axes[0].set_title("Recombined Image")
        #axes[0].set_axis_off()
        axes[0].set_yticklabels([])
        axes[0].set_xticklabels([])

        axes[1].imshow(mask_breast, cmap="gray")
        axes[1].set_title("Mask Breast")
        #axes[1].set_axis_off()
        axes[1].set_yticklabels([])
        axes[1].set_xticklabels([])

        axes[2].imshow(mask_bg, cmap="gray")
        axes[2].set_title("Mask Background")
        #axes[2].set_axis_off()
        axes[2].set_yticklabels([])
        axes[2].set_xticklabels([])

        plt.savefig(fname = os.path.join( dir_save, f"{im}.png"), dpi=250, format="png")
        plt.close("all")
  



path_csv_bening    = "birads_1_2_3_4.csv"
path_csv_maling    = "birads_5_6.csv"
dir_data    = "/media/mirplab/TB2/Experiments-Mammography/02-CDD-CESM/images-1/"
dir_save    = "/media/mirplab/TB2/Experiments-Mammography/01_Data/TEST/test_mask_09"


os.makedirs( os.path.join( dir_save ), exist_ok=True )

data_bening        = pd.read_csv(path_csv_bening, sep=",")
data_bening        = data_bening["Image_Name"].to_list()

data_malign        = pd.read_csv(path_csv_maling, sep=",")
data_malign        = data_malign["Image_Name"].to_list()

data_malign_train, data_malign_test         = train_test_split( np.array(data_malign), train_size= 68, random_state=42 )
data_malign_test, data_malign_val           = train_test_split( data_malign_test, train_size= 15, random_state=42 )

data_bening_train, data_bening_test         = train_test_split( np.array(data_bening), train_size= 272, random_state=42 )
data_bening_test, data_bening_val           = train_test_split( data_bening_test, test_size= 15, random_state=42 )

train   = data_bening_train.tolist() + data_malign_train.tolist()
test    = data_bening_test.tolist() + data_malign_test.tolist()
val     = data_bening_val.tolist() + data_malign_val.tolist()

save_images(
    dir_save    = dir_save,
    list_data   = train,
    dir_data    = dir_data,
    subset      = "train"
)
print("Done CC train")

save_images(
    dir_save    = dir_save,
    list_data   = test,
    dir_data    = dir_data,
    subset      = "test"
)
print("Done CC test")

save_images(
    dir_save    = dir_save,
    list_data   = val,
    dir_data    = dir_data,
    subset      = "val"
)
print("Done CC val")
print("Done!")

