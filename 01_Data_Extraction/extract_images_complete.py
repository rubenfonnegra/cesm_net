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
    
    for im in list_data:
        im = im.replace(" ", "")
        img1 = np.array(Image.open( os.path.join(dir_data, "LE", f"{im}.jpg") ).convert('L'))
        img2 = np.array(Image.open( os.path.join(dir_data, "RC", f"{im}.jpg") ).convert('L'))
        
        img1 = convert_squared(img1, im)
        img2 = convert_squared(img2, im)
        
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        img1 = np.array(img1.resize((512,512), Image.BICUBIC))
        img2 = np.array(img2.resize((512,512), Image.BICUBIC))
        
        img1 = scaler(
            im = img1,
            range_in= [np.amin(img1), np.amax(img1)],
            range_out=[0,1]
        )

        img2 = scaler(
            im = img2,
            range_in= [np.amin(img2), np.amax(img2)],
            range_out=[0,1]
        )

        img1 = Image.fromarray(img1)
        img1.save( os.path.join (dir_save, "LE", subset, f"{im}.tif") )

        img2 = Image.fromarray(img2)
        img2.save( os.path.join (dir_save, "RC", subset, f"{im}.tif") )

        fig, axes = plt.subplots(1,2)

        axes[0].imshow(np.array(img1), cmap="gray")
        axes[0].set_title("Low Energy")
        axes[0].set_axis_off()

        axes[1].imshow(np.array(img2), cmap="gray")
        axes[1].set_title("Recombined")
        axes[1].set_axis_off()

        plt.savefig(fname = os.path.join( dir_save, "vis", subset, f"{im}.png"), dpi=250, format="png")
        plt.close("all")
  



path_csv_bening    = "birads_1_2_3_4.csv"
path_csv_maling    = "birads_5_6.csv"
dir_data    = "/media/mirplab/TB2/Experiments-Mammography/02-CDD-CESM/images-1/"
dir_save    = "/media/mirplab/TB2/Experiments-Mammography/01_Data/cesm_images_complete_512_val"
train_ratio = 0.8
validation_ratio = 0.10
test_ratio = 0.10

os.makedirs( os.path.join( dir_save, "LE"), exist_ok=True )
os.makedirs( os.path.join( dir_save, "RC"), exist_ok=True )
os.makedirs( os.path.join( dir_save, "vis"), exist_ok=True )

data_bening        = pd.read_csv(path_csv_bening, sep=",")
data_bening        = data_bening["Image_Name"].to_list()

data_malign        = pd.read_csv(path_csv_maling, sep=",")
data_malign        = data_malign["Image_Name"].to_list()

# data_CC     = [image for image in data if (image.find("CC")) != -1]

# data_CC_train, data_CC_test                 = train_test_split( np.array(data_CC), train_size= 350, random_state=42 )
# data_CC_test, data_CC_val                   = train_test_split( data_CC_test, test_size=20, random_state=42)
# data_CC_train, data_CC_test, data_CC_val    = data_CC_train.tolist(), data_CC_test.tolist(), data_CC_val.tolist()

data_malign_train, data_malign_test         = train_test_split( np.array(data_malign), train_size= 60, random_state=42 )
data_malign_test, data_malign_val           = train_test_split( data_malign_test, train_size= 23, random_state=42 )

data_bening_train, data_bening_test         = train_test_split( np.array(data_bening), train_size= 290, random_state=42 )
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

