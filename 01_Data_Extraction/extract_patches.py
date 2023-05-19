import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

def scaler(im, range_in = [], range_out = []):
    if range_in == []:  min_, max_ = im.min(), im.max()
    else: min_, max_ = np.min(range_in), np.max(range_in)
    range_out = np.asarray(range_out)
    return (range_out.max() - range_out.min()) * (im - min_) / (max_ - min_) + range_out.min()

def convert_squared(im, name):

    im_width  = im.shape[1]
    im_height = im.shape[0]
    dim_faltante = im_height - im_width

    if ((name.find("L")) != -1):
        im = np.concatenate( ( im, np.zeros((im_height, dim_faltante))), axis= 1 )
    else:
        im = np.concatenate( ( np.zeros((im_height, dim_faltante)), im), axis=1 )
    return im

def extract_patches(img, name, patch_size, dir_save):

    img = scaler(img, range_in= [np.amin(img), np.amax(img)], range_out=[0,1] )

    sum_higth = patch_size - (img.shape[0] % patch_size)
    sum_widht = patch_size - (img.shape[1] % patch_size)

    if ((name.find("L")) != -1):
        img = np.concatenate( (img, np.zeros((img.shape[0], sum_widht))), axis = 1)
        
    else:
        img = np.concatenate( (np.zeros((img.shape[0], sum_widht)), img), axis = 1)
    
    img = np.concatenate( (img, np.zeros((sum_higth, img.shape[1]))), axis= 0)
    

    num_patches_width = img.shape[1] // patch_size
    num_patches_height = img.shape[0] // patch_size

    patches = []
    # Extrae los parches
    for i in range(num_patches_height):
        for j in range(num_patches_width):
            x = j * patch_size
            y = i * patch_size

            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            im_1_c = cv2.rectangle(img,  (x,y), (x+patch_size, y+patch_size), (1), (10))

    for i, patch_ in enumerate(patches):

        patch = Image.fromarray(patch_)
        patch.save( os.path.join (dir_save, f"{name}_{i}.tif") )
        
    return im_1_c

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def save_images(dir_save = None, list_data = [], dir_data = None, subset = None, patch_size = 128, image_complete = False ):

    os.makedirs(os.path.join( dir_save, "LE", subset), exist_ok=True)
    os.makedirs(os.path.join( dir_save, "RC", subset), exist_ok=True)
    os.makedirs(os.path.join( dir_save, "vis", subset), exist_ok=True)
    
    if(image_complete == False):
        for im in list_data:

            im = im.replace(" ", "")
            img1 = np.array(Image.open( os.path.join(dir_data, "LE", f"{im}.jpg") ).convert('L'))
            img2 = np.array(Image.open( os.path.join(dir_data, "RC", f"{im}.jpg") ).convert('L'))
            

            im1 = extract_patches(img1, im, patch_size, os.path.join( dir_save, "LE", subset))
            im2 = extract_patches(img2, im, patch_size, os.path.join( dir_save, "RC", subset))

            fig, axes = plt.subplots(1,4)

            axes[0].imshow(img1, cmap="gray")
            axes[0].set_title("Low Energy")
            axes[0].set_axis_off()

            axes[1].imshow(img2, cmap="gray")
            axes[1].set_title("Recombined")
            axes[1].set_axis_off()

            axes[2].imshow(im1, cmap="gray")
            axes[2].set_title("Patches LE")
            axes[2].set_axis_off()

            axes[3].imshow(im2, cmap="gray")
            axes[3].set_title("Patches RC")
            axes[3].set_axis_off()

            plt.savefig(fname = os.path.join( dir_save, "vis", subset, f"{im}.png"), dpi=250, format="png")
            plt.close("all")

    else:
        for im in list_data:
            im = im.replace(" ", "")
            img1 = np.array(Image.open( os.path.join(dir_data, "LE", f"{im}.jpg") ).convert('L'))
            img2 = np.array(Image.open( os.path.join(dir_data, "RC", f"{im}.jpg") ).convert('L'))
            
            img1 = convert_squared(img1, im)
            img2 = convert_squared(img2, im)
            
            img1 = Image.fromarray(img1)
            img2 = Image.fromarray(img2)

            img1 = np.array(img1.resize((256,256), Image.BICUBIC))
            img2 = np.array(img2.resize((256,256), Image.BICUBIC))
            
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
dir_save    = "/media/mirplab/TB2/Experiments-Mammography/01_Data/cesm_patch_riednet_val"

os.makedirs( os.path.join( dir_save, "LE"), exist_ok=True )
os.makedirs( os.path.join( dir_save, "RC"), exist_ok=True )
os.makedirs( os.path.join( dir_save, "vis"), exist_ok=True )

data_bening        = pd.read_csv(path_csv_bening, sep=",")
data_bening        = data_bening["Image_Name"].to_list()

data_malign        = pd.read_csv(path_csv_maling, sep=",")
data_malign        = data_malign["Image_Name"].to_list()

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
    subset      = "test_patches"
)
print("Done CC test patches")

save_images(
    dir_save        = dir_save,
    list_data       = test,
    dir_data        = dir_data,
    subset          = "test_image_complete",
    image_complete  = True
)
print("Done CC test")

save_images(
    dir_save    = dir_save,
    list_data   = val,
    dir_data    = dir_data,
    subset      = "val_image_complete",
    image_complete = True
)

save_images(
    dir_save    = dir_save,
    list_data   = val,
    dir_data    = dir_data,
    subset      = "val_patches",
    image_complete = True
)
print("Done CC val")
print("Done!")

