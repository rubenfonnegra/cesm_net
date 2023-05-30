import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from numpy.lib.stride_tricks import as_strided
import numbers
from patchify import patchify
from patchify import patchify, unpatchify
from empatches import EMPatches


def scaler(im, range_in = [], range_out = []):
    if range_in == []:  min_, max_ = im.min(), im.max()
    else: min_, max_ = np.min(range_in), np.max(range_in)
    range_out = np.asarray(range_out)
    return ((im - min_) / (max_ - min_)) * ((range_out.max() - range_out.min()))+range_out.min()

def extract_patches(imgs, name, patch_size, step_size, dir_save, subset):

    le_img = imgs[0]
    rc_img = imgs[1]

    sum_higth = patch_size - (le_img.shape[0] % patch_size)
    sum_widht = patch_size - (le_img.shape[1] % patch_size)

    if ((name.find("L")) != -1):
        le_img = np.concatenate( (le_img, np.ones((le_img.shape[0], sum_widht))*-1.), axis = 1)
        rc_img = np.concatenate( (rc_img, np.ones((rc_img.shape[0], sum_widht))*-1.), axis = 1)
        
    else:
        le_img = np.concatenate( (np.ones((le_img.shape[0], sum_widht))*-1., le_img), axis = 1)
        rc_img = np.concatenate( (np.ones((rc_img.shape[0], sum_widht))*-1., rc_img), axis = 1)
    
    le_img = np.concatenate( (le_img, np.ones((sum_higth, le_img.shape[1]))*-1.), axis= 0)
    rc_img = np.concatenate( (rc_img, np.ones((sum_higth, rc_img.shape[1]))*-1.), axis= 0)


    img_h           = le_img.shape[0]          # height of the full image
    img_w           = le_img.shape[1]          # width of the full image
    patches_le         = []                    # list where save patches
    color_p         = (1)                   # color of patch
    tickness        = 10                    # tickness of patch in image      

    iter = 0
    indices_le = []
    for h in range((img_h - patch_size) // step_size + 1):
        for w in range((img_w - patch_size) // step_size + 1):
            
            plot_img_le        = le_img.copy()            # Copy image for plot in cv2
            plot_img_rc        = rc_img.copy()            # Copy image for plot in cv2

            """ Extract Patches """
            patch_le = le_img[
                        h * step_size:(h * step_size) + patch_size,
                        w * step_size:(w * step_size) + patch_size
                    ]
            
            patches_le.append(patch_le)
            indices_le.append((h * step_size, (h * step_size) + patch_size, w * step_size, (w * step_size) + patch_size ))
            iter = iter + 1
            
            # patch_rc = rc_img[
            #             h * step_size:(h * step_size) + patch_size,
            #             w * step_size:(w * step_size) + patch_size
            #         ]
            
            

            
            # """ Create Mask for delete patches with black background """
            # mask            = (patch_le <= -0.95 )* 1.
            # porcent_zeros   = 100*np.sum(mask)// (patch_le.shape[0]*patch_le.shape[1])

            # if(porcent_zeros < 40):

            #     im_1_c = cv2.rectangle(
            #         img         = plot_img_le,
            #         pt1         = (w* step_size, h * step_size), 
            #         pt2         = ((w * step_size)+patch_size, (h * step_size)+patch_size ),
            #         color       = color_p,
            #         thickness   = tickness
            #     )

            #     im_2_c = cv2.rectangle(
            #         img         = plot_img_rc,
            #         pt1         = (w* step_size, h * step_size), 
            #         pt2         = ((w * step_size)+patch_size, (h * step_size)+patch_size ),
            #         color       = color_p,
            #         thickness   = tickness
            #     )

            #     patch_le_ = Image.fromarray(patch_le)
            #     patch_le_.save( os.path.join (dir_save, "LE", subset, f"{name}_{iter}.tif") )
            #     patch_rc_ = Image.fromarray(patch_rc)
            #     patch_rc_.save( os.path.join (dir_save, "RC", subset, f"{name}_{iter}.tif") )

            #     dir_save_vis = os.path.join(dir_save, "vis", subset)
            #     os.makedirs(dir_save_vis, exist_ok=True)
                

            #     fig, axes = plt.subplots(1,4)
            #     axes[0].imshow(im_1_c, cmap="gray", vmin= -1., vmax = 1.)
            #     axes[0].set_axis_off()
            #     axes[1].imshow(patch_le, cmap="gray", vmin= -1., vmax = 1.)
            #     axes[1].set_axis_off()
            #     axes[2].imshow(im_2_c, cmap="gray", vmin= -1., vmax = 1.)
            #     axes[2].set_axis_off()
            #     axes[3].imshow(patch_rc, cmap="gray", vmin= -1., vmax = 1.)
            #     axes[3].set_axis_off()

            #     fig.suptitle(f"Image: {name}, Patch: {iter}")
            #     #fig.tight_layout()
            #     plt.savefig(fname = os.path.join( dir_save_vis, f"{name}_{iter}.png"), dpi=250, format="png")
            #     plt.close("all")

                #iter = iter + 1
            # else:
            #     continue
    emp = EMPatches()
    img_patches, indices = emp.extract_patches(le_img, patchsize=256, stride=254)
    print()

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def save_images(dir_save = None, list_data = [], dir_data = None, subset = None ):

    os.makedirs(os.path.join( dir_save, "LE", subset), exist_ok=True)
    os.makedirs(os.path.join( dir_save, "RC", subset), exist_ok=True)
    os.makedirs(os.path.join( dir_save, "vis", subset), exist_ok=True)
    
    for im in list_data:
        im = im.replace(" ", "")
        img1 = np.array(Image.open( os.path.join(dir_data, "LE", f"{im}.jpg") ).convert('L'))
        img2 = np.array(Image.open( os.path.join(dir_data, "RC", f"{im}.jpg") ).convert('L'))
        
        img1 = scaler(
            im = img1,
            range_in= [0,255],
            range_out=[-1,1]
        )

        img2 = scaler(
            im = img2,
            range_in= [0, 255],
            range_out=[-1,1]
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

def save_patches(dir_save = None, list_data = [], dir_data = None, subset = None, patch_size = 256, step_size = 64):

    os.makedirs(os.path.join( dir_save, "LE", subset), exist_ok=True)
    os.makedirs(os.path.join( dir_save, "RC", subset), exist_ok=True)
    os.makedirs(os.path.join( dir_save, "vis", subset), exist_ok=True)

    for im in list_data:

        im = im.replace(" ", "")
        img1 = np.array(Image.open( os.path.join(dir_data, "LE", f"{im}.jpg") ).convert('L'))
        img2 = np.array(Image.open( os.path.join(dir_data, "RC", f"{im}.jpg") ).convert('L'))

        if(img1.shape[0] < img2.shape[0]):
            sum_dif = img2.shape[0] - img1.shape[0]
            img1 = np.concatenate( (img1, np.zeros((sum_dif, img1.shape[1]))), axis=0 )
        elif(img1.shape[0] > img2.shape[0]):
            sum_dif = img1.shape[0] - img2.shape[0]
            img2 = np.concatenate( (img2, np.zeros((sum_dif, img2.shape[1]))), axis=0 )
        
        if(img1.shape[1] < img2.shape[1]):
            sum_dif = img2.shape[1] - img1.shape[1]

            if ((im.find("L")) != -1):
                img1 = np.concatenate( (img1, np.zeros((img1.shape[0], sum_dif))), axis = 1)  
            else:
                img1 = np.concatenate( ((np.zeros((img1.shape[0], sum_dif))), img1), axis = 1)
        
        elif(img1.shape[1] > img2.shape[1]):
            sum_dif = img1.shape[1] - img2.shape[1]

            if ((im.find("L")) != -1):
                img2 = np.concatenate( (img2, np.zeros((img2.shape[0], sum_dif))), axis = 1)  
            else:
                img2 = np.concatenate( ((np.zeros((img2.shape[0], sum_dif))), img2), axis = 1)

        # sum_higth = patch_size - (img.shape[0] % patch_size)
        # sum_widht = patch_size - (img.shape[1] % patch_size)

        
        

#         if ((im.find("L")) != -1):
#     new_img[ new_img.shape[0]-img2.shape[0]:new_img.shape[0], new_img.shape[1]-img2.shape[1]:new_img.shape[1]]  = img2 
# else:
#     new_img[ new_img.shape[0]-img2.shape[0]:new_img.shape[0], new_img.shape[1]-img2.shape[1]:new_img.shape[1]]  = img2 
        # img = np.concatenate( (img, (np.ones((sum_higth, img.shape[1])))*-1.), axis= 0) 
        

        # if(img1.shape != img2.shape):
        #     print(f"Shape not same: {img1.shape, img2.shape}")
        #     continue
            
        img1 = scaler(img1, range_in= [0,255], range_out=[-1,1] )
        img2 = scaler(img2, range_in= [0,255], range_out=[-1,1] )

        extract_patches([img1, img2], im, patch_size, step_size, dir_save, subset)
        # extract_patches(img2, im, patch_size, step_size, os.path.join( dir_save, "RC", subset), os.path.join( dir_save, "vis", "RC", subset))

path_csv_bening    = "birads_1_2_3_4.csv"
path_csv_maling    = "birads_5_6.csv"
dir_data    = "/media/mirplab/TB2/Experiments-Mammography/02-CDD-CESM/images-1/"
dir_save    = "/media/mirplab/TB2/Experiments-Mammography/01_Data/cesm_patch_256_64"

os.makedirs( os.path.join( dir_save, "LE"), exist_ok=True )
os.makedirs( os.path.join( dir_save, "RC"), exist_ok=True )
os.makedirs( os.path.join( dir_save, "vis"), exist_ok=True )

data_bening        = pd.read_csv(path_csv_bening, sep=",")
data_bening        = data_bening["Image_Name"].to_list()

data_malign        = pd.read_csv(path_csv_maling, sep=",")
data_malign        = data_malign["Image_Name"].to_list()

data_malign_train, data_malign_test         = train_test_split( np.array(data_malign), test_size = 30, random_state=42 )
data_bening_train, data_bening_test         = train_test_split( np.array(data_bening), test_size = 30, random_state=42 )

data_train                                  = data_bening_train.tolist() + data_malign_train.tolist()
data_test                                   = data_bening_test.tolist() + data_malign_test.tolist()

data_train, data_val                        = train_test_split( np.array(data_train), test_size = 9, random_state=42 )
data_train, data_val                        = data_train.tolist(), data_val.tolist()


save_patches(
    dir_save    = dir_save,
    list_data   = data_train,
    dir_data    = dir_data,
    patch_size  = 256,
    step_size   = 254, 
    subset      = "train"
)
print("Done CC train")

save_images(
    dir_save    = dir_save,
    list_data   = data_test,
    dir_data    = dir_data,
    subset      = "test"
)
print("Done CC test")

save_patches(
    dir_save    = dir_save,
    list_data   = data_val,
    dir_data    = dir_data,
    patch_size  = 128,
    step_size   = 64, 
    subset      = "train"
)
print("Done CC train")

print("Done!")
