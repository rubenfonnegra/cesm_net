import os
import cv2
import pydicom
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from natsort import natsorted, ns
import argparse
from skimage.transform import resize


def crop_images(self, im_input_, im_output_):

        max_heigth = []
        max_widht = []

        for j in range( im_input_.shape[1]):
            
            if(( im_input_[:,j] != 0.).any() ):
                max_widht.append(j)
                
        
        for j in range( im_input_.shape[0]):

            if(( im_input_[j,:] != 0.).any() ):
                max_heigth.append(j)                        
    
        if ( (im_input_[:, 0] != 0.).any() ):

            max_widht = max_widht[-1]
            im_input_   = im_input_[ max_heigth[0]: max_heigth[-1], 0: max_widht]
            im_output_  = im_output_[ max_heigth[0]: max_heigth[-1], 0: max_widht]
        
        else:
            max_widht = max_widht[0]
            im_input_   = im_input_[ max_heigth[0]: max_heigth[-1],  max_widht:im_input_.shape[0]]
            im_output_  = im_output_[ max_heigth[0]: max_heigth[-1], max_widht:im_input_.shape[0]] 

        # im_input_   = resize(im_input_, (256, 256) )
        # im_output_  = resize(im_output_, (256, 256) )

        return im_input_, im_output_

def clamp_histogram(im_, range_ = [2020, 2280]):
    #
    im_fg = []
    # min_, max_ = 2020, 2280
    min_, max_ = range_
    
    for px in im_.ravel():
        if   px < min_: im_fg.append(min_) 
        elif min_ <= px <= max_: im_fg.append(px)
        elif px > max_: im_fg.append(max_)
    
    im_fg = np.array(im_fg).reshape(im_.shape)
    return im_fg


def scaler(im, range_in = [], range_out = []):
    if range_in == []:  min_, max_ = im.min(), im.max()
    else: min_, max_ = np.min(range_in), np.max(range_in)
    range_out = np.asarray(range_out)
    return (range_out.max() - range_out.min()) * (im - min_) / (max_ - min_) + range_out.min()


def extract_patches (images, n_patches = 25, patch_size = 256, return_patch_locs=False, porcent_bg = 10, porcent_borde = 10):
    #
    # n_patches = 25
    # patch_size = 256
    
    """ Calcular el numero de parches correspondiente a cada uno """
    n_patches_black_bg      = (n_patches * porcent_bg) // 100
    n_patches_borde         = (n_patches * porcent_borde) // 100
    n_patches_without_bg    = n_patches - (n_patches_black_bg + n_patches_borde)
    
    im_1, im_2 = images
    locs = np.array(np.where(im_1 >= 0)).T
    
    """ Creacion de lista en donde se guardan los parches que cumplan con las condiciones """
    patches_im1, patches_im2                = [], []
    patches_bg_im1, patches_bg_im2          = [], []
    patches_borde_im1, patches_borde_im2    = [], []
    
    im_1_c = im_1.copy() 
    im_2_c = im_2.copy()
    np.random.seed(0)
    i = 0
    
    #while (len(patches_im1) < n_patches_without_bg) and i < 200:
    while (
            (len(patches_im1) < n_patches_without_bg)   and
            (len(patches_bg_im1) < n_patches_black_bg)  and
            (len(patches_borde_im1) < n_patches_borde)  or 
            (i<2000)
        ):
        #
        i += 1
        lucky = np.random.randint(0, len(locs))
        patch_le = im_1[locs[lucky][0]:locs[lucky][0]+patch_size, locs[lucky][1]:locs[lucky][1]+patch_size]
        patch_rc = im_2[locs[lucky][0]:locs[lucky][0]+patch_size, locs[lucky][1]:locs[lucky][1]+patch_size]
        
        porcent_zero = np.mean(patch_rc == 0)
        
        if (patch_le.shape != (patch_size, patch_size)) or (patch_rc.shape != (patch_size, patch_size)):
            continue
        
        if (porcent_zero >= 0.4 ) and (porcent_zero <= 0.6) and (len(patches_borde_im1) < n_patches_borde):
            im_1_c = cv2.rectangle(im_1_c, (locs[lucky][1], locs[lucky][0]), (locs[lucky][1]+patch_size, locs[lucky][0]+patch_size), (1), (10))
            im_2_c = cv2.rectangle(im_2_c, (locs[lucky][1], locs[lucky][0]), (locs[lucky][1]+patch_size, locs[lucky][0]+patch_size), (1), (10))
            patches_borde_im1.append(patch_le)
            patches_borde_im2.append(patch_rc)
               
        #elif (not(0 in patch_rc)) and ((len(patches_im1) < n_patches_without_bg)):
        elif (porcent_zero < 0.05) and ((len(patches_im1) < n_patches_without_bg)):
            im_1_c = cv2.rectangle(im_1_c, (locs[lucky][1], locs[lucky][0]), (locs[lucky][1]+patch_size, locs[lucky][0]+patch_size), (1), (10))
            im_2_c = cv2.rectangle(im_2_c, (locs[lucky][1], locs[lucky][0]), (locs[lucky][1]+patch_size, locs[lucky][0]+patch_size), (1), (10))
            patches_im1.append(patch_le)
            patches_im2.append(patch_rc)
        
        elif (porcent_zero == 1.0) and ((len(patches_bg_im1) < n_patches_black_bg)):
            im_1_c = cv2.rectangle(im_1_c, (locs[lucky][1], locs[lucky][0]), (locs[lucky][1]+patch_size, locs[lucky][0]+patch_size), (1), (10))
            im_2_c = cv2.rectangle(im_2_c, (locs[lucky][1], locs[lucky][0]), (locs[lucky][1]+patch_size, locs[lucky][0]+patch_size), (1), (10))
            patches_bg_im1.append(patch_le)
            patches_bg_im2.append(patch_rc)
            
    patches_im1+=patches_bg_im1
    patches_im2+=patches_bg_im2
    patches_im1+=patches_borde_im1
    patches_im2+=patches_borde_im2
    
    patches_im1, patches_im2 = np.array(patches_im1), np.array(patches_im2)
    
    if not return_patch_locs: return patches_im1, patches_im2
    else: return patches_im1, patches_im2, im_1_c, im_2_c

def extract_patches_without_background (images, n_patches = 25, patch_size = 256, return_patch_locs=False):
    
    # n_patches = 25
    # patch_size = 256
    
    im_1, im_2 = images
    locs = np.array(np.where(im_1 > 0)).T
    
    """ Creacion de lista en donde se guardan los parches que cumplan con las condiciones """
    patches_im1, patches_im2                = [], []
    
    im_1_c = im_1.copy() 
    im_2_c = im_2.copy()
    np.random.seed(0)
    i = 0
    
    #while (len(patches_im1) < n_patches_without_bg) and i < 200:
    while (
            (len(patches_im1) < n_patches)
        ):
        #
        i += 1
        lucky = np.random.randint(0, len(locs))
        patch_le = im_1[locs[lucky][0]:locs[lucky][0]+patch_size, locs[lucky][1]:locs[lucky][1]+patch_size]
        patch_rc = im_2[locs[lucky][0]:locs[lucky][0]+patch_size, locs[lucky][1]:locs[lucky][1]+patch_size]
        
        porcent_zero = np.mean(patch_rc == 0)
        
        if (patch_le.shape != (patch_size, patch_size)) or (patch_rc.shape != (patch_size, patch_size)):
            continue
        
        if ( porcent_zero <= 0.15 ):
            im_1_c = cv2.rectangle(im_1_c, (locs[lucky][1], locs[lucky][0]), (locs[lucky][1]+patch_size, locs[lucky][0]+patch_size), (1), (10))
            im_2_c = cv2.rectangle(im_2_c, (locs[lucky][1], locs[lucky][0]), (locs[lucky][1]+patch_size, locs[lucky][0]+patch_size), (1), (10))
            patches_im1.append(patch_le)
            patches_im2.append(patch_rc)
                
    patches_im1, patches_im2 = np.array(patches_im1), np.array(patches_im2)
    
    if not return_patch_locs: return patches_im1, patches_im2
    else: return patches_im1, patches_im2, im_1_c, im_2_c

def save_images(patches, name, output_path, subset, name_data, side=None, proj=None):
    #
    # os.makedirs(output_path, exist_ok = True)
    os.makedirs(output_path + "/LE/" + subset, exist_ok = True)
    os.makedirs(output_path + "/RC/" + subset, exist_ok = True)
    
    patches_le, patches_rc = patches
    
    if(name_data == "cesm"):
        for i, (patch_le, patch_rc) in enumerate(zip(patches_le, patches_rc)):
            _ = Image.fromarray(patch_le).save('{0}/LE/{1}/{2}_{3}_{4}_{5}.tif'.format(output_path, subset, name, proj, side, i))
            _ = Image.fromarray(patch_rc).save('{0}/RC/{1}/{2}_{3}_{4}_{5}.tif'.format(output_path, subset, name, proj, side, i))
    elif(name_data == "cdd-cesm"):
        for i, (patch_le, patch_rc) in enumerate(zip(patches_le, patches_rc)):
            _ = Image.fromarray(patch_le).save('{0}/LE/{1}/{2}_{3}.tif'.format(output_path, subset, name, i))
            _ = Image.fromarray(patch_rc).save('{0}/RC/{1}/{2}_{3}.tif'.format(output_path, subset, name, i))



parser = argparse.ArgumentParser()

parser.add_argument("--b_id", type=str, default="L")
parser.add_argument("--pr_id", type=str, default="MLO")
parser.add_argument("--porcent_bg", type=int, default=0)
parser.add_argument("--porcent_borde", type=int, default=0)
parser.add_argument("--patches", default=False, action="store_true")
parser.add_argument("--name", type=str, default="prueba")
parser.add_argument("--name_data", type=str, default="cdd-cesm")
parser.add_argument("--path_data", type=str, default = "/media/mirplab/TB2/Experiments-Mammography/02-CDD-CESM/images-1/")
parser.add_argument("--path_csv", type=str, default = "/media/mirplab/TB2/Experiments-Mammography/02-CDD-CESM/images-1/")
parser.add_argument("--path_output", type=str, default = "/media/mirplab/TB2/Experiments-Mammography/01_Data/")
parser.add_argument("--plot_resize", default=True, action="store_true")
parser.add_argument("--crop", default=True, action="store_true")

args = parser.parse_args()

print(args.__dict__)

os.makedirs(args.path_output, exist_ok=True)

if args.patches:
    subsets = [ "train", "test", "val"]
else:
    subsets = ["train", "test"]

path_output = os.path.join(args.path_output, args.name)
    
if(args.name_data == "cesm"):
    
    for subset in subsets:
        
        if (args.patches) and subset == "val":
            meta_ = pd.read_csv( os.path.join( args.path_csv, f"test_{args.pr_id}.csv" ) )
        
        meta_ = pd.read_csv( os.path.join( args.path_csv, f"{subset}_{args.pr_id}.csv" ) )

        for i in tqdm(meta_.index, ncols = 100):
            
            name_p = meta_["le_file"].iloc[i] [:meta_["le_file"].iloc[i].find("/")-4]
                
            # if( name_p == "SCEDM030") or (name_p == "SCEDM053"):
            #     continue
            
            print (name_p + args.b_id + args.pr_id)
            
            dcm1 = pydicom.dcmread(args.path + meta_["le_file"].iloc[i]).pixel_array
            dcm2 = pydicom.dcmread(args.path + meta_["rec_file"].iloc[i]).pixel_array
            
            wl, ww = 2020, 2280
            
            dcm2 = clamp_histogram(dcm2, range_ = [wl, ww])
            
            dcm1 = scaler(dcm1, range_out = [0,1])
            dcm2 = scaler(dcm2, range_out = [0,1])
            
            if((args.patches) and ((args.porcent_bg == 0) or args.porcent_borde == 0)):
                patches_le, patches_rc, im_1_c, im_2_c = extract_patches_without_background ([dcm1, dcm2], n_patches = 100, patch_size = 256, return_patch_locs=True)
            elif ((args.patches) and ((args.porcent_bg != 0) or args.porcent_borde != 0)):
                patches_le, patches_rc, im_1_c, im_2_c = extract_patches ([dcm1, dcm2], n_patches = 100, patch_size = 256, return_patch_locs=True)
            
            if( args.crop ):
                dcm1, dcm2 = crop_images(dcm1, dcm2)
                        
            _, axes = plt.subplots(1,2, figsize=(12, 8))
            axes[0].imshow(dcm1, cmap="gray")
            axes[1].imshow(dcm2, cmap="gray")
            
            for ax in axes: ax.set_axis_off()
            plt.tight_layout()
            os.makedirs( os.path.join (path_output, "vis", subset), exist_ok=True)
            plt.savefig( os.path.join (path_output, "vis", subset, f"{name_p}_{args.b_id}_{args.pr_id}.png") )

            if(args.plot_resize):

                im_1_c = resize(dcm1, (256,256))
                im_2_c = resize(dcm2, (256,256))
                _, axes = plt.subplots(1,2, figsize=(12, 8))
                axes[0].imshow(im_1_c, cmap="gray")
                axes[0].set_title("Low Energy")
                axes[1].imshow(im_2_c, cmap="gray")
                axes[1].set_title("Recombined")
                
                for ax in axes: ax.set_axis_off()
                plt.tight_layout()
                os.makedirs( os.path.join (path_output, "resize", subset), exist_ok=True)
                plt.savefig( os.path.join (path_output, "resize", subset, f"{name_p}.png") )
                plt.close("all")
            
            if(args.patches):    
                save_images([patches_le, patches_rc], name = name_p, output_path = path_output, subset = subset, side = args.b_id, proj = args.pr_id)
            else:
                save_images([[dcm1], [dcm2]], name = name_p, output_path = path_output, subset = subset, side = args.b_id, proj = args.pr_id)

elif(args.name_data == "cdd-cesm"):

    for subset in subsets:
        
        if (args.patches) and (subset == "val"):
            meta_ = pd.read_csv( os.path.join( args.path_csv, f"test_{args.pr_id}.csv" ) )
        else:
            meta_ = pd.read_csv( os.path.join( args.path_csv, f"{subset}_{args.pr_id}.csv" ) )

        for i in tqdm(meta_.index, ncols = 100):
            
            name_p = meta_["LE"].iloc[i] [:meta_["LE"].iloc[i].find("_")+2]
            name_p = f"{name_p}_{args.pr_id}"
                
            print (f"{name_p}")
            
            img1 = np.asarray( Image.open( os.path.join( args.path_data, "LE", meta_["LE"].iloc[i])).convert("F"))
            img2 = np.asarray( Image.open( os.path.join( args.path_data, "RC", meta_["RC"].iloc[i])).convert("F"))
                    
            img1 = scaler(img1, range_out = [0,1])
            img2 = scaler(img2, range_out = [0,1])
            
            im_1_c = img1
            im_2_c = img2

            if(subset != "val"):
                if((args.patches) and ((args.porcent_bg == 0) or args.porcent_borde == 0)):
                    patches_le, patches_rc, im_1_c, im_2_c = extract_patches_without_background ([img1, img2], n_patches = 50, patch_size = 256, return_patch_locs=True)
                elif ((args.patches) and ((args.porcent_bg != 0) or args.porcent_borde != 0)):
                    patches_le, patches_rc, im_1_c, im_2_c = extract_patches ([img1, img2], n_patches = 50, patch_size = 256, return_patch_locs=True)                
                

            _, axes = plt.subplots(1,2, figsize=(12, 8))
            axes[0].imshow(im_1_c, cmap="gray")
            axes[1].imshow(im_2_c, cmap="gray")
            
            for ax in axes: ax.set_axis_off()
            plt.tight_layout()
            os.makedirs( os.path.join (path_output, "vis", subset), exist_ok=True)
            plt.savefig( os.path.join (path_output, "vis", subset, f"{name_p}.png") )
            plt.close("all")

            if(args.plot_resize):

                im_1_c = resize(img1, (256,256))
                im_2_c = resize(img2, (256,256))
                _, axes = plt.subplots(1,2, figsize=(12, 8))
                axes[0].imshow(im_1_c, cmap="gray")
                axes[0].set_title("Low Energy")
                axes[1].imshow(im_2_c, cmap="gray")
                axes[1].set_title("Recombined")
                
                for ax in axes: ax.set_axis_off()
                plt.tight_layout()
                os.makedirs( os.path.join (path_output, "resize", subset), exist_ok=True)
                plt.savefig( os.path.join (path_output, "resize", subset, f"{name_p}.png") )
                plt.close("all")

            
            if(args.patches) and (subset!="val"):    
                save_images([patches_le, patches_rc], name = name_p, output_path = path_output, subset = subset, name_data=args.name_data)
            elif(not(args.patches)) and (subset!="val"):
                save_images([[img1], [img2]], name = name_p, output_path = path_output, subset = subset, name_data=args.name_data)
            elif(args.patches) and (subset =="val"):
                save_images([[img1], [img2]], name = name_p, output_path = path_output, subset = subset, name_data=args.name_data)

# subset = "test"
# meta_ = pd.read_csv( os.path.join( path_csv, f"test_{b_id}_{pr_id}.csv" ) )

# for i in tqdm(meta_.index, ncols = 100): #range(5): #
#     # print(path + meta_["le_file"].iloc[i], path + meta_["rec_file"].iloc[i])
#     name_p = meta_["le_file"].iloc[i] [:meta_["le_file"].iloc[i].find("/")-4]
#     print (name_p + b_id + pr_id)
    
#     dcm1 = pydicom.dcmread(path + meta_["le_file"].iloc[i]).pixel_array
#     dcm2 = pydicom.dcmread(path + meta_["rec_file"].iloc[i]).pixel_array
    
#     wl, ww = 2020, 2280
    
#     dcm2 = clamp_histogram(dcm2, range_ = [wl, ww])
    
#     dcm1 = scaler(dcm1, range_out = [0,1])
#     dcm2 = scaler(dcm2, range_out = [0,1])
       
#     _, axes = plt.subplots(1,2, figsize=(12, 8))
#     axes[0].imshow(dcm1, cmap="gray")
#     axes[1].imshow(dcm2, cmap="gray")
    
#     for ax in axes: ax.set_axis_off()
#     plt.tight_layout()
#     os.makedirs( os.path.join (path_output, "vis", subset), exist_ok=True)
#     plt.savefig( os.path.join (path_output, "vis", subset, f"{name_p}_{b_id}_{pr_id}.png") )

#     save_images([[dcm1], [dcm2]], name = name_p, output_path = path_output, subset = subset, side = b_id, proj = pr_id)
# subset = "test"

# # meta_ = pd.read_csv("train_L_MLO.csv")
# #meta_ = pd.read_csv("{0}_{1}_{2}.csv".format(subset, b_id, pr_id))
# meta_ = pd.read_csv( os.path.join( path_csv, f"{subset}_{b_id}_{pr_id}.csv" ) )

# for i in tqdm(meta_.index, ncols = 100):
#     # print(path + meta_["le_file"].iloc[i], path + meta_["rec_file"].iloc[i])
#     name_p = meta_["le_file"].iloc[i] [:meta_["le_file"].iloc[i].find("/")-4]
        
#     if( name_p == "SCEDM030") or (name_p == "SCEDM053"):
#         continue
    
#     print (name_p + b_id + pr_id)
    
#     dcm1 = pydicom.dcmread(path + meta_["le_file"].iloc[i]).pixel_array
#     dcm2 = pydicom.dcmread(path + meta_["rec_file"].iloc[i]).pixel_array
    
#     wl, ww = 2020, 2280
    
#     dcm2 = clamp_histogram(dcm2, range_ = [wl, ww])
    
#     dcm1 = scaler(dcm1, range_out = [0,1])
#     dcm2 = scaler(dcm2, range_out = [0,1])
    
#     #patches_le, patches_rc, im_1_c, im_2_c = extract_patches_without_background ([dcm1, dcm2], n_patches = 20, patch_size = 256, return_patch_locs=True)
    
#     _, axes = plt.subplots(1,2, figsize=(12, 8))
#     axes[0].imshow(im_1_c, cmap="gray")
#     axes[1].imshow(im_2_c, cmap="gray")
    
#     for ax in axes: ax.set_axis_off()
#     plt.tight_layout()
#     #os.makedirs("cesm_patches/vis/{0}/".format(subset), exist_ok = True)
#     #plt.savefig("cesm_patches/vis/{0}/{1}_{2}_{3}.png".format(subset, name_p, b_id, pr_id))
#     os.makedirs( os.path.join (path_output, "vis", subset), exist_ok=True)
#     plt.savefig( os.path.join (path_output, "vis", subset, f"{name_p}_{b_id}_{pr_id}.png") )
    
# #    #plt.show()
    
#     save_images([patches_le, patches_rc], name = name_p, output_path = path_output, subset = subset, side = b_id, proj = pr_id)

    


