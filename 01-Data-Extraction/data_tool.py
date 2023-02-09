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

def save_images(patches, name, output_path, subset, side, proj):
    #
    # os.makedirs(output_path, exist_ok = True)
    os.makedirs(output_path + "/LE/" + subset, exist_ok = True)
    os.makedirs(output_path + "/RC/" + subset, exist_ok = True)
    
    patches_le, patches_rc = patches
    
    for i, (patch_le, patch_rc) in enumerate(zip(patches_le, patches_rc)):
        _ = Image.fromarray(patch_le).save('{0}/LE/{1}/{2}_{3}_{4}_{5}.tif'.format(output_path, subset, name, proj, side, i))
        _ = Image.fromarray(patch_rc).save('{0}/RC/{1}/{2}_{3}_{4}_{5}.tif'.format(output_path, subset, name, proj, side, i))


parser = argparse.ArgumentParser()

parser.add_argument("--b_id", type=str, default="L")
parser.add_argument("--pr_id", type=str, default="CC")
parser.add_argument("--porcent_bg", type=int)
parser.add_argument("--porcent_borde", type=int)
parser.add_argument("--name", type=str, default="data_without_background")

args = parser.parse_args()

print(args.__dict__)

b_id = args.b_id
pr_id = args.pr_id
porcent_bg = args.porcent_bg
porcent_borde = args.porcent_borde
name = args.name

path        = "/media/labmirp/Datos/Proyecto_Colciencias_Mamas/OtrosDatasets/cesm_data/"
path_csv    = "/media/labmirp/Datos/workspaces/cesm_net/Data/"
path_output = os.path.join("/media/labmirp/Datos/Proyecto_Colciencias_Mamas/OtrosDatasets/cesm_patches/cesm_patches/", name)
os.makedirs(path_output, exist_ok=True)

subset = "train"

# meta_ = pd.read_csv("train_L_MLO.csv")
#meta_ = pd.read_csv("{0}_{1}_{2}.csv".format(subset, b_id, pr_id))
meta_ = pd.read_csv( os.path.join( path_csv, f"{subset}_{b_id}_{pr_id}.csv" ) )

for i in tqdm(meta_.index, ncols = 100):
    # print(path + meta_["le_file"].iloc[i], path + meta_["rec_file"].iloc[i])
    name_p = meta_["le_file"].iloc[i] [:meta_["le_file"].iloc[i].find("/")-4]
        
    if( name_p == "SCEDM030") or (name_p == "SCEDM053"):
        continue
    
    print (name_p + b_id + pr_id)
    
    dcm1 = pydicom.dcmread(path + meta_["le_file"].iloc[i]).pixel_array
    dcm2 = pydicom.dcmread(path + meta_["rec_file"].iloc[i]).pixel_array
    
    wl, ww = 2020, 2280
    
    dcm2 = clamp_histogram(dcm2, range_ = [wl, ww])
    
    dcm1 = scaler(dcm1, range_out = [0,1])
    dcm2 = scaler(dcm2, range_out = [0,1])
    
    patches_le, patches_rc, im_1_c, im_2_c = extract_patches_without_background ([dcm1, dcm2], n_patches = 100, patch_size = 256, return_patch_locs=True)
    
    _, axes = plt.subplots(1,2, figsize=(12, 8))
    axes[0].imshow(im_1_c, cmap="gray")
    axes[1].imshow(im_2_c, cmap="gray")
    
    for ax in axes: ax.set_axis_off()
    plt.tight_layout()
    #os.makedirs("cesm_patches/vis/{0}/".format(subset), exist_ok = True)
    #plt.savefig("cesm_patches/vis/{0}/{1}_{2}_{3}.png".format(subset, name_p, b_id, pr_id))
    os.makedirs( os.path.join (path_output, "vis", subset), exist_ok=True)
    plt.savefig( os.path.join (path_output, "vis", subset, f"{name_p}_{b_id}_{pr_id}.png") )
    
#    #plt.show()
    
    save_images([patches_le, patches_rc], name = name_p, output_path = path_output, subset = subset, side = b_id, proj = pr_id)


subset = "test"

# meta_ = pd.read_csv("train_L_MLO.csv")
#meta_ = pd.read_csv("{0}_{1}_{2}.csv".format(subset, b_id, pr_id))
meta_ = pd.read_csv( os.path.join( path_csv, f"{subset}_{b_id}_{pr_id}.csv" ) )

for i in tqdm(meta_.index, ncols = 100):
    # print(path + meta_["le_file"].iloc[i], path + meta_["rec_file"].iloc[i])
    name_p = meta_["le_file"].iloc[i] [:meta_["le_file"].iloc[i].find("/")-4]
        
    if( name_p == "SCEDM030") or (name_p == "SCEDM053"):
        continue
    
    print (name_p + b_id + pr_id)
    
    dcm1 = pydicom.dcmread(path + meta_["le_file"].iloc[i]).pixel_array
    dcm2 = pydicom.dcmread(path + meta_["rec_file"].iloc[i]).pixel_array
    
    wl, ww = 2020, 2280
    
    dcm2 = clamp_histogram(dcm2, range_ = [wl, ww])
    
    dcm1 = scaler(dcm1, range_out = [0,1])
    dcm2 = scaler(dcm2, range_out = [0,1])
    
    patches_le, patches_rc, im_1_c, im_2_c = extract_patches_without_background ([dcm1, dcm2], n_patches = 20, patch_size = 256, return_patch_locs=True)
    
    _, axes = plt.subplots(1,2, figsize=(12, 8))
    axes[0].imshow(im_1_c, cmap="gray")
    axes[1].imshow(im_2_c, cmap="gray")
    
    for ax in axes: ax.set_axis_off()
    plt.tight_layout()
    #os.makedirs("cesm_patches/vis/{0}/".format(subset), exist_ok = True)
    #plt.savefig("cesm_patches/vis/{0}/{1}_{2}_{3}.png".format(subset, name_p, b_id, pr_id))
    os.makedirs( os.path.join (path_output, "vis", subset), exist_ok=True)
    plt.savefig( os.path.join (path_output, "vis", subset, f"{name_p}_{b_id}_{pr_id}.png") )
    
#    #plt.show()
    
    save_images([patches_le, patches_rc], name = name_p, output_path = path_output, subset = subset, side = b_id, proj = pr_id)


subset = "val"
#meta_ = pd.read_csv("{0}_{1}_{2}.csv".format(subset, b_id, pr_id))
meta_ = pd.read_csv( os.path.join( path_csv, f"test_{b_id}_{pr_id}.csv" ) )


for i in tqdm(meta_.index, ncols = 100): #range(5): #
    # print(path + meta_["le_file"].iloc[i], path + meta_["rec_file"].iloc[i])
    name_p = meta_["le_file"].iloc[i] [:meta_["le_file"].iloc[i].find("/")-4]
    print (name_p + b_id + pr_id)
    
    dcm1 = pydicom.dcmread(path + meta_["le_file"].iloc[i]).pixel_array
    dcm2 = pydicom.dcmread(path + meta_["rec_file"].iloc[i]).pixel_array
    
    wl, ww = 2020, 2280
    
    dcm2 = clamp_histogram(dcm2, range_ = [wl, ww])
    
    dcm1 = scaler(dcm1, range_out = [0,1])
    dcm2 = scaler(dcm2, range_out = [0,1])
       
    _, axes = plt.subplots(1,2, figsize=(12, 8))
    axes[0].imshow(dcm1, cmap="gray")
    axes[1].imshow(dcm2, cmap="gray")
    
    for ax in axes: ax.set_axis_off()
    plt.tight_layout()
    # os.makedirs("cesm_patches/vis/{0}/".format(subset), exist_ok = True)
    # plt.savefig("cesm_patches/vis/{0}/{1}_{2}_{3}.png".format(subset, name_p, b_id, pr_id))
    os.makedirs( os.path.join (path_output, "vis", subset), exist_ok=True)
    plt.savefig( os.path.join (path_output, "vis", subset, f"{name_p}_{b_id}_{pr_id}.png") )
#    #plt.show()
    
    save_images([[dcm1], [dcm2]], name = name_p, output_path = path_output, subset = subset, side = b_id, proj = pr_id)
