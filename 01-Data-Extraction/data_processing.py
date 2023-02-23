from git import base
from matplotlib import path
import numpy as np
from PIL import Image
import argparse
from glob import glob
from sklearn.model_selection import train_test_split
from os.path import basename, join
from os import listdir, makedirs
from pathlib import Path
import csv
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize



def scaler(im, range_in = [], range_out = []):
    if range_in == []:  min_, max_ = im.min(), im.max()
    else: min_, max_ = np.min(range_in), np.max(range_in)
    range_out = np.asarray(range_out)
    return (range_out.max() - range_out.min()) * (im - min_) / (max_ - min_) + range_out.min()

def extract_data(args, view):

    path_save = args.path_save
    path_data = args.path_data

    list = [basename(x) for x in glob(f"{path_data}/LE/*{view}*.jpg")]

    train, test = train_test_split(list, test_size=0.2, random_state=0, shuffle=True)

    names_le_train = []
    names_si_train = []
    names_le_test  = []
    names_si_test  = []  

    makedirs( join(path_save, "LE", "train"), exist_ok=True)
    makedirs( join(path_save, "LE", "test"), exist_ok=True)
    makedirs( join(path_save, "SI", "train"), exist_ok=True)
    makedirs( join(path_save, "SI", "test"), exist_ok=True)
    makedirs( join(path_save, "vis", "train"), exist_ok=True)
    makedirs( join(path_save, "vis", "test"), exist_ok=True)


    for image in train:

        le_path =   glob(f"{path_data}/LE/{image}")[0]
        si_path =   glob(f"{path_data}/SI/{image.replace('DM','CM')}")[0]

        img_le         = np.array(Image.open(le_path).convert("L"))
        img_si         = np.array(Image.open(si_path).convert("L"))

        img_le = scaler(img_le, range_out=[0,1])
        img_si = scaler(img_si, range_out=[0,1])

        le_name = Path(le_path).stem
        si_name = Path(si_path).stem

        names_le_train.append(f"{le_name}.tif")
        names_si_train.append(f"{si_name}.tif")

        Image.fromarray(img_le).save('{0}/LE/{1}/{2}.tif'.format(path_save, "train", le_name))
        Image.fromarray(img_si).save('{0}/SI/{1}/{2}.tif'.format(path_save, "train", si_name))

        fig, axes = plt.subplots(2,2)
        axes[0][0].imshow(img_le, cmap="gray")
        axes[0][0].set_title(f"Low Energy Image\n{le_name}")
        axes[0][1].imshow(img_si, cmap="gray")
        axes[0][1].set_title(f"Subtracted Image\n{si_name}")
        axes[1][0].imshow( resize( img_le, (256,256) ), cmap="gray" )
        axes[1][0].set_title(f"Resize Low Energy\n{le_name}")
        axes[1][1].imshow( resize( img_si, (256,256) ), cmap="gray")
        axes[1][1].set_title(f"Resize Subtracted \n{si_name}")
        for ax in axes.ravel(): ax.set_axis_off()
        plt.tight_layout()
        plt.savefig( join (path_save, "vis", "train", f"{le_name.split('_')[0]}_{view}.png") )
        plt.clf()
        plt.close(fig)
        plt.clf()
        print(f"Training data extraction: {le_name}")


    for image in test:

        le_path =   glob(f"{path_data}/LE/{image}")[0]
        si_path =   glob(f"{path_data}/SI/{image.replace('DM','CM')}")[0]

        img_le         = np.array(Image.open(le_path).convert("L"))
        img_si         = np.array(Image.open(si_path).convert("L"))

        img_le = scaler(img_le, range_out=[0,1])
        img_si = scaler(img_si, range_out=[0,1])

        le_name = Path(le_path).stem
        si_name = Path(si_path).stem

        names_le_test.append(f"{le_name}.tif")
        names_si_test.append(f"{si_name}.tif")

        Image.fromarray(img_le).save('{0}/LE/{1}/{2}.tif'.format(path_save, "test", le_name))
        Image.fromarray(img_si).save('{0}/SI/{1}/{2}.tif'.format(path_save, "test", si_name))

        fig, axes = plt.subplots(2,2)
        axes[0][0].imshow(img_le, cmap="gray")
        axes[0][0].set_title(f"Low Energy Image\n{le_name}")
        axes[0][1].imshow(img_si, cmap="gray")
        axes[0][1].set_title(f"Subtracted Image\n{si_name}")
        axes[1][0].imshow( resize( img_le, (256,256) ), cmap="gray" )
        axes[1][0].set_title(f"Resize Low Energy\n{le_name}")
        axes[1][1].imshow( resize( img_si, (256,256) ), cmap="gray")
        axes[1][1].set_title(f"Resize Subtracted \n{si_name}")
        for ax in axes.ravel(): ax.set_axis_off()
        plt.tight_layout()
        plt.savefig( join (path_save, "vis", "test", f"{le_name.split('_')[0]}_{view}.png") )
        plt.clf()
        plt.close(fig)
        plt.clf()
        print(f"Testing data extraction: {le_name}")
    
    dict_train = {
        "LE": names_le_train,
        "SI": names_si_train 
    }

    dict_test = {
        "LE": names_le_test,
        "SI": names_si_test 
    }

    df_train    = pd.DataFrame.from_dict(dict_train)
    df_test     = pd.DataFrame.from_dict(dict_test)
    df_train.to_csv(    join ( path_save, f"train_{view}.csv"), index = False, header=True, sep=",")
    df_test.to_csv(     join ( path_save, f"test_{view}.csv"), index = False, header=True, sep=",")

    print()


if __name__ == "__main__":

    np.random.seed(0)

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--path_save', type=str, default="/media/mirplab/TB2/Experiments-Mammography/01_Data/cdd-cesm/", help='an integer for the accumulator')
    parser.add_argument('--path_data', type=str, default="/media/mirplab/TB2/Experiments-Mammography/02-CDD-CESM/images-1", help='an integer for the accumulator')
    
    args = parser.parse_args()
    extract_data(args, "CC")
    extract_data(args, "MLO")


