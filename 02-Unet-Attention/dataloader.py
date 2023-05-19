 
from ast import Raise
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from natsort import natsorted, ns


class ImageDataset(Dataset):
    def __init__(self, inputs, outputs,
                 proj,
                 batch_size=10, image_size = (512,512), 
                 n_channels = 1,
                 shuffle = True,
                 format = "png", 
                 transforms_=None, name="cesm", **kwargs):
        #
        super(ImageDataset).__init__()
        'Initialization'
        self.files          = [] 
        self.targs          = [] 
        self.proj           = proj
        self.dim            = image_size
        self.batch_size     = batch_size
        self.n_channels     = n_channels
        self.shuffle        = shuffle
        self.name           = name
        
        files = natsorted(glob.glob(f"{inputs}/*.{format}") , alg=ns.PATH)
        targs = natsorted(glob.glob(f"{outputs}/*.{format}"), alg=ns.PATH)
        
        for file_ in files: 
            if self.proj in file_: self.files.append(file_)
        for targ_ in targs: 
            if self.proj in targ_: self.targs.append(targ_)
        
        self.transforms = T.Compose(transforms_)
    
    def __random_shuffle__(self): 
        #
        self.files, self.targs = shuffle(self.files, self.targs)


    def __getitem__(self, index):
        #
        if isinstance( index, int ) :
            sample = index % len(self.files)

            im_input_  = Image.open(self.files[sample])
            im_output_ = Image.open(self.targs[sample])
            
            im_input_  = self.transforms(im_input_)
            im_output_ = self.transforms(im_output_)

            return im_input_, im_output_

    def __len__(self):
        return len(self.files)



class ValImageDataset(Dataset):
    def __init__(self, inputs, outputs, proj, format = "png", transforms = None):
        #
        'Initialization'
        self.files          = [] 
        self.targs          = [] 
        self.proj           = proj
        self.transforms     = transforms
        
        files = natsorted(glob.glob(f"{inputs}/*.{format}") , alg=ns.PATH)
        targs = natsorted(glob.glob(f"{outputs}/*.{format}"), alg=ns.PATH)
        
        for file_ in files: 
            if self.proj in file_: self.files.append(file_)
        for targ_ in targs: 
            if self.proj in targ_: self.targs.append(targ_)     
        
    def __random_shuffle__(self): 
        
        """ For image complete exp """
        self.files, self.targs = shuffle(self.files, self.targs)
    

    def __getitem__(self, index):
        #
        if isinstance( index, int ) :
            sample = index % len(self.files)
            
            im_input_  = Image.open(self.files[sample])
            im_output_ = Image.open(self.targs[sample])
            im_input_  = self.transforms(im_input_)
            im_output_ = self.transforms(im_output_)
            
            return im_input_, im_output_

    def __len__(self):
        return len(self.files)
    

class Loader(pl.LightningDataModule):
    def __init__(self, data_path, proj, format = "png", img_res=(128, 128), n_channels = 1, img_complete = True, batch_size = 10):
        super(Loader).__init__()

        self.data_path      = data_path        
        self.format         = format
        self.img_res        = img_res
        self.n_channels     = n_channels
        self.proj           = proj
        self.img_complete   = img_complete
        self.batch_size     = batch_size
        self.prepare_data_per_node = True
        self.allow_zero_length_dataloader_with_multiple_devices = True
        self.save_hyperparameters()

    def prepare_data(self):
    
        self.train_i = self.data_path + "/LE/train/"
        self.train_o = self.data_path + "/RC/train/"
        self.test_i  = self.data_path + "/LE/test/"
        self.test_o  = self.data_path + "/RC/test/"
        self.val_i   = self.data_path + "/LE/val/"
        self.val_o   = self.data_path + "/RC/val/"

        self.transforms = T.Compose([
                            T.ToTensor(),
                        ])
        
        self.transforms = T.Compose([
                            T.ToTensor(),
                        ])
        
        self.transforms = T.Compose([
                            T.ToTensor(),
                        ])

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.train_dataset = ValImageDataset(
                inputs      = self.train_i, 
                outputs     = self.train_o, 
                proj        = self.proj, 
                format      = self.format, 
                transforms  = self.transforms
            )
            self.val_dataset = ValImageDataset(
                inputs      = self.val_i,  
                outputs     = self.val_o, 
                proj        = self.proj, 
                format      = self.format,
                transforms  = self.transforms
            )
        elif stage == "test":
            self.test_dataset = ValImageDataset(
                inputs      = self.test_i, 
                outputs     = self.test_o, 
                proj        = self.proj, 
                format      = self.format,
                transforms  = self.transforms
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, drop_last=False)

