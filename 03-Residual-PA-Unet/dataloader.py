 
from ast import Raise
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from natsort import natsorted, ns

from utils import min_max_scaling



class ImageDataset(Dataset):
    def __init__(self, inputs, outputs,
                 proj,
                 batch_size=10, image_size = (512,512), 
                 n_channels = 1,
                 shuffle = True,
                 format = "png", num_workers = 10, 
                 transforms_=None, name="cesm", **kwargs):
        #
        super(ImageDataset).__init__()
        'Initialization'
        self.files = [] #inputs  #np.array(inputs)  # list_IDs
        self.targs = [] #outputs #np.array(outputs)   # labels
        self.proj = proj
        self.dim = image_size
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.name = name
        self.num_workers = num_workers
        
        files = natsorted(glob.glob(f"{inputs}/*.{format}") , alg=ns.PATH)
        targs = natsorted(glob.glob(f"{outputs}/*.{format}"), alg=ns.PATH)
        
        # self.files = [file_  ]
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

            if self.name == "cesm":
                im_input_  = Image.open(self.files[sample]).convert("F")
                im_output_ = Image.open(self.targs[sample]).convert("F")
            
            im_input_  = self.transforms(im_input_)
            im_output_ = self.transforms(im_output_)

            return {"in": im_input_, "out": im_output_}
        elif isinstance( index, slice ) :
            batch_size_ = np.abs(index.indices(len(self))[0] - index.indices(len(self))[1])
            batch_input_  = torch.empty([batch_size_, self.n_channels, *self.dim]); 
            batch_output_ = torch.empty([batch_size_, self.n_channels, *self.dim]); 
            for i, idx in enumerate(range(*index.indices(len(self)))): 
                sample = idx % len(self.files)
                
                if self.name == "cesm":
                    im_input_  = Image.open(self.files[sample]).convert("F")
                    im_output_ = Image.open(self.targs[sample]).convert("F")
                    if np.array(im_output_).min()==0 and np.array(im_output_).max()==0:
                        im_input_  = Image.open(self.files[sample-1]).convert("F")
                        im_output_ = Image.open(self.targs[sample-1]).convert("F")

                im_input_  = self.transforms(im_input_)
                im_output_ = self.transforms(im_output_)
                batch_input_[i] = im_input_; batch_output_[i] = im_output_
                
            return {"in": batch_input_, "out": batch_output_}

    def __len__(self):
        return len(self.files)



class ValImageDataset(Dataset):
    def __init__(self, inputs, outputs,
                 proj,
                 batch_size=10, image_size = (512,512), 
                 n_channels = 1,
                 shuffle = True,
                 format = "png", num_workers = 10, 
                 transforms_=None, name="cesm", **kwargs):
        #
        'Initialization'
        self.files = [] #inputs  #np.array(inputs)  # list_IDs
        self.targs = [] #outputs #np.array(outputs)   # labels
        self.proj = proj
        self.dim = image_size
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.name = name
        self.num_workers = num_workers
        
        files = natsorted(glob.glob(f"{inputs}/*.{format}") , alg=ns.PATH)
        targs = natsorted(glob.glob(f"{outputs}/*.{format}"), alg=ns.PATH)
        
        # self.files = [file_  ]
        for file_ in files: 
            if self.proj in file_: self.files.append(file_)
        for targ_ in targs: 
            if self.proj in targ_: self.targs.append(targ_)
        
        # self.transforms = T.Compose(transforms_)
        self.transforms = T.Compose([
                                T.Resize((256, 256), Image.BICUBIC),
                                T.ToTensor()
                            ])
        

    
    def __random_shuffle__(self): 
        #
        self.files, self.targs, self.metadata = shuffle(self.files, self.targs, self.metadata )
        self.metadata.index = np.arange(len(self.files)) 
    

    def __getitem__(self, index):
        #
        if isinstance( index, int ) :
            sample = index % len(self.files)
            
            if self.name == "cesm":
                im_input_  = Image.open(self.files[sample]).convert("F")
                im_output_ = Image.open(self.targs[sample]).convert("F")
            
            im_input_  = self.transforms(im_input_)
            im_output_ = self.transforms(im_output_)
            
            return {"in": im_input_, "out": im_output_}

        elif isinstance( index, slice ) :
            batch_size_ = np.abs(index.indices(len(self))[0] - index.indices(len(self))[1])
            batch_input_  = torch.empty([batch_size_, self.n_channels, *self.dim]); 
            batch_output_ = torch.empty([batch_size_, self.n_channels, *self.dim]); 
            
            for i, idx in enumerate(range(*index.indices(len(self)))): 
                sample = idx % len(self.files)
                
                if self.name == "cesm":
                    im_input_  = Image.open(self.files[sample]).convert("F")
                    im_output_ = Image.open(self.targs[sample]).convert("F")
                
                im_input_  = self.transforms(im_input_)
                im_output_ = self.transforms(im_output_)
                
                batch_input_[i] = im_input_; batch_output_[i] = im_output_
                
            return {"in": batch_input_, "out": batch_output_}

    def __len__(self):
        return len(self.files)





class Loader():
    def __init__(self, data_path, proj, 
                 batch_size=50, dataset_name = "cesm", format = "png", num_workers = 10,
                 img_res=(128, 128), transforms = None, n_channels = 3, **kwargs):
        #
        #data_path = "/media/labmirp/Datos/Proyecto_Colciencias_Mamas/Estudios_A2/"
        #data_path = "/media/ruben-kubuntu/Datos/breast_data/"
        self.data_path = data_path
        
        self.batch_size = batch_size
        self.dataset_name = dataset_name.lower()
        self.format = format
        self.img_res = img_res
        self.transforms = transforms
        self.n_channels = n_channels
        self.proj = proj
        self.num_workers = num_workers


        self.img_res = img_res
        
        if dataset_name.lower() == "cesm":
            # (train_i, train_o, train_meta), (test_i, test_o, test_meta), (val_i, val_o, val_meta) = self.get_duke_metadata()
            train_i = data_path + "/LE/train/"
            train_o = data_path + "/RC/train/"
            test_i  = data_path + "/LE/test/"
            test_o  = data_path + "/RC/test/"
            val_i   = data_path + "/LE/val/"
            val_o   = data_path + "/RC/val/"
        
        else: 
            raise NotImplementedError (dataset_name, "Database not implemented")
        

        self.train_generator = ImageDataset(inputs = train_i, outputs = train_o, proj = self.proj,
                                            name=self.dataset_name, format=self.format,
                                            batch_size=batch_size, num_workers = self.num_workers, 
                                            image_size=img_res, n_channels=n_channels, 
                                            shuffle = True, transforms_ = self.transforms)
        

        self.test_patch_generator = ImageDataset ( inputs = test_i, outputs = test_o, proj = self.proj,
                                                name=self.dataset_name, format=self.format,
                                                batch_size=batch_size, num_workers = self.num_workers, 
                                                image_size=img_res, n_channels=n_channels, 
                                                shuffle = True, transforms_ = self.transforms)
        
        self.test_img_complete_generator  = ValImageDataset ( inputs = val_i, outputs = val_o, proj = self.proj,
                                                name=self.dataset_name, format=self.format,
                                                batch_size=batch_size, num_workers = self.num_workers, 
                                                image_size=img_res, n_channels=n_channels, 
                                                shuffle = True, transforms_ = self.transforms)
    

    def on_epoch_end(self, shuffle = "train"): 
        #
        assert shuffle.lower() in ["train", "test", "val"], "Check subset to shuffle"

        if shuffle.lower() == "train": 
            self.train_generator.__random_shuffle__()
        elif shuffle.lower() == "test": 
            self.test_patch_generator.__random_shuffle__()
        elif shuffle.lower() == "val": 
            self.test_img_complete_generator.__random_shuffle__()
    

    def __len__(self):
        return len(self.train_generator)


"""

# Configure dataloaders
transforms = [
    # T.Resize((256, 256), Image.BICUBIC),
    T.ToTensor(),
    # min_max_scaling(range = [-1,1]),
    #T.Normalize((0.5,), (0.5,)),
]


# Configure data loader
dataset_name = 'CESM'
data_loader = Loader(data_path = 'Data/cesm_patches/', proj = "MLO", 
                     dataset_name = dataset_name, format = "tif",
                     batch_size = 5, img_res=(256, 256), n_channels = 1, 
                     transforms = transforms)


images = data_loader.train_generator[20:30]
print (images["in"].shape, images["out"].shape)
sample_in = images["in"][0]
sample_out = images["out"][0]

import matplotlib.pyplot as plt
_, axes = plt.subplots(2,2,figsize=(10,10))
axes = axes.ravel()
axes[0].imshow(np.squeeze(images["in"][5].detach().numpy()), cmap="gray")
axes[1].imshow(np.squeeze(images["out"][5].detach().numpy()), cmap="gray")
axes[2].imshow(np.squeeze(images["in"][7].detach().numpy()), cmap="gray")
axes[3].imshow(np.squeeze(images["out"][7].detach().numpy()), cmap="gray")
plt.savefig("im1.png"); plt.close()
"""