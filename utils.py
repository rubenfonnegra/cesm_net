
import os
import csv
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

import json
import numpy as np
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
import wandb

from torch.utils.tensorboard import SummaryWriter

## Additional dependencies
from metrics import *

##
## Tensorboard logs
## 
class Monitor:
    #
    def __init__(self, logs_path = "Logs/"):
        #
        self.logs_path = logs_path
        self.tb = self.create_logs(logs_path)

    def create_logs(self, log_path = "/Logs"):
        #tensorboard = create_file_writer(log_path)
        tensorboard = SummaryWriter(log_dir = log_path)
        return tensorboard

    def write_logs(self, names, logs, current_batch):
        #
        if self.tb:
            for name, value in zip(names, logs):
                self.tb.add_scalar(name, value, current_batch)
                #self.tb.add_image('images', grid, 0)
            """
            with self.tb.as_default():
                for name, value in zip(names, logs):
                    scalar(name, value, step = current_batch)
                    self.tb.flush()
            """
        else:
            raise NameError
            pass


##
## Custom scaling transform
## 
class min_max_scaling (torch.nn.Module):
    """Scales a tensor image within [range.min, range.max]
    This transform does not support PIL Image.
    Given range: ``range[0], range[1]`` this transform will scale each input
    ``torch.*Tensor`` i.e.,
    ``output = (range.max - range.min) * ( (input - input.min ) / (tensor.max - tensor.min) ) - range.min ``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        range (sequence): Sequence of min-max values to scale.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, range = [-1,1], inplace=False):
        """"""
        super().__init__()
        self.inplace = inplace
        self.range = np.asarray(range)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be scaled.
        Returns:
            Tensor: Scaled Tensor image.
        """
        #return F.normalize(tensor, self.mean, self.std, self.inplace)
        return (self.range.max()-self.range.min())*(tensor-tensor.min()) / (tensor.max()-tensor.min()) + self.range.min()
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(range min={self.range.min()}, range max={self.range.max()})"

##
## Function to create image grid
## 
def save_images (list_images, output_dir, diffmap = None, image_ax = [0,1,2,4,5,6], diffmap_ax = None, plot_shape = None, figsize=(14, 3) ): 
    #
    if not plot_shape: 
        num_plots = len(list_images)+1 if diffmap is not None else len(list_images)
        plot_shape = (1,num_plots)
    
    _, axes = plt.subplots(plot_shape[0], plot_shape[1], figsize=figsize) #, figsize = (5,2)
    axes = axes.ravel()

    for i, image in zip(image_ax, list_images):
        image = np.squeeze(image).transpose(1,2,0) if image.shape[1] > 1 else np.squeeze(image)
        axes[i].imshow(np.squeeze(image), cmap = "gray", vmin=0, vmax=1) #, vmin=-1, vmax=1
        axes[i].set_axis_off(); #print(image.min(), image.max())
    
    if diffmap is not None:
        if diffmap_ax == None: diffmap_ax = -1

        for i, dm in zip(diffmap_ax, diffmap):
            sns.heatmap(np.squeeze(dm), cmap = "hot", ax=axes[i], vmin=0, vmax=1)
            axes[i].set_axis_off(); axes[i].set_title("Avg diff: {0:0.3f}".format(np.mean(np.squeeze(dm))))

    #plt.margins(0)
    plt.tight_layout(pad=0)
    plt.subplots_adjust(hspace=.0)
    #plt.savefig(output_dir)
    fig = plt.gcf()
    fig = fig2img(fig)
    fig.save(output_dir)
    plt.close('all')

##
## Network params and configuration saving
## 
def save_configs (args):
    #
    with Path("%s/%s" %(args.result_dir, "config.txt")).open("a") as f:        
        f.write("\n##############\n   Settings\n##############\n\n")
        args_dict = vars(args)
        for key in args_dict.keys():
            f.write("{0}: {1}, \n".format(json.dumps(key), json.dumps(args_dict[key])))
        f.write("\n")

##
## Image and stats saving in generation mode
## THIS FUNCTION MUST BE CHANGED BEFORE COMPUTING METRICS 
## 
def generate_images_with_stats(args, dataloader, generator, epoch, shuffled = True, \
                               output_dir = None, write_log = False, img_complete = True):

        if (img_complete):
            dataloader_c = dataloader.test_generator
        else:
            dataloader_c = dataloader.test_generator
            dataloader_p = dataloader.val_generator
        
        #dataloader_ = dataloader.test_generator if (img_complete) else dataloader.val_generator
        
        """Saves a generated sample from the validation set"""
        Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        difference = True

        if output_dir == None:
            
            if(img_complete): 
                output_dir_c = "%s/images/ep%s/" % (args.result_dir, epoch)
                os.makedirs(output_dir_c, exist_ok = True)
            else:
                output_dir_c = "%s/images/ep%s/img_complete" % (args.result_dir, epoch)
                output_dir_p = "%s/images/ep%s/patches" % (args.result_dir, epoch)
                os.makedirs(output_dir_c, exist_ok = True)
                os.makedirs(output_dir_p, exist_ok = True)

        if shuffled:
            if(not(img_complete)): 
                lucky_c = np.random.randint(0, len(dataloader_c), args.sample_size)
                lucky_p = np.random.randint(0, len(dataloader_p), args.sample_size)
            else:
                lucky_c = np.random.randint(0, len(dataloader_c), args.sample_size)
        
        else: 
            if(not(img_complete)): 
                lucky_c = np.arange(0, args.sample_size)
                lucky_p = np.arange(0, args.sample_size)
            else:
                lucky_c = np.arange(0, args.sample_size)
        
        # if shuffled: 
        #     lucky = np.random.randint(0, len(dataloader_), args.sample_size)
        # else: 
        #     lucky = np.arange(0, args.sample_size)
        
        # out_dir = output_dir+"imgs_completa/" if (img_complete) else output_dir+"imgs_parches/"
        # os.makedirs(out_dir, exist_ok = True)
        
        if args.type_model == "attention":
            
            if(img_complete):
                output_attn_c = os.path.join( output_dir_c, "attention_maps", "imgs_complete" )
                os.makedirs( output_attn_c, exist_ok= True )
            else:
                output_attn_c = os.path.join( output_dir_c, "attention_maps", "imgs_complete" )
                output_attn_p = os.path.join( output_dir_p, "attention_maps", "patches" )
                os.makedirs( output_attn_c, exist_ok= True )
                os.makedirs( output_attn_p, exist_ok= True ) 
        
            # output_attn = output_dir+"attn_maps/"+"imgs_completa/" if (img_complete) else output_dir+"attn_maps/"+"imgs_parches/"
        
        if(not(img_complete)):
            
            """ Plot and save pathes"""
            plot_imgs(args          = args,
                      lucky         = lucky_p,
                      dataloader    = dataloader_p,
                      output_dir    = output_dir_p,
                      generator     = generator,
                      difference    = True,
                      output_attn   = output_attn_c,
                      attention     = True,
                      write_log     = True)
            
            """ Plot and save images complete"""
            plot_imgs(args          = args,
                      lucky         = lucky_c,
                      dataloader    = dataloader_c,
                      output_dir    = output_dir_c,
                      generator     = generator,
                      difference    = True,
                      output_attn   = output_attn_c,
                      attention     = True,
                      write_log     = True)
        
        else:
            
            """ Plot and save images complete"""
            plot_imgs(args          = args,
                      lucky         = lucky_c,
                      dataloader    = dataloader_c,
                      output_dir    = output_dir_c,
                      generator     = generator,
                      difference    = True,
                      output_attn   = output_attn_c,
                      attention     = True,
                      write_log     = True)
               
##
## Image saving during training
## 
def sample_images(args, dataloader, generator, epoch, difference = True, output_dir = None, shuffled = True, write_log = False, img_complete=False):

        if output_dir == None:
            if(img_complete): 
                output_dir_c = "%s/images/ep%s/image_complete" % (args.result_dir, epoch)
            else:
                output_dir_p = "%s/images/ep%s/patches" % (args.result_dir, epoch)
                output_dir_c = "%s/images/ep%s/image_complete" % (args.result_dir, epoch)
                
        if shuffled:
            if(not(img_complete)): 
                lucky_c = np.random.randint(0, len(dataloader.val_generator), args.sample_size)
                lucky_p = np.random.randint(0, len(dataloader.test_generator), args.sample_size)
            else:
                lucky_c = np.random.randint(0, len(dataloader.test_generator), args.sample_size)
        
        else: 
            if(not(img_complete)): 
                lucky_c = np.arange(0, args.sample_size)
                lucky_p = np.arange(0, args.sample_size)
            else:
                lucky_c = np.arange(0, args.sample_size)
        
        if(not(img_complete)):
            
            """ Plot and save pathes"""
            plot_imgs(args          = args,
                      lucky         = lucky_p,
                      dataloader    = dataloader.test_generator,
                      output_dir    = output_dir_p,
                      generator     = generator,
                      difference    = True)

            """ Plot and save images complete"""
            plot_imgs(args          = args,
                      lucky         = lucky_c,
                      dataloader    = dataloader.val_generator,
                      output_dir    = output_dir_c,
                      generator     = generator,
                      difference    = True)
        
        else:
            
            """ Plot and save images complete"""
            plot_imgs(args          = args,
                      lucky         = lucky_c,
                      dataloader    = dataloader.test_generator,
                      output_dir    = output_dir_c,
                      generator     = generator,
                      difference    = True)

def plot_imgs(args, lucky, dataloader, output_dir, generator, difference=True, output_attn=None, attention = False, write_log = False):
    
    """Saves a generated sample from the validation set"""
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
    os.makedirs(output_dir, exist_ok = True)
    
    m_fi, s_fi, p_fi= [], [], []

    for k, l in tqdm(enumerate(lucky), ncols=100):
        
        img = dataloader[int(l)]

        real_in  = Variable(img["in" ].type(Tensor)); real_in = real_in[None, :]
        real_out = Variable(img["out"].type(Tensor)); real_out = real_out[None, :]
        
        if(args.type_model == "attention"):

            fake_out, dictOutput = generator(real_in)
            
            if (attention):
                
                os.makedirs(output_attn+str(l), exist_ok = True)
                for key, value in dictOutput.items():
                    value = value.cpu().detach().numpy()
                    np.save(f"{output_attn}{str(l)}/{key}.npy", value)
    
        else:
            fake_out = generator(real_in)

        if difference:
            diffmap = abs(real_out.data - fake_out.data) 
            img_sample = [real_in.data.cpu().numpy(), real_out.data.cpu().numpy(), fake_out.data.cpu().numpy()]
            diffmaps = [diffmap.cpu().numpy()]
            save_images(img_sample, output_dir = os.path.join(output_dir, f"{k}.png"), \
                        diffmap = diffmaps, diffmap_ax = [3], plot_shape = (1,4), figsize=(12,3))
            
            ##---- Metrics -----
            m_, s_, p_ = pixel_metrics((real_out.data.cpu().numpy()+1)/2, (fake_out.data.cpu().numpy()+1)/2)
            m_fi.append(m_), s_fi.append(s_), p_fi.append(p_)

        else:
            img_sample = torch.cat((real_in.data, real_out.data, fake_out.data), -1)
            save_image(img_sample, output_dir + "%s.png" % (k), normalize=True)
        
    if write_log == True: 
        
        stats_fi = "{0},{1:.6f},{2:.6f},{3:.6f}".format(args.exp_name, \
                                            np.mean(m_fi),np.mean(s_fi),np.mean(p_fi))
        
        dict = {args.exp_name + "avg_fim" : stats_fi}

        output_dir = "%s/metrics/" % (args.result_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        if args.img_complete:
            w = csv.writer(open("{0}/{1}_stats_img_complete.csv".format(output_dir, args.exp_name), "a"))
        else:
            w = csv.writer(open("{0}/{1}_stats_patch.csv".format(output_dir, args.exp_name), "a"))            
            
        for key, val in dict.items(): w.writerow([key, val])
        print ("\n [!] -> Results saved in: Results/{0}_stats.csv \n".format(args.exp_name))
