 
from gc import callbacks
import os
import numpy as np
import warnings
import lightning as L
from utils import save_configs
from modules.getModel import get_model

warnings.filterwarnings("ignore")
from torchinfo import summary
import torch
import torch.nn.functional as F
import shutil
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from callbacks import Custom_Callbacks
from pytorch_lightning.loggers import WandbLogger


#from utils import *
from dataloader import *

def setup_configs(args):
    
    """ Save Code Of Experiment """
    path        = os.path.join(args.result_dir, args.exp_name, "code")
    path_models = os.path.join(args.result_dir, args.exp_name, "code", "models")
    os.makedirs( path_models, exist_ok=True)
    
    for name in glob.glob('./*.py'):
        shutil.copyfile( os.path.join(".", name), os.path.join(path, name))
    for name in glob.glob('./models/*.py'):
        shutil.copyfile( os.path.join(".", name), os.path.join(path_models, os.path.basename(name)))
    for name in glob.glob('*.sh'):
        shutil.copyfile( os.path.join(".", name), os.path.join(path, name))

    save_configs(args)
    
def run_model(args):

    """ Weigths and Bias initial configuration """
    wandb_logger = WandbLogger(
        project     ="master-degree-thesis",
        entity      ="kevin-osorno-castillo",
        name        = args.exp_name,
        config      = vars(args),
        tags        = [args.tag_exp]
    )
    
    # Inicializacion de semilla de aletoriedad
    L.seed_everything(42)

    # Initialize model
    module = get_model(args= args)    
    summary(module.model, input_size=(5, args.channels, args.image_size, args.image_size))    
    
    # Initialize data loader
    data_module = Loader ( 
        data_path = args.data_dir, proj = args.projection, format = args.format, batch_size = args.batch_size, 
        img_res=(args.image_size, args.image_size), n_channels = args.channels, img_complete = args.img_complete
    )

    callbacks = Custom_Callbacks(args)
    callbacks_l = callbacks.get_callbacks()

    trainer = L.Trainer(
        default_root_dir    = os.path.join( args.result_dir, args.exp_name),
        accelerator         = "auto",
        max_epochs          = args.n_epochs,
        callbacks           = callbacks_l,
        logger              = wandb_logger,
        log_every_n_steps   = 1,
        reload_dataloaders_every_n_epochs = 1
    )

    trainer.fit( model = module, datamodule = data_module)
    trainer.test ( model = module, datamodule = data_module)
    
    
    print ("\n [✓] -> Done! \n\n")

