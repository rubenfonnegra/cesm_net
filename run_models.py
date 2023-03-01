
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
import wandb

from training import setup_configs, run_model


def main(): 
    #
    parser = argparse.ArgumentParser(description= "Training GANs using CA loss")
    
    # Configs  
    parser.add_argument("--exp_name", type=str, default="prueba3", help="name of the experiment")
    parser.add_argument("--gpus", type = str, default = "0", help="GPUs to use")
    parser.add_argument("--data_dir", type = str, default = "/home/mirplab/Documents/kevin/01-cesm_net/Data/data_img_complete/", help="Data dir path")
    parser.add_argument("--result_dir", type = str, default = "Results/", help = "Results path. Default = %(default)s")
    parser.add_argument("--generate", help="Image generation mode (default: False)", default=None, action="store_true")
    parser.add_argument("--img_complete", help="Image complete or patches exp?", default=True, action="store_true")
    # parser.add_argument("--ce_metrics", help="CE metrics computation mode (default: False)", default=None, action="store_true")

    # Custom configs
    ### Main adv loss
    parser.add_argument("--type_model", help="Type model to use.", default = "UNet", choices=["UNet", "GAN", "Attention"])
    parser.add_argument("--model", help="Model to use.", default = "Residual-PA-Unet", choices=["UNet_Deep", "UNet_Not_Deep" "GAN", "Residual-PA-Unet", "PA-Unet", "SA-Unet"])

    # Dataset params
    parser.add_argument("--projection", type=str, default="CC", help="Projection in which mammograms were taken")
    parser.add_argument("--image_size", type=int, default = 256, help = "Input image size")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--dataset_name", type=str, default="cesm", help="Dataset name")
    parser.add_argument("--format", type=str, default="tif", help="Image format") # "png"
    parser.add_argument("--workers", type=int, default=12, help="number of workers")
    
    # Training params
    parser.add_argument("--epoch", type=int, default=0, help="Epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=401, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--weigth_init", type=str, default="normal", choices = ['normal', 'glorot'], help="weights initializer")
    parser.add_argument("--sample_size", type=int, default=50, help="interval of sampled images to generate")
    parser.add_argument("--sample_interval", type=int, default=10, help="interval between sampling of images from generators")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints. Default = %(default)s (no save)")
    parser.add_argument("--use_wandb", type=bool, default=False, help="if is False use tensorboard, if is True use weigths and biases")
    parser.add_argument("--lambda_pixel", type=int, default=100, help="The weight of pixel loss, default = 100")


    # Initial configs
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args_dict = vars(args)
    
    """ Weigths and Bias initial configuration """
    if args.use_wandb:
        wandb.init(
            project     ="master-degree-thesis",
            entity      ="kevin-osorno-castillo",
            save_code   = True,
            name        = args.exp_name,
            config      = vars(args),
            settings    = wandb.Settings(start_method="fork")
        )
    
    setup_configs(args)
    run_model(args)


if __name__ == "__main__":
    
    main()
