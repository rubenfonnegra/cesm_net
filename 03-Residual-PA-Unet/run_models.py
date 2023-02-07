
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

from training import setup_configs, run_model


def main(): 
    #
    parser = argparse.ArgumentParser(description= "Training GANs using CA loss")
    
    # Configs  
    parser.add_argument("--exp_name", type=str, default="exp_1", help="name of the experiment")
    parser.add_argument("--gpus", type = str, default = "0", help="GPUs to use")
    parser.add_argument("--data_dir", type = str, default = "Data/cesm_patches/borde_bg_30/", help="Data dir path")
    parser.add_argument("--result_dir", type = str, default = "Results/", help = "Results path. Default = %(default)s")
    parser.add_argument("--generate", help="Image generation mode (default: False)", default=None, action="store_true")
    # parser.add_argument("--ce_metrics", help="CE metrics computation mode (default: False)", default=None, action="store_true")

    # Custom configs
    ### Main adv loss
    parser.add_argument("--model", help="Model to use.", default = "UNet", choices=["UNet", "GAN"])

    # Dataset params
    parser.add_argument("--projection", type=str, default="CC", help="Projection in which mammograms were taken")
    parser.add_argument("--image_size", type=int, default = 256, help = "Input image size")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--dataset_name", type=str, default="cesm", help="Dataset name")
    parser.add_argument("--format", type=str, default="tif", help="Image format") # "png"
    parser.add_argument("--workers", type=int, default=12, help="number of workers")
    
    # Training params
    parser.add_argument("--epoch", type=int, default=0, help="Epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=201, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=30, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--weigth_init", type=str, default="glorot", choices = ['normal', 'glorot'], help="weights initializer")
    parser.add_argument("--sample_size", type=int, default=10, help="interval of sampled images to generate")
    parser.add_argument("--sample_interval", type=int, default=10, help="interval between sampling of images from generators")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints. Default = %(default)s (no save)")


    # Initial configs
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    setup_configs(args)
    run_model(args)


if __name__ == "__main__":
    #
    """
    param = sys.argv.append
    
    
    args = "--gpus 0 \
            --dataset_name cesm \
            --projection MLO \
            --exp_name tryout \
            --data_dir Data/cesm_patches/ \
            --image_size 256 --channels 1 \
            --batch_size 25 \
            --n_epochs 201 \
            --sample_interval 100 --checkpoint_interval 50 \
            --model UNet"

    # args = "--gpus 0 \
    #         --generate \
    #         --exp_name tryout \
    #         --epoch 200 \
    #         --sample_size 10 \
    #         --dataset_name cesm \
    #         --data_dir Data/cesm_patches/ \
    #         --image_size 256 --channels 1"
    
    for arg in args.split(" "): 
        if arg: param(arg)
    """
    main()
