
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")


from training import setup_configs, run_model


def main(): 
    #
    parser = argparse.ArgumentParser(description= "Training models")
    
    # Configs  
    parser.add_argument("--exp_name", type=str, default="pruebaWSES", help="name of the experiment")
    parser.add_argument("--data_dir", type = str, default = "/media/labmirp/Datos/workspaces/cesm_net/Data/cesm_images_complete_256_11/", help="Data dir path")
    parser.add_argument("--result_dir", type = str, default = "/media/labmirp/Datos/workspaces/cesm_net/Results/06-Exp-May19-May26/WeightedSum_EdgeSobel_Loss", help = "Results path. Default = %(default)s")
    parser.add_argument("--img_complete", help="Image complete or patches exp?", default=True, action="store_true")
    parser.add_argument("--tag_exp", type=str, default="Exp", help="Tag for wandb")
    
    # Model configs
    parser.add_argument("--model", help="Model to use.", default = "RPA-Unet", choices=["riedNet", "RPA-Unet", "SA-Unet-v1", "SA-Unet-v2"])
    parser.add_argument("--act_out", type=str, default="LeackyReLU", help="Activation out model", choices=["Linear", "ReLU", "Sigmoid", "LeackyReLU"])
    parser.add_argument("--lambda_pixel", type=float, default=100., help="The weight of pixel loss, default = 100")
    parser.add_argument("--lambda_edge", type=float, default=10., help="The weight of pixel loss, default = 100")
    parser.add_argument("--alpha_breast", type=float, default=0.8, help="The weight of breast in Weigthed loss, default = 0.8")
    parser.add_argument("--alpha_background", type=float, default=0.2, help="The weight of background in Weigthed loss, default = 0.2")
    parser.add_argument("--gamma_loss", type=float, default=100., help="The weight of losses, default = 100.")
    parser.add_argument("--gamma", type=float, default=0.0, help="Gamma Self-Attention layer")

    # Dataset params
    parser.add_argument("--projection", type=str, default="CC", help="Projection in which mammograms were taken")
    parser.add_argument("--image_size", type=int, default = 256, help = "Input image size")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--format", type=str, default="tif", help="Image format") # "png"
    
    # Training params
    parser.add_argument("--epoch", type=int, default=0, help="Epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=401, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--weigth_init", type=str, default="normal", choices = ['normal', 'glorot'], help="weights initializer")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints. Default = %(default)s (no save)")
    parser.add_argument("--loss", type=str, default="WeightSum", help="Loss of model", choices=["MAE", "WeightSum", "WeightSumEdgeSobel", "MAEEdgeSobel"])
    

    # Initial configs
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    setup_configs(args)
    run_model(args)


if __name__ == "__main__":
    
    main()
