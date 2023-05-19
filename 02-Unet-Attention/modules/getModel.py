from torch import nn
from modules.ried_net_module import riedNet
from modules.RPA_Unet_module import RPA_Unet_Module
from modules.SA_Unet_v1_Module import SA_Unet_v1_Module
from modules.SA_Unet_v2_Module import SA_Unet_v2_Module


def get_model(args):
    if(args.act_out == "ReLU"):
        actOut = nn.ReLU()
    elif(args.act_out == "Sigmoid"):
        actOut = nn.Sigmoid()
    elif(args.act_out == "Linear"):
        actOut = None
    
    if(args.model == "riedNet"):
        return riedNet( config = args, actOut = actOut)
    elif(args.model == "RPA-Unet"):
        return RPA_Unet_Module(config = args, actOut = actOut)
    elif(args.model == "SA-Unet-v1"):
        return SA_Unet_v1_Module(config = args, actOut = actOut)
    elif(args.model == "SA-Unet-v2"):
        return SA_Unet_v2_Module(config = args, actOut = actOut)

    
    


    
