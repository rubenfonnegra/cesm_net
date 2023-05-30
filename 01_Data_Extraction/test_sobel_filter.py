import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os, glob
from pathlib import Path

class GradLayer(nn.Module):

    # def __init__(self):
    #     super(GradLayer, self).__init__()
    #     kernel_v = [[0, -1, 0],
    #                 [0, 0, 0],
    #                 [0, 1, 0]]
    #     kernel_h = [[0, 0, 0],
    #                 [-1, 0, 1],
    #                 [0, 0, 0]]
    #     kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
    #     kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
    #     self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
    #     self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[2.0, 4.0, 2.0], 
                    [0.0, 0.0, 0.0],
                    [-2.0, -4.0, -2.0]]
        kernel_h = [[2.0, 0.0, -2.0],
                    [4.0, 0.0, -4.0],
                    [2.0, 0.0, -2.0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x


if __name__ == "__main__":

    
    net = GradLayer()    

    dir_data    = "/home/mirplab/Documents/kevin/01-cesm_net/Data/cesm_images_complete_val"
    dir_save    = f"/home/mirplab/Documents/kevin/01-cesm_net/Data/cesm_images_complete_val/sobel_filter_1"

    os.makedirs( dir_save, exist_ok=True )
    data = glob.glob( os.path.join( dir_data, 'RC/**/*.tif') )

    for im in data:

        img = Image.open( im )
        img = np.array(img)
        img = torch.from_numpy(img[np.newaxis,...])

        img_edge = net(img)
        img_edge = img_edge.numpy()

        fig, axes = plt.subplots(1,2)

        axes[0].imshow(img[0,...], cmap="gray")
        axes[0].set_title("Recombined Image")
        axes[0].set_axis_off()

        axes[1].imshow(img_edge[0,...], cmap="gray")
        axes[1].set_title("Recombined Image Edge Sobel")
        axes[1].set_axis_off()

        image_name = Path(im).stem
        plt.savefig(fname = os.path.join( dir_save, f"{image_name}.png"), dpi=250, format="png")
        print(f"Done: {image_name}")
        plt.close("all")

    print("Done!")


