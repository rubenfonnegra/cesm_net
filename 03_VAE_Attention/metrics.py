import os
import cv2
import numpy as np
import seaborn as sns
from PIL import Image 
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
import torch




# Structural Similarity Measure
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def mae(im_true, im_pred):
    im_true, im_pred = np.array(im_true).reshape([-1]), np.array(im_pred).reshape([-1])
    return np.mean(np.abs(im_true - im_pred))


def pixel_metrics (im_true, im_pred):
    #
    p = psnr(np.squeeze(im_true), np.squeeze(im_pred), data_range=1)
    s = ssim(np.squeeze(im_true), np.squeeze(im_pred))
    m = mae (np.squeeze(im_true), np.squeeze(im_pred))
    return m, s, p

""" Define Jensen Shannon Distance loss """
class jensen_shannon_distance(torch.nn.Module):
    def __init__(self):
        super(jensen_shannon_distance, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)).log_softmax(-1), q.view(-1, q.size(-1)).log_softmax(-1)
        m = (0.5 * (p + q))
        return 0.5 * (self.kl(m, p) + self.kl(m, q))