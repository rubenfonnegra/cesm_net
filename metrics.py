

import os
import cv2
import numpy as np
import seaborn as sns
from PIL import Image 
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')

# Structural Similarity Measure
from sklearn.metrics import confusion_matrix
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from skimage.morphology import square, dilation



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
    p = psnr(np.squeeze(im_true), np.squeeze(im_pred))
    s = ssim(np.squeeze(im_true), np.squeeze(im_pred))
    m = mae (np.squeeze(im_true), np.squeeze(im_pred))
    return m, s, p
