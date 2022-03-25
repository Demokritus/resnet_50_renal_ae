import torch
from torchvision.utils import save_image
from typing import List
import numpy as np
from PIL import Image, ImageDraw
import os
import torchvision
import torchvision.transforms as transforms



# utility functions
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def make_dir():
    image_dir = 'renal_encoder_images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)


def save_decoded_image_V0(img_out: torch.Tensor,
                          img_src: torch.Tensor,
                          epoch: int, name: List[str],
                          batch_size: int = None,
                          img_size: tuple = (256, 256)) -> None:
    # assert batch_size == img.size(0), 'Batch size inconsistency found in validation set!!'
    if batch_size == None:
        batch_size = img_out.size(0)
    elif batch_size != img_out.size(0):
        raise Exception('Batch size inconsistency in validation set!')

    for idx in range(batch_size):
        img_out = img_out.view(batch_size, 1, *img_size)
        # img_out = img_out[idx, 0, :, :]
        img_src = img_src.view(batch_size, 1, *img_size)
        # img_src = img_src[idx, 0, :, :]
        img = torch.stack([img_src[idx, 0, :, :], img_out[idx, 0, :, :]], dim=2)
        save_image(img, './renal_encoder_images/{0}_epoch_{1}.png'.format(name[idx], epoch))


def save_decoded_image(img_out: torch.Tensor,
                       img_src: torch.Tensor,
                       epoch: int, name: List[str],
                       batch_size: int = None) -> None:
    # IMG_SIZE = (256, 256)
    IMG_SIZE = (512, 512)
    # assert batch_size == img.size(0), 'Batch size inconsistency found in validation set!!'
    if batch_size == None:
        batch_size = img_out.size(0)
    elif batch_size != img_out.size(0):
        raise Exception('Batch size inconsistency in validation set!')

    # img_out = img_out.view(batch_size, 3, *IMG_SIZE)
    # img_src = img_src.view(batch_size, 3, *IMG_SIZE)
    img_out = img_out.view(batch_size, 1, *IMG_SIZE)
    img_src = img_src.view(batch_size, 1, *IMG_SIZE)

    for idx in range(batch_size):
        pic_out = transforms.ToPILImage()(img_out[idx, 0, :, :]).convert("RGB")
        sep = Image.new("RGB", (10, 512), (127, 255, 127))
        pic_src = transforms.ToPILImage()(img_src[idx, 0, :, :]).convert("RGB")
        draw_src = ImageDraw.Draw(pic_src)
        draw_out = ImageDraw.Draw(pic_out)
        draw_src.text((0, 0), "Original image", (127, 255, 127))
        draw_out.text((0, 0), "Output image", (127, 127, 255))
        pic_save = Image.new("RGB", (512 + 512 + 10, 512))
        pic_save.paste(pic_src, (0, 0))
        pic_save.paste(sep, (512, 0))
        pic_save.paste(pic_out, (512 + 10, 0))
        pic_save.save('./renal_encoder_images/{0}_epoch_{1}.png'.format(name[idx], epoch))


def sum_square_diff(A, B):
    assert A.shape == B.shape
    A = A.detach().cpu().numpy()
    B = B.detach().cpu().numpy()
    
    sum_square = 0
    norm_A = 0
    norm_B = 0
    eps = 1e-5

    for i in range(A.shape[2]):
        for j in range(A.shape[3]):
            sum_square += (A[:, 0, i, j] - B[:, 0, i, j]) ** 2
            norm_A += A[:, 0, i, j] ** 2
            norm_B += B[:, 0, i, j] ** 2

    denom = np.sqrt(norm_A * norm_B)
    try:
        sum_square_norm = (sum_square + eps) / (denom + eps)
    except ZeroDivisionError:
        sum_square_norm = None
        print("DIVISION BY ZERO!!!")
    return np.mean(sum_square_norm)


def cross_correlation(A, B):
    assert A.shape == B.shape
    A = A.detach().cpu().numpy()
    B = B.detach().cpu().numpy()

    corr = 0
    norm_A = 0
    norm_B = 0
    eps = 1e-5

    for i in range(A.shape[2]):
        for j in range(A.shape[3]):
            corr += (A[:, 0, i, j] * B[:, 0, i, j]) ** 2
            norm_A += A[:, 0, i, j] ** 2
            norm_B += B[:, 0, i, j] ** 2

    denom = np.sqrt(norm_A * norm_B)
    try:
        corr_norm = (corr + eps) / (denom + eps)
    except ZeroDivisionError:
        corr_norm = None
        print("DIVISION BY ZERO!!!")
    return np.mean(corr_norm)


