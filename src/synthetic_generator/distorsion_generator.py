import math
import random
import numpy as np
from typing import Tuple
from PIL import Image
import cv2
import numpy as np

def apply_distorsion(
    image: Image, mask: Image, vertical: bool, horizontal: bool, max_offset: int, func
) -> Tuple:
    if not vertical and not horizontal:
        return image, mask

    img_arr = np.array(image.convert("RGBA"))
    mask_arr = np.array(mask.convert("RGB"))
    h, w = img_arr.shape[:2]

    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    if vertical:
        v_offsets = np.array([func(i) for i in range(w)], dtype=np.float32)
        map_y += v_offsets

    if horizontal:
        h_offsets = np.array([func(i) for i in range(h)], dtype=np.float32)
        map_x += h_offsets[:, np.newaxis] 

    distorted_img = cv2.remap(
        img_arr, map_x, map_y, 
        interpolation=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    distorted_mask = cv2.remap(
        mask_arr, map_x, map_y, 
        interpolation=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_REPLICATE
    )

    return (
        Image.fromarray(distorted_img).convert("RGBA"),
        Image.fromarray(distorted_mask).convert("RGB")
    )


def sin(
    image: Image, mask: Image, vertical: bool = False, horizontal: bool = False
) -> Tuple:
    max_offset = int(image.height * 0.04)
    return apply_distorsion(
        image,
        mask,
        vertical,
        horizontal,
        max_offset,
        (lambda x: (math.sin(math.radians(x)) * max_offset)),
    )


def cos(
    image: Image, mask: Image, vertical: bool = False, horizontal: bool = False
) -> Tuple:
    max_offset = int(image.height * 0.04)
    return apply_distorsion(
        image,
        mask,
        vertical,
        horizontal,
        max_offset,
        (lambda x: (math.cos(math.radians(x)) * max_offset)),
    )


def rnd(
    image: Image, mask: Image, vertical: bool = False, horizontal: bool = False
) -> Tuple:
    max_offset = int(image.height * 0.04)
    return apply_distorsion(
        image,
        mask,
        vertical,
        horizontal,
        max_offset,
        (lambda x: random.uniform(-max_offset,max_offset)),
    )