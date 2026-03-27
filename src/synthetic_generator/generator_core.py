from PIL import Image, ImageFilter
from . import printed_text_generate, background_generator
from .distorsion_generator import sin, cos, rnd
from typing import Tuple
import numpy as np
import random
class LineGerenator(object):
    def __init__(self):
        super().__init__()

    # def apply_lighting(self, img):
    #     H, W = img.height, img.width

    #     x = np.linspace(0, 1, W)
    #     y = np.linspace(0, 1, H)
    #     xv, yv = np.meshgrid(x, y)

    #     angle = random.uniform(0, 2 * np.pi)
    #     grad = np.cos(angle) * xv + np.sin(angle) * yv
    #     grad = (grad - grad.min()) / (grad.max() - grad.min())

    #     illumination = random.uniform(0.45, 0.75) + grad * random.uniform(0.3, 0.6)
    #     illumination = illumination ** random.uniform(0.8, 1.3)
    #     radial = 1 - 0.3 * ((xv-0.5)**2 + (yv-0.5)**2)
    #     illumination *= radial

    #     img_np = np.array(img, dtype=np.float32)
    #     img_np *= illumination
    #     img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    #     return Image.fromarray(img_np, mode="L")

    def apply_noise(self, image: Image) -> Image:
        img_np = np.array(image).astype(np.float32)
        noise_type = random.choice(["gaussian", "speckle", "none"])
        
        if noise_type == "gaussian":
            sigma = random.uniform(2, 12)
            noise = np.random.normal(0, sigma, img_np.shape)
            img_np += noise
            
        elif noise_type == "speckle":
            prob = random.uniform(0.005, 0.02)
            thres = 1 - prob
            rdn = np.random.random(img_np.shape)
            img_np[rdn < prob] = 0
            img_np[rdn > thres] = 255

        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)

    def generate(
        self,
        text:str,
        font:str,
        size:int,
        margins:Tuple[int, int, int, int],
        space_width: int,
        character_spacing: int,
        fit:bool,
        split_to_word:bool,
        stroke_width:int,
        rotate_angle:float,
        blur:int,
        distorsion:bool = True,
        text_color:str = "#000000",
        stroke_color:str = "#000000",
        orientation:str = "horizontal"
    ):
        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom
        image, mask = printed_text_generate.generate_text(
            text,
            font,
            size,
            space_width,
            character_spacing,
            fit,
            split_to_word,
            stroke_width,
            text_color,
            stroke_color,
            orientation
        )
        angle = random.randint(-rotate_angle, rotate_angle)
        rotated_image = image.rotate(angle, expand=1)
        rotated_mask = mask.rotate(angle, expand=1)

        if distorsion:
            distorsion_mode = random.randint(0,3)
            distorsion_orientation = random.randint(0,2)
            vertical = distorsion_orientation == 0 or distorsion_orientation == 2
            horizontal = distorsion_orientation == 1 or distorsion_orientation == 2
        else:
            distorsion_mode = 0
        
        if distorsion_mode == 0:
            distorted_img = rotated_image
            distorted_mask = rotated_mask
        elif distorsion_mode == 1:
            distorted_img, distorted_mask = sin(rotated_image, rotated_mask, vertical, horizontal)
        elif distorsion_mode == 2:
            distorted_img, distorted_mask = cos(rotated_image, rotated_mask, vertical, horizontal)
        else:
            distorted_img, distorted_mask = rnd(rotated_image, rotated_mask, vertical, horizontal)


        height, width = distorted_img.size
        extend = 0
        background_img = background_generator.generate_paper((width + horizontal_margin + extend, height + vertical_margin + extend))
        background_mask = Image.new(
            "RGB", (width, height ), (0, 0, 0)
        )
        alignment = random.randint(0,2)
        if alignment == 0:
            background_img.paste(distorted_img, (margin_left, margin_top), distorted_img)
            background_mask.paste(distorted_mask, (margin_left, margin_top))
        elif alignment == 1:
            background_img.paste(distorted_img, (int((width + horizontal_margin)/2 - width/2), margin_top), distorted_img)
            background_mask.paste(distorted_mask, (int((width + horizontal_margin)/2 - width/2), margin_top))
        else:
            background_img.paste(distorted_img, (horizontal_margin + extend - margin_right, margin_top), distorted_img)
            background_mask.paste(distorted_mask, (horizontal_margin + extend - margin_right, margin_top))
        
        background_img = background_img.convert("RGB")
        background_mask = background_mask.convert("RGB")

        blur = random.uniform(0, blur)
        filter = ImageFilter.GaussianBlur(blur)
        final_image = background_img.filter(filter)
        final_mask = background_mask.filter(filter)
        
        return final_image, final_mask
    


