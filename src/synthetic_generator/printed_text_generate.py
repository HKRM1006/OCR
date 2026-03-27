from typing import Tuple
from PIL import Image, ImageColor, ImageDraw, ImageFont
import random
def _get_height(font: ImageFont.FreeTypeFont, text: str) -> float:
    _, top, _, bottom = font.getbbox(text)
    return abs(top-bottom) + 30

def _generate_horizontal_text(
    text: str,
    font: str,
    font_size: int,
    space_width: int,
    character_spacing: int,
    fit: bool,
    split_to_word: bool,
    stroke_width: int = 0,
    text_color: str = "#000000",
    stroke_color: str = "#000000",
) -> Tuple[Image.Image, Image.Image]:
    image_font = ImageFont.truetype(font, font_size)
    space_width = int(image_font.getlength(" ") * space_width)
    
    if split_to_word:
        splitted_text = []
        for w in text.split(" "):
            splitted_text.append(w)
            splitted_text.append(" ")
        splitted_text.pop()
    else:
        splitted_text = text
    
    part_size = [image_font.getlength(p) if p != " " else space_width for p in splitted_text]
    text_width = int(sum(part_size))
    if not split_to_word:
        text_width += character_spacing * (len(text) - 1)
    text_height = _get_height(image_font, "".join(splitted_text) if split_to_word else splitted_text)

    img = Image.new("RGBA", (text_width, text_height), (0,0,0,0))
    img_mask = Image.new("RGB", (text_width, text_height), (0,0,0))

    img_draw = ImageDraw.Draw(img)
    mask_draw = ImageDraw.Draw(img_mask)

    if text_color:
        fill = ImageColor.getrgb(text_color)
    else:
        fill = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    if stroke_color:
        stroke_fill = ImageColor.getrgb(stroke_color)
    else:
        stroke_fill = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

    for i, p in enumerate(splitted_text):
        img_draw.text(
            (sum(part_size[0:i]) + i * character_spacing * int(not split_to_word), 0),
            p,
            fill=fill,
            font=image_font,
            stroke_fill=stroke_fill,
            stroke_width=stroke_width
        )
        mask_draw.text(
            (sum(part_size[0:i]) + i * character_spacing * int(not split_to_word), 0),
            p,
            fill=(i // (255*255), i // 255, i % 255),
            font=image_font,
            stroke_fill=stroke_fill,
            stroke_width=stroke_width
        )
    
    if fit:
        return img.crop(img.getbbox()), img_mask.crop(img_mask.getbbox())
    else:
        return img, img_mask

def generate_text(
    text: str,
    font: str,
    font_size: int,
    space_width: int,
    character_spacing: int,
    fit: bool,
    split_to_word: bool,
    stroke_width: int = 0,
    text_color: str = "#000000",
    stroke_color: str = "#000000",
    orientation: str = "horizontal"
) -> Tuple[Image.Image, Image.Image]:
    if orientation == "horizontal":
        return _generate_horizontal_text(
            text, 
            font, 
            font_size, 
            space_width, 
            character_spacing,
            fit,
            split_to_word,
            stroke_width,
            text_color,
            stroke_color)
    #Add more later
    else:
        return _generate_horizontal_text(
            text, 
            font, 
            font_size, 
            space_width, 
            character_spacing,
            fit,
            split_to_word,
            stroke_width,
            text_color,
            stroke_color)
        
