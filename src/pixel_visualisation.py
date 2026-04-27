import os 
from collections import Counter
import math
from PIL import Image
from pixelfy import Pixelfy, PixelfyOps
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

PX_SIZE = 0.78
DINA_SIZES = {0 : (84.1, 118.9),
              1 : (59.4, 84.1),
              2 : (42.0, 59.4),
              3 : (29.7, 42.0), 
              4 : (21.0, 29.7)
             }

def ax_gridded_imshow(ax, img, scaling=10, mayor={5:'tab:blue',20:'darkblue',48:'tab:red'}):
    width, height = img.size
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = PixelfyOps.scale_up(img, scaling=scaling)
    ax.imshow(img)
    for v in [i*scaling for i in range( width+2)]: ax.axvline(v, alpha=0.3)
    for h in [i*scaling for i in range(height+2)]: ax.axhline(h, alpha=0.3)
    for k,color in mayor.items():
        if k < width:
            for v in [i*scaling*k for i in range(int( width/k)+1)]: ax.axvline(v, color=color)
        if k < height:
            for h in [i*scaling*k for i in range(int(height/k)+1)]: ax.axhline(h, color=color)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def separate_colors(img: Image.Image, background=(0, 0, 0, 0)):
    """
    Split an image into one image per exact RGB color.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image.
    background : tuple
        RGBA color to use for non-matching pixels in each output image.
        Default is fully transparent.

    Returns
    -------
    dict
        {(R, G, B): PIL.Image.Image, ...}
    """
    rgba = img.convert("RGBA") # in case of transparency
    pixels = list(rgba.get_flattened_data())
    unique_colors = sorted({(r, g, b) for (r, g, b, a) in pixels if a != 0}) # but ignoring alpha for the list of colors

    result = {}
    width, height = rgba.size

    for color in unique_colors:
        out = Image.new("RGBA", (width, height), background)
        out_pixels = []

        for px in pixels:
            r, g, b, a = px
            if a != 0 and (r, g, b) == color:
                out_pixels.append((r, g, b, a))
            else:
                out_pixels.append(background)

        out.putdata(out_pixels)
        result[color] = out
    return result

def nearest_css4_color_name(rgb):
    """
    Return the nearest CSS4 named color from matplotlib.
    
    Example:
        (200, 200, 200) -> 'silver' or 'lightgray'
    """
    CSS4_RGB = {
        name: tuple(int(round(c * 255)) for c in mcolors.to_rgb(hex_value))
        for name, hex_value in mcolors.CSS4_COLORS.items()
    }
    r, g, b = rgb
    best_name = None
    best_dist = float("inf")

    for name, (cr, cg, cb) in CSS4_RGB.items():
        # 3D euclidian distance 
        dist = math.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name

def color_histogram(img: Image.Image, flatten_bg=None) -> dict:
    """
    Return a dict mapping (R,G,B) -> count.
    If flatten_bg=(r,g,b) is provided and the image has alpha, it is composited over that color first.
    """
    im = img
    if flatten_bg is not None:
        im = Image.alpha_composite(
            Image.new('RGBA', im.size, (*flatten_bg, 255)),
            im.convert('RGBA')
        ).convert('RGB')
    else:
        if im.mode != 'RGB':
            im = im.convert('RGB')
    return dict(Counter(im.get_flattened_data()))

def hexcolor(rgb):
    """takes color as rgb integer tuple and returns hexcode string"""
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"

def fig_seperated_colors(img, scaling=10, fig_width=4):
    channels = separate_colors(img)
    count = color_histogram(img)
    fig, axs = plt.subplots(1, len(channels)+1, figsize=((len(channels)+1)*fig_width,fig_width*1.5))
    axs[0] = ax_gridded_imshow(axs[0], img)
    axs[0].set_title(f'size: {img.size[1]*PX_SIZE:.1f} x {img.size[0]*PX_SIZE:.1f} cm')
    for ax, (color, img) in zip(axs[1::], channels.items()):
        # blackify the image (channel):
        r, g, b, a = img.split()
        z = Image.new('L', img.size, 0)   # zero channel
        img = Image.merge('RGBA', (z, z, z, a))
        ax = ax_gridded_imshow(ax, img)
        ax.set_title(f"{nearest_css4_color_name(color)} ({hexcolor(color)}): {(count[color])}px")
    return fig, axs
