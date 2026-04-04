from PIL import Image, ImageEnhance
import warnings
try:    import matplotlib
except: warnings.warn('matplotlib not available.')
    
__all__ = ['PixelfyOps', 'Pixelfy']

class PixelfyOps:
    """
    Stateless toolkit of image processing operations for the pixel art effect.

    All methods are static and can be used independently of the :class:`Pixelfy` workflow.
    Constants :attr:`PALETTES`, :attr:`DOWNSCALE_METHODS`, and :attr:`INFER_METHODS` are
    exposed for inspection and use by custom subclasses or external code.

    Example::

        from PIL import Image
        img = Image.open("photo.jpg")
        small = PixelfyOps.shrink(img, resolution=(64, 64))
        result = PixelfyOps.variable_dither(small, strength=0.5)
    """

    PALETTES = {
        # colorpicked from screenshot on https://www.pixlesscamera.com/pixless-camera-mk1
        'default' : [(202, 203, 161), (204, 159, 93), (173, 106, 71) , (139, 63, 73), (85, 50, 68), (81, 83, 98), (100, 119, 125), (143, 159, 146)],
        'mdnight' : [(255, 130, 117), (214, 60, 106), (124, 24, 60), (71, 14, 43), (48, 5, 32), (31, 5, 16), (19, 2, 8)],
        'ammo'    : [(3, 13, 5), (17, 34, 24), (30, 59, 41), (48, 92, 65), (77, 128, 97), (138, 162, 88), (191, 220, 129), (238, 255, 203)],
        'ancient' : [(219, 183, 183), (184, 145, 148), (144, 108, 107), (109, 73, 73), (255, 254, 183), (255, 218, 145), (255, 145, 39), (220, 109, 39), (183, 36, 36), (109, 36, 37), (1, 0, 2)],
        'other'   : [(253, 235, 189), (252, 206, 92), (255, 152, 36), (234, 88, 38), (132, 97, 3), (62, 71, 14), (29, 39, 12), (7, 15, 4)]
    }
    """Named palettes colorpicked from the Pixless Camera Mk1. Each entry is a list of RGB 3-tuples."""

    DOWNSCALE_METHODS = {'nearest':Image.NEAREST, 'bilinear':Image.BILINEAR,'bicubic':Image.BICUBIC, 'lanczos':Image.LANCZOS}
    """Mapping of algorithm name strings to PIL resampling filters used during downscaling."""

    INFER_METHODS = {'mediancut': Image.MEDIANCUT, 'maxcoverage': Image.MAXCOVERAGE, 'fastoctree': Image.FASTOCTREE}
    """Mapping of method name strings to PIL quantization methods used for palette inference."""

    @staticmethod
    def _validate_shrink_algorithm(value):
        """Validate that *value* is a key in :attr:`DOWNSCALE_METHODS`. Returns *value* unchanged."""
        valid = set(PixelfyOps.DOWNSCALE_METHODS.keys())
        if value not in valid:
            raise ValueError(f"shrink_algorithm must be one of {valid}, got '{value}'")
        return value

    @staticmethod
    def _validate_infer_method(value):
        """Validate that *value* is a key in :attr:`INFER_METHODS`. Returns *value* unchanged."""
        if value not in PixelfyOps.INFER_METHODS:
            raise ValueError(f"infer method must be one of {PixelfyOps.INFER_METHODS}, got '{value}'")
        return value

    @staticmethod
    def _validate_image(value):
        """Validate that *value* is a PIL :class:`Image.Image`. Returns *value* unchanged."""
        if not isinstance(value, Image.Image):
            raise TypeError(f"image must be a PIL Image, got {type(value).__name__}")
        return value

    @staticmethod
    def _validate_rgb(value):
        """
        Validate that *value* is a 3-tuple of ints each in the range 0–255.
        Returns *value* unchanged.
        """
        if not isinstance(value, tuple) or len(value) != 3:
            raise TypeError(f"RGB value must be a 3-tuple, got {value!r}")
        if not all(isinstance(v, int) and 0 <= v <= 255 for v in value):
            raise ValueError(f"RGB channels must be ints in 0–255, got {value!r}")
        return value

    @staticmethod
    def _validate_palette(value):
        """
        Validate a palette argument.

        *value* may be:

        - A ``str`` naming a key in :attr:`PALETTES` or :attr:`INFER_METHODS`.
        - A non-empty ``list`` of RGB 3-tuples, each validated by :meth:`_validate_rgb`.

        Returns *value* unchanged.
        """
        if isinstance(value, str):
            valid = set(PixelfyOps.PALETTES.keys()) | PixelfyOps.INFER_METHODS.keys()
            if value not in valid:
                raise ValueError(f"palette string must be one of {valid}, got '{value}'")
        elif isinstance(value, list):
            if len(value) == 0:
                raise ValueError("palette list must not be empty")
            for i, color in enumerate(value):
                try:
                    PixelfyOps._validate_rgb(color)
                except (TypeError, ValueError) as e:
                    raise type(e)(f"palette[{i}]: {e}") from e
        else:
            raise TypeError(f"palette must be a str or list of RGB tuples, got {type(value).__name__}")
        return value

    @staticmethod
    def _validate_saturation(value):
        """Validate that *value* is a non-negative number. Returns it as a ``float``."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"saturation must be a number, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"saturation must be >= 0, got {value}")
        return float(value)

    @staticmethod
    def _validate_dithering(value):
        """Validate that *value* is a number in the range [0, 1]. Returns it as a ``float``."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"dithering must be a number, got {type(value).__name__}")
        if not 0 <= value <= 1:
            raise ValueError(f"dithering must be between 0 and 1, got {value}")
        return float(value)

    @staticmethod
    def _validate_resolution(value):
        """Validate that *value* is a 2-tuple of positive ints. Returns *value* unchanged."""
        if (
            not isinstance(value, tuple) or len(value) != 2 or
            not all(isinstance(d, int) and d > 0 for d in value)
        ):
            raise ValueError(f"resolution must be a tuple of two positive ints, got {value!r}")
        return value

    @staticmethod
    def _validate_upscaling(value):
        """Validate that *value* is a positive int. Returns *value* unchanged."""
        if not isinstance(value, int) or value < 1:
            raise TypeError(f"upscaling must be a positive int, got {value!r}")
        return value

    @staticmethod
    def _palette_image2palette(image):
        """
        Extract the palette from a PIL palette-mode image as a list of RGB 3-tuples.

        :param image: A PIL image in ``'P'`` (palette) mode.
        :returns: List of 256 ``(R, G, B)`` tuples.
        """
        palette_data = image.getpalette()  # returns flat [R,G,B, R,G,B, ...] list of 256*3 values
        return [tuple(palette_data[i:i+3]) for i in range(0, len(palette_data), 3)]

    @staticmethod
    def infer_palette(image, ncolors=8, method='mediancut'):
        """
        Derive a palette from an image using PIL quantization.

        :param image: Source PIL image.
        :param ncolors: Number of colors to extract (default 8).
        :param method: Quantization method — one of ``'mediancut'``, ``'maxcoverage'``,
            ``'fastoctree'`` (default ``'mediancut'``).
        :returns: List of ``(R, G, B)`` tuples representing the inferred palette.
        """
        method = PixelfyOps._validate_infer_method(method)
        return PixelfyOps._palette_image2palette(image.quantize(colors=ncolors, method=PixelfyOps.INFER_METHODS[method]))

    @staticmethod
    def palette_from_hex_file(path):
        """
        Load a palette from a plain-text hex file, as provided by `Lospec <https://lospec.com/palette-list>`_.

        Each line of the file should contain exactly one 6-character hex color code (e.g. ``ff8800``),
        with no leading ``#``.

        :param path: Path to the ``.hex`` palette file.
        :returns: List of ``(R, G, B)`` tuples.
        """
        with open(path) as f:
            return [tuple(int(line.strip()[i:i+2], 16) for i in (0, 2, 4)) for line in f if line.strip()]

    @staticmethod
    def palette_from_mpl(cmap_name, n=8):
        """
        Sample a Matplotlib colormap as a palette.

        Samples *n* evenly spaced colors from the colormap identified by *cmap_name*.
        Requires ``matplotlib`` to be installed.

        :param cmap_name: Name of a Matplotlib colormap (e.g. ``'viridis'``, ``'plasma'``).
        :param n: Number of colors to sample (default 8).
        :returns: List of ``(R, G, B)`` tuples.
        :raises ImportError: If ``matplotlib`` is not installed.
        """
        try: import matplotlib
        except ImportError:
            raise ImportError("matplotlib is required for palette_from_mpl()") from None
        cmap = matplotlib.colormaps[cmap_name]
        return [tuple(int(v * 255) for v in cmap(i / (n - 1))[:3]) for i in range(n)]
            
    @staticmethod
    def build_palette_img(colors):
        """
        Build a PIL palette-mode image from a list of colors.

        Used internally to prepare a quantization target for PIL's ``quantize()``.
        The palette is padded to 256 entries with black if fewer colors are provided.

        :param colors: List of ``(R, G, B)`` tuples (up to 256).
        :returns: A 1×1 PIL image in ``'P'`` mode carrying the given palette.
        """
        pal_img = Image.new("P", (1, 1))
        flat = [v for rgb in colors for v in rgb]
        flat += [0] * (256 * 3 - len(flat))
        pal_img.putpalette(flat)
        return pal_img

    @staticmethod
    def shrink(image, resolution=(128, 256), algorithm='lanczos'):
        """
        Downscale an image to fit within *resolution*, preserving aspect ratio.

        Uses :meth:`Image.thumbnail` internally, so *resolution* is a bounding box,
        not an exact target size.

        :param image: Source PIL image.
        :param resolution: Maximum ``(width, height)`` bounding box (default ``(128, 256)``).
        :param algorithm: Resampling algorithm — one of ``'nearest'``, ``'bilinear'``,
            ``'bicubic'``, ``'lanczos'`` (default ``'lanczos'``).
        :returns: A downscaled copy of the image.
        """
        algorithm = PixelfyOps._validate_shrink_algorithm(algorithm)
        result = image.copy()
        result.thumbnail(resolution, PixelfyOps.DOWNSCALE_METHODS[algorithm])
        return result

    @staticmethod
    def adjust_saturation(image, value=1.0):
        """
        Adjust the color saturation of an image.

        :param image: Source PIL image.
        :param value: Saturation multiplier. ``0`` produces greyscale, ``1`` leaves the
            image unchanged, values above ``1`` oversaturate (default ``1.0``).
        :returns: Saturation-adjusted PIL image.
        """
        value = PixelfyOps._validate_saturation(value)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(value)
        
    @staticmethod
    def variable_dither(image, strength=0.75, palette=None):
        """
        Quantize an image to a palette with variable dithering strength.

        Blends a fully dithered (Floyd-Steinberg) quantization with a flat (no dither)
        quantization according to *strength*, then re-quantizes the blend. This allows
        fine-grained control between the structured look of flat quantization and the
        smoother gradients of full dithering.

        :param image: Source PIL image.
        :param strength: Dithering blend factor in [0, 1]. ``0`` = no dithering,
            ``1`` = full Floyd-Steinberg (default ``0.75``).
        :param palette: Target palette as a list of ``(R, G, B)`` tuples, or a palette
            name string. Defaults to :attr:`PixelfyOps.PALETTES['default'] <PALETTES>`.
        :returns: Quantized PIL image in ``'RGB'`` mode.
        """
        if palette is None: palette = PixelfyOps.PALETTES['default']
        palette = PixelfyOps._validate_palette(palette)
        strength = PixelfyOps._validate_dithering(strength)
        pal_image = PixelfyOps.build_palette_img(palette)
        dith = image.quantize(palette=pal_image, dither=Image.Dither.FLOYDSTEINBERG)
        flat = image.quantize(palette=pal_image, dither=Image.Dither.NONE)
        blended = Image.blend(flat.convert("RGB"), dith.convert("RGB"), strength)
        return blended.quantize(palette=pal_image, dither=Image.Dither.NONE).convert("RGB")

    @staticmethod
    def scale_up(image, scaling=8):
        """
        Upscale an image by an integer factor using nearest-neighbor (box) resampling.

        Nearest-neighbor is intentional here — it preserves the hard pixel edges
        characteristic of pixel art.

        :param image: Source PIL image.
        :param scaling: Integer scale factor (default ``8``).
        :returns: Upscaled PIL image.
        """
        scaling = PixelfyOps._validate_upscaling(scaling)
        return image.resize(tuple(x * scaling for x in image.size), resample=Image.BOX)

    @staticmethod
    def palette_visualisation(colors, swatch_size=42, padding=3, background=(255, 255, 255)):
        """
        Render a palette as a row of color swatches.
    
        :param colors: List of ``(R, G, B)`` tuples to display.
        :param swatch_size: Width and height of each color square in pixels (default ``42``).
        :param padding: Gap in pixels between swatches and around the image border (default ``3``).
        :param background: Background color as an ``(R, G, B)`` tuple (default white).
        :returns: PIL image containing the rendered palette.
        :raises TypeError: If any color or the background is not a valid RGB 3-tuple.
        :raises ValueError: If *colors* is empty or any channel value is out of range.
        """
        if not isinstance(swatch_size, int) or swatch_size < 1:
            raise ValueError(f"swatch_size must be a positive int, got {swatch_size!r}")
        if not isinstance(padding, int) or padding < 0:
            raise ValueError(f"padding must be a non-negative int, got {padding!r}")
    
        PixelfyOps._validate_rgb(background)
        color_list = []
        for i, c in enumerate(colors):
            try:
                PixelfyOps._validate_rgb(c)
            except (TypeError, ValueError) as e:
                raise type(e)(f"colors[{i}]: {e}") from e
            color_list.append(c)
        if not color_list: raise ValueError("colors must not be empty")
    
        cols = len(color_list)
        width = cols * swatch_size + (cols + 1) * padding
        height = swatch_size + 2 * padding
    
        img = Image.new("RGB", (width, height), background)
        for i, color in enumerate(color_list):
            x = padding + i * (swatch_size + padding)
            y = padding
            img.paste(color, (x, y, x + swatch_size, y + swatch_size))
        return img
        
        

class Pixelfy:
    """
    Applies a retro pixel art effect to a PIL Image.

    Bundles the stateless operations in :class:`PixelfyOps` into a configured,
    reusable workflow. Parameters are validated on construction and stored as
    instance attributes, then applied in sequence by :meth:`make`.

    A custom ops class can be injected via the *ops* parameter to override
    individual processing steps without subclassing :class:`Pixelfy` itself.

    This is heavily inspired by the `Pixless Camera project <https://www.pixlesscamera.com/>`_.

    Usage::

        from PIL import Image

        im = Image.open("photo.jpg")
        result = Pixelfy(im).make()
        result.save("out.jpg")
    """

    def __init__(self, 
                 image, 
                 palette='default', 
                 saturation=1.0, 
                 dithering=0.75,
                 resolution=(128, 256), 
                 shrink_algorithm='lanczos',
                 upscaling=8, 
                 ops=PixelfyOps):
        """
        :param image: Source PIL image to process.
        :param palette: Palette to quantize to. Either a name string from
            :attr:`PixelfyOps.PALETTES`, an infer method string from
            :attr:`PixelfyOps.INFER_METHODS`, or a list of ``(R, G, B)`` tuples
            (default ``'default'``).
        :param saturation: Saturation multiplier applied before quantization.
            ``0`` = greyscale, ``1`` = unchanged (default ``1.0``).
        :param dithering: Dithering blend strength in [0, 1]. ``0`` = flat quantization,
            ``1`` = full Floyd-Steinberg (default ``0.75``).
        :param resolution: Maximum ``(width, height)`` bounding box for downscaling
            (default ``(128, 256)``).
        :param shrink_algorithm: Resampling algorithm for downscaling — one of
            ``'nearest'``, ``'bilinear'``, ``'bicubic'``, ``'lanczos'``
            (default ``'lanczos'``).
        :param upscaling: Integer factor by which the processed image is upscaled
            at the end of the pipeline (default ``8``).
        :param ops: Ops class supplying the processing methods. Defaults to
            :class:`PixelfyOps`. Swap this to inject custom behaviour.
        """
        self.ops = ops
        self.image            = self.ops._validate_image(image)
        self.saturation       = self.ops._validate_saturation(saturation)
        self.dithering        = self.ops._validate_dithering(dithering)
        self.resolution       = self.ops._validate_resolution(resolution)
        self.upscaling        = self.ops._validate_upscaling(upscaling)
        self.shrink_algorithm = self.ops._validate_shrink_algorithm(shrink_algorithm)
        self.set_palette(palette)
        
    def set_palette(self, palette):
        """
        Set the active palette, resolving name strings and inferring from the image if needed.

        Can be called after construction to swap the palette without creating a new instance.

        :param palette: A palette name string, an infer method string, or a list of
            ``(R, G, B)`` tuples. See :meth:`PixelfyOps._validate_palette` for accepted values.
        """
        palette = self.ops._validate_palette(palette)
        if isinstance(palette, list):
            self.palette = palette
        elif palette in self.ops.PALETTES:
            self.palette = self.ops.PALETTES[palette]
        elif palette in self.ops.INFER_METHODS:
            self.palette = self.ops.infer_palette(self.image, method=palette)
        else:
            warnings.warn('given palette not known. falling back to default.')
            self.palette = self.ops.PALETTES['default']

    def make(self):
        """
        Run the full pixel art pipeline and return the processed image.

        Steps applied in order:

        1. :meth:`~PixelfyOps.shrink` — downscale to target resolution
        2. :meth:`~PixelfyOps.adjust_saturation` — apply saturation adjustment
        3. :meth:`~PixelfyOps.variable_dither` — quantize to palette with dithering
        4. :meth:`~PixelfyOps.scale_up` — upscale with nearest-neighbor resampling

        :returns: Processed PIL image in ``'RGB'`` mode.
        """
        img = self.ops.shrink(self.image, resolution=self.resolution, algorithm=self.shrink_algorithm)
        img = self.ops.adjust_saturation(img, self.saturation)
        img = self.ops.variable_dither(img, strength=self.dithering, palette=self.palette)
        img = self.ops.scale_up(img, scaling=self.upscaling)
        return img

    def preview_palette(self, swatch_size=42, padding=3, background=(255, 255, 255)):
        """
        Render the active palette as a row of color swatches.
    
        Delegates to :meth:`PixelfyOps.palette_visualisation` using the palette
        currently set on this instance.
    
        :param swatch_size: Width and height of each swatch in pixels (default ``42``).
        :param padding: Gap in pixels between swatches and around the border (default ``3``).
        :param background: Background color as an ``(R, G, B)`` tuple (default white).
        :returns: PIL image containing the rendered palette.
        """
        return self.ops.palette_visualisation(
            self.palette,
            swatch_size=swatch_size,
            padding=padding,
            background=background
        )