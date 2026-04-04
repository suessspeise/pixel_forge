"""
Taken from https://github.com/philgyford/python-halftone, interface modfied.

As stated there, the bulk of this is taken from this Stack Overflow answer by fraxel:
http://stackoverflow.com/a/10575940/250962
"""
import numpy as np
from PIL import Image, ImageDraw, ImageStat

class Halftone:
    """
    Applies a halftone effect to a PIL Image.

    Usage:
        from PIL import Image
        from halftone import Halftone

        im = Image.open("photo.jpg")
        ht = Halftone(im)
        result = ht.make()                       # Returns a CMYK PIL Image
        result.save("out.jpg")

        result = ht.make(style="grayscale")      # Grayscale

        result, channels = ht.make(return_channels=True)  # With channel images
    """

    ANTIALIAS_SCALE = 4

    def __init__(self, image: Image.Image):
        """
        Args:
            image: A PIL Image object to halftone.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"image must be a PIL Image object, not '{type(image)}'.")
        self.image = image

    def make(
        self,
        sample: int = 10,
        scale: int = 1,
        percentage: float = 0,
        angles: list = None,
        style: str = "color",
        antialias: bool = False,
        return_channels: bool = False,
    ) -> Image.Image:
        """
        Apply the halftone effect and return the resulting PIL Image.

        Args:
            sample:          Sample box size from the original image, in pixels.
            scale:           Max output dot diameter is sample * scale (also the
                             number of possible dot sizes).
            percentage:      How much of the gray component to remove from the CMY
                             channels and place in the K channel (0–100).
            angles:          Screen angles per channel. Must be a list of 4 ints for
                             'color' style, or at least 1 int for 'grayscale'.
                             Defaults to [0, 15, 30, 45].
            style:           'color' or 'grayscale'.
            antialias:       Apply antialiasing to the dots.
            return_channels: If True, return a tuple of (merged_image, channel_list)
                             where channel_list contains the individual channel images.
                             For 'grayscale', channel_list contains the single channel.

        Returns:
            A PIL Image (CMYK for color, L for grayscale), or a tuple of
            (PIL Image, list[PIL Image]) when return_channels=True.
        """
        if angles is None:
            angles = [0, 15, 30, 45]

        self._validate(
            angles=angles,
            antialias=antialias,
            percentage=percentage,
            sample=sample,
            scale=scale,
            style=style,
        )

        if style == "grayscale":
            gray_im = self.image.convert("L")
            channel_images = self._halftone(gray_im, sample, scale, angles[:1], antialias)
            result = channel_images[0]
        else:
            cmyk = self._gcr(percentage)
            channel_images = self._halftone(cmyk, sample, scale, angles, antialias)
            result = Image.merge("CMYK", channel_images)

        if return_channels:
            return result, channel_images
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check(condition: bool, exc_type: type, message: str):
        """Raise exc_type(message) if condition is False."""
        if not condition:
            raise exc_type(message)

    def _validate(self, angles, antialias, percentage, sample, scale, style):
        """Validate all arguments; raises TypeError or ValueError on bad input."""

        # style first — other checks branch on its value.
        self._check(
            style in ("color", "grayscale"),
            ValueError,
            f"style must be 'color' or 'grayscale', not '{style}'.",
        )
        self._check(
            isinstance(angles, list),
            TypeError,
            f"angles must be a list of integers, not '{angles}'.",
        )
        if style == "grayscale":
            self._check(
                len(angles) >= 1,
                ValueError,
                f"angles must contain at least 1 integer for grayscale, but has {len(angles)}.",
            )
        else:
            self._check(
                len(angles) == 4,
                ValueError,
                f"angles must be a list of exactly 4 integers for color, but has {len(angles)}.",
            )
        self._check(
            all(isinstance(a, int) for a in angles),
            ValueError,
            f"All elements of angles must be integers, but got {angles}.",
        )
        self._check(
            isinstance(antialias, bool),
            TypeError,
            f"antialias must be a boolean, not '{antialias}'.",
        )
        self._check(
            isinstance(percentage, (int, float)),
            TypeError,
            f"percentage must be a number, not '{percentage}'.",
        )
        self._check(
            0 <= percentage <= 100,
            ValueError,
            f"percentage must be between 0 and 100, but got {percentage}.",
        )
        self._check(isinstance(sample, int), TypeError,
                    f"sample must be an integer, not '{sample}'.")
        self._check(isinstance(scale, int), TypeError,
                    f"scale must be an integer, not '{scale}'.")

    def _gcr(self, percentage: float) -> Image.Image:
        """
        Gray Component Replacement. Returns a CMYK image with `percentage`
        of the gray component moved from the CMY channels into K.

        Example: percentage=100, (41, 100, 255, 0) → (0, 59, 214, 41)
        """
        cmyk_im = self.image.convert("CMYK")
        if not percentage:
            return cmyk_im

        cmyk = np.array(cmyk_im, dtype=np.float32)   # shape: (H, W, 4)
        c, m, y, k = cmyk[..., 0], cmyk[..., 1], cmyk[..., 2], cmyk[..., 3]

        gray = np.minimum(np.minimum(c, m), y) * (percentage / 100)

        cmyk[..., 0] = c - gray
        cmyk[..., 1] = m - gray
        cmyk[..., 2] = y - gray
        cmyk[..., 3] = gray

        cmyk = np.clip(cmyk, 0, 255).astype(np.uint8)
        return Image.fromarray(cmyk, mode="CMYK")

    def _halftone(
        self,
        source: Image.Image,
        sample: int,
        scale: int,
        angles: list,
        antialias: bool,
    ) -> list:
        """
        Core halftone routine. Returns a list of channel images (PIL 'L' mode).

        Each channel is rotated by its corresponding angle, sampled at `sample`
        pixel intervals, and a dot is drawn whose diameter is proportional to
        the mean brightness of each sample box.
        """
        if antialias:
            scale = scale * self.ANTIALIAS_SCALE

        channels = source.split()
        dots = []

        for channel, angle in zip(channels, angles):
            channel = channel.rotate(angle, expand=True)
            size = channel.size[0] * scale, channel.size[1] * scale
            half_tone = Image.new("L", size)
            draw = ImageDraw.Draw(half_tone)

            for x in range(0, channel.size[0], sample):
                for y in range(0, channel.size[1], sample):
                    box = channel.crop((x, y, x + sample, y + sample))
                    mean = ImageStat.Stat(box).mean[0]
                    diameter = (mean / 255) ** 0.5
                    box_size = sample * scale
                    draw_diameter = diameter * box_size

                    box_x, box_y = x * scale, y * scale
                    x1 = box_x + (box_size - draw_diameter) / 2
                    y1 = box_y + (box_size - draw_diameter) / 2
                    x2 = x1 + draw_diameter
                    y2 = y1 + draw_diameter

                    draw.ellipse([(x1, y1), (x2, y2)], fill=255)

            half_tone = half_tone.rotate(-angle, expand=True)
            width_half, height_half = half_tone.size

            xx1 = (width_half - self.image.size[0] * scale) / 2
            yy1 = (height_half - self.image.size[1] * scale) / 2
            xx2 = xx1 + self.image.size[0] * scale
            yy2 = yy1 + self.image.size[1] * scale

            half_tone = half_tone.crop((xx1, yy1, xx2, yy2))

            if antialias:
                w = int((xx2 - xx1) / self.ANTIALIAS_SCALE)
                h = int((yy2 - yy1) / self.ANTIALIAS_SCALE)
                half_tone = half_tone.resize((w, h), resample=Image.LANCZOS)

            dots.append(half_tone)

        return dots