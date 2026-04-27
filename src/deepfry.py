from PIL import Image, ImageFilter, ImageOps
import io
import cv2
import numpy as np
import sklearn.cluster

__all__ = ['Deepfryer']

class Deepfryer:
    """
    Stateless toolkit of image processing operations for *deepfrying* effects.

    Example::

        from PIL import Image
        img = Image.open("photo.jpg")
        blurred = Deepfrier.jpeg_compression(img)
    """

    @staticmethod
    def reduce_color_space(img: Image.Image, n_color_bits: int = 1) -> Image.Image:
        """
        Glitchy, cheap color reduction using per-channel bit depth.
        - n_color_bits: 1..8 (per channel). 1 is harsh, 8 is no reduction.
        """
        b = max(1, min(8, int(n_color_bits)))
        return ImageOps.posterize(img.convert("RGB"), b)
    
    @staticmethod
    def posterize(img: Image.Image, levels: int = 8) -> Image.Image:
        """
        Posterize a PIL image by color quantization via K-means clustering.
    
        Parameters:
            img (PIL.Image.Image): Input image.
            levels (int, optional): Number of color clusters/levels to reduce to.
                Higher values preserve more colors. Default is 8.
    
        Returns:
            PIL.Image.Image: Posterized image.
        """
        array = np.asarray(img.convert("RGB"), dtype=np.float32)  # (H, W, 3)
        h, w, c = array.shape
        pixels = array.reshape(-1, c)
        kmeans = sklearn.cluster.KMeans(n_clusters=levels, n_init=10, random_state=0)
        labels = kmeans.fit_predict(pixels)
        centers = np.clip(np.rint(kmeans.cluster_centers_), 0, 255).astype(np.uint8)
        quantized = centers[labels].reshape(h, w, c)
        return Image.fromarray(quantized, mode="RGB")

    @staticmethod    
    def tone_curving(img: Image.Image, gamma: float = 10.0, crush_factor: float = 0.2) -> Image.Image:
        """
        Apply a per-channel tone curve (gamma + midtone crush) to an 8-bit PIL image.
    
        Remaps 8-bit pixel values using a 256-entry lookup table (LUT).
    
        Parameters:
            img: PIL image.
            gamma: Gamma exponent (default 2.0).
            crush_factor: Midtone emphasis in (default 0.2).
    
        Returns:
            PIL.Image.Image: Remapped image
        """
        x = np.arange(256, dtype=np.float32)
        lut = 255.0 * (x / 255.0) ** gamma
        lut = np.clip(lut * (1.0 - crush_factor) + (crush_factor * 128.0), 0, 255).astype(np.uint8)
        lut_list = lut.tolist()
        rgb = img.convert("RGB")
        return rgb.point(lut_list * 3)

    @staticmethod
    def blend_overlay(base: Image.Image, blend: Image.Image) -> Image.Image:
        """
        Overlay-blend two images
    
        - Images must have the same size.
        - If modes differ, both are converted to RGBA; otherwise the original mode is preserved.
        - Overlay is applied per channel (including alpha if present), mirroring the NumPy version.
        """
        if base.size != blend.size: blend = blend.resize(base.size)
        if base.mode == blend.mode: 
            mode = base.mode
        else:
            mode = "RGBA"
            base = base.convert(mode)
            blend = blend.convert(mode)
    
        b = np.asarray(base, dtype=np.float32) / 255.0
        s = np.asarray(blend, dtype=np.float32) / 255.0
    
        out = np.where(b < 0.5, 2.0 * b * s, 1.0 - 2.0 * (1.0 - b) * (1.0 - s))
        out = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode)

    @staticmethod
    def saturation_contrast(img: Image.Image, sat_factor: float = 2.0, clip_limit: float = 4.0) -> Image.Image:
        """
        Boost saturation and apply CLAHE to a PIL image.
    
        - Preserves alpha; operates on color/lightness only. Other modes convert to RGB.
        - sat_factor ≥ 0 scales saturation; clip_limit > 0 controls contrast strength.
    
        Parameters:
            img: PIL image ("L", "RGB", "LA", "RGBA" preferred).
            sat_factor: Saturation multiplier (default 2.0).
            clip_limit: CLAHE clip limit (default 4.0).
    
        Returns:
            PIL.Image.Image: Enhanced image; mode preserved when possible.
        """
        sat_factor = float(max(sat_factor, 0.0))
        clip_limit = float(max(clip_limit, 1e-6))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s.astype(np.float32) * sat_factor, 0, 255).astype(np.uint8)
        v = clahe.apply(v)
        out = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)
        return Image.fromarray(out, "RGB")

    @staticmethod
    def multilayer_overlay(img, function=saturation_contrast, layers=3):
        """
    Apply an effect repeatedly and combine results using Overlay blending.
    
    Parameters
    ----------
    img : PIL.Image.Image or numpy.ndarray
        Input image. The function assumes `function` and `blend_overlay` can handle
        this type and that all intermediate images share the same size and mode/shape.
    function : callable, default saturation_contrast
        A unary callable `f(image) -> image` that takes the current result and
        returns a new image (same size and type) to be blended via Overlay.
    layers : int, default 3
        Number of images in the stack including the base. Must be >= 1 for multiple
        overlay steps to occur. Values < 1 effectively behave like 1 (returns a copy).
    
    Returns
    -------
    PIL.Image.Image or numpy.ndarray
        The final composited image, matching the type/shape/mode of `img`.
    """
        result = img.copy()
        for _ in range(layers - 1):
            layer = function(result)  # Or other effect
            result = Deepfryer.blend_overlay(result, layer)
        return result

    @staticmethod
    def jpeg_compression(img: Image.Image, quality: int = 5) -> Image.Image:
        """
        Repeatedly save/reload as low-quality JPEG to produce classic “deep-fry” artifacts.
    
        Parameters:
            img: PIL image (converted to RGB).
            quality: JPEG quality per cycle (lower = harsher).
    
        Returns:
            PIL.Image.Image: Deep-fried RGB image.
        """
        current = img.convert("RGB")
        with io.BytesIO() as buf:
            current.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).copy()
    
    @staticmethod
    def repeated_compression(img, iterations=20, q_min=2, q_max=10):
        current = img.convert("RGB")
        for _ in range(iterations):
            q = np.random.randint(q_min, q_max + 1)  # jitter quality
            with io.BytesIO() as buf:
                current.save(buf, "JPEG", quality=q, subsampling=2, optimize=False)
                buf.seek(0)
                current = Image.open(buf).copy()
        return current

    @staticmethod
    def unsharp_masking(img: Image.Image, sigma: float = 1.5, amount: float = 3.0, iterations: int = 3) -> Image.Image:
        """
        Apply iterative unsharp masking (per-channel) to enhance edges and halos.
        Preserves alpha when present; other modes convert to RGB.
    
        Parameters:
            img: PIL image ("L", "RGB", "LA", "RGBA" preferred).
            sigma: Gaussian blur sigma for the low-pass filter.
            amount: Sharpening strength (1.0 = mild, >2.0 = aggressive).
            iterations: Number of sharpening passes.
    
        Returns:
            PIL.Image.Image: Sharpened image.
        """
        arr = np.array(img.convert("RGB"))
        cur = arr.astype(np.float32)
        for _ in range(iterations):
            blur = cv2.GaussianBlur(cur, (0, 0), sigmaX=sigma, sigmaY=sigma)
            cur = cv2.addWeighted(cur, 1.0 + amount, blur, -amount, 0.0)
            cur = np.clip(cur, 0, 255)
        return Image.fromarray(cur.astype(np.uint8), "RGB")

    @staticmethod
    def add_noise(img: Image.Image, mean: float = 0.0, sigma: float = 100.0, amount: float = 0.4) -> Image.Image:
        """
        Add Gaussian noise by blending the image with random noise (image*(1-amount) + noise*amount).
        Preserves alpha for LA/RGBA; other modes convert to RGB.
    
        Parameters:
            img: PIL image ("L", "RGB", "LA", "RGBA" preferred).
            mean: Noise mean (same units as 8-bit pixel values).
            sigma: Noise stddev.
            amount: Blend factor (0=noise off, 1=pure noise).
    
        Returns:
            PIL.Image.Image: Noisy image.
        """
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
    
        cur = arr.astype(np.float32)
        n = np.random.normal(mean, sigma, size=arr.shape).astype(np.float32)
        out = cur * (1.0 - amount) + n * amount
        out = np.clip(out, 0, 255).astype(np.uint8)
    
        return Image.fromarray(out, "RGB")
    
    @staticmethod
    def blur(img, method='blur'):
        if method == 'minfilter':
            return img.filter(ImageFilter.MinFilter(5))
        else:
            return img.filter(ImageFilter.BLUR)

    @staticmethod
    def add_blob(
        image: Image.Image,
        x: int,
        y: int,
        radius: int,
        zoom_factor: float = 7,
        strength: float = 1.0,
        blend_mode: str = "smooth",
        feather_exponent: float = 2.0,
        interpolation: str = "bicubic",
    ) -> Image.Image:
        """
        Apply a localized spherical bulge (fisheye) distortion to a circular region.
    
        Acts as a convex lens placed over the image: pixels near the centre are
        sampled from a zoomed-in source coordinate, while pixels toward the edge
        of the blob radius transition smoothly back to their undistorted position.
        The distortion reaches zero exactly at ``radius`` so there is no visible seam.
    
        Parameters
        ----------
        image : PIL.Image.Image
            Source image to distort. Any mode supported by Pillow (``"RGB"``,
            ``"RGBA"``, ``"L"``, …). The original object is not modified.
        x : int
            Horizontal centre of the blob in pixels (0 = left edge).
        y : int
            Vertical centre of the blob in pixels (0 = top edge).
        radius : int
            Radius of the affected circular region in pixels. Must be > 0.
        zoom_factor : float, optional
            Magnification applied at the exact centre of the blob. Values > 1
            produce a convex / zoom-in bulge; values in ``(0, 1)`` produce a
            concave / pinch effect. Must be > 0. Default is ``1.5``.
        strength : float, optional
            Overall intensity of the bulge effect in the range ``[0.0, 1.0]``.
            ``0.0`` returns the image unchanged; ``1.0`` applies the full
            distortion as defined by ``zoom_factor``. Default is ``1.0``.
        blend_mode : {"smooth", "linear", "cosine"}, optional
            Shape of the radial falloff curve that transitions between the
            distorted centre and the undistorted surroundings:
    
            - ``"smooth"``  – smoothstep (cubic Hermite); C¹-continuous, no edge.
            - ``"linear"``  – simple linear ramp; faster but may show a faint ring.
            - ``"cosine"``  – cosine ease; softer than linear, slightly harder than smooth.
    
            Default is ``"smooth"``.
        feather_exponent : float, optional
            Exponent applied to the normalised radial distance before the falloff
            function. Higher values concentrate the distortion toward the centre,
            leaving more of the blob periphery undistorted. Must be > 0.
            Default is ``2.0`` (quadratic, i.e. a spherical profile).
    
        Returns
        -------
        PIL.Image.Image
            A new image of the same size and mode as ``image`` with the bulge
            distortion applied to the specified circular region.
    
        Raises
        ------
        ValueError
            If ``zoom_factor`` <= 0, ``radius`` <= 0, ``strength`` is outside
            ``[0, 1]``, or ``blend_mode`` / ``interpolation`` receive an
            unsupported value.
        TypeError
            If ``image`` is not a ``PIL.Image.Image`` instance.
    
        Notes
        -----
        The distortion is implemented as an **inverse warp**: for every output
        pixel inside the bounding circle, the algorithm computes the source
        coordinate via:
    
        .. math::
    
            r_{\\text{norm}} = \\frac{r}{R}, \\quad
            \\alpha(r) = \\text{falloff}\\!\\left(r_{\\text{norm}}^{p}\\right), \\quad
            r_{\\text{src}} = r \\cdot \\left(1 - \\alpha(r)\\right)
                            + \\frac{r}{z} \\cdot \\alpha(r)
    
        where :math:`r` is the Euclidean distance from the blob centre, :math:`R`
        the blob radius, :math:`p` the ``feather_exponent``, :math:`z` the
        ``zoom_factor``, and :math:`\\alpha` the chosen ``blend_mode`` falloff.
        Pixels outside the blob radius are copied verbatim.
    
        Examples
        --------
        >>> from PIL import Image
        >>> img = Image.open("photo.jpg")
        >>> result = add_blob(img, x=120, y=80, radius=60, zoom_factor=2.0, strength=0.8)
        >>> result.save("distorted.jpg")
        """
        # --- input validation ---------------------------------------------------
        if not isinstance(image, Image.Image):
            raise TypeError(f"image must be a PIL.Image.Image, got {type(image)}")
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")
        if zoom_factor <= 0:
            raise ValueError(f"zoom_factor must be > 0, got {zoom_factor}")
        if not (0.0 <= strength <= 1.0):
            raise ValueError(f"strength must be in [0, 1], got {strength}")
        if blend_mode not in ("smooth", "linear", "cosine"):
            raise ValueError(f"blend_mode must be 'smooth', 'linear', or 'cosine', got {repr(blend_mode)}")
    
        # short-circuit: zero strength means no change
        if strength == 0.0:
            return image.copy()
    
        _interp_map = {
            "bicubic":  Image.BICUBIC,
            "bilinear": Image.BILINEAR,
            "nearest":  Image.NEAREST,
        }
        pil_interp = _interp_map[interpolation]
    
        W, H = image.size
    
        # --- build pixel coordinate grids ---------------------------------------
        # We only process pixels inside the bounding box of the circle for speed.
        x0 = max(0, x - radius)
        y0 = max(0, y - radius)
        x1 = min(W, x + radius)
        y1 = min(H, y + radius)
    
        # Absolute pixel coordinates in the bounding box
        cols = np.arange(x0, x1, dtype=np.float64)   # shape (bw,)
        rows = np.arange(y0, y1, dtype=np.float64)   # shape (bh,)
        px, py = np.meshgrid(cols, rows)              # shape (bh, bw)
    
        # Offset from blob centre
        dx = px - x
        dy = py - y
        r  = np.sqrt(dx**2 + dy**2)                  # Euclidean distance
    
        # Mask: only pixels strictly inside the radius
        inside = r < radius
    
        # --- falloff / alpha computation ----------------------------------------
        r_norm = np.where(inside, r / radius, 1.0)   # normalised [0, 1]
        t = np.clip(r_norm ** feather_exponent, 0.0, 1.0)
    
        if blend_mode == "smooth":
            alpha = 1.0 - (3 * t**2 - 2 * t**3)     # smoothstep, 1 at centre → 0 at edge
        elif blend_mode == "linear":
            alpha = 1.0 - t
        else:  # cosine
            alpha = 0.5 * (1.0 + np.cos(np.pi * t))
    
        alpha *= strength                             # apply master strength
    
        # --- inverse warp -------------------------------------------------------
        # At each output pixel, compute where to sample in the *source* image.
        # r_src = lerp(r, r/zoom_factor, alpha)  →  centre is zoomed in
        with np.errstate(divide="ignore", invalid="ignore"):
            unit_x = np.where(r > 0, dx / r, 0.0)
            unit_y = np.where(r > 0, dy / r, 0.0)
    
        r_src = r * (1.0 - alpha) + (r / zoom_factor) * alpha
    
        src_x = np.where(inside, x + unit_x * r_src, px)
        src_y = np.where(inside, y + unit_y * r_src, py)
    
        # Clamp source coordinates to image bounds
        src_x = np.clip(src_x, 0, W - 1)
        src_y = np.clip(src_y, 0, H - 1)
    
        # --- sample source image ------------------------------------------------
        # Build a flat mesh of (src_x, src_y) coords for PIL's transform.
        # PIL's MESH transform maps output tiles to source quads; instead we use
        # getdata/putdata via scipy for full per-pixel control, falling back to a
        # manual bilinear/nearest implementation to avoid adding a hard dependency.
    
        src_array = np.array(image, dtype=np.float64)  # (H, W) or (H, W, C)
        result_array = src_array.copy()
    
        bh, bw = src_y.shape
    
        # Flatten for vectorised sampling
        flat_sx = src_x.ravel()
        flat_sy = src_y.ravel()
    
        # interpolation
        x0f = np.floor(flat_sx).astype(int)
        y0f = np.floor(flat_sy).astype(int)
        x1f = np.clip(x0f + 1, 0, W - 1)
        y1f = np.clip(y0f + 1, 0, H - 1)
        x0f = np.clip(x0f, 0, W - 1)
        y0f = np.clip(y0f, 0, H - 1)
    
        wx = (flat_sx - np.floor(flat_sx))[..., np.newaxis] if src_array.ndim == 3 else (flat_sx - np.floor(flat_sx))
        wy = (flat_sy - np.floor(flat_sy))[..., np.newaxis] if src_array.ndim == 3 else (flat_sy - np.floor(flat_sy))
    
        top    = src_array[y0f, x0f] * (1 - wx) + src_array[y0f, x1f] * wx
        bottom = src_array[y1f, x0f] * (1 - wx) + src_array[y1f, x1f] * wx
        sampled = top * (1 - wy) + bottom * wy
    
        # Write sampled values back into the bounding-box region
        flat_mask = inside.ravel()
        patch = result_array[y0:y1, x0:x1]
        patch_flat = patch.reshape(-1, *patch.shape[2:]) if src_array.ndim == 3 else patch.ravel()
        patch_flat[flat_mask] = sampled[flat_mask]
        if src_array.ndim == 3:
            result_array[y0:y1, x0:x1] = patch_flat.reshape(bh, bw, src_array.shape[2])
        else:
            result_array[y0:y1, x0:x1] = patch_flat.reshape(bh, bw)
    
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)
        return Image.fromarray(result_array, mode=image.mode)

    @staticmethod
    def add_glow_point(
        image: Image.Image,
        x: int,
        y: int,
        color: tuple[int, int, int] = (255, 60, 20),
        radius: int = 10,
        intensity: float = 30,
        core_radius: int = 0.2,
        core_brightness: float = 1.0,
        num_layers: int = 8,
        darken_background: float = 0.0,
        desaturate_background: float = 0.0,
        streak_angle: float = None,
        streak_length: float = 0.15,
        streak_intensity: float = 0.1,
    ) -> Image.Image:
        """
        Add a colored bloom/glow at (x, y) with optional white-hot core and anamorphic streaks.
        
        This composites a soft, multi-scale Gaussian glow over the input image. You can
        optionally desaturate or darken the background to increase perceived contrast,
        add a white-hot core, and draw lens-like streaks through the glow center.
        
        Parameters
        ----------
        image : PIL.Image.Image
            Base image. Converted to RGB internally.
        x : int
            X coordinate of the glow center (pixels, origin at top-left).
        y : int
            Y coordinate of the glow center (pixels, origin at top-left).
        color : tuple[int, int, int], default (255, 60, 20)
            Glow hue as 8-bit RGB.
        radius : float, default 10
            Overall glow scale in pixels. Controls the Gaussian blur sigmas used per layer.
        intensity : float, default 30
            Linear brightness multiplier for the glow contribution.
        core_radius : float, default 0.2
            Radius (px) of the white-hot core. Set 0 to disable the core.
        core_brightness : float, default 1.0
            How much the core blends toward white: 0 = use `color`, 1 = white-hot.
            Ignored if `core_radius` is 0.
        num_layers : int, default 8
            Number of Gaussian blur layers (increasing sigma). Higher is smoother but slower.
        darken_background : float in [0, 1], default 0.0
            Multiplies the base image before compositing (1.0 makes the base fully black).
        desaturate_background : float in [0, 1], default 0.0
            Blends the base image toward luma (grayscale) before compositing.
        streak_angle : float or None, default None
            If provided (degrees), draws two opposite thin streaks through (x, y)
            at this angle before blurring, so they also bloom. None disables streaks.
        streak_length : float, default 0.15
            Streak half-length as a fraction of max(image width, image height).
        streak_intensity : float, default 0.1
            Relative brightness of the streak seed (before layering and intensity).
        
        Returns
        -------
        PIL.Image.Image
            New RGB image with the glow composited.
        
        Raises
        ------
        ValueError
            If any parameter is out of range:
            - radius <= 0
            - core_radius < 0
            - intensity < 0
            - core_brightness < 0
            - darken_background or desaturate_background not in [0, 1]
        
        Notes
        -----
        - The glow is built by seeding a colored core and optional streaks, then
          blurring across `num_layers` with increasing sigma and weights 1/(k+1);
          the sum is normalized by the harmonic number for stable brightness.
        - The result is tinted toward `color` per channel and re-normalized to
          compensate for hue bias, then scaled by `intensity` and added to the base.
        - If (x, y) lies outside the image, the core may not render; streak segments
          that intersect the image can still appear.
        """
        if radius <= 0:                               raise ValueError(f"radius must be > 0, got {radius}")
        if core_radius < 0:                           raise ValueError(f"core_radius must be >= 0, got {core_radius}")
        if intensity < 0:                             raise ValueError(f"intensity must be >= 0, got {intensity}")
        if core_brightness < 0:                       raise ValueError(f"core_brightness must be >= 0, got {core_brightness}")
        if not (0.0 <= darken_background <= 1.0):     raise ValueError(f"darken_background must be in [0, 1], got {darken_background}")
        if not (0.0 <= desaturate_background <= 1.0): raise ValueError(f"desaturate_background must be in [0, 1], got {desaturate_background}")
    
        base = image.convert("RGB")
        W, H = base.size
        base_f = np.array(base, dtype=np.float64)  # (H, W, 3)
    
        if desaturate_background > 0.0:
            # ITU-R BT.601 luma weights
            luma = (0.299 * base_f[..., 0]
                  + 0.587 * base_f[..., 1]
                  + 0.114 * base_f[..., 2])[..., np.newaxis]
            base_f = base_f * (1.0 - desaturate_background) + luma * desaturate_background
        if darken_background > 0.0: base_f = base_f * (1.0 - darken_background)
    
        
        seed = np.zeros((H, W, 3), dtype=np.float64) # A black RGB image with the hard core painted at (x, y).
    
        if core_radius > 0 and core_brightness > 0.0:
            # Rasterise a filled circle for the hard core.
            ys, xs = np.ogrid[:H, :W]
            dist = np.sqrt((xs - x) ** 2 + (ys - y) ** 2)
            core_mask = dist <= core_radius
            # Core colour: lerp from glow colour (edge) to white (centre)
            for ch in range(3):
                # White-hot centre: blend glow colour toward 255
                core_color_ch = color[ch] + (255 - color[ch]) * core_brightness
                seed[..., ch] = np.where(core_mask, core_color_ch, 0.0)
    
        # --- build streak lines (before blurring so they bloom too) -------------
        if not streak_angle is None:
            angle_rad = np.radians(streak_angle)
            dx = np.cos(angle_rad)
            dy = np.sin(angle_rad)
    
            max_len = int(max(W, H) * streak_length)
    
            # Paint two thin lines (±direction) with falloff along length
            for sign in (1.0, -1.0):
                for step in range(max_len):
                    px = int(round(x + sign * dx * step))
                    py = int(round(y + sign * dy * step))
                    if not (0 <= px < W and 0 <= py < H):
                        break
                    # Perpendicular softening: paint 3 pixels wide
                    falloff = (1.0 - step / max_len) ** 2
                    for perp in (-1, 0, 1):
                        perp_weight = 1.0 if perp == 0 else 0.3
                        ppx = int(round(px - sign * dy * perp))
                        ppy = int(round(py + sign * dx * perp))
                        if 0 <= ppx < W and 0 <= ppy < H:
                            for ch in range(3):
                                seed[ppy, ppx, ch] = max(
                                    seed[ppy, ppx, ch],
                                    color[ch] * falloff * perp_weight * streak_intensity
                                )
    
        glow = np.zeros((H, W, 3), dtype=np.float64)
        for k in range(num_layers):
            sigma = (radius / num_layers) * (k + 1)
            weight = 1.0 / (k + 1)
            # Blur each channel independently via PIL's GaussianBlur
            seed_img = Image.fromarray(np.clip(seed, 0, 255).astype(np.uint8), mode="RGB")
            blurred  = seed_img.filter(ImageFilter.GaussianBlur(radius=sigma))
            layer    = np.array(blurred, dtype=np.float64)
            glow += layer * weight
    
        # Normalise layers (sum of weights = H_n, the n-th harmonic number)
        harmonic = sum(1.0 / (k + 1) for k in range(num_layers))
        glow /= harmonic  # now in roughly [0, 255] range before intensity
    
        # Apply colour tint (glow map is already coloured from the seed,
        # but we normalise per-channel to reinforce the hue)
        color_norm = np.array(color, dtype=np.float64) / 255.0
        glow *= color_norm[np.newaxis, np.newaxis, :]  # tint
        glow /= (color_norm.mean() + 1e-6) / 1.0 # Restore overall brightness lost to tinting
        glow *= intensity
        
        result_f = base_f + glow
        result   = np.clip(result_f, 0, 255).astype(np.uint8)
        return Image.fromarray(result, mode="RGB")

    @staticmethod
    def add_chromatic_aberration(image: Image.Image, channel_weights: tuple[float, float, float] = (10,0,-10), angle: float = 0.0,) -> Image.Image:
        """
        Apply a glitch-style chromatic aberration by shifting RGB channels with wrap-around edges.
    
        Each channel (R, G, B) is translated by an integer offset derived from its
        weight and the given direction.
    
        Parameters
        ----------
        image : PIL.Image.Image
            Source image. Converted internally to "RGB"; the original is not modified.
        channel_weights : tuple[float, float, float], optional
            Per-channel shift magnitudes in pixels along the given angle: (w_R, w_G, w_B).
            Positive values move in the angle direction; negative values move opposite.
            Default (10, 0, -10) yields R right, G anchored, B left when angle=0.
        angle : float, optional
            Shift direction in degrees, counter-clockwise from the positive x-axis
            (0 = right, 90 = up).
    
        Returns
        -------
        PIL.Image.Image
            A new "RGB" image with the wrapped channel shifts applied.
        """
        theta = np.radians(angle)
        ux =  np.cos(theta)
        uy = -np.sin(theta)
        
        channels = np.array(image.convert("RGB"), dtype=np.uint8)  # (H, W, 3)
        H, W = channels.shape[:2]
        result = np.zeros_like(channels)
        
        for ch_idx, weight in enumerate(channel_weights):
            raw_dx = weight * ux
            raw_dy = weight * uy
            dx = int(round(raw_dx))   # columns  (positive → right)
            dy = int(round(raw_dy))   # rows     (positive → down)
            plane = channels[..., ch_idx]  # (H, W)
            shifted = np.roll(np.roll(plane, dy, axis=0), dx, axis=1)
            result[..., ch_idx] = shifted
        
        return Image.fromarray(result, mode="RGB")