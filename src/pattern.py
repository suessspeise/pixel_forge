"""
patterns.py — generative 32×32 greyscale PIL.Image patterns
Each generator returns mode 'L', size (32, 32), values 0–255.
"""

import math
import random
from PIL import Image, ImageOps

__all__ = [ 'PlasmaGenerator',
            'VoronoiGenerator',
            'WaveGenerator',
            'NoiseGenerator',
            'MazeGenerator',
            'LissajousGenerator',
            'TruchetGenerator',
            'ReactionDiffusionGenerator',
            'GameOfLifeGenerator',
            'DLAGenerator',
            'ChladniGenerator']

def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))

def to_byte(v: float) -> int:
    return int(clamp(v) * 255)

class Generator:
    SIZE = 32

    def pixel(self, x: int, y: int) -> int:
        """Return a greyscale value in [0, 255]."""
        raise NotImplementedError

    def generate(self) -> Image.Image:
        img = Image.new("L", (self.SIZE, self.SIZE))
        img.putdata([self.pixel(x, y)
                     for y in range(self.SIZE)
                     for x in range(self.SIZE)])
        return img

    def show_scaled(self, scale: int = 8) -> Image.Image:
        return self.generate().resize(
            (self.SIZE * scale, self.SIZE * scale),
            resample=Image.NEAREST,
        )

class PlasmaGenerator(Generator):
    """
    Sine-wave interference pattern.

    Creates a pattern by summing several simple sine waves that travel in 
    different directions and then interfe with each other. It emulates the
    [classic plasma effect from 1990s](https://en.wikipedia.org/wiki/Plasma_effect)
    """

    def __init__(self, freqs=(0.4, 0.3, 0.5), seed: int | None = None):
        self.freqs = freqs
        self.phase = random.Random(seed).uniform(0, math.tau) if seed is not None else 0.0

    def pixel(self, x, y):
        f1, f2, f3 = self.freqs
        v = (math.sin(x * f1 + self.phase)
             + math.sin(y * f2)
             + math.sin((x + y) * f3)
             + math.sin(math.hypot(x - 16, y - 16) * 0.5))
        return to_byte((v + 4) / 8)

class VoronoiGenerator(Generator):
    """
    Cell-boundary ridges via nearest-neighbour distance difference.

    [wiki](https://en.wikipedia.org/wiki/Voronoi_diagram)
    """

    def __init__(self, n_seeds: int = 8, seed: int | None = None):
        rng = random.Random(seed)
        self.seeds = [(rng.uniform(0, 31), rng.uniform(0, 31))
                      for _ in range(n_seeds)]

    def pixel(self, x, y):
        dists = sorted(math.hypot(x - px, y - py) for px, py in self.seeds)
        ridge = (dists[1] - dists[0]) / 8
        return to_byte(clamp(ridge))

class WaveGenerator(Generator):
    """Product of two orthogonal cosine waves."""

    def __init__(self, freq_x: float = 0.5, freq_y: float = 0.4,
                 phase_x: float = 0.0, phase_y: float = 0.0):
        self.fx, self.fy = freq_x, freq_y
        self.px, self.py = phase_x, phase_y

    def pixel(self, x, y):
        u = math.cos(x * self.fx + self.px)
        v = math.cos(y * self.fy + self.py)
        return to_byte((u * v + 1) / 2)

class NoiseGenerator(Generator):
    """Two-octave value noise with cosine interpolation. No external deps."""

    def __init__(self, scale: float = 8.0, octaves: int = 2,
                 seed: int | None = None):
        self.scale = scale
        self.octaves = octaves
        rng = random.Random(seed)
        self._lat = [[rng.random() for _ in range(64)] for _ in range(64)]

    def _sample(self, fx: float, fy: float) -> float:
        x0, y0 = int(fx) % 64, int(fy) % 64
        x1, y1 = (x0 + 1) % 64, (y0 + 1) % 64
        tx = (1 - math.cos((fx - int(fx)) * math.pi)) * 0.5
        ty = (1 - math.cos((fy - int(fy)) * math.pi)) * 0.5
        l = self._lat
        return (l[y0][x0] * (1 - tx) * (1 - ty)
                + l[y0][x1] * tx       * (1 - ty)
                + l[y1][x0] * (1 - tx) * ty
                + l[y1][x1] * tx       * ty)

    def pixel(self, x, y):
        v, amp, freq = 0.0, 1.0, 1.0
        for _ in range(self.octaves):
            v += amp * self._sample(x / self.scale * freq,
                                    y / self.scale * freq)
            amp *= 0.5
            freq *= 2
        return to_byte(v / 1.5)

class MazeGenerator(Generator):
    """
    Recursive DFS backtracking on a 16×16 cell grid.
    Passages are white (255), walls are black (0).
    Naturally binary, but returned as mode 'L' for pipeline consistency.
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed

    def generate(self) -> Image.Image:
        CELLS = 16
        grid = [[0] * self.SIZE for _ in range(self.SIZE)]
        visited = [[False] * CELLS for _ in range(CELLS)]
        rng = random.Random(self.seed)

        def carve(cx, cy):
            visited[cy][cx] = True
            grid[cy * 2][cx * 2] = 255
            dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            rng.shuffle(dirs)
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < CELLS and 0 <= ny < CELLS and not visited[ny][nx]:
                    grid[cy * 2 + dy][cx * 2 + dx] = 255
                    carve(nx, ny)

        carve(0, 0)
        img = Image.new("L", (self.SIZE, self.SIZE))
        img.putdata([grid[y][x]
                     for y in range(self.SIZE)
                     for x in range(self.SIZE)])
        return img

    def pixel(self, x, y):
        pass  # generate() is overridden directly

    
class LissajousGenerator(Generator):
    """
    Rasterises a Lissajous curve as black on white.
    Each sample increments the nearest single pixel (no bilinear spread),
    then the accumulator is gamma-compressed and inverted so the background
    is white and the curve is a sharp dark line.
    """

    def __init__(self, a: int = 3, b: int = 4, delta: float = math.pi / 4,
                 samples: int = 12000, gamma: float = 0.3):
        self.a = a
        self.b = b
        self.delta = delta
        self.samples = samples
        self.gamma = gamma   # <1 compresses highlights, making thin lines darker
        self._grid: list[list[float]] | None = None

    def _build(self):
        S = self.SIZE
        acc = [[0.0] * S for _ in range(S)]
        for i in range(self.samples):
            t = i / self.samples * math.tau
            cx = (math.sin(self.a * t + self.delta) + 1) / 2 * (S - 1)
            cy = (math.cos(self.b * t) + 1) / 2 * (S - 1)
            x, y = round(cx), round(cy)
            if 0 <= x < S and 0 <= y < S:
                acc[y][x] += 1.0
        peak = max(acc[y][x] for y in range(S) for x in range(S)) or 1.0
        # normalise → gamma compress → invert
        self._grid = [
            [1.0 - (acc[y][x] / peak) ** self.gamma for x in range(S)]
            for y in range(S)
        ]

    def generate(self) -> Image.Image:
        self._build()
        S = self.SIZE
        img = Image.new("L", (S, S))
        img.putdata([to_byte(self._grid[y][x])
                     for y in range(S) for x in range(S)])
        return img

    def pixel(self, x, y):
        if self._grid is None:
            self._build()
        return to_byte(self._grid[y][x])


class TruchetGenerator(Generator):
    """
    Each cell gets a randomly chosen tile from two styles, controlled by
    the `style` parameter:

        'diagonal' — two triangles separated by a diagonal; orientation
                     0 = top-left/bottom-right, 1 = top-right/bottom-left.
        'arc'      — two quarter-circle arcs connecting adjacent edge
                     midpoints; orientation 0 = top-bottom pair,
                     1 = left-right pair. Produces the classic Smith (1987)
                     flowing curve pattern.

    Orientations are drawn from a seeded RNG so the distribution is
    genuinely random rather than hash-uniform.
    """

    def __init__(self, cell_size: int = 4, style: str = "arc",
                 seed: int | None = None):
        self.cell_size = cell_size
        self.style = style
        self.seed = seed
        self._orientations: dict[tuple[int, int], int] | None = None

    def _build(self):
        S = self.SIZE
        cs = self.cell_size
        cols = math.ceil(S / cs)
        rows = math.ceil(S / cs)
        rng = random.Random(self.seed)
        self._orientations = {
            (cx, cy): rng.randint(0, 1)
            for cy in range(rows)
            for cx in range(cols)
        }

    def _arc_value(self, lx, ly, cs, orientation) -> int:
        """
        Inside a cell of size cs, return 255 (white) or 0 (black) for
        the arc-style Truchet tile.  The arc is approximated analytically:
        orientation 0 connects top-mid↔left-mid and bottom-mid↔right-mid,
        orientation 1 connects top-mid↔right-mid and bottom-mid↔left-mid.
        A pixel is white if it is closer to its nearer arc than `line_r`.
        """
        mid = (cs - 1) / 2
        line_r = cs * 0.18          # half-thickness of the arc line

        # fractional coords centred on the cell
        fx, fy = lx - mid, ly - mid

        if orientation == 0:
            # arc 1: quarter-circle centred on top-left corner (0, 0) of cell
            # arc 2: quarter-circle centred on bottom-right corner (cs-1, cs-1)
            d1 = abs(math.hypot(lx, ly) - mid)
            d2 = abs(math.hypot(lx - (cs - 1), ly - (cs - 1)) - mid)
        else:
            # arc 1: centred on top-right corner
            # arc 2: centred on bottom-left corner
            d1 = abs(math.hypot(lx - (cs - 1), ly) - mid)
            d2 = abs(math.hypot(lx, ly - (cs - 1)) - mid)

        return 0 if min(d1, d2) < line_r else 255

    def _diagonal_value(self, lx, ly, cs, orientation) -> int:
        mid = (cs - 1) / 2
        if orientation == 0:
            return 0 if lx < ly else 255
        else:
            return 0 if lx + ly < cs - 1 else 255

    def pixel(self, x, y):
        if self._orientations is None:
            self._build()
        cs = self.cell_size
        cx, cy = x // cs, y // cs
        lx, ly = x % cs, y % cs
        orientation = self._orientations.get((cx, cy), 0)
        if self.style == "arc":
            return self._arc_value(lx, ly, cs, orientation)
        else:
            return self._diagonal_value(lx, ly, cs, orientation)


class ReactionDiffusionGenerator(Generator):
    """
    Gray-Scott reaction-diffusion system run for `steps` iterations starting
    from a small central seed perturbation. Returns the V-chemical concentration
    mapped to greyscale. Parameter presets (feed, kill) control the morphology:

        spots:   feed=0.035, kill=0.065
        stripes: feed=0.060, kill=0.062
        worms:   feed=0.078, kill=0.061
        holes:   feed=0.039, kill=0.058
    """

    def __init__(self, feed: float = 0.055, kill: float = 0.062,
                 steps: int = 2000, du: float = 0.16, dv: float = 0.08,
                 dt: float = 1.0, seed: int | None = None):
        self.feed = feed
        self.kill = kill
        self.steps = steps
        self.du = du
        self.dv = dv
        self.dt = dt
        self.seed = seed
        self._grid: list[list[float]] | None = None

    def _laplacian(self, grid, x, y, S):
        """5-point stencil with wrap-around."""
        return (grid[(y - 1) % S][x] + grid[(y + 1) % S][x] +
                grid[y][(x - 1) % S] + grid[y][(x + 1) % S] -
                4 * grid[y][x])

    def _build(self):
        S = self.SIZE
        rng = random.Random(self.seed)
        U = [[1.0] * S for _ in range(S)]
        V = [[0.0] * S for _ in range(S)]
        # seed a small noisy region near the centre
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                x, y = (S // 2 + dx) % S, (S // 2 + dy) % S
                V[y][x] = 0.5 + rng.uniform(-0.1, 0.1)
                U[y][x] = 0.25 + rng.uniform(-0.1, 0.1)

        for _ in range(self.steps):
            U2 = [row[:] for row in U]
            V2 = [row[:] for row in V]
            for y in range(S):
                for x in range(S):
                    u, v = U[y][x], V[y][x]
                    uvv = u * v * v
                    lu = self._laplacian(U, x, y, S)
                    lv = self._laplacian(V, x, y, S)
                    U2[y][x] = u + (self.du * lu - uvv + self.feed * (1 - u)) * self.dt
                    V2[y][x] = v + (self.dv * lv + uvv - (self.kill + self.feed) * v) * self.dt
            U, V = U2, V2

        lo = min(V[y][x] for y in range(S) for x in range(S))
        hi = max(V[y][x] for y in range(S) for x in range(S))
        span = hi - lo or 1.0
        self._grid = [[(V[y][x] - lo) / span for x in range(S)] for y in range(S)]

    def generate(self) -> Image.Image:
        self._build()
        S = self.SIZE
        img = Image.new("L", (S, S))
        img.putdata([to_byte(self._grid[y][x])
                     for y in range(S) for x in range(S)])
        return  ImageOps.invert(img)

    def pixel(self, x, y):
        if self._grid is None:
            self._build()
        return to_byte(self._grid[y][x])


class GameOfLifeGenerator(Generator):
    """
    Runs Conway's Game of Life for `steps` generations from a random initial
    state with the given `density`. Returns the final state as a binary image
    (white = alive). The greyscale mode is preserved for pipeline consistency.
    """

    def __init__(self, steps: int = 30, density: float = 0.45,
                 seed: int | None = None):
        self.steps = steps
        self.density = density
        self.seed = seed
        self._state: list[list[int]] | None = None

    def _build(self):
        S = self.SIZE
        rng = random.Random(self.seed)
        grid = [[1 if rng.random() < self.density else 0
                 for _ in range(S)] for _ in range(S)]
        for _ in range(self.steps):
            next_grid = [[0] * S for _ in range(S)]
            for y in range(S):
                for x in range(S):
                    neighbours = sum(
                        grid[(y + dy) % S][(x + dx) % S]
                        for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                        if (dx, dy) != (0, 0)
                    )
                    alive = grid[y][x]
                    next_grid[y][x] = int(
                        (alive and neighbours in (2, 3)) or
                        (not alive and neighbours == 3)
                    )
            grid = next_grid
        self._state = grid

    def generate(self) -> Image.Image:
        self._build()
        S = self.SIZE
        img = Image.new("L", (S, S))
        img.putdata([self._state[y][x] * 255
                     for y in range(S) for x in range(S)])
        return ImageOps.invert(img)

    def pixel(self, x, y):
        if self._state is None:
            self._build()
        return self._state[y][x] * 255


class DLAGenerator(Generator):
    """
    Particles random-walk from the border until they contact the growing
    cluster, then freeze in place. Produces branching, dendritic structures.

    `seed_mode`:
        'centre'  — single seed pixel at (16, 16)
        'bottom'  — full bottom row as seed (grows upward like a crystal floor)
        'cross'   — seed cross at centre

    `max_particles` controls density; more particles = denser, slower.
    """

    def __init__(self, max_particles: int = 300, seed_mode: str = "centre",
                 seed: int | None = None):
        self.max_particles = max_particles
        self.seed_mode = seed_mode
        self.seed = seed
        self._grid: list[list[int]] | None = None

    def _build(self):
        S = self.SIZE
        rng = random.Random(self.seed)
        grid = [[0] * S for _ in range(S)]

        if self.seed_mode == "bottom":
            for x in range(S):
                grid[S - 1][x] = 1
        elif self.seed_mode == "cross":
            for i in range(S):
                grid[S // 2][i] = 1
                grid[i][S // 2] = 1
        else:
            grid[S // 2][S // 2] = 1

        def is_adjacent(x, y):
            for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < S and 0 <= ny < S and grid[ny][nx]:
                    return True
            return False

        for _ in range(self.max_particles):
            # spawn on a random border pixel
            edge = rng.randint(0, 3)
            if edge == 0:   x, y = rng.randint(0, S-1), 0
            elif edge == 1: x, y = rng.randint(0, S-1), S-1
            elif edge == 2: x, y = 0, rng.randint(0, S-1)
            else:           x, y = S-1, rng.randint(0, S-1)

            for _ in range(S * S * 4):         # max walk steps
                if is_adjacent(x, y):
                    grid[y][x] = 1
                    break
                dx, dy = rng.choice([(-1,0),(1,0),(0,-1),(0,1)])
                x = max(0, min(S-1, x + dx))
                y = max(0, min(S-1, y + dy))
            else:
                continue

        self._grid = grid

    def generate(self) -> Image.Image:
        self._build()
        S = self.SIZE
        img = Image.new("L", (S, S))
        img.putdata([self._grid[y][x] * 255
                     for y in range(S) for x in range(S)])
        return ImageOps.invert(img)

    def pixel(self, x, y):
        if self._grid is None:
            self._build()
        return self._grid[y][x] * 255


class ChladniGenerator(Generator):
    """
    Nodal lines of a vibrating square plate:
        f(x, y) = cos(m·π·x) · cos(n·π·y) - cos(n·π·x) · cos(m·π·y)

    Pixels near the zero-contour (|f| < threshold) are white (sand collects
    there on a real plate); the rest are black.

    (m, n) pairs to try: (1,2), (2,3), (3,4), (2,5), (3,5), (4,7)
    """

    def __init__(self, m: int = 4, n: int = 3, threshold: float = 0.15):
        self.m = m
        self.n = n
        self.threshold = threshold

    def pixel(self, x, y):
        S = self.SIZE
        # normalise coords to [0, 1]
        nx = x / (S - 1)
        ny = y / (S - 1)
        v = (math.cos(self.m * math.pi * nx) * math.cos(self.n * math.pi * ny)
             - math.cos(self.n * math.pi * nx) * math.cos(self.m * math.pi * ny))
        return 255 if abs(v) < self.threshold else 0

    def generate(self):
        return ImageOps.invert(super().generate())
