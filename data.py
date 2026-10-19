"""Deterministic offline world of contours; train/test seeds are disjoint.

This synthetic corpus is a mechanics smoke test, NOT proof on natural images.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def contour_image(seed):
    rng = np.random.default_rng(seed)
    im = Image.new("L", (32, 32), 0)
    d = ImageDraw.Draw(im)
    family = int(rng.integers(0, 5))
    thick = int(rng.integers(2, 5))
    x = int(rng.integers(6, 11))
    y = int(rng.integers(6, 11))
    w = int(rng.integers(12, 19))
    h = int(rng.integers(12, 19))
    if family == 0:
        d.arc((x, y, x+w, y+h), start=int(rng.integers(-20, 20)), end=int(rng.integers(300, 360)), fill=255, width=thick)
    elif family == 1:
        d.rectangle((x, y, x+w, y+h), outline=255, width=thick)
    elif family == 2:
        d.line([(x, y+h), (x+w//2, y), (x+w, y+h)], fill=255, width=thick, joint="curve")
    elif family == 3:
        d.line([(x, y), (x+w, y+h//2), (x, y+h)], fill=255, width=thick, joint="curve")
    else:
        d.line([(x, y+h), (x+w//3, y+1), (x+2*w//3, y+h-1), (x+w, y)], fill=255, width=thick, joint="curve")
    if rng.random() < .5:
        d.line((x+2, y+h//2, x+w-2, y+h//2), fill=255, width=thick)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(.45)), dtype=np.float32)/255.


def train_set(n=320, seed=0):
    return np.array([contour_image(seed*100000 + i) for i in range(n)])


def held_out(n=12, seed=17):
    return train_set(n, seed)


def damage(img, seed=0, kind="gap"):
    rng = np.random.default_rng(seed)
    mask = np.ones_like(img, bool)
    if kind == "gap":
        start = int(rng.integers(11, 17))
        mask[:, start:start+5] = False
    elif kind == "hole":
        x = int(rng.integers(11, 18))
        y = int(rng.integers(11, 18))
        mask[y:y+8, x:x+8] = False
    else:
        raise ValueError("unknown damage type")
    obs = np.where(mask, img, .5)
    return obs, mask
