# Outfit painted into the Rocketbox m002 body texture: black polo with the Orange logo on
# the left chest, grey trousers instead of the khaki shorts, white sneakers.
# Coordinates are pixels of the 2048x2048 atlas; the right leg mirrors the left one.
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
src, tex = sys.argv[1], sys.argv[2]
col = np.asarray(Image.open(f'{src}/m002_body_color.tga').convert('RGB')).astype(np.float32)
H, W, _ = col.shape
r, g, b = col[..., 0], col[..., 1], col[..., 2]
lum = 0.299 * r + 0.587 * g + 0.114 * b
skin = ((r - b) > 45) & (lum > 95)
bg = col.max(-1) < 12


def rect(y0, y1, x0, x1, mirror=False):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    if mirror: m |= m[:, ::-1]
    return m


def closing(a, size):
    """Grey closing: fills thin dark lines (stripes, seams), keeps the large folds."""
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size)).filter(ImageFilter.GaussianBlur(2))).astype(np.float32)


out = col.copy()

# ---- polo
torso = rect(40, 1750, 668, 1381) | rect(1750, 1805, 700, 1290)  # front, back, collar, side panels, hem
sleeves = rect(890, 1285, 0, 640, mirror=True)
polo = (torso | (sleeves & ~skin)) & ~bg
closed = closing(lum, 17)
shade = np.clip(closed / np.median(closed[polo]), 0.55, 1.3)
knit = np.clip(lum / np.maximum(closed, 1), 0.85, 1.05)  # keep a little fabric grain
base = np.array([30, 30, 32], np.float32)
out[polo] = base[None, :] * (shade[polo] * (0.6 + 0.4 * knit[polo]))[:, None]

# Orange logo: orange square, "orange" in white, bottom left. ~4.3 cm on the chest (1.2 px/mm).
S, k = 52, 8
big = Image.new('RGB', (S * k, S * k), (255, 121, 0))
d = ImageDraw.Draw(big)
font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', int(S * k * 0.215))
d.text((int(S * k * 0.095), int(S * k * 0.9)), 'orange', font=font, fill=(255, 255, 255), anchor='ls')
logo = np.asarray(big.resize((S, S), Image.LANCZOS)).astype(np.float32)
cx, cy = 1136, 1108
ys, xs = slice(cy - S // 2, cy - S // 2 + S), slice(cx - S // 2, cx - S // 2 + S)
out[ys, xs] = logo * (0.88 + 0.12 * shade[ys, xs])[..., None]

# ---- trousers: the shorts recoloured, the bare shins covered with the same cloth
GREY = np.array([112, 114, 118], np.float32)
shorts = rect(0, 505, 0, 640, mirror=True) & ~bg
# Shin: all of it down to the sleeve islands, then only skin next to their curved edge.
shin = (rect(505, 885, 0, 640, mirror=True) & ~bg & ~torso) | (rect(885, 960, 0, 640, mirror=True) & skin)
cloth_shade = np.clip(lum / np.median(lum[shorts]), 0.45, 1.45)
cloth = GREY[None, None, :] * cloth_shade[..., None]  # grey cloth, folds of the shorts
form_all = closing(lum, 41)
for side in (0, 1):
    # Mirrored views put either leg at columns 0..625; writes go through to `cloth`.
    c = cloth[:, ::-1] if side else cloth
    f = form_all[:, ::-1] if side else form_all
    # The shorts body (rows 0..440) stays; its turned-up hem (440..505) and the shin
    # (505..960) get cloth from the middle of the shorts, stretched over that height,
    # cross-faded over 40 rows so the join does not show.
    patch = Image.fromarray(np.clip(c[150:430, 0:575], 0, 255).astype(np.uint8)).resize((640, 960 - 400), Image.BICUBIC)
    form = np.clip(f[400:960, 0:640] / np.median(f[400:960, 0:640]), 0.8, 1.15)  # the leg's baked shading
    filled = np.asarray(patch).astype(np.float32) * form[..., None]
    t = np.clip((np.arange(400, 960) - 400) / 40, 0, 1)[:, None, None]
    c[400:960, 0:640] = c[400:960, 0:640] * (1 - t) + filled * t
trousers = shorts | shin
out[trousers] = cloth[trousers]
# Ankle showing at the shoe opening: same cloth, a shade darker, as if tucked in.
ankle = (rect(1930, 2048, 520, 728) | rect(1930, 2048, 1320, 1560)) & skin
out[ankle] = GREY * 0.8 * np.clip(lum[ankle] / np.median(lum[ankle]), 0.7, 1.2)[:, None]

# ---- white sneakers: canvas, soles and laces brought to white, details kept faintly
shoes = (rect(1400, 2048, 450, 870) | rect(1400, 2048, 1180, 1600)) & ~skin & ~bg & ~torso
w = np.clip(195 + 0.24 * lum, 0, 255)
out[shoes] = w[shoes][:, None] * np.array([1.0, 1.0, 0.985], np.float32)[None, :]

Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(f'{tex}/body_color.jpg', quality=92)

# Matte cloth; sneakers a little smoother.
rough = np.asarray(Image.open(f'{tex}/body_rough_orig.jpg').convert('RGB').resize((W, H))).copy()
rough[..., 1][polo | trousers | ankle] = 220
rough[..., 1][shoes] = 170
rough[ys, xs, 1] = 200
Image.fromarray(rough).resize((1024, 1024), Image.LANCZOS).save(f'{tex}/body_rough.jpg', quality=90)
print('polo', polo.sum(), 'trousers', trousers.sum(), 'shoes', shoes.sum(), 'ankle', ankle.sum())
