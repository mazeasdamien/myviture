import sys
from PIL import Image, ImageOps
src, dst = sys.argv[1], sys.argv[2]
for part in ('body', 'head'):
    c = Image.open(f'{src}/m002_{part}_color.tga'); print(part, c.size, c.mode)
    c.convert('RGB').save(f'{dst}/{part}_color.jpg', quality=90)
    Image.open(f'{src}/m002_{part}_normal.tga').convert('RGB').save(f'{dst}/{part}_normal.jpg', quality=92)
    s = ImageOps.grayscale(Image.open(f'{src}/m002_{part}_specular.tga').convert('RGB')).resize((1024, 1024), Image.LANCZOS)
    r = s.point(lambda v: int(255 * (0.92 - 0.5 * v / 255)))  # shiny where specular is high
    Image.merge('RGB', (Image.new('L', r.size, 255), r, Image.new('L', r.size, 0))).save(f'{dst}/{part}_rough.jpg', quality=90)
o = Image.open(f'{src}/m002_opacity_color.tga'); print('opacity', o.size, o.mode)
o.convert('RGBA').resize((1024, 1024), Image.LANCZOS).save(f'{dst}/opacity.png', optimize=True)
