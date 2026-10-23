"""
Cuts the two hands out of the Rocketbox human (public/models/human.glb, MIT, Microsoft)
for the native app's realistic hands and forearms: native/models/hand_l.bin, hand_r.bin
and the skin texture native/models/hand_skin.jpg (smoothed: a younger skin).

A vertex belongs to a hand when at least half of its skin weight is on that hand's bones
(Hand, the 15 finger bones and the forearm); weight on other bones (the upper arm) goes to
the forearm. Each file holds, in metres, in the model's bind pose:
  'MVHD', u32 version (2), u32 vertices, u32 indices
  22 x 3 f32: the bind positions of MediaPipe's 21 hand landmarks (wrist, then thumb,
              index, middle, ring, little: base to tip; tips from the farthest vertices),
              then the elbow
  vertices: 3 f32 position, 3 f32 normal, 2 f32 uv, 4 u8 bone, 4 f32 weight
  indices: u16
Bones: 0 Hand, then Finger<f><k> for f = 0..4 (thumb..little), k = 0..2 (base..tip), 16 Forearm.

    python tools/hands/extract_hands.py
"""
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC = os.path.join(ROOT, 'public', 'models', 'human.glb')
OUT = os.path.join(ROOT, 'native', 'models')

b = open(SRC, 'rb').read()
jlen = struct.unpack_from('<I', b, 12)[0]
gl = json.loads(b[20:20 + jlen])
bin_off = 20 + jlen + 8
BIN = b[bin_off:]

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def accessor(i):
    a = gl['accessors'][i]
    v = gl['bufferViews'][a['bufferView']]
    dt = np.dtype(COMP[a['componentType']])
    n = NCOMP[a['type']]
    start = v.get('byteOffset', 0) + a.get('byteOffset', 0)
    stride = v.get('byteStride', dt.itemsize * n)
    raw = np.frombuffer(BIN, dtype=np.uint8, count=stride * (a['count'] - 1) + dt.itemsize * n, offset=start)
    out = np.lib.stride_tricks.as_strided(raw, shape=(a['count'], dt.itemsize * n), strides=(stride, 1)).copy()
    return out.view(dt).reshape(a['count'], n)


skin = gl['skins'][0]
joints = skin['joints']
names = [gl['nodes'][j].get('name', '') for j in joints]
ibm = accessor(skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)  # column-major -> row-major
bind = np.linalg.inv(ibm)  # bone -> model, at bind
heads = bind[:, :3, 3]

# The mesh and its bind matrices are in metres (the body is 1.58 m tall); only the node
# offsets are in centimetres.
UNIT = 1.0

prims = gl['meshes'][0]['primitives']
body = [p for p in prims if p.get('material') == 0][0]  # m002_body: arms and hands
P = accessor(body['attributes']['POSITION']).astype(np.float64)
N = accessor(body['attributes']['NORMAL']).astype(np.float64)
UV = accessor(body['attributes']['TEXCOORD_0']).astype(np.float64)
J = accessor(body['attributes']['JOINTS_0']).astype(np.int64)
W = accessor(body['attributes']['WEIGHTS_0']).astype(np.float64)
IDX = accessor(body['indices']).reshape(-1, 3).astype(np.int64)


def one(side):
    S = 'L' if side == 'l' else 'R'
    bones = ['Bip01 %s Hand' % S] + ['Bip01 %s Finger%d%s' % (S, f, '' if k == 0 else str(k)) for f in range(5) for k in range(3)] + ['Bip01 %s Forearm' % S]
    bi = [names.index(n) for n in bones]
    local = {g: k for k, g in enumerate(bi)}
    share = np.zeros(len(P))
    for c in range(4):
        share += np.where(np.isin(J[:, c], bi), W[:, c], 0)
    keep = share >= 0.5
    tris = IDX[keep[IDX].all(1)]
    used = np.unique(tris)
    remap = -np.ones(len(P), dtype=np.int64)
    remap[used] = np.arange(len(used))
    # Bones and weights: other bones' weight onto Hand, 4 per vertex, normalised.
    vb, vw = np.zeros((len(used), 4), np.uint8), np.zeros((len(used), 4), np.float32)
    for n_, vi in enumerate(used):
        acc = {}
        for c in range(4):
            if W[vi, c] <= 0:
                continue
            k = local.get(int(J[vi, c]), 16)  # the upper arm's share: the forearm
            acc[k] = acc.get(k, 0) + W[vi, c]
        items = sorted(acc.items(), key=lambda t: -t[1])[:4]
        tot = sum(w for _, w in items)
        for c, (k, w) in enumerate(items):
            vb[n_, c] = k
            vw[n_, c] = w / tot
    pos = P[used] * UNIT
    # Landmarks: the bone heads, and each finger's tip as its last bone's farthest vertex.
    lm = [heads[bi[0]] * UNIT]
    for f in range(5):
        h = [heads[bi[1 + f * 3 + k]] * UNIT for k in range(3)]
        lm += h
        last = 1 + f * 3 + 2
        mine = (vb[:, 0] == last) & (vw[:, 0] > 0.5)
        d = h[2] - h[1]
        d = d / np.linalg.norm(d)
        if mine.any():
            proj = (pos[mine] - h[2]) @ d
            lm.append(h[2] + d * proj.max())
        else:
            lm.append(h[2] + d * np.linalg.norm(h[2] - h[1]) * 0.8)
    lm.append(heads[bi[16]] * UNIT)  # the elbow
    lm = np.array(lm)
    out = open(os.path.join(OUT, 'hand_%s.bin' % side), 'wb')
    out.write(b'MVHD' + struct.pack('<III', 2, len(used), tris.size))
    out.write(lm.astype(np.float32).tobytes())
    for n_ in range(len(used)):
        vi = used[n_]
        out.write(struct.pack('<8f', *pos[n_], *N[vi], *UV[vi]))
        out.write(bytes(vb[n_]))
        out.write(vw[n_].tobytes())
    out.write(remap[tris].astype(np.uint16).tobytes())
    out.close()
    size = np.linalg.norm(lm[9] - lm[0])
    print('hand %s: %d vertices, %d triangles, wrist to middle knuckle %.1f cm, forearm %.1f cm' % (side, len(used), len(tris), size * 100, np.linalg.norm(lm[21] - lm[0]) * 100))


os.makedirs(OUT, exist_ok=True)
one('l')
one('r')
# The body colour texture (the hands' skin).
mat = gl['materials'][body['material']]
img = gl['images'][gl['textures'][mat['pbrMetallicRoughness']['baseColorTexture']['index']]['source']]
v = gl['bufferViews'][img['bufferView']]
# A younger skin: wrinkles and veins smoothed (blur blended back), a little lighter and warmer.
import io
from PIL import Image, ImageEnhance, ImageFilter
src = Image.open(io.BytesIO(BIN[v.get('byteOffset', 0):v.get('byteOffset', 0) + v['byteLength']])).convert('RGB')
smooth = Image.blend(src, src.filter(ImageFilter.GaussianBlur(3)), 0.75)
smooth = ImageEnhance.Brightness(smooth).enhance(1.10)
smooth = ImageEnhance.Color(smooth).enhance(1.08)
smooth.save(os.path.join(OUT, 'hand_skin.jpg'), quality=92)
print('skin texture:', img.get('name'), v['byteLength'] // 1024, 'KB')
