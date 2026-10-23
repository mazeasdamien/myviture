"""
Bakes the Rocketbox human (public/models/human.glb, MIT, Microsoft) and its idle clip
(public/models/anims/idle.glb) for the native app: native/models/human.bin and the three
base-colour textures native/models/human_<material>.<jpg|png>.

human.bin, little endian:
  'MVHU', u32 version (1)
  u32 nodes N; per node: i32 parent (-1 root), 3 f32 rest translation, 4 f32 rest rotation
      (x y z w), 3 f32 rest scale, 32-byte name (zero padded)
  u32 joints J; per joint: u32 node, 16 f32 inverse bind matrix (row-major)
  16 f32: the mesh node's global matrix inverse at rest (row-major)
  u32 parts P; per part: u32 vertices, u32 indices, u32 alpha mode (0 opaque, 1 mask),
      32-byte texture file name; vertices: 3 f32 position, 3 f32 normal, 2 f32 uv,
      4 u8 joint, 4 f32 weight; indices: u32
  f32 fps, u32 frames F; per frame, per node: 3 f32 translation, 4 f32 rotation, 3 f32 scale
      (the idle clip sampled; nodes it does not animate keep their rest values)
  3 f32: the character's forward direction in model space (heel -> toe, level), then
  f32 the lowest foot height at rest (model units)

    python tools/avatar/bake_native.py
"""
import json
import os
import struct

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC = os.path.join(ROOT, 'public', 'models', 'human.glb')
IDLE = os.path.join(ROOT, 'public', 'models', 'anims', 'idle.glb')
OUT = os.path.join(ROOT, 'native', 'models')
FPS = 30.0

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def load(path):
    b = open(path, 'rb').read()
    jlen = struct.unpack_from('<I', b, 12)[0]
    gl = json.loads(b[20:20 + jlen])
    return gl, b[20 + jlen + 8:]


def accessor(gl, BIN, i):
    a = gl['accessors'][i]
    v = gl['bufferViews'][a['bufferView']]
    dt = np.dtype(COMP[a['componentType']])
    n = NCOMP[a['type']]
    start = v.get('byteOffset', 0) + a.get('byteOffset', 0)
    stride = v.get('byteStride', dt.itemsize * n)
    raw = np.frombuffer(BIN, dtype=np.uint8, count=stride * (a['count'] - 1) + dt.itemsize * n, offset=start)
    out = np.lib.stride_tricks.as_strided(raw, shape=(a['count'], dt.itemsize * n), strides=(stride, 1)).copy()
    out = out.view(dt).reshape(a['count'], n)
    if a.get('normalized') and dt != np.float32:
        out = out.astype(np.float32) / np.iinfo(dt).max
    return out


def trs_matrix(t, r, s):
    x, y, z, w = r
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3] = R * np.array(s)[None, :]
    M[:3, 3] = t
    return M


def slerp(a, b, u):
    a = np.array(a, float); b = np.array(b, float)
    d = float(np.dot(a, b))
    if d < 0: b = -b; d = -d
    if d > 0.9995:
        q = a + u * (b - a)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - u) * th) * a + np.sin(u * th) * b) / np.sin(th)
    return q / np.linalg.norm(q)


gl, BIN = load(SRC)
nodes = gl['nodes']
N = len(nodes)
parent = [-1] * N
for i, nd in enumerate(nodes):
    for c in nd.get('children', []):
        parent[c] = i
rest = []
for nd in nodes:
    rest.append((list(nd.get('translation', [0, 0, 0])), list(nd.get('rotation', [0, 0, 0, 1])), list(nd.get('scale', [1, 1, 1]))))
names = [nd.get('name', '') for nd in nodes]


def globals_of(local):
    G = [None] * N
    def g(i):
        if G[i] is None:
            M = trs_matrix(*local[i])
            G[i] = M if parent[i] < 0 else g(parent[i]) @ M
        return G[i]
    for i in range(N):
        g(i)
    return G


skin = gl['skins'][0]
joints = skin['joints']
ibm = accessor(gl, BIN, skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)  # column-major -> row-major
mesh_node = next(i for i, nd in enumerate(nodes) if 'mesh' in nd)
G0 = globals_of(rest)
mesh_inv = np.linalg.inv(G0[mesh_node])

# The idle clip, by node name.
agl, ABIN = load(IDLE)
anim = agl['animations'][0]
by_name = {n: i for i, n in enumerate(names)}
tracks = []  # (node, path, times, values)
tmax = 0
for ch in anim['channels']:
    tn = agl['nodes'][ch['target']['node']].get('name', '')
    # Bones only, as the web app does: the clip's root (Bip01) carries Blender's own
    # scale and offset, which would make the body 100 times too big.
    if tn not in by_name or by_name[tn] not in joints:
        continue
    smp = anim['samplers'][ch['sampler']]
    times = accessor(agl, ABIN, smp['input'])[:, 0]
    vals = accessor(agl, ABIN, smp['output'])
    tracks.append((by_name[tn], ch['target']['path'], times, vals))
    tmax = max(tmax, float(times[-1]))
F = max(1, int(round(tmax * FPS)))
frames = []
for f in range(F):
    t = f / FPS
    local = [list(map(list, r)) for r in rest]
    for node, path, times, vals in tracks:
        k = int(np.searchsorted(times, t, side='right')) - 1
        if k < 0: v = vals[0]
        elif k >= len(times) - 1: v = vals[-1]
        else:
            u = (t - times[k]) / max(1e-9, times[k + 1] - times[k])
            v = slerp(vals[k], vals[k + 1], u) if path == 'rotation' else vals[k] + u * (vals[k + 1] - vals[k])
        idx = {'translation': 0, 'rotation': 1, 'scale': 2}.get(path)
        if idx is not None: local[node][idx] = list(map(float, v))
    frames.append(local)

# Forward: heel (foot) -> toe, level, at rest.
def pos(G, name): return G[by_name[name]][:3, 3]
fwd = (pos(G0, 'Bip01 L Toe0') - pos(G0, 'Bip01 L Foot')) + (pos(G0, 'Bip01 R Toe0') - pos(G0, 'Bip01 R Foot'))
fwd[1] = 0; fwd /= np.linalg.norm(fwd)
foot_y = min(pos(G0, 'Bip01 L Toe0')[1], pos(G0, 'Bip01 R Toe0')[1])

os.makedirs(OUT, exist_ok=True)
out = bytearray(b'MVHU' + struct.pack('<I', 1))
out += struct.pack('<I', N)
for i in range(N):
    t, r, s = rest[i]
    out += struct.pack('<i3f4f3f', parent[i], *t, *r, *s) + names[i].encode()[:31].ljust(32, b'\0')
out += struct.pack('<I', len(joints))
for j, node in enumerate(joints):
    out += struct.pack('<I', node) + struct.pack('<16f', *ibm[j].reshape(-1))
out += struct.pack('<16f', *mesh_inv.reshape(-1))
prims = gl['meshes'][nodes[mesh_node]['mesh']]['primitives']
out += struct.pack('<I', len(prims))
for p in prims:
    A = p['attributes']
    P = accessor(gl, BIN, A['POSITION']); Nn = accessor(gl, BIN, A['NORMAL']); UV = accessor(gl, BIN, A['TEXCOORD_0'])
    J = accessor(gl, BIN, A['JOINTS_0']).astype(np.uint8); W = accessor(gl, BIN, A['WEIGHTS_0']).astype(np.float32)
    W = W / np.maximum(1e-9, W.sum(1, keepdims=True))
    I = accessor(gl, BIN, p['indices'])[:, 0].astype(np.uint32)
    mat = gl['materials'][p['material']]
    tex = gl['textures'][mat['pbrMetallicRoughness']['baseColorTexture']['index']]
    img = gl['images'][tex['source']]
    ext = 'png' if img.get('mimeType') == 'image/png' else 'jpg'
    fname = 'human_' + mat.get('name', 'part').split('_')[-1] + '.' + ext
    bv = gl['bufferViews'][img['bufferView']]
    open(os.path.join(OUT, fname), 'wb').write(BIN[bv.get('byteOffset', 0):bv.get('byteOffset', 0) + bv['byteLength']])
    alpha = 1 if mat.get('alphaMode') == 'MASK' else 0
    out += struct.pack('<III', len(P), len(I), alpha) + fname.encode().ljust(32, b'\0')
    for k in range(len(P)):
        out += struct.pack('<3f3f2f4B4f', *P[k], *Nn[k], *UV[k], *J[k], *W[k])
    out += I.astype('<u4').tobytes()
    print('part', mat.get('name'), len(P), 'vertices', len(I) // 3, 'triangles', fname)
out += struct.pack('<fI', FPS, F)
for local in frames:
    for t, r, s in local:
        out += struct.pack('<3f4f3f', *t, *r, *s)
out += struct.pack('<3ff', *fwd, foot_y)
open(os.path.join(OUT, 'human.bin'), 'wb').write(out)
print('nodes', N, 'joints', len(joints), 'idle frames', F, 'forward', fwd, 'foot height', foot_y, 'bytes', len(out))
for nm in ['Bip01', 'Bip01 Pelvis', 'Bip01 L Thigh', 'Bip01 L Calf', 'Bip01 L Foot', 'Bip01 L UpperArm', 'Bip01 Head']:
    print(nm, np.round(pos(G0, nm), 3))

G1 = globals_of([tuple(map(list, r)) for r in frames[0]])
print("idle frame 0 head", np.round(pos(G1, "Bip01 Head"), 3), "foot", np.round(pos(G1, "Bip01 L Foot"), 3))
