import bpy, sys
fbx, tex, out = sys.argv[sys.argv.index('--')+1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx)
for a in list(bpy.data.actions): bpy.data.actions.remove(a)
for o in bpy.data.objects:
    if o.animation_data: o.animation_data_clear()
mesh = next(o for o in bpy.data.objects if o.type == 'MESH')
mesh.name = 'Body'
# keep ARKit (AK_) and visemes (AA_VI_) only
for kb in list(mesh.data.shape_keys.key_blocks)[1:]:
    if not kb.name.startswith(('AK_', 'AA_VI_')):
        mesh.shape_key_remove(kb)
for kb in mesh.data.shape_keys.key_blocks[1:]:
    kb.name = kb.name.split('_', 2)[2] if kb.name.startswith('AK_') else 'viseme_' + kb.name.split('_', 3)[3]
print('KEPT', [k.name for k in mesh.data.shape_keys.key_blocks[1:]])


def widen_shins(o):
    """Bare shins -> straight trouser legs (outfit.py paints them as cloth).

    The shin vertices (body faces whose UVs fall in the shin islands, below the knee)
    are pushed away from each leg's axis to at least a straight profile: 6.2 cm at 20 cm
    up to 7.6 cm at 44 cm, then back towards the ankle inside the shoe; above the hem
    (47 cm) they stay put, inside the shorts, which are not round higher up. The
    flared, uneven hem of the shorts is made round at 7.8 cm, fading out by 62 cm, so the
    shorts end 2 mm outside the shin and the leg goes straight past the knee. Same offset
    on every shape key, so the face keys stay relative.
    """
    import math
    from mathutils import Vector
    me = o.data; uv = me.uv_layers.active.data; M = o.matrix_world
    body = [i for i, m in enumerate(me.materials) if m.name.endswith('body')][0]
    shin, hem = set(), set()
    for p in me.polygons:
        if p.material_index != body: continue
        for l, vi in zip(p.loop_indices, p.vertices):
            u, v = uv[l].uv
            x, y = u * 2048, (1 - v) * 2048
            z = (M @ me.vertices[vi].co).z
            if min(x, 2048 - x) < 640 and 505 <= y < 960 and z < 0.6: shin.add(vi)
            elif min(x, 2048 - x) < 640 and y < 505 and z < 0.62: hem.add(vi)
    world = {i: M @ me.vertices[i].co for i in shin}
    to_local = M.inverted().to_3x3()
    moved = 0

    def move(i, dx, dy, r, target):
        delta = to_local @ Vector((dx / r * (target - r), dy / r * (target - r), 0))
        for kb in me.shape_keys.key_blocks:
            kb.data[i].co += delta
        me.vertices[i].co += delta

    for side in (1, -1):
        pts = [(i, p) for i, p in world.items() if p.x * side > 0]
        # Leg axis: least-squares line x(z), y(z) through the shin vertices.
        n = len(pts); zs = [p.z for _, p in pts]; zm = sum(zs) / n
        var = sum((z - zm) ** 2 for z in zs)
        def fit(c):
            cm = sum(c(p) for _, p in pts) / n
            k = sum((c(p) - cm) * (p.z - zm) for _, p in pts) / var
            return lambda z: cm + k * (z - zm)
        ax, ay = fit(lambda p: p.x), fit(lambda p: p.y)
        for i, p in pts:
            dx, dy = p.x - ax(p.z), p.y - ay(p.z)
            r = math.hypot(dx, dy)
            if r < 1e-4: continue
            if p.z >= 0.47: continue
            if p.z >= 0.2: target = max(r, 0.062 + (0.076 - 0.062) * min(1, (p.z - 0.2) / 0.24))
            else: target = max(r, r * 1.1 + (0.062 - r * 1.1) * max(0, (p.z - 0.12) / 0.08))
            if target <= r: continue
            move(i, dx, dy, r, target)
            moved += 1
        for i in hem:
            p = M @ me.vertices[i].co
            if p.x * side <= 0: continue
            dx, dy = p.x - ax(p.z), p.y - ay(p.z)
            r = math.hypot(dx, dy)
            if r < 1e-4: continue
            w = min(1, max(0, (0.62 - p.z) / 0.12))  # full at the hem, nothing from 62 cm up
            move(i, dx, dy, r, r + (0.078 - r) * w)
            moved += 1
    print('SHINS', len(shin), 'HEM', len(hem), 'moved', moved)


widen_shins(mesh)

def img(name, colour):
    i = bpy.data.images.load(f'{tex}/{name}')
    i.colorspace_settings.name = 'sRGB' if colour else 'Non-Color'
    return i

def build(mat, part):
    nt = mat.node_tree; nt.nodes.clear()
    outn = nt.nodes.new('ShaderNodeOutputMaterial'); bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new(bsdf.outputs[0], outn.inputs[0])
    bsdf.inputs['Metallic'].default_value = 0
    if part == 'opacity':
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = img('opacity.png', True)
        nt.links.new(t.outputs['Color'], bsdf.inputs['Base Color'])
        r = nt.nodes.new('ShaderNodeMath'); r.operation = 'ROUND'  # exporter reads this as alphaMode MASK
        nt.links.new(t.outputs['Alpha'], r.inputs[0]); nt.links.new(r.outputs[0], bsdf.inputs['Alpha'])
        bsdf.inputs['Roughness'].default_value = 0.7
        # no specular: at grazing angles the Fresnel sheen turned the hair cards into grey sheets
        bsdf.inputs['Specular IOR Level'].default_value = 0
        mat.use_backface_culling = False
        return
    c = nt.nodes.new('ShaderNodeTexImage'); c.image = img(f'{part}_color.jpg', True)
    nt.links.new(c.outputs['Color'], bsdf.inputs['Base Color'])
    n = nt.nodes.new('ShaderNodeTexImage'); n.image = img(f'{part}_normal.jpg', False)
    nm = nt.nodes.new('ShaderNodeNormalMap'); nt.links.new(n.outputs['Color'], nm.inputs['Color']); nt.links.new(nm.outputs[0], bsdf.inputs['Normal'])
    r = nt.nodes.new('ShaderNodeTexImage'); r.image = img(f'{part}_rough.jpg', False)
    sep = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(r.outputs['Color'], sep.inputs[0]); nt.links.new(sep.outputs['Green'], bsdf.inputs['Roughness'])
    mat.use_backface_culling = True

for m in mesh.data.materials:
    build(m, m.name.split('_', 1)[1])
bpy.ops.export_scene.gltf(filepath=out, export_format='GLB', export_animations=False, export_morph=True,
    export_morph_normal=False, export_skins=True, export_image_format='AUTO', export_yup=True, export_apply=False)
