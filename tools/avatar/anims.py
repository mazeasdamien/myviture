# Rocketbox body animations (FBX, in-place "motextr_static" set) -> one small GLB per clip.
# Only the Bip01 skeleton and its action are kept; the clip is named after the output file.
import bpy, sys, os
args = sys.argv[sys.argv.index('--')+1:]
src_dir, out_dir, pairs = args[0], args[1], args[2:]
for pair in pairs:
    src, name = pair.split('=')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=os.path.join(src_dir, src))
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    for o in list(bpy.data.objects):
        if o is not arm: bpy.data.objects.remove(o)
    act = arm.animation_data.action
    for a in list(bpy.data.actions):
        if a is not act: bpy.data.actions.remove(a)
    act.name = name
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = int(act.frame_range[0]), int(act.frame_range[1])
    bpy.ops.export_scene.gltf(filepath=os.path.join(out_dir, name + '.glb'), export_format='GLB',
        export_animations=True, export_animation_mode='ACTIONS', export_yup=True, export_skins=False,
        export_morph=False, export_optimize_animation_size=True)
    print('DONE', name, act.frame_range[:], sc.render.fps)
