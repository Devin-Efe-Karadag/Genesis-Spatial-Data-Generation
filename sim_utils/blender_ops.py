import bpy
import gc
import re
import random

FALLBACK_MAT_NAME = "FallbackMaterial"
VERTEX_COLOR_MAT_NAME = "VertexColorMaterial"


def _mesh_has_image_textures(obj):
    for mat_slot in obj.material_slots:
        mat = mat_slot.material
        if not mat or not mat.use_nodes or not mat.node_tree:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                return True
    return False


def _mesh_has_vertex_colors(obj):
    color_attrs = getattr(obj.data, "color_attributes", None)
    if color_attrs and len(color_attrs) > 0:
        return True
    vertex_colors = getattr(obj.data, "vertex_colors", None)
    if vertex_colors and len(vertex_colors) > 0:
        return True
    return False


def _ensure_fallback_material(obj, color=None):
    if color is None:
        color = (random.random(), random.random(), random.random(), 1.0)
    mat = bpy.data.materials.new(name=FALLBACK_MAT_NAME)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.6
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def _ensure_vertex_color_material(obj):
    color_attrs = getattr(obj.data, "color_attributes", None)
    vertex_colors = getattr(obj.data, "vertex_colors", None)
    if color_attrs and len(color_attrs) > 0:
        layer_name = color_attrs[0].name
    elif vertex_colors and len(vertex_colors) > 0:
        layer_name = vertex_colors[0].name
    else:
        return False

    mat = bpy.data.materials.new(name=VERTEX_COLOR_MAT_NAME)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    vcol = nodes.new(type="ShaderNodeVertexColor")
    vcol.layer_name = layer_name
    links.new(vcol.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return True


def _bounds_volume(obj):
    dims = obj.dimensions
    return float(dims.x * dims.y * dims.z)


def _remove_small_loose_parts(
    obj, min_volume_ratio=0.005, min_face_ratio=0.01, min_faces=50
):
    if bpy.context.active_object and bpy.context.active_object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")

    mesh_parts = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if len(mesh_parts) <= 1:
        return obj

    metrics = []
    for part in mesh_parts:
        vol = _bounds_volume(part)
        faces = len(part.data.polygons)
        metrics.append((part, vol, faces))

    largest_by_vol = max(metrics, key=lambda x: x[1])
    largest_vol = largest_by_vol[1]
    largest_faces = max(metrics, key=lambda x: x[2])[2]

    face_threshold = max(min_faces, int(largest_faces * min_face_ratio))
    keep = []
    remove = []
    for part, vol, faces in metrics:
        keep_by_vol = largest_vol > 0 and vol >= largest_vol * min_volume_ratio
        keep_by_faces = faces >= face_threshold
        if part == largest_by_vol[0] or keep_by_vol or keep_by_faces:
            keep.append(part)
        else:
            remove.append(part)

    for part in remove:
        bpy.data.objects.remove(part, do_unlink=True)

    if len(keep) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for part in keep:
            part.select_set(True)
        bpy.context.view_layer.objects.active = keep[0]
        bpy.ops.object.join()
        return bpy.context.view_layer.objects.active

    return keep[0]


def _cleanup_mesh(obj, allow_fill_holes=False):
    if bpy.context.active_object and bpy.context.active_object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.merge_by_distance(distance=1.0e-6)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    if allow_fill_holes:
        bpy.ops.mesh.fill_holes(sides=32)
    bpy.ops.object.mode_set(mode="OBJECT")


def cleanup_blender():
    """
    Clean up Blender data blocks to prevent memory accumulation.
    """
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

    for block in bpy.data.actions:
        if block.users == 0:
            bpy.data.actions.remove(block)

    gc.collect()


def merge_glb_submeshes(
    src_file,
    dest_file,
    anim_frame=None,
    anim_action_name=None,
    random_anim_action=False,
):
    """
    Merges meshes, removes tiny loose parts, preserves textures/UVs when present,
    and exports a clean GLB for simulation.
    """
    try:
        if bpy.context.active_object and bpy.context.active_object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        bpy.ops.import_scene.gltf(filepath=str(src_file))

        # --- Animation Baking ---
        if random_anim_action:
            animations = [
                (a.name, int(a.frame_range[0]), int(a.frame_range[1]))
                for a in bpy.data.actions
                if not re.search(r"\.\d+$", a.name)
            ]
            if animations:
                anim_action_name, frame_start, frame_end = random.choice(animations)
                anim_frame = random.randint(frame_start, frame_end)

        if anim_frame is not None and anim_action_name:
            armature = next(
                (obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"),
                None,
            )
            action = bpy.data.actions.get(anim_action_name)
            if armature and action:
                armature.animation_data.action = action
                bpy.context.scene.frame_set(anim_frame)
                print(f"  → Set animation to '{anim_action_name}', frame {anim_frame}")

        # --- Geometry Merging & Pose Application ---
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        if not mesh_objects:
            print("  ⚠ No mesh objects found.")
            return False

        for obj in mesh_objects:
            bpy.context.view_layer.objects.active = obj
            if obj.data.shape_keys:
                obj.shape_key_clear()
            for mod in obj.modifiers:
                if mod.type == "ARMATURE":
                    bpy.ops.object.modifier_apply(modifier=mod.name)

        bpy.ops.object.select_all(action="DESELECT")
        for obj in mesh_objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objects[0]
        if len(mesh_objects) > 1:
            bpy.ops.object.join()
        merged_obj = bpy.context.view_layer.objects.active
        print(f"  → Merged {len(mesh_objects)} meshes into one.")

        # --- Cleanup & Artifact Removal ---
        merged_obj = _remove_small_loose_parts(merged_obj)
        has_textures = _mesh_has_image_textures(merged_obj)
        has_vertex_colors = _mesh_has_vertex_colors(merged_obj)
        has_uvs = len(merged_obj.data.uv_layers) > 0

        _cleanup_mesh(
            merged_obj, allow_fill_holes=not (has_textures or has_vertex_colors)
        )

        bpy.ops.object.select_all(action="DESELECT")
        merged_obj.select_set(True)
        bpy.context.view_layer.objects.active = merged_obj

        # --- Texture Preservation / Fallback ---
        if has_textures and has_uvs:
            print("  → Preserving existing textures and UVs (skip baking).")
        elif has_textures and not has_uvs:
            print("  ⚠ Textures detected but no UVs; applying fallback material.")
            _ensure_fallback_material(merged_obj)
        elif has_vertex_colors:
            if not merged_obj.material_slots or all(
                s.material is None for s in merged_obj.material_slots
            ):
                print("  → Vertex colors detected; creating material.")
                _ensure_vertex_color_material(merged_obj)
        else:
            if not merged_obj.material_slots or all(
                s.material is None for s in merged_obj.material_slots
            ):
                print("  → No textures detected; applying fallback material.")
                _ensure_fallback_material(merged_obj)

        bpy.ops.export_scene.gltf(
            filepath=str(dest_file),
            export_format="GLB",
            use_selection=True,
            export_materials="EXPORT",
            export_image_format="AUTO",
        )
        print(f"  ✓ Exported successfully to {dest_file}")
        return True

    except Exception as e:
        print(f"  ⚠ An error occurred during GLB processing: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        cleanup_blender()
