import bpy
import os
import gc
import re
import random

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
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.separate(type="LOOSE")
        bpy.ops.object.mode_set(mode="OBJECT")

        mesh_parts = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        if len(mesh_parts) > 1:
            metrics = []
            for part in mesh_parts:
                dims = part.dimensions
                vol = float(dims.x * dims.y * dims.z)
                faces = len(part.data.polygons)
                metrics.append((part, vol, faces))

            largest_by_vol = max(metrics, key=lambda x: x[1])
            largest_vol = largest_by_vol[1]
            largest_faces = max(metrics, key=lambda x: x[2])[2]

            min_volume_ratio = 0.005
            min_face_ratio = 0.01
            min_faces = 50
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
                merged_obj = bpy.context.view_layer.objects.active

        has_textures = False
        for mat_slot in merged_obj.material_slots:
            mat = mat_slot.material
            if not mat or not mat.use_nodes or not mat.node_tree:
                continue
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image is not None:
                    has_textures = True
                    break
            if has_textures:
                break
        has_uvs = len(merged_obj.data.uv_layers) > 0
        materials = [slot.material for slot in merged_obj.material_slots if slot.material]
        if len(materials) <= 1:
            has_multiple_mats = False
        else:
            unique_names = {mat.name for mat in materials}
            has_multiple_mats = len(unique_names) > 1 or len(materials) > 1

        bpy.ops.object.select_all(action="DESELECT")
        merged_obj.select_set(True)
        bpy.context.view_layer.objects.active = merged_obj

        # --- Handle Material-less Objects ---
        if has_textures and has_uvs and not has_multiple_mats:
            if len(merged_obj.material_slots) > 1:
                for poly in merged_obj.data.polygons:
                    poly.material_index = 0
                while len(merged_obj.material_slots) > 1:
                    merged_obj.active_material_index = len(merged_obj.material_slots) - 1
                    try:
                        bpy.ops.object.material_slot_remove()
                    except Exception:
                        break
            bpy.ops.export_scene.gltf(
                filepath=str(dest_file), export_format="GLB", use_selection=True
            )
            return True

        if not has_textures or not has_uvs:
            mat = bpy.data.materials.new(name="FallbackMaterial")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            bsdf = nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (
                    random.random(),
                    random.random(),
                    random.random(),
                    1.0,
                )
                bsdf.inputs["Roughness"].default_value = 0.6
            merged_obj.data.materials.clear()
            merged_obj.data.materials.append(mat)

            if len(merged_obj.material_slots) > 1:
                for poly in merged_obj.data.polygons:
                    poly.material_index = 0
                while len(merged_obj.material_slots) > 1:
                    merged_obj.active_material_index = len(merged_obj.material_slots) - 1
                    try:
                        bpy.ops.object.material_slot_remove()
                    except Exception:
                        break

            bpy.ops.export_scene.gltf(
                filepath=str(dest_file), export_format="GLB", use_selection=True
            )
            return True

        # --- Professional Texture Baking ---
        if has_multiple_mats:
            print("  → Starting professional texture baking process...")
            bpy.context.scene.render.engine = "CYCLES"
            bpy.context.scene.cycles.device = "GPU"
            bpy.context.scene.cycles.samples = 1
            bpy.context.scene.render.bake.margin = 16

            uv_map_name = "BakeUVMap"
            bpy.context.view_layer.objects.active = merged_obj
            merged_obj.select_set(True)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
            bpy.ops.object.mode_set(mode="OBJECT")
            if merged_obj.data.uv_layers.active:
                merged_obj.data.uv_layers.active.name = uv_map_name

            bake_image_name = "BakedTextureAtlas"
            img_size = 2048
            bake_image = bpy.data.images.new(
                bake_image_name, width=img_size, height=img_size
            )

            for mat_slot in merged_obj.material_slots:
                if mat_slot.material and mat_slot.material.node_tree:
                    nodes = mat_slot.material.node_tree.nodes
                    image_node = nodes.new(type="ShaderNodeTexImage")
                    image_node.image = bake_image
                    nodes.active = image_node

            bpy.ops.object.bake(
                type="DIFFUSE",
                pass_filter={"COLOR"},
                uv_layer=uv_map_name,
                cage_extrusion=0.1,
                max_ray_distance=1.0,
            )

            temp_dir = bpy.app.tempdir
            temp_file_path = os.path.join(temp_dir, f"{bake_image_name}.png")
            bake_image.filepath_raw = temp_file_path
            bake_image.file_format = "PNG"
            bake_image.save()

            # --- Finalization & Export ---
            final_mat = bpy.data.materials.new(name="BakedMaterial")
            final_mat.use_nodes = True
            nodes = final_mat.node_tree.nodes
            bsdf = nodes.get("Principled BSDF")
            tex_node = nodes.new("ShaderNodeTexImage")
            tex_node.image = bpy.data.images.load(temp_file_path)
            final_mat.node_tree.links.new(
                bsdf.inputs["Base Color"], tex_node.outputs["Color"]
            )

            merged_obj.data.materials.clear()
            merged_obj.data.materials.append(final_mat)
            merged_obj.data.uv_layers[uv_map_name].active_render = True

            if len(merged_obj.material_slots) > 1:
                for poly in merged_obj.data.polygons:
                    poly.material_index = 0
                while len(merged_obj.material_slots) > 1:
                    merged_obj.active_material_index = len(merged_obj.material_slots) - 1
                    try:
                        bpy.ops.object.material_slot_remove()
                    except Exception:
                        break

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
