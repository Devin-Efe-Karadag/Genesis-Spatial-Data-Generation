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
    Merges, bakes, and exports a GLB with a single, artifact-free texture atlas.
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

        # --- Handle Material-less Objects ---
        if not merged_obj.material_slots or all(
            s.material is None for s in merged_obj.material_slots
        ):
            print("  → Preserving material-less object. Exporting as is.")
            bpy.ops.export_scene.gltf(
                filepath=str(dest_file), export_format="GLB", use_selection=True
            )
            return True

        # --- Professional Texture Baking ---
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
