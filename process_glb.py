import bpy
import argparse
import os
import glob
import tqdm
import mathutils


def setup_white_world_lighting():
    """
    Set up world lighting to pure white for baking.
    """
    # Ensure we have a world
    if not bpy.context.scene.world:
        bpy.context.scene.world = bpy.data.worlds.new("World")

    world = bpy.context.scene.world
    world.use_nodes = True

    # Clear existing nodes
    world.node_tree.nodes.clear()

    # Create background shader with pure white
    bg_node = world.node_tree.nodes.new(type="ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)  # Pure white
    bg_node.inputs["Strength"].default_value = 1.0

    # Create world output
    output_node = world.node_tree.nodes.new(type="ShaderNodeOutputWorld")

    # Connect background to output
    world.node_tree.links.new(
        bg_node.outputs["Background"], output_node.inputs["Surface"]
    )


def clear_all_lights():
    """
    Remove all light objects from the scene.
    """
    lights_to_remove = [obj for obj in bpy.context.scene.objects if obj.type == "LIGHT"]
    for light in lights_to_remove:
        bpy.data.objects.remove(light, do_unlink=True)


def normalize_mesh_to_cube(obj, cube_size=1.0):
    """
    Normalize mesh to fit within a cube of specified size centered at origin.
    cube_size=1.0 means the object will fit in [-0.5, 0.5] range for all axes.
    """
    if not obj or obj.type != "MESH" or not obj.data.vertices:
        print(f"Warning: Object {obj.name if obj else 'None'} has no valid mesh data")
        return False

    # Get all vertex coordinates
    coords = [v.co.copy() for v in obj.data.vertices]

    if not coords:
        print(f"Warning: Object {obj.name} has no vertices")
        return False

    # Calculate bounding box
    min_corner = mathutils.Vector((min(v[i] for v in coords) for i in range(3)))
    max_corner = mathutils.Vector((max(v[i] for v in coords) for i in range(3)))

    # Calculate current center and size
    center = (min_corner + max_corner) / 2.0
    size = max_corner - min_corner
    max_dim = max(size)

    print(f"Object {obj.name}: center={center}, size={size}, max_dim={max_dim}")

    # Check for degenerate geometry
    if max_dim < 1e-6:  # Very small object
        print(
            f"Warning: Object {obj.name} has very small dimensions (max_dim={max_dim})"
        )
        return False

    # Step 1: Center the object at origin
    for v in obj.data.vertices:
        v.co -= center

    # Step 2: Scale to fit within the cube
    # We want the largest dimension to be cube_size, so scale_factor = cube_size / max_dim
    scale_factor = cube_size / max_dim
    for v in obj.data.vertices:
        v.co *= scale_factor

    # Verify the result
    new_coords = [v.co.copy() for v in obj.data.vertices]
    new_min = mathutils.Vector((min(v[i] for v in new_coords) for i in range(3)))
    new_max = mathutils.Vector((max(v[i] for v in new_coords) for i in range(3)))

    print(f"After normalization: min={new_min}, max={new_max}")

    # Double-check bounds
    for i in range(3):
        if new_min[i] < -cube_size / 2 - 1e-6 or new_max[i] > cube_size / 2 + 1e-6:
            print(
                f"Warning: Object {obj.name} exceeds expected bounds after normalization"
            )
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Input folder containing GLB files",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Output folder for processed GLB files",
    )
    args = parser.parse_args()

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs("./tmp", exist_ok=True)

    failed_files = []

    for input_path in tqdm.tqdm(glob.glob(os.path.join(args.input_folder, "*.glb"))):
        filename = os.path.basename(input_path)
        output_path = os.path.join(args.output_folder, filename)

        print(f"\nProcessing: {filename}")

        try:
            # Clear existing objects
            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.delete(use_global=False)

            # Import GLB file
            bpy.ops.import_scene.gltf(filepath=input_path)

            # Get all mesh objects
            mesh_objects = [
                obj for obj in bpy.context.scene.objects if obj.type == "MESH"
            ]

            if not mesh_objects:
                print(f"Warning: No mesh objects found in {filename}")
                failed_files.append(filename)
                continue

            # Join all mesh objects into one if there are multiple
            if len(mesh_objects) > 1:
                bpy.ops.object.select_all(action="DESELECT")
                for obj in mesh_objects:
                    obj.select_set(True)
                bpy.context.view_layer.objects.active = mesh_objects[0]
                bpy.ops.object.join()

            obj = bpy.context.view_layer.objects.active

            # Only do texture baking if there are multiple materials
            if len(mesh_objects) > 1:
                # Switch to Cycles
                bpy.context.scene.render.engine = "CYCLES"
                bpy.context.scene.cycles.device = "GPU"

                # Set up lighting environment for baking
                clear_all_lights()  # Remove all directional/point lights
                setup_white_world_lighting()  # Set world to pure white

                # Create new image for baking
                img = bpy.data.images.new("BakedTexture", width=2048, height=2048)
                img.colorspace_settings.name = "sRGB"

                # Add bake target to all materials
                for mat in obj.data.materials:
                    if mat is None:
                        continue
                    mat.use_nodes = True
                    nodes = mat.node_tree.nodes
                    texnode = nodes.new(type="ShaderNodeTexImage")
                    texnode.image = img
                    nodes.active = texnode

                # Ensure active object
                bpy.context.view_layer.objects.active = obj

                # Make sure UVs exist
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.03)
                bpy.ops.object.mode_set(mode="OBJECT")

                # Bake only base color with white diffuse lighting
                bpy.ops.object.bake(
                    type="DIFFUSE", use_clear=True, pass_filter={"COLOR"}, margin=1
                )

                # Save baked texture
                img.filepath_raw = "./tmp/baked_texture.png"
                img.file_format = "PNG"
                img.save()

                # Replace materials with a single baked one
                baked_mat = bpy.data.materials.new(name="BakedMat")
                baked_mat.use_nodes = True
                nodes = baked_mat.node_tree.nodes
                links = baked_mat.node_tree.links
                nodes.clear()

                output = nodes.new(type="ShaderNodeOutputMaterial")
                diffuse = nodes.new(type="ShaderNodeBsdfPrincipled")
                new_texnode = nodes.new(type="ShaderNodeTexImage")
                new_texnode.image = img
                links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
                links.new(new_texnode.outputs["Color"], diffuse.inputs["Base Color"])

                obj.data.materials.clear()
                obj.data.materials.append(baked_mat)

            # Apply transforms to all mesh objects
            for mesh_obj in [o for o in bpy.context.scene.objects if o.type == "MESH"]:
                bpy.context.view_layer.objects.active = mesh_obj
                mesh_obj.select_set(True)
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                mesh_obj.select_set(False)

            # Join all meshes again (in case transforms created multiple)
            mesh_objects = [
                obj for obj in bpy.context.scene.objects if obj.type == "MESH"
            ]
            if len(mesh_objects) > 1:
                bpy.ops.object.select_all(action="DESELECT")
                for mesh_obj in mesh_objects:
                    mesh_obj.select_set(True)
                bpy.context.view_layer.objects.active = mesh_objects[0]
                bpy.ops.object.join()

            obj = bpy.context.view_layer.objects.active

            # Normalize the mesh to fit in [-0.5, 0.5] cube
            success = normalize_mesh_to_cube(obj, cube_size=1.0)
            if not success:
                print(f"Failed to normalize {filename}")
                failed_files.append(filename)
                continue

            # Apply final transform
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Export only mesh objects (exclude empties, lights, etc.)
            bpy.ops.object.select_all(action="DESELECT")
            final_mesh_objects = [
                obj
                for obj in bpy.context.scene.objects
                if obj.type == "MESH" and obj.data and obj.data.vertices
            ]
            for mesh_obj in final_mesh_objects:
                mesh_obj.select_set(True)

            # Export the processed GLB with only selected mesh objects
            bpy.ops.export_scene.gltf(
                filepath=output_path,
                export_format="GLB",
                use_selection=True,  # Only export selected objects
                export_materials="EXPORT",
                export_texcoords=True,
                export_normals=True,
            )

            print(f"Successfully processed: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            failed_files.append(filename)

    print(f"\n=== Processing Complete ===")
    print(f"Failed files ({len(failed_files)}): {failed_files}")
