#!/usr/bin/env python3
"""
Test rendering script for merged meshes with animation support.

Randomly samples GLB files from a source directory, merges submeshes,
handles animations, and renders from multiple camera angles.
"""

import argparse
import atexit
import gc
import os
import random
import sys
from pathlib import Path
from typing import List

import cv2
import genesis as gs
import numpy as np
import torch

# Import bpy for mesh merging
try:
    import bpy
    HAS_BPY = True

    # Suppress bpy cleanup warnings at exit
    def suppress_bpy_warnings():
        """Suppress harmless bpy unregister warnings during exit."""
        sys.stderr = open(os.devnull, 'w')

    atexit.register(suppress_bpy_warnings)

except ImportError:
    HAS_BPY = False
    print("Warning: bpy not available. Cannot merge submeshes.")


def find_glb_files(source_dir: Path) -> List[Path]:
    """
    Recursively find all .glb files in the source directory.

    Args:
        source_dir: Path to the directory to search

    Returns:
        List of Path objects pointing to .glb files
    """
    glb_files = list(source_dir.rglob("*.glb"))
    return glb_files


def select_random_files(files: List[Path], count: int) -> List[Path]:
    """
    Randomly select a specified number of files from the list.

    Args:
        files: List of file paths
        count: Number of files to select

    Returns:
        List of randomly selected file paths

    Raises:
        ValueError: If count exceeds the number of available files
    """
    if count > len(files):
        raise ValueError(
            f"Requested {count} files but only {len(files)} available"
        )
    return random.sample(files, count)


def get_glb_animations(src_file):
    """
    Get animation information from GLB file.

    Args:
        src_file: Source GLB file path

    Returns:
        List of tuples (anim_index, anim_name, frame_start, frame_end)
        Empty list if no animations
    """
    if not HAS_BPY:
        return []

    try:
        # Clear the scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # Import the GLB file
        bpy.ops.import_scene.gltf(filepath=str(src_file))

        # Check for animations
        animations = []
        if bpy.data.actions:
            for idx, action in enumerate(bpy.data.actions):
                frame_start = int(action.frame_range[0])
                frame_end = int(action.frame_range[1])
                # Store action name so we can reference it later
                animations.append(
                    (idx, action.name, frame_start, frame_end)
                )

        return animations

    except Exception as e:
        print(f"  ⚠ Failed to get animations: {e}")
        return []


def merge_glb_submeshes(src_file, dest_file, anim_frame=None,
                        anim_action_name=None):
    """
    Merge all submeshes in a GLB file using bpy.
    Also consolidates materials to ensure single mesh output.
    Optionally bake a specific animation frame.

    Args:
        src_file: Source GLB file path
        dest_file: Destination GLB file path
        anim_frame: If provided, bake mesh at this frame of animation
        anim_action_name: Name of the action (animation) to use

    Returns:
        True if successful, False otherwise
    """
    if not HAS_BPY:
        return False

    try:
        # Clear the scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # Import the GLB file
        bpy.ops.import_scene.gltf(filepath=str(src_file))

        # Set to specific animation and frame if requested
        if anim_frame is not None and anim_action_name is not None:
            # Find the armature object
            armature = None
            for obj in bpy.context.scene.objects:
                if obj.type == 'ARMATURE':
                    armature = obj
                    break

            if armature and armature.animation_data:
                # Find the action by name
                action = bpy.data.actions.get(anim_action_name)
                if action:
                    # Assign this action to the armature
                    armature.animation_data.action = action
                    print(f"  → Assigned action: '{anim_action_name}'")
                else:
                    print(f"  ⚠ Action '{anim_action_name}' not found")

            # Now set the frame
            bpy.context.scene.frame_set(anim_frame)
            print(f"  → Set to frame {anim_frame}")

        # Get all mesh objects
        mesh_objects = [
            obj for obj in bpy.context.scene.objects
            if obj.type == 'MESH'
        ]

        if len(mesh_objects) == 0:
            print(f"  ⚠ No mesh objects found")
            return False

        # If animating, apply armature modifiers to bake pose
        if anim_frame is not None:
            for obj in mesh_objects:
                # Apply modifiers (armature, shape keys, etc.)
                bpy.context.view_layer.objects.active = obj
                for modifier in obj.modifiers[:]:
                    if modifier.type == 'ARMATURE':
                        bpy.ops.object.modifier_apply(modifier=modifier.name)

        # Join all meshes into one object
        if len(mesh_objects) > 1:
            print(f"  → Merging {len(mesh_objects)} mesh objects...")

            # Deselect all first
            bpy.ops.object.select_all(action='DESELECT')

            # Select all mesh objects
            for obj in mesh_objects:
                obj.select_set(True)

            # Set the first mesh as active
            bpy.context.view_layer.objects.active = mesh_objects[0]

            # Join all meshes
            bpy.ops.object.join()

            print(f"  → Merged into single mesh object")

        # Now we have one mesh object, but it may have multiple materials
        # Get the merged object (should be the only mesh object now)
        merged_obj = [obj for obj in bpy.context.scene.objects
                      if obj.type == 'MESH'][0]

        # Check material count
        num_materials = len(merged_obj.data.materials)
        if num_materials > 1:
            print(f"  → Found {num_materials} materials, "
                  f"consolidating to single material...")

            # Keep only the first material, remove others
            # This ensures Genesis sees it as a single mesh
            while len(merged_obj.data.materials) > 1:
                merged_obj.data.materials.pop(index=1)

            # Assign all faces to material slot 0
            for poly in merged_obj.data.polygons:
                poly.material_index = 0

            print(f"  → Consolidated to 1 material")

        # Export as GLB (without animation data)
        bpy.ops.export_scene.gltf(
            filepath=str(dest_file),
            export_format='GLB',
            use_selection=False,
            export_animations=False  # Export only the current pose
        )

        # Verify export succeeded
        import os
        if not os.path.exists(dest_file):
            print(f"  ⚠ Export failed - file not created: {dest_file}")
            return False

        file_size = os.path.getsize(dest_file)
        if file_size == 0:
            print(f"  ⚠ Export failed - file is empty: {dest_file}")
            return False

        print(f"  ✓ Exported successfully ({file_size} bytes)")
        return True

    except Exception as e:
        print(f"  ⚠ bpy merge failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def orbit_camera_position(elevation_deg, azimuth_deg, radius):
    """
    Compute camera position using orbit camera convention.

    Args:
        elevation_deg: Elevation angle in degrees (0=horizontal, +up)
        azimuth_deg: Azimuth angle in degrees (rotation around Z axis)
        radius: Distance from origin

    Returns:
        Camera position as numpy array
    """
    elevation_rad = np.deg2rad(elevation_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)

    x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = radius * np.sin(elevation_rad)

    return np.array([x, y, z])


def render_mesh_static(glb_path, output_folder, mesh_name, anim_frame=None,
                       anim_action_name=None):
    """
    Render a single mesh from multiple camera angles without simulation.

    Args:
        glb_path: Path to GLB file
        output_folder: Output directory for images
        mesh_name: Name for organizing output files
        anim_frame: Optional frame number to bake from animation
        anim_action_name: Optional action name to use for animation
    """
    print(f"\n{'='*60}")
    print(f"Rendering: {mesh_name}")
    print(f"{'='*60}")

    # Create output directory
    mesh_output = os.path.join(output_folder, mesh_name)
    os.makedirs(mesh_output, exist_ok=True)

    # Merge submeshes (MANDATORY - Genesis requires single mesh)
    mesh_to_use = glb_path
    merged_file_path = None

    if not HAS_BPY:
        raise RuntimeError("bpy module not available - cannot merge submeshes. "
                          "Genesis requires single mesh for MPM entities.")

    print(f"  → Merging submeshes...")
    # Save merged mesh to output folder
    if anim_frame is not None:
        merged_filename = f"merged_frame_{anim_frame:04d}.glb"
    else:
        merged_filename = "merged.glb"

    merged_file_path = os.path.join(mesh_output, merged_filename)

    success = merge_glb_submeshes(
        glb_path,
        merged_file_path,
        anim_frame,
        anim_action_name
    )

    if not success:
        # Clean up if merge failed
        if os.path.exists(merged_file_path):
            os.remove(merged_file_path)
        raise RuntimeError(
            f"Failed to merge submeshes for {glb_path}. "
            f"Cannot proceed without merged mesh."
        )

    mesh_to_use = merged_file_path
    print(f"  → Using merged mesh: {merged_filename}")

    # Scene parameters
    center = np.array([0.0, 0.0, 0.0])
    scale = 0.6
    camera_radius = 1.5
    resolution = 512
    fov = 49.1

    # Camera configurations (8 viewpoints)
    camera_configs = [
        {'elevation': 0, 'azimuth': 0, 'name': 'front'},
        {'elevation': 0, 'azimuth': 90, 'name': 'right'},
        {'elevation': 0, 'azimuth': 180, 'name': 'back'},
        {'elevation': 0, 'azimuth': 270, 'name': 'left'},
        {'elevation': 30, 'azimuth': 45, 'name': 'top_front_right'},
        {'elevation': -15, 'azimuth': 135, 'name': 'bottom_back_right'},
        {'elevation': 45, 'azimuth': 225, 'name': 'top_back_left'},
        {'elevation': 15, 'azimuth': 315, 'name': 'mid_front_left'},
    ]

    # Simulation parameters for MPM
    dt = 2.5e-4
    substeps = 1
    gravity = (0, 0, -9.81)
    lower_bound = np.array([-5.0, -5.0, -0.7])
    upper_bound = np.array([5.0, 5.0, 2.0])
    particle_size = None  # Auto
    grid_density = 64

    # Create scene with MPM options
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=substeps,
            gravity=gravity,
            requires_grad=False,
        ),
        mpm_options=gs.options.MPMOptions(
            enable_CPIC=False,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            use_sparse_grid=False,
            grid_density=grid_density,
            particle_size=particle_size,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            shadow=True,
            background_color=(1.0, 1.0, 1.0),
        ),
        show_viewer=False,
    )

    # Create MPM material (elastic)
    E = 1e5  # Young's modulus
    nu = 0.3  # Poisson's ratio
    rho = 1e3  # Density
    mat_elastic = gs.materials.MPM.Elastic(E=E, nu=nu, rho=rho,
                                            model="neohookean")

    # Add mesh entity (MPM, not rigid)
    surface = gs.surfaces.Default() # vis_mode="recon_simple")

    scene.add_entity(
        material=mat_elastic,
        morph=gs.morphs.Mesh(
            file=str(mesh_to_use),
            scale=scale,
            pos=center,
            decimate=False,
            normalize=True,
            group_by_material=True,  # Merge primitives with same material
        ),
        surface=surface,
    )

    # Add cameras
    cameras = []
    for cam_config in camera_configs:
        cam_pos = orbit_camera_position(
            cam_config['elevation'],
            cam_config['azimuth'],
            camera_radius
        )
        cam_pos += center

        cam = scene.add_camera(
            res=(resolution, resolution),
            pos=cam_pos,
            lookat=center,
            up=np.array([0., 0., 1.]),
            fov=fov,
            GUI=False,
        )
        cameras.append((cam, cam_config['name']))

    # Build scene
    scene.build()

    # Render from all camera angles (no simulation, just static)
    print(f"  → Rendering {len(cameras)} viewpoints...")
    saved_count = 0

    for cam, view_name in cameras:
        # Render returns tuple: (rgb,) for basic render
        render_output = cam.render()

        # Extract RGB from tuple if needed
        if isinstance(render_output, tuple):
            rgb = render_output[0]
        else:
            rgb = render_output

        # Ensure it's a numpy array and uint8
        if not isinstance(rgb, np.ndarray):
            print(f"    ⚠ Warning: rgb is not numpy array, "
                  f"type: {type(rgb)}")
            continue

        rgb = np.asarray(rgb, dtype=np.uint8)

        # Save as JPEG
        output_path = os.path.join(mesh_output, f"{view_name}.jpg")
        cv2.imwrite(
            output_path,
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )
        print(f"    ✓ Saved: {view_name}.jpg")
        saved_count += 1

    # Cleanup scene (Genesis will be destroyed in main)
    try:
        scene.destroy()
    except Exception as e:
        print(f"    ⚠ Warning: scene.destroy() failed: {e}")

    del scene
    del cameras

    print(f"  → Completed: {saved_count} images saved to {mesh_output}")
    if merged_file_path and os.path.exists(merged_file_path):
        if anim_frame is not None:
            print(f"  → Merged mesh saved: merged_frame_{anim_frame:04d}.glb")
        else:
            print(f"  → Merged mesh saved: merged.glb")


def main():
    """Main function to render test GLB files with animation support."""
    parser = argparse.ArgumentParser(
        description="Render randomly selected GLB files with animation support"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("filtered_objs"),
        help="Source directory containing GLB files (default: filtered_objs)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_rendering"),
        help="Output directory (default: test_rendering)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of files to select (default: 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)"
    )
    parser.add_argument(
        "--max-anims",
        type=int,
        default=5,
        help="Maximum animations per object (default: 5). "
             "If object has more, randomly sample this many."
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Validate source directory
    if not args.source.exists():
        raise FileNotFoundError(f"Source directory not found: {args.source}")

    # Find all GLB files
    print(f"Searching for GLB files in {args.source}...")
    all_glb_files = find_glb_files(args.source)
    print(f"Found {len(all_glb_files)} GLB files")

    if len(all_glb_files) == 0:
        print(f"No GLB files found in {args.source}")
        return

    # Select random files
    try:
        selected_files = select_random_files(all_glb_files, args.count)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"Selected {len(selected_files)} random files:")
    for idx, glb_path in enumerate(selected_files):
        print(f"  [{idx}] {glb_path.name}")

    # Create test_files list with (path, name) tuples
    test_files = []
    for idx, glb_path in enumerate(selected_files):
        # Use stem (filename without extension) as base name
        mesh_name = f"mesh_{idx}"
        test_files.append((str(glb_path), mesh_name))

    output_folder = str(args.output)

    print(f"\n{'#'*60}")
    print(f"# Test Rendering for Merged Meshes with Animation Support")
    print(f"# Selected {len(test_files)} GLB files")
    print(f"# Output: {output_folder}")
    print(f"{'#'*60}")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Render each mesh
    success_count = 0
    total_renders = 0

    for idx, (glb_path, mesh_name) in enumerate(test_files, 1):
        print(f"\n[{idx}/{len(test_files)}] Processing: {mesh_name}")

        # Check for animations
        if HAS_BPY:
            animations = get_glb_animations(glb_path)
            print(f"  → Found {len(animations)} animation(s)")

            if len(animations) > 0:
                # Show all animations found
                for anim_idx, anim_name, frame_start, frame_end in animations:
                    print(f"      [{anim_idx}] '{anim_name}': "
                          f"frames {frame_start}-{frame_end} "
                          f"({frame_end - frame_start + 1} frames)")

            # Limit number of animations if too many
            if len(animations) > args.max_anims:
                print(f"  → Limiting to {args.max_anims} animations "
                      f"(randomly sampled)")
                animations = random.sample(animations, args.max_anims)
                # Sort by index for consistent output folder naming
                animations = sorted(animations, key=lambda x: x[0])
                print(f"  → Selected animations: "
                      f"{', '.join([f'{a[0]}:{a[1]}' for a in animations])}")
        else:
            animations = []

        if len(animations) == 0:
            # No animations - process as static mesh
            gs.init(precision="32")
            try:
                render_mesh_static(glb_path, output_folder, mesh_name)
                success_count += 1
                total_renders += 1
            except Exception as e:
                print(f"\n✗ Failed to render {mesh_name}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                try:
                    gs.destroy()
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as cleanup_error:
                    print(f"    ⚠ Cleanup warning: {cleanup_error}")

        else:
            # Process each animation separately
            for anim_idx, anim_name, frame_start, frame_end in animations:
                # Pick random frame from this animation
                random_frame = np.random.randint(frame_start, frame_end + 1)
                output_name = f"{mesh_name}_anim{anim_idx}"

                print(f"\n  Animation {anim_idx}: '{anim_name}'")
                print(f"    Frame range: [{frame_start}, {frame_end}]")
                print(f"    Selected frame: {random_frame}")

                gs.init(precision="32")
                try:
                    render_mesh_static(
                        glb_path,
                        output_folder,
                        output_name,
                        anim_frame=random_frame,
                        anim_action_name=anim_name
                    )
                    success_count += 1
                    total_renders += 1
                except Exception as e:
                    print(f"\n✗ Failed to render {output_name}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    try:
                        gs.destroy()
                        torch.cuda.empty_cache()
                        gc.collect()
                    except Exception as cleanup_error:
                        print(f"    ⚠ Cleanup warning: {cleanup_error}")

    print(f"\n{'#'*60}")
    print(f"# Rendering Complete!")
    print(f"# Output folder: {output_folder}")
    print(f"# Files processed: {len(test_files)}")
    print(f"# Successful renders: {success_count}/{total_renders}")
    print(f"# Total images: {success_count * 8} (8 views per render)")
    if args.max_anims < 999:
        print(f"# Animation limit: {args.max_anims} per object")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()

