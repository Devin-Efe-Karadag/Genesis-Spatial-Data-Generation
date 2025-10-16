import kaolin # need to be called before genesis

import argparse
import gc
import json
import math
import os
import tarfile
from collections import namedtuple
from glob import glob
from pathlib import Path

import cv2
import genesis as gs
import bpy
import re
import random
import numpy as np
import taichi as ti
import torch
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as R

# Import mesh reconstruction cache patch
import genesis_mesh_cache_patch as gmc


def save_points_as_ply(points, filename):
    """Save particle positions as PLY file."""
    points = np.asarray(points)
    points = points[:, 0, [1, 2, 0]]  # due to genesis-specific coord system
    point_cloud = trimesh.points.PointCloud(vertices=points)
    point_cloud.export(file_obj=filename, file_type='ply', encoding='ascii')


def sample_unit_vector(min_elevation_radian, max_elevation_radian):
    """
    Samples a 3D unit vector with azimuth uniformly distributed in [0, 2π)
    and elevation uniformly distributed in [min_elevation_radian, max_elevation_radian].
    """
    if min_elevation_radian == max_elevation_radian:
        cos_theta = np.cos([min_elevation_radian])
    else:
        cos_theta = np.random.uniform(np.cos(min_elevation_radian),
                                      np.cos(max_elevation_radian), 1)

    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2 * np.pi, 1)

    sin_theta = np.sqrt(1 - cos_theta**2)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta

    return np.concatenate((x, y, z), axis=-1)


def orbit_camera_position(elevation_deg, azimuth_deg, radius):
    """
    Compute camera position using orbit camera convention.
    elevation_deg: elevation angle in degrees (0 = horizontal, positive = up)
    azimuth_deg: azimuth angle in degrees (rotation around Z axis)
    radius: distance from origin
    """
    elevation_rad = np.deg2rad(elevation_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)

    # Standard spherical to Cartesian conversion
    x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = radius * np.sin(elevation_rad)

    return np.array([x, y, z])


def create_tar_files(save_root, uid, output_folder):
    """
    Create tar files following the reference format:
    - random_clip-{uid}: contains views 000-031 (random cameras)
    - fixed_16_clip-{uid}: contains views 032-047 (fixed cameras)
    """
    # Create tar for random cameras (views 0-31)
    random_tar_path = os.path.join(output_folder, f'random_clip-{uid}')
    with tarfile.open(random_tar_path, 'w') as tar:
        for view_idx in range(32):
            view_folder = os.path.join(save_root, f'{view_idx:03d}')
            if os.path.exists(view_folder):
                tar.add(view_folder, arcname=f'{view_idx:03d}')

    print(f"Created tar file: {random_tar_path}")

    # Create tar for fixed cameras (views 32-47)
    fixed_tar_path = os.path.join(output_folder, f'fixed_16_clip-{uid}')
    with tarfile.open(fixed_tar_path, 'w') as tar:
        for view_idx in range(32, 48):
            view_folder = os.path.join(save_root, f'{view_idx:03d}')
            if os.path.exists(view_folder):
                tar.add(view_folder, arcname=f'{view_idx:03d}')

    print(f"Created tar file: {fixed_tar_path}")


def cleanup_temp_mesh_files(
        mesh_cleaned,
        mesh_repaired,
        mesh_file_to_use,
        obj_path):
    """Helper function to clean up temporary mesh files."""
    if mesh_cleaned and os.path.exists(
            mesh_file_to_use) and mesh_file_to_use != obj_path:
        try:
            os.remove(mesh_file_to_use)
            print(f"  → Cleaned up temporary file: {mesh_file_to_use}")
        except Exception as e:
            pass  # Silently ignore cleanup errors


# ============================================================================
# HELPER DATA STRUCTURES
# ============================================================================

EntityCreationResult = namedtuple('EntityCreationResult', [
    'success', 'scene', 'cameras', 'n_particles', 'actual_particle_size',
    'mesh_file_to_use', 'mesh_cleaned', 'mesh_repaired'
])


# ============================================================================
# CONFIGURATION HELPER FUNCTIONS
# ============================================================================

def setup_camera_configs(n_random, n_fixed, elevation_range_random,
                         rotation_range, elevation_fixed):
    """
    Setup camera configurations for both random and fixed cameras.

    Parameters
    ----------
    n_random : int
        Number of random cameras
    n_fixed : int
        Number of fixed cameras
    elevation_range_random : tuple
        (min, max) elevation in degrees for random cameras
    rotation_range : tuple
        (min, max) rotation in degrees
    elevation_fixed : float
        Elevation in degrees for fixed cameras

    Returns
    -------
    list of dict
        Camera configuration dictionaries
    """
    camera_configs = []

    # Random cameras
    for i in range(n_random):
        elevation_deg = np.random.uniform(elevation_range_random[0], elevation_range_random[1])
        rotation_deg = np.random.uniform(rotation_range[0], rotation_range[1])
        camera_configs.append({
            'mode': 'random',
            'elevation': elevation_deg,
            'rotation': rotation_deg,
        })

    # Fixed cameras
    stepsize = 360.0 / n_fixed
    for i in range(n_fixed):
        rotation_deg = i * stepsize
        camera_configs.append({
            'mode': 'fixed',
            'elevation': elevation_fixed,
            'rotation': rotation_deg,
        })

    return camera_configs


def setup_lights(n_base_lights, n_variation, center):
    """
    Setup scene lights with random positions and intensities.

    Parameters
    ----------
    n_base_lights : int
        Base number of lights
    n_variation : int
        Variation in number of lights (n ± variation)
    center : np.ndarray
        Scene center position

    Returns
    -------
    list of dict
        Light configuration dictionaries
    """
    n_lights = n_base_lights + np.random.randint(-n_variation, n_variation + 1)
    n_lights = max(1, n_lights)  # at least 1 light

    lights = []
    for i in range(n_lights):
        # Place lights far away to simulate directional lighting
        light_distance = np.random.uniform(8.0, 20.0)
        light_pos = sample_unit_vector(0, np.pi) * light_distance
        light_pos += center

        # Random intensity
        light_intensity = np.random.uniform(8.0, 20.0)
        light_radius = np.random.uniform(2.0, 6.0)

        lights.append({
            "pos": tuple(light_pos),
            "radius": light_radius,
            "color": (light_intensity, light_intensity, light_intensity)
        })

    return lights


def create_random_material():
    """
    Create random elastic material parameters.

    Returns
    -------
    tuple
        (E, nu, rho, mat_elastic) where:
        - E: Young's modulus
        - nu: Poisson's ratio
        - rho: density
        - mat_elastic: Genesis material object
    """
    E = 10 ** np.random.uniform(4.0, 7.0)
    nu = np.random.uniform(0.0, 0.49)
    rho = 1e3
    mat_elastic = gs.materials.MPM.Elastic(E=E, nu=nu, rho=rho, model="neohookean")
    return E, nu, rho, mat_elastic


def create_random_velocity(scale=0.25):
    """
    Create random initial velocity.

    Returns
    -------
    np.ndarray
        3D velocity vector
    """
    return np.random.randn(3) * scale


def cleanup_scene(scene, cameras):
    """
    Clean up scene and cameras, free GPU memory.

    Parameters
    ----------
    scene : gs.Scene or None
        Scene to clean up
    cameras : list
        List of cameras to clean up
    """
    if scene is not None:
        # Properly destroy the scene to release Taichi resources
        try:
            scene.destroy()
        except Exception as e:
            # If destroy fails, just log and continue
            print(f"  → Warning: scene.destroy() failed: {e}")
        del scene
    if cameras:
        del cameras
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def compute_grid_cell_size(grid_density):
    """
    Compute the grid cell size for MPM solver.

    Genesis uses uniform grid spacing: dx = 1.0 / grid_density
    This is independent of domain bounds and uniform across all dimensions.

    For MPM stability, each grid cell should contain ~8 particles on average,
    which requires: grid_cell_size >= 2 * particle_size

    Parameters
    ----------
    grid_density : int
        Grid density parameter (Genesis computes dx = 1.0 / grid_density)

    Returns
    -------
    float
        Grid cell size in meters (uniform across all dimensions)
    """
    return 1.0 / grid_density


def check_particle_count(n_particles, actual_particle_size, min_count, min_size,
                         max_count=None, max_size=None, grid_density=None, is_stuck=False):
    """
    Validate particle count and suggest new particle size if needed.

    Handles both minimum (too few particles) and maximum (too many particles) cases.
    For max case, ensures particle_size <= max_size to maintain grid stability.
    Also adjusts grid_density if needed to maintain 8-64 particles per grid cell.

    Parameters
    ----------
    n_particles : int
        Current number of particles
    actual_particle_size : float
        Current particle size
    min_count : int
        Minimum required particle count
    min_size : float
        Minimum allowed particle size
    max_count : int, optional
        Maximum allowed particle count (for performance)
    max_size : float, optional
        Maximum allowed particle size (for grid stability: particle_size <= grid_cell_size/2)
    grid_density : int, optional
        Current grid density (for grid adjustment)
    is_stuck : bool, optional
        If True, particle count hasn't changed from previous attempt (cache issue)

    Returns
    -------
    tuple
        (is_valid: bool, suggested_particle_size: float or None, suggested_grid_density: int or None)

        If is_valid=True: particle count is acceptable, no changes needed
        If is_valid=False and both suggestions are None: cannot be fixed, skip object
        If is_valid=False and suggestions provided: retry with new parameters
    """
    # Check minimum particle count
    if n_particles < min_count and actual_particle_size > min_size:
        # Calculate required particle size to reach target particles
        # Particle count scales as (1/particle_size)^3
        ratio = (min_count / n_particles) ** (1.0 / 3.0)
        new_particle_size = max(actual_particle_size / ratio * 0.9, min_size)

        print(f"  ⚠ Particle count too low ({n_particles} < {min_count})")
        print(f"  → Will retry with particle_size={new_particle_size:.4f}m (smaller particles)")

        return False, new_particle_size, grid_density  # Keep grid_density unchanged

    # Check maximum particle count (for performance)
    if max_count is not None and n_particles > max_count:
        # Target particle count (aim for 75% of max to have some buffer)
        target_count = int(max_count * 0.75)

        # Calculate required particle size to reduce to target
        # Particle count scales as (1/particle_size)^3
        ratio = (n_particles / target_count) ** (1.0 / 3.0)
        new_particle_size = actual_particle_size * ratio

        # If stuck (cache cycle), apply aggressive increase to break out
        if is_stuck:
            # Force a 20% increase minimum to ensure cache invalidation
            min_increase = actual_particle_size * 1.2
            if new_particle_size < min_increase:
                print(f"  → Stuck! Forcing particle_size increase: {new_particle_size:.6f}m → {min_increase:.6f}m")
                new_particle_size = min_increase

        # Ensure we don't exceed max_size (grid stability constraint)
        # if max_size is not None:
        #     before_clamp = new_particle_size
        #     new_particle_size = min(new_particle_size, max_size)
        #     if new_particle_size < before_clamp:
        #         print(f"  → Clamped by max_size: {before_clamp:.6f}m → {new_particle_size:.6f}m (max={max_size:.6f}m)")

        print(f"  ⚠ Particle count too high ({n_particles} > {max_count})")
        print(f"  → Current actual_particle_size from Genesis: {actual_particle_size:.6f}m")
        print(f"  → Computed new_particle_size: {new_particle_size:.6f}m (ratio={ratio:.3f})")
        print(f"  → Target particle count: {target_count}")

        # Check if we need to adjust grid density
        new_grid_density = grid_density
        if grid_density is not None:
            current_grid_cell_size = compute_grid_cell_size(grid_density)

            # Grid cells should contain 8-64 particles on average
            # This requires: particle_size * 2 <= grid_cell_size <= particle_size * 4
            min_required_cell_size = new_particle_size * 2  # For max 64 particles per cell
            max_required_cell_size = new_particle_size * 4  # For min 8 particles per cell

            if current_grid_cell_size < max_required_cell_size:
                # Need to reduce grid_density (increase cell size) to maintain 8-64 particles per cell
                print(f"  → Current grid cell size ({current_grid_cell_size:.4f}m) too small for new particle size")
                print(f"  → Need grid cell size in range [{min_required_cell_size:.4f}m, {max_required_cell_size:.4f}m]")

                # Try powers of 2 in descending order: 32, 16, 8, 4, 2, 1
                # (Don't increase density, only decrease)
                found_valid = False
                for candidate_density in [32, 16, 8, 4, 2, 1]:
                    if candidate_density >= grid_density:
                        continue  # Skip if not reducing density

                    candidate_cell_size = compute_grid_cell_size(candidate_density)

                    # Check if this gives 8-64 particles per cell
                    if min_required_cell_size <= candidate_cell_size <= max_required_cell_size:
                        new_grid_density = candidate_density
                        found_valid = True
                        print(f"  → Reducing grid_density: {grid_density} → {new_grid_density}")
                        print(f"  → New grid cell size: {candidate_cell_size:.4f}m")
                        print(f"  → Expected particles per cell: {(candidate_cell_size / new_particle_size)**3:.1f}")
                        break

                if not found_valid:
                    # Even grid_density=1 doesn't satisfy constraint
                    min_cell_size_at_1 = compute_grid_cell_size(1)
                    print(f"  ⚠ Cannot satisfy grid constraint even at grid_density=1 (cell_size={min_cell_size_at_1:.4f}m)")
                    print(f"  → Skipping object: too many particles, cannot reduce spatial resolution further")
                    return False, None, None  # Signal to skip object

        return False, new_particle_size, new_grid_density

    return True, None, None  # Valid, no changes needed


# ============================================================================
# MESH AND SCENE CREATION FUNCTIONS
# ============================================================================
def merge_glb_submeshes(src_file, dest_file, anim_frame=None,
                        anim_action_name=None,
                        random_anim_action=False):
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
    try:
        # Clear the scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # Import the GLB file with explicit texture/material import
        bpy.ops.import_scene.gltf(
            filepath=str(src_file),
            import_pack_images=True,  # Pack images into blend file
            merge_vertices=False,  # Keep original mesh structure
            import_shading='NORMALS',  # Import with proper shading
        )

        # random animations
        animations = []
        if bpy.data.actions:
            for idx, action in enumerate(bpy.data.actions):
                # Filter out animations ending with .001, .002, etc.
                # These are typically auto-generated variants we want to skip
                if re.search(r'\.\d+$', action.name):
                    continue

                frame_start = int(action.frame_range[0])
                frame_end = int(action.frame_range[1])
                # Store action name so we can reference it later
                animations.append(
                    (idx, action.name, frame_start, frame_end)
                )

        if len(animations) > 0 and random_anim_action:
            animation = random.choice(animations)
            anim_action_name = animation[1]
            frame_start = animation[2]
            frame_end = animation[3]
            anim_frame = random.randint(frame_start, frame_end)

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
                bpy.context.view_layer.objects.active = obj

                # Remove shape keys if present (they prevent modifier application)
                if obj.data.shape_keys:
                    obj.shape_key_clear()
                    print(f"  → Removed shape keys from {obj.name}")

                # Apply armature modifiers to bake the current pose
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

        # Check material count and handle multi-material properly
        num_materials = len(merged_obj.data.materials)

        # Check if we have UV maps
        has_uvs = merged_obj.data.uv_layers and len(merged_obj.data.uv_layers) > 0

        if has_uvs:
            print(f"  → Found {len(merged_obj.data.uv_layers)} UV layer(s)")
            # Ensure the first UV layer is active
            merged_obj.data.uv_layers[0].active = True
            merged_obj.data.uv_layers[0].active_render = True
            print(f"  → Set active UV layer: {merged_obj.data.uv_layers[0].name}")

        if num_materials >= 1:
            print(f"  → Found {num_materials} material(s)")

            # Collect all textures and colors from materials
            material_textures = {}  # material_index -> image
            material_colors = {}  # material_index -> (r, g, b, a) for materials without textures

            for mat_idx, mat in enumerate(merged_obj.data.materials):
                found_texture = False
                if mat and mat.use_nodes:
                    # Look for image texture nodes
                    for node in mat.node_tree.nodes:
                        if node.type == 'TEX_IMAGE' and node.image:
                            material_textures[mat_idx] = node.image
                            found_texture = True
                            break

                # If no texture found, try to get base color
                if not found_texture:
                    if mat and mat.use_nodes:
                        for node in mat.node_tree.nodes:
                            if node.type == 'BSDF_PRINCIPLED':
                                color = node.inputs['Base Color'].default_value
                                material_colors[mat_idx] = (color[0], color[1], color[2], 1.0)
                                break
                    else:
                        # Fallback to diffuse color
                        if mat:
                            material_colors[mat_idx] = (0.8, 0.8, 0.8, 1.0)  # Default gray

            # Deduplicate textures (same texture used by multiple materials)
            unique_textures = {}  # image_name -> image
            texture_to_unique_idx = {}  # image_name -> unique_index

            for mat_idx, image in material_textures.items():
                if image.name not in unique_textures:
                    unique_textures[image.name] = image

            # Map material indices to unique texture indices
            image_name_to_idx = {name: idx for idx, name in enumerate(unique_textures.keys())}
            material_to_texture_idx = {mat_idx: image_name_to_idx[img.name]
                                       for mat_idx, img in material_textures.items()}

            has_multiple_textures = len(unique_textures) > 1
            has_any_textures = len(material_textures) > 0
            has_untextured_materials = len(material_colors) > 0

            print(f"  → Unique textures: {len(unique_textures)}, "
                  f"Textured materials: {len(material_textures)}, "
                  f"Untextured materials: {len(material_colors)}")

            if has_multiple_textures or (has_any_textures and has_untextured_materials):
                print(f"  → Creating texture atlas...")

                import math
                import numpy as np

                # Create solid color textures for materials without textures
                for mat_idx, color in material_colors.items():
                    # Create a small 64x64 solid color texture
                    color_img = bpy.data.images.new(
                        f"SolidColor_{mat_idx}",
                        width=64,
                        height=64,
                        alpha=True
                    )
                    # Fill with solid color
                    pixels = np.full((64, 64, 4), color, dtype=np.float32)
                    color_img.pixels = pixels.flatten().tolist()
                    color_img.update()

                    # Add to textures
                    material_textures[mat_idx] = color_img
                    if color_img.name not in unique_textures:
                        unique_textures[color_img.name] = color_img
                        image_name_to_idx[color_img.name] = len(image_name_to_idx)
                        material_to_texture_idx[mat_idx] = image_name_to_idx[color_img.name]

                # Get all unique images
                unique_images = list(unique_textures.values())

                # Calculate atlas layout (simple grid layout)
                n_textures = len(unique_images)
                grid_size = int(math.ceil(math.sqrt(n_textures)))

                # Get texture size statistics
                texture_sizes = [(img.size[0], img.size[1]) for img in unique_images]
                min_width = min(w for w, h in texture_sizes)
                max_width = max(w for w, h in texture_sizes)
                min_height = min(h for w, h in texture_sizes)
                max_height = max(h for w, h in texture_sizes)

                print(f"  → Texture resolution range: "
                      f"{min_width}x{min_height} to {max_width}x{max_height}")

                # Warn if there's significant size mismatch (>2x difference)
                if max_width > 2 * min_width or max_height > 2 * min_height:
                    wasted_space_pct = (1 - (min_width * min_height) / (max_width * max_height)) * 100
                    print(f"  ⚠ Warning: Large texture size variance detected. "
                          f"Smallest texture will use ~{100 - wasted_space_pct:.1f}% of its tile space.")

                # Create atlas with consistent tile size (using max dimensions)
                atlas_width = max_width * grid_size
                atlas_height = max_height * grid_size

                print(f"  → Creating {atlas_width}x{atlas_height} atlas "
                      f"({grid_size}x{grid_size} grid, {max_width}x{max_height} per tile)")
                print(f"  → NOTE: Textures will NOT be resized - original resolutions preserved")

                # Create new image for atlas
                atlas_image = bpy.data.images.new(
                    "TextureAtlas",
                    width=atlas_width,
                    height=atlas_height,
                    alpha=True
                )

                # Initialize atlas with white (so empty areas look reasonable)
                pixels = np.ones((atlas_height, atlas_width, 4), dtype=np.float32)

                # Copy each texture into the atlas
                texture_positions = {}  # texture_idx -> (u_offset, v_offset, u_scale, v_scale)

                for tex_idx, (img_name, image) in enumerate(unique_textures.items()):
                    # Calculate grid position
                    grid_x = tex_idx % grid_size
                    grid_y = tex_idx // grid_size

                    # Calculate pixel offsets in atlas
                    x_offset = grid_x * max_width
                    y_offset = grid_y * max_height

                    # IMPORTANT: Get original image pixels WITHOUT resizing
                    # We use image.size to get the ORIGINAL dimensions
                    orig_width = image.size[0]
                    orig_height = image.size[1]

                    # Extract raw pixel data at original resolution (optimized: use foreach_get)
                    img_pixels = np.empty((orig_height, orig_width, 4), dtype=np.float32)
                    image.pixels.foreach_get(img_pixels.ravel())

                    # Verify no accidental resizing occurred
                    assert img_pixels.shape[0] == orig_height, \
                        f"Height mismatch: {img_pixels.shape[0]} != {orig_height}"
                    assert img_pixels.shape[1] == orig_width, \
                        f"Width mismatch: {img_pixels.shape[1]} != {orig_width}"

                    # Flip vertically (Blender stores images bottom-up, but OpenGL expects top-down)
                    img_pixels = np.flipud(img_pixels)

                    # Copy to atlas at original resolution (NO RESIZING)
                    y_end = y_offset + orig_height
                    x_end = x_offset + orig_width
                    pixels[y_offset:y_end, x_offset:x_end] = img_pixels

                    # Calculate UV scale and offset based on ORIGINAL size
                    u_scale = orig_width / atlas_width
                    v_scale = orig_height / atlas_height
                    u_offset = x_offset / atlas_width
                    v_offset = y_offset / atlas_height

                    texture_positions[tex_idx] = (u_offset, v_offset, u_scale, v_scale)

                    # Calculate tile utilization percentage
                    tile_utilization = (orig_width * orig_height) / (max_width * max_height) * 100

                    print(f"    → Texture {tex_idx} ('{img_name[:30]}...'): "
                          f"tile ({grid_x}, {grid_y}), "
                          f"resolution {orig_width}x{orig_height} (ORIGINAL), "
                          f"tile usage {tile_utilization:.1f}%")

                # Flatten and assign pixels to atlas (optimized: use foreach_set)
                pixels = np.flipud(pixels)  # Flip back for Blender
                atlas_image.pixels.foreach_set(pixels.ravel())
                atlas_image.update()

                # Verify atlas was created at correct resolution
                print(f"  → Atlas created successfully: {atlas_image.size[0]}x{atlas_image.size[1]}")
                assert atlas_image.size[0] == atlas_width, \
                    f"Atlas width mismatch: {atlas_image.size[0]} != {atlas_width}"
                assert atlas_image.size[1] == atlas_height, \
                    f"Atlas height mismatch: {atlas_image.size[1]} != {atlas_height}"
                print(f"  ✓ All textures preserved at original resolution (no downsampling)")

                # Remap UVs based on original material assignment
                if has_uvs:
                    uv_layer = merged_obj.data.uv_layers[0]
                    remapped_count = 0
                    skipped_count = 0

                    for poly in merged_obj.data.polygons:
                        mat_idx = poly.material_index

                        # Map material index to texture index
                        if mat_idx in material_to_texture_idx:
                            tex_idx = material_to_texture_idx[mat_idx]

                            if tex_idx in texture_positions:
                                u_offset, v_offset, u_scale, v_scale = texture_positions[tex_idx]

                                # Remap UVs for this polygon
                                for loop_idx in poly.loop_indices:
                                    uv = uv_layer.data[loop_idx].uv
                                    # Scale and offset to new position in atlas
                                    uv[0] = uv[0] * u_scale + u_offset
                                    uv[1] = uv[1] * v_scale + v_offset

                                remapped_count += 1
                            else:
                                skipped_count += 1
                        else:
                            skipped_count += 1

                    print(f"  → Remapped UVs for {remapped_count} polygons "
                          f"(skipped {skipped_count} without textures)")

                # Replace first material's texture with atlas
                first_mat = merged_obj.data.materials[0]
                if first_mat and first_mat.use_nodes:
                    for node in first_mat.node_tree.nodes:
                        if node.type == 'TEX_IMAGE':
                            node.image = atlas_image
                            print(f"  → Assigned atlas to first material")
                            break

            # Keep only the first material, remove others
            # This ensures Genesis sees it as a single mesh (required for MPM)
            first_material = merged_obj.data.materials[0] if merged_obj.data.materials else None

            while len(merged_obj.data.materials) > 1:
                merged_obj.data.materials.pop(index=1)

            # Assign all faces to material slot 0
            for poly in merged_obj.data.polygons:
                poly.material_index = 0

            if has_multiple_textures or has_untextured_materials:
                print(f"  → Consolidated to 1 material with atlas texture")
            else:
                print(f"  → Consolidated to 1 material")

        # Final UV validation after all remapping
        if has_uvs:
            uv_layer = merged_obj.data.uv_layers[0]
            uv_data = uv_layer.data

            invalid_uvs = 0
            nan_uvs = 0
            for uv in uv_data:
                # Check for NaN
                if not (uv.uv[0] == uv.uv[0] and uv.uv[1] == uv.uv[1]):
                    nan_uvs += 1
                    uv.uv[0] = 0.5
                    uv.uv[1] = 0.5
                # Check for extremely large values
                elif not (-100 <= uv.uv[0] <= 100 and -100 <= uv.uv[1] <= 100):
                    invalid_uvs += 1
                    # Clamp to [0, 1] range
                    uv.uv[0] = max(0.0, min(1.0, uv.uv[0]))
                    uv.uv[1] = max(0.0, min(1.0, uv.uv[1]))

            if nan_uvs > 0:
                print(f"  → Fixed {nan_uvs} NaN UV coordinates")
            if invalid_uvs > 0:
                print(f"  → Clamped {invalid_uvs} out-of-range UV coordinates")

        # Export as GLB (without animation data)
        # Ensure textures and materials are exported correctly
        bpy.ops.export_scene.gltf(
            filepath=str(dest_file),
            export_format='GLB',
            use_selection=False,
            export_animations=False,  # Export only the current pose
            export_materials='EXPORT',  # Export materials
            export_image_format='AUTO',  # Let Blender choose best format (PNG for quality)
            export_texcoords=True,  # Export UV coordinates
            export_colors=True,  # Export vertex colors if present
            export_apply=False,  # Don't apply modifiers (already done)
            export_texture_dir='',  # Embed textures in GLB
            export_keep_originals=False,  # Don't create separate texture files
        )

        print(f"  → Exported with embedded textures (no external files)")

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


def load_and_preprocess_mesh(obj_path, model_identifier,
                             animation_idx, output_folder):
    """
    Load mesh and optionally filter bounding box helpers.

    Parameters
    ----------
    obj_path : str
        Path to mesh file
    model_identifier : str
        Model identifier for temp file naming
    animation_idx : int
        Animation index for temp file naming
    output_folder : str
        Output folder for temp files

    Returns
    -------
    tuple
        (mesh_file_to_use, mesh_cleaned, mesh_repaired)
    """
    mesh_file_to_use = obj_path
    mesh_cleaned = False
    mesh_repaired = False

    # Only export if we actually removed something
    temp_clean_dir = os.path.join(output_folder, '.temp_clean')
    os.makedirs(temp_clean_dir, exist_ok=True)
    temp_mesh_path = os.path.join(
        temp_clean_dir, f"{model_identifier}_{animation_idx}_merged.glb")

    # Sanity check: ensure we're not overwriting the original
    assert temp_mesh_path != obj_path, "ERROR: Would overwrite original file!"

    merge_glb_submeshes(obj_path, temp_mesh_path, random_anim_action=True)

    mesh_file_to_use = temp_mesh_path
    mesh_cleaned = True
    print(f"  → Using merged GLB: {temp_mesh_path}")

    return mesh_file_to_use, mesh_cleaned, mesh_repaired


def try_create_entity_with_position_retries(
        mesh_file_to_use, scale, pos, quat, material_params,
        camera_configs, center, particle_size, grid_density,
        lower_bound, upper_bound, dt, substeps, gravity,
        width, height, fov, camera_radius, max_position_retries,
        model_identifier, animation_idx, output_folder, obj_path):
    """
    Try to create entity with position retries and optional mesh repair.

    Returns
    -------
    EntityCreationResult
        Result containing scene, cameras, and metadata
    """
    _, _, _, mat_elastic = material_params
    mesh_cleaned = (mesh_file_to_use != obj_path)
    mesh_repaired = False
    position_retry = 0

    while position_retry < max_position_retries:
        try:
            # Log what we're passing to Genesis
            ps_debug = f"{particle_size:.6f}" if particle_size is not None else "None (auto)"
            print(f"    [DEBUG] Passing to Genesis: particle_size={ps_debug}, grid_density={grid_density}")

            # Create scene
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

            # Surface for mesh - provide fallback color for objects without textures
            # Genesis will use GLB textures if available, otherwise fall back to this color
            # Random color ensures visual distinction between objects
            random_color = tuple(np.random.uniform(0.0, 1.0, size=3))
            surface = gs.surfaces.Default(color=random_color, vis_mode="recon_simple")

            # Add entity
            scene.add_entity(
                material=mat_elastic,
                morph=gs.morphs.Mesh(
                    file=mesh_file_to_use,
                    scale=scale,
                    pos=pos,
                    quat=quat,
                    decimate=False,
                    normalize=True,
                ),
                surface=surface,
            )

            # Add cameras
            cameras = []
            for i, cam_config in enumerate(camera_configs):
                cam_pos = orbit_camera_position(
                    cam_config['elevation'],
                    cam_config['rotation'],
                    camera_radius
                )
                cam_pos += center

                lookat = center
                up = np.array([0., 0., 1.])

                cam = scene.add_camera(
                    res=(width, height),
                    pos=cam_pos,
                    lookat=lookat,
                    up=up,
                    fov=fov,
                    GUI=False,
                )
                cameras.append(cam)

            # Build scene to initialize particles
            scene.build()

            # Get particle count and actual particle size
            n_particles = scene._sim.active_solvers[-1].particles.pos.shape[1]
            actual_particle_size = scene.mpm_options.particle_size

            # Check if Genesis overrode our particle_size
            if particle_size is not None and abs(actual_particle_size - particle_size) > 1e-6:
                print(f"    [WARNING] Genesis overrode particle_size!")
                print(f"    [WARNING] Requested: {particle_size:.6f}m, Got: {actual_particle_size:.6f}m")

            # Check if particle_size precision might be causing cache issues
            if particle_size is not None:
                print(f"    [DEBUG] particle_size type: {type(particle_size)}, value: {particle_size!r}")

            return EntityCreationResult(
                success=True,
                scene=scene,
                cameras=cameras,
                n_particles=n_particles,
                actual_particle_size=actual_particle_size,
                mesh_file_to_use=mesh_file_to_use,
                mesh_cleaned=mesh_cleaned,
                mesh_repaired=mesh_repaired
            )

        except Exception as e:
            error_msg = str(e)

            # Check for unrecoverable errors
            if "sub-mesh" in error_msg.lower() or "multiple sub-meshes" in error_msg.lower():
                print(f"  ⚠ Skipping object: mesh has multiple sub-meshes (not supported)")
                return EntityCreationResult(False, None, [], 0, 0.0, mesh_file_to_use,
                                             mesh_cleaned, mesh_repaired)

            # Handle boundary errors - try different positions
            elif "bound" in error_msg.lower() or "outside" in error_msg.lower():
                position_retry += 1
                if position_retry < max_position_retries:
                    print(f"  ⚠ Boundary error: {error_msg}")
                    print(f"  → Retrying with new position ({position_retry}/{max_position_retries})...")
                    # Generate new position for retry
                    pos = np.clip(
                        center + np.array([0., 0., 0.2]) + scale * 0.1 * np.random.randn(3),
                        lower_bound + scale / 2,
                        upper_bound - scale / 2
                    )
                    quat = np.random.randn(4)
                    quat = quat / np.linalg.norm(quat)
                    continue
                else:
                    print(f"  ⚠ Skipping object: boundary error persists after retries")
                    return EntityCreationResult(False, None, [], 0, 0.0, mesh_file_to_use,
                                                 mesh_cleaned, mesh_repaired)
            else:
                # Unknown error - skip
                print(f"  ⚠ Skipping object due to error: {error_msg}")
                return EntityCreationResult(False, None, [], 0, 0.0, mesh_file_to_use,
                                             mesh_cleaned, mesh_repaired)

    print(f"  ⚠ Failed to create entity after {max_position_retries} position retries")
    return EntityCreationResult(False, None, [], 0, 0.0, mesh_file_to_use,
                                 mesh_cleaned, mesh_repaired)


def run_simulation_and_save(scene, cameras, save_root, init_vel,
                             n_sim_steps, vis_substeps):
    """
    Run simulation and save frames.

    Parameters
    ----------
    scene : gs.Scene
        Scene to simulate
    cameras : list
        List of cameras
    save_root : str
        Root directory for saving frames
    init_vel : np.ndarray
        Initial velocity
    n_sim_steps : int
        Number of simulation steps
    vis_substeps : int
        Visualization substeps

    Returns
    -------
    tuple
        (success: bool, n_frames: int)
    """
    # Create output directories
    os.makedirs(save_root, exist_ok=True)

    # Pre-create all subdirectories (optimization: avoid repeated makedirs in render loop)
    for c in range(len(cameras)):
        view_folder = os.path.join(save_root, f'{c:03d}')
        os.makedirs(os.path.join(view_folder, 'img'), exist_ok=True)
        os.makedirs(os.path.join(view_folder, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'particles'), exist_ok=True)

    # Reset scene
    scene.reset()

    # Set initial velocity
    with torch.no_grad():
        for entity in scene.entities:
            if not isinstance(entity, gs.engine.entities.MPMEntity):
                continue
            for i in range(entity.particle_start, entity.particle_end):
                scene._sim.active_solvers[-1].particles[0, i, 0].vel = ti.Vector(init_vel)

    # Run simulation and save frames
    frame_idx = 0
    is_wrong = False

    for i in range(n_sim_steps):
        if i % vis_substeps == 0:
            # Clear mesh reconstruction cache for new frame
            # This ensures we reconstruct for new particle positions
            # and prevents memory leaks from accumulating cached meshes
            gmc.clear_cache()

            is_wrong = float(scene.sim.active_solvers[0].particles_ng.active.to_numpy().mean()) < 1
            if is_wrong:
                break

            # Render all cameras
            for c, cam in enumerate(cameras):
                rgb, depth, seg, normal = cam.render(depth=True, segmentation=True)

                # Free unused render outputs immediately
                del depth, normal

                # Create alpha mask
                alpha = (seg == 1).astype(rgb.dtype)

                # Get folder paths for this view (directories already created)
                view_folder = os.path.join(save_root, f'{c:03d}')
                img_folder = os.path.join(view_folder, 'img')
                mask_folder = os.path.join(view_folder, 'mask')

                # Save white-background image
                mask_3ch = alpha.reshape(*alpha.shape, 1)
                white_img = rgb * mask_3ch + (1 - mask_3ch) * 255
                white_img = np.clip(white_img, 0, 255).astype(np.uint8)

                # Save as JPEG
                cv2.imwrite(
                    os.path.join(img_folder, f'{frame_idx:03d}.jpg'),
                    cv2.cvtColor(white_img, cv2.COLOR_RGB2BGR)
                )

                # Save mask as PNG
                cv2.imwrite(
                    os.path.join(mask_folder, f'{frame_idx:03d}.png'),
                    (alpha * 255).astype(np.uint8)
                )

                # Free memory after saving
                del rgb, seg, alpha, mask_3ch, white_img

            # Save particles (same for all cameras at this frame)
            particles_folder = os.path.join(save_root, 'particles')

            # Store particle positions in variable for explicit cleanup
            particle_pos = scene._sim.active_solvers[-1].particles.pos.to_numpy()[0]
            save_points_as_ply(
                particle_pos,
                os.path.join(particles_folder, f"{frame_idx:03d}.ply")
            )
            del particle_pos

            frame_idx += 1

            # Periodic garbage collection to prevent memory buildup
            if frame_idx % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        if is_wrong:
            break

        scene.step()

    if is_wrong:
        print("Simulation failed: object disappeared")
        return False, frame_idx

    return True, frame_idx


def process_single_object(
        obj_path,
        synset_idx,
        model_identifier,
        animation_idx,
        args):
    """Process a single object file and generate simulation data."""

    # Constants
    N_CAMERAS_RANDOM = 32
    N_CAMERAS_FIXED = 16
    N_CAMERAS_TOTAL = N_CAMERAS_RANDOM + N_CAMERAS_FIXED  # 48 total

    # Simulation parameters
    N_SIM_STEPS = args.n_sim_steps
    SUBSTEPS = 1
    SIM_REQUIRES_GRAD = False
    DT = 2.5e-4
    GRAVITY = (0, 0, -9.81)
    LOWER_BOUND = np.array([-5.0, -5.0, -0.7]) - 0.046875
    UPPER_BOUND = np.array([5.0, 5.0, 2.0]) + 0.046875
    CENTER = np.array([0.0, 0.0, args.center_z])

    # Rendering parameters
    FPS = args.fps
    HEIGHT = args.resolution
    WIDTH = args.resolution
    FOV = args.fov
    VIS_SUBSTEPS = int(1 / FPS / (DT * SUBSTEPS))  # num of sim steps per vis

    # Setup paths following reference format
    # save_root = output_folder / synset_idx / model_identifier / animation_idx
    save_root = os.path.join(
        args.output_folder,
        synset_idx,
        model_identifier,
        f'{animation_idx:03d}'
    )
    # Don't create directories yet - wait until entity is successfully created

    # ========================================================================
    # CONSTANTS
    # ========================================================================
    MAX_SIMULATION_RETRIES = 3
    MAX_ADJUSTMENT_RETRIES = 10
    MAX_POSITION_RETRIES = 5
    MIN_PARTICLE_COUNT = 16384
    MIN_PARTICLE_SIZE = 0.001  # Minimum particle size (1mm)
    MAX_PARTICLE_COUNT = 131072  # 128k particles (limit for performance)
    INITIAL_GRID_DENSITY = 64  # Starting grid density (resets for each object)

    # Compute maximum particle size from grid stability constraint
    # Each grid cell should contain ~8 particles on average
    # This requires: particle_size <= grid_cell_size / 2
    GRID_CELL_SIZE = compute_grid_cell_size(INITIAL_GRID_DENSITY)
    MAX_PARTICLE_SIZE = GRID_CELL_SIZE / 2.0
    print(f"  → Initial grid_density: {INITIAL_GRID_DENSITY}")
    print(f"  → Grid cell size: {GRID_CELL_SIZE:.4f}m (uniform, independent of domain)")
    print(f"  → Grid constraint: particle_size <= {MAX_PARTICLE_SIZE:.4f}m")

    # ========================================================================
    # ONE-TIME SETUP (fixed for all retries)
    # ========================================================================
    # Setup camera configurations
    camera_configs = setup_camera_configs(
        n_random=N_CAMERAS_RANDOM,
        n_fixed=N_CAMERAS_FIXED,
        elevation_range_random=(-5, 30),
        rotation_range=(0, 360),
        elevation_fixed=0.0
    )

    # Setup lights
    lights = setup_lights(
        n_base_lights=args.n_lights,
        n_variation=args.n_lights_variation,
        center=CENTER
    )

    # Load and preprocess mesh (only once)
    mesh_file_to_use, mesh_cleaned, mesh_repaired = load_and_preprocess_mesh(
        obj_path=obj_path,
        model_identifier=model_identifier,
        animation_idx=animation_idx,
        output_folder=args.output_folder
    )

    # ========================================================================
    # LEVEL 1: SIMULATION RETRY (change material & velocity on failure)
    # ========================================================================
    for sim_retry in range(MAX_SIMULATION_RETRIES):
        print(f"\n{'='*60}")
        print(f"Simulation attempt {sim_retry + 1}/{MAX_SIMULATION_RETRIES}")
        print(f"{'='*60}")

        # Randomize physics once per simulation attempt
        E, nu, rho, mat_elastic = create_random_material()
        init_vel = create_random_velocity()
        material_params = (E, nu, rho, mat_elastic)

        # ====================================================================
        # LEVEL 2: PARTICLE/VOLUME ADJUSTMENT (change particle_size or scale)
        # ====================================================================
        particle_size = None  # Start with auto
        grid_density = INITIAL_GRID_DENSITY  # Reset grid density for each simulation attempt
        min_scale_bound = args.min_scale
        scale = None  # Will be sampled on first attempt
        prev_n_particles = None  # Track if we're making progress

        for adjustment_retry in range(MAX_ADJUSTMENT_RETRIES):
            # Ensure min_scale_bound doesn't exceed max_scale (due to floating point errors)
            min_scale_bound = min(min_scale_bound, args.max_scale - 1e-6)

            # Sample scale only on first attempt or when explicitly needed
            # Do NOT resample scale when retrying due to particle_size adjustments!
            if scale is None:
                scale = np.random.uniform(min_scale_bound, args.max_scale)
                print(f"  → Sampled scale: {scale:.3f}")

            # Generate initial position and orientation
            pos = np.clip(
                CENTER + np.array([0., 0., 0.2]) + scale * 0.1 * np.random.randn(3),
                LOWER_BOUND + scale / 2,
                UPPER_BOUND - scale / 2
            )
            quat = np.random.randn(4)
            quat = quat / np.linalg.norm(quat)

            # ================================================================
            # LEVEL 3 & 4: POSITION RETRY + MESH REPAIR
            # ================================================================
            ps_str = f"{particle_size:.6f}m" if particle_size else "auto"
            print(f"  → Attempt {adjustment_retry + 1}/{MAX_ADJUSTMENT_RETRIES}: "
                  f"particle_size={ps_str}, grid_density={grid_density}")

            result = try_create_entity_with_position_retries(
                mesh_file_to_use=mesh_file_to_use,
                scale=scale,
                pos=pos,
                quat=quat,
                material_params=material_params,
                camera_configs=camera_configs,
                center=CENTER,
                particle_size=particle_size,
                grid_density=grid_density,  # Use variable grid_density (can change during retries)
                lower_bound=LOWER_BOUND,
                upper_bound=UPPER_BOUND,
                dt=DT,
                substeps=SUBSTEPS,
                gravity=GRAVITY,
                width=WIDTH,
                height=HEIGHT,
                fov=FOV,
                camera_radius=args.camera_radius,
                max_position_retries=MAX_POSITION_RETRIES,
                model_identifier=model_identifier,
                animation_idx=animation_idx,
                output_folder=args.output_folder,
                obj_path=obj_path
            )

            if not result.success:
                cleanup_temp_mesh_files(
                    mesh_cleaned, mesh_repaired, mesh_file_to_use, obj_path
                )
                return  # Skip object (unrecoverable error)

            # Update mesh tracking if repair happened
            mesh_file_to_use = result.mesh_file_to_use
            mesh_repaired = result.mesh_repaired

            # Log what Genesis actually used
            print(f"  → Genesis created: {result.n_particles} particles, "
                  f"actual_particle_size={result.actual_particle_size:.6f}m")

            # Check if we're stuck (same particle count as before)
            is_stuck = False
            if prev_n_particles is not None and result.n_particles == prev_n_particles:
                print(f"  ⚠ WARNING: Particle count unchanged ({result.n_particles})")
                print(f"  ⚠ This suggests Genesis loaded a cached .ptc file!")
                print(f"  ⚠ Will apply aggressive particle_size increase to break cache cycle")
                is_stuck = True
            prev_n_particles = result.n_particles

            # ================================================================
            # VALIDATION: Check particle count
            # ================================================================
            # Compute max_particle_size based on CURRENT grid_density (not initial)
            current_grid_cell_size = compute_grid_cell_size(grid_density)
            current_max_particle_size = current_grid_cell_size / 2.0

            if grid_density != INITIAL_GRID_DENSITY:
                print(f"  → Updated max_particle_size: {MAX_PARTICLE_SIZE:.6f}m (grid={INITIAL_GRID_DENSITY}) "
                      f"→ {current_max_particle_size:.6f}m (grid={grid_density})")

            particle_valid, new_particle_size, new_grid_density = check_particle_count(
                n_particles=result.n_particles,
                actual_particle_size=result.actual_particle_size,
                min_count=MIN_PARTICLE_COUNT,
                min_size=MIN_PARTICLE_SIZE,
                max_count=MAX_PARTICLE_COUNT,
                max_size=current_max_particle_size,  # Use current, not initial!
                grid_density=grid_density,
                is_stuck=is_stuck
            )

            if not particle_valid:
                # Check if object should be skipped (unfixable)
                if new_particle_size is None and new_grid_density is None:
                    print(f"  ⚠ Skipping object: cannot satisfy constraints")
                    cleanup_scene(result.scene, result.cameras)
                    cleanup_temp_mesh_files(
                        mesh_cleaned, mesh_repaired, mesh_file_to_use, obj_path
                    )
                    return  # Skip this object

                # Update parameters for retry
                old_particle_size = particle_size
                if new_particle_size is not None:
                    particle_size = float(new_particle_size)  # Ensure it's a float
                    old_str = f"{old_particle_size:.6f}m" if old_particle_size else "auto"
                    print(f"  → Updated particle_size: {old_str} → {particle_size:.6f}m")
                if new_grid_density is not None:
                    old_grid_density = grid_density
                    grid_density = new_grid_density
                    print(f"  → Updated grid_density: {old_grid_density} → {grid_density}")

                cleanup_scene(result.scene, result.cameras)
                continue  # Retry with adjusted particle_size and/or grid_density

            # ================================================================
            # SUCCESS: Entity created with valid particle count
            # ================================================================
            break
        else:
            # Exhausted all adjustment retries
            print(f"  ⚠ Failed after {MAX_ADJUSTMENT_RETRIES} adjustment retries")
            cleanup_temp_mesh_files(
                mesh_cleaned, mesh_repaired, mesh_file_to_use, obj_path
            )
            return

        # ====================================================================
        # RUN SIMULATION
        # ====================================================================
        success, num_frames = run_simulation_and_save(
            scene=result.scene,
            cameras=result.cameras,
            save_root=save_root,
            init_vel=init_vel,
            n_sim_steps=N_SIM_STEPS,
            vis_substeps=VIS_SUBSTEPS
        )

        if success:
            # ================================================================
            # SAVE METADATA
            # ================================================================
            print(f"Simulation completed with {num_frames} frames")

            # Save camera parameters (FOV and extrinsics)
            for c, cam in enumerate(result.cameras):
                camera_folder = os.path.join(save_root, f'{c:03d}', 'camera')
                os.makedirs(camera_folder, exist_ok=True)

                # Get FOV (field of view in degrees)
                fov = cam.fov
                # Replicate for all frames (FOV doesn't change)
                fov_frames = np.full(num_frames, fov, dtype=np.float32)
                np.save(
                    os.path.join(camera_folder, 'fov.npy'),
                    fov_frames
                )

                # Get extrinsics matrix (4x4 camera-to-world transform)
                extrinsics = cam.transform
                extrinsics_frames = np.tile(extrinsics, (num_frames, 1, 1))
                np.save(
                    os.path.join(camera_folder, 'extrinsics.npy'),
                    extrinsics_frames
                )

            # Save physical parameters
            physics_folder = os.path.join(save_root, 'physics')
            os.makedirs(physics_folder, exist_ok=True)

            np.save(os.path.join(physics_folder, 'youngs_modulus.npy'), E)
            np.save(os.path.join(physics_folder, 'poisson_ratio.npy'), nu)
            np.save(os.path.join(physics_folder, 'initial_velocity.npy'), init_vel)
            np.save(os.path.join(physics_folder, 'gravity.npy'), np.array(GRAVITY))
            np.save(os.path.join(physics_folder, 'particle_size.npy'), result.actual_particle_size)

            # Create tar files
            uid = f"{synset_idx}-{model_identifier}-{animation_idx:03d}"
            create_tar_files(save_root, uid, args.output_folder)

            print(f"Dataset created successfully at {save_root}")
            print(f"Total cameras: {N_CAMERAS_TOTAL} ({N_CAMERAS_RANDOM} random + {N_CAMERAS_FIXED} fixed)")
            print(f"Total frames: {num_frames}")

            # Clean up scene and temporary files
            cleanup_scene(result.scene, result.cameras)
            cleanup_temp_mesh_files(mesh_cleaned, mesh_repaired, mesh_file_to_use, obj_path)
            return  # SUCCESS!

        else:
            # Simulation failed (object disappeared), retry with new material
            print("  → Retrying with new material and velocity...")
            cleanup_scene(result.scene, result.cameras)
            continue  # Next sim_retry with new material/velocity

    # All retries exhausted
    print(f"  ⚠ Failed after {MAX_SIMULATION_RETRIES} simulation retries")
    cleanup_temp_mesh_files(mesh_cleaned, mesh_repaired, mesh_file_to_use, obj_path)


def main(args):
    """Main function that finds all glb files and processes them."""
    # Find all .glb files in the input folder
    glb_pattern = os.path.join(args.input_folder, '**', '*.glb')
    all_glb_files = sorted(glob(glb_pattern, recursive=True))

    print(f"Found {len(all_glb_files)} glb files in {args.input_folder}")

    # Filter files using idx, stride, and n_samples
    if args.n_samples is not None:
        all_glb_files = all_glb_files[:args.n_samples]
    all_glb_files = all_glb_files[args.idx::args.stride]

    if len(all_glb_files) == 0:
        print(f"No .glb files found in {args.input_folder}")
        return

    print(f"Processing {len(all_glb_files)} objects...")

    # Enable mesh reconstruction caching for performance
    # This caches mesh reconstruction per frame, avoiding redundant work
    # across 48 cameras viewing the same particle positions
    gmc.enable_cache()

    for idx, glb_path in enumerate(all_glb_files):
        # Extract synset_idx and model_identifier from path
        # Path structure: input_folder/synset_idx/model_identifier.glb
        path_obj = Path(glb_path)
        model_identifier = path_obj.stem  # filename without extension
        synset_idx = path_obj.parent.name  # parent directory name

        print(
            f"\n[{idx+1}/{len(all_glb_files)}] Processing: {synset_idx}/{model_identifier}")

        # Check if output already exists BEFORE initializing Genesis
        animation_idx = 0
        uid = f"{synset_idx}-{model_identifier}-{animation_idx:03d}"
        random_tar_path = os.path.join(args.output_folder, f'random_clip-{uid}')
        fixed_tar_path = os.path.join(args.output_folder, f'fixed_16_clip-{uid}')

        if os.path.exists(random_tar_path) and os.path.exists(fixed_tar_path):
            if not args.overwrite:
                print(f"  → Output already exists, skipping (use --overwrite to regenerate)")
                continue
            else:
                print(f"  → Output exists but overwrite=True, regenerating...")

        # Initialize Genesis before processing each object
        # This ensures clean Taichi state for each object
        gs.init(precision="32")

        # Reset cache stats for this object
        gmc.clear_cache()

        try:
            process_single_object(
                obj_path=glb_path,
                synset_idx=synset_idx,
                model_identifier=model_identifier,
                animation_idx=animation_idx,
                args=args
            )
            print(f"✓ Successfully processed {synset_idx}/{model_identifier}")

            # Show cache performance stats
            stats = gmc.get_cache_stats()
            if stats['hits'] > 0 or stats['misses'] > 0:
                cache_rate = 100 * stats['hits'] / (stats['hits'] + stats['misses'])
                print(f"  → Mesh cache: {stats['hits']} hits, {stats['misses']} misses ({cache_rate:.1f}% hit rate)")
        except Exception as e:
            print(f"✗ Failed to process {synset_idx}/{model_identifier}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Properly destroy Genesis/Taichi after each object
            # This releases all GPU memory and resets Taichi's FieldsBuilder
            gs.destroy()
            torch.cuda.empty_cache()
            gc.collect()
            print(
                f"  → Genesis destroyed and GPU memory cleared after processing object {idx+1}/{len(all_glb_files)}")

    print(f"\n{'='*60}")
    print(f"Completed processing {len(all_glb_files)} objects")
    print(f"Output folder: {args.output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate physics simulation dataset in reference format')

    # Input/Output arguments
    parser.add_argument(
        '--input_folder',
        type=str,
        default='filtered_objs/glbs',
        help='Input folder containing glb files (default: filtered_objs/glbs)')
    parser.add_argument('-o', '--output_folder', type=str, default="black_hole",
                        help='Output folder for dataset')

    # Simulation arguments
    parser.add_argument('--n_sim_steps', type=int, default=3001, # 3200, # 4800,
                        help='Number of simulation steps')
    parser.add_argument('--fps', type=int, default=20,
                        help='Frames per second for recording')
    parser.add_argument('--center_z', type=float, default=-0.2,
                        help='Z coordinate of scene center')

    # Camera arguments
    parser.add_argument('--resolution', type=int, default=512,
                        help='Image resolution (width and height)')
    parser.add_argument('--fov', type=float, default=49.1,
                        help='Camera field of view in degrees')
    parser.add_argument('--camera_radius', type=float, default=1.5,
                        help='Camera distance from center')

    # Lighting arguments
    parser.add_argument('--n_lights', type=int, default=4,
                        help='Base number of lights for multi-light mode')
    parser.add_argument('--n_lights_variation', type=int, default=1,
                        help='Variation in number of lights (n ± m)')

    # Overwrite argument
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output files')

    parser.add_argument('--idx', type=int, default=0,
                        help='Starting index for processing files (default: 0)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for processing files (default: 1)')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of samples to process (default: None for all files)')

    parser.add_argument('--min_scale', type=float, default=0.6,
                        help='Minimum scale for object (default: 0.6)')
    parser.add_argument('--max_scale', type=float, default=0.9,
                        help='Maximum scale for object (default: 0.9)')

    args = parser.parse_args()

    main(args)

