"""
Genesis compatibility utilities.

This module provides wrapper functions that replicate functionality from older modified 
Genesis versions, allowing use with pip-installed Genesis without source modifications.
"""

import os
import trimesh
import numpy as np


def normalize_mesh_file(input_path, output_path=None):
    """
    Normalize a mesh file so that the maximum extent along x, y, z axes is 1.0
    and the mesh is centered at the origin (0, 0, 0).

    This replicates the `normalize=True` parameter from older Genesis versions.

    Parameters
    ----------
    input_path : str
        Path to the input mesh file
    output_path : str, optional
        Path to save the normalized mesh. If None, creates a temp file in the same
        directory with '_normalized' suffix.

    Returns
    -------
    str
        Path to the normalized mesh file
    """
    # Load scene/mesh while preserving textures/materials
    loaded = trimesh.load(input_path, force=None, skip_texture=False, process=False)

    if isinstance(loaded, trimesh.Scene):
        bounds = loaded.bounds
    else:
        bounds = loaded.bounds

    center = (bounds[0] + bounds[1]) / 2.0
    extents = bounds[1] - bounds[0]
    max_extent = extents.max()

    if max_extent == 0:
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_normalized{ext}"
        if isinstance(loaded, trimesh.Scene):
            loaded.export(output_path)
        else:
            loaded.export(output_path)
        return output_path

    # Normalize: center at origin and max extent = 1.0
    scale = 1.0 / max_extent
    translate = np.eye(4)
    translate[:3, 3] = -center
    scale_m = np.eye(4)
    scale_m[:3, :3] *= scale
    transform = scale_m @ translate

    if isinstance(loaded, trimesh.Scene):
        loaded.apply_transform(transform)
    else:
        loaded.apply_transform(transform)

    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_normalized{ext}"

    # Export normalized mesh/scene
    loaded.export(output_path)

    return output_path


def preprocess_mesh_for_genesis(mesh_file, normalize=False, temp_dir=None):
    """
    Preprocess a mesh file for Genesis, applying transformations that were
    previously handled by modified Genesis source code.

    Parameters
    ----------
    mesh_file : str
        Path to the mesh file
    normalize : bool, optional
        Whether to normalize the mesh to have max extent 1.0 centered at origin
    temp_dir : str, optional
        Directory to store temporary processed meshes. Defaults to same directory as input.

    Returns
    -------
    str
        Path to the preprocessed mesh file (may be the original if no processing needed)
    """
    if not normalize:
        return mesh_file

    # Determine output directory
    if temp_dir is None:
        temp_dir = os.path.dirname(mesh_file)

    # Create output path
    basename = os.path.basename(mesh_file)
    name, ext = os.path.splitext(basename)
    output_path = os.path.join(temp_dir, f"{name}_normalized{ext}")

    # Normalize if it doesn't already exist
    if not os.path.exists(output_path):
        normalize_mesh_file(mesh_file, output_path)

    return output_path
