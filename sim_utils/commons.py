import os
import numpy as np
import trimesh
import tarfile


def sample_unit_vector(min_elevation_radian, max_elevation_radian):
    """
    Samples a 3D unit vector with azimuth uniformly distributed in [0, 2π)
    and elevation uniformly distributed in [min_elevation_radian, max_elevation_radian].
    """
    if min_elevation_radian == max_elevation_radian:
        cos_theta = np.cos([min_elevation_radian])
    else:
        cos_theta = np.random.uniform(
            np.cos(min_elevation_radian), np.cos(max_elevation_radian), 1
        )

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


def save_points_as_ply(points, filename):
    """Save particle positions as PLY file."""
    points = np.asarray(points)
    points = points[:, 0, [1, 2, 0]]  # due to genesis-specific coord system
    point_cloud = trimesh.points.PointCloud(vertices=points)
    point_cloud.export(file_obj=filename, file_type="ply", encoding="ascii")


def create_tar_files(save_root, uid, output_folder):
    """
    Create tar files following the reference format:
    - random_clip-{uid}: contains views 000-031 (random cameras)
    - fixed_16_clip-{uid}: contains views 032-047 (fixed cameras)
    """
    # Create tar for random cameras (views 0-31)
    random_tar_path = os.path.join(output_folder, f"random_clip-{uid}")
    with tarfile.open(random_tar_path, "w") as tar:
        for view_idx in range(32):
            view_folder = os.path.join(save_root, f"{view_idx:03d}")
            if os.path.exists(view_folder):
                tar.add(view_folder, arcname=f"{view_idx:03d}")

    print(f"Created tar file: {random_tar_path}")

    # Create tar for fixed cameras (views 32-47)
    fixed_tar_path = os.path.join(output_folder, f"fixed_16_clip-{uid}")
    with tarfile.open(fixed_tar_path, "w") as tar:
        for view_idx in range(32, 48):
            view_folder = os.path.join(save_root, f"{view_idx:03d}")
            if os.path.exists(view_folder):
                tar.add(view_folder, arcname=f"{view_idx:03d}")

    print(f"Created tar file: {fixed_tar_path}")


def cleanup_temp_mesh_files(mesh_cleaned, mesh_repaired, mesh_file_to_use, obj_path):
    """Helper function to clean up temporary mesh files."""
    if (
        mesh_cleaned
        and os.path.exists(mesh_file_to_use)
        and mesh_file_to_use != obj_path
    ):
        try:
            os.remove(mesh_file_to_use)
            print(f"  → Cleaned up temporary file: {mesh_file_to_use}")
        except Exception as e:
            pass  # Silently ignore cleanup errors


def validate_saved_frames(save_root, n_cameras_total, expected_n_frames):
    """
    Validate that all camera folders have the expected number of frames saved.
    """
    print(f"\nValidating saved frames...")
    print(f"  Expected: {n_cameras_total} cameras × {expected_n_frames} frames")

    # Check that all camera folders exist
    for cam_idx in range(n_cameras_total):
        cam_folder = os.path.join(save_root, f"{cam_idx:03d}")
        img_folder = os.path.join(cam_folder, "img")
        mask_folder = os.path.join(cam_folder, "mask")

        if not os.path.exists(cam_folder):
            print(f"  ✗ Missing camera folder: {cam_folder}")
            return False

        if not os.path.exists(img_folder):
            print(f"  ✗ Missing img folder in camera {cam_idx:03d}")
            return False

        if not os.path.exists(mask_folder):
            print(f"  ✗ Missing mask folder in camera {cam_idx:03d}")
            return False

        # Count frames in img and mask folders
        img_files = sorted([f for f in os.listdir(img_folder) if f.endswith(".jpg")])
        mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith(".png")])

        n_img_frames = len(img_files)
        n_mask_frames = len(mask_files)

        if n_img_frames != expected_n_frames:
            print(
                f"  ✗ Camera {cam_idx:03d}: Expected {expected_n_frames} images, found {n_img_frames}"
            )
            return False

        if n_mask_frames != expected_n_frames:
            print(
                f"  ✗ Camera {cam_idx:03d}: Expected {expected_n_frames} masks, found {n_mask_frames}"
            )
            return False

    # Check particles folder
    particles_folder = os.path.join(save_root, "particles")
    if not os.path.exists(particles_folder):
        print(f"  ✗ Missing particles folder")
        return False

    particle_files = sorted(
        [f for f in os.listdir(particles_folder) if f.endswith(".ply")]
    )
    n_particle_frames = len(particle_files)

    if n_particle_frames != expected_n_frames:
        print(
            f"  ✗ Expected {expected_n_frames} particle files, found {n_particle_frames}"
        )
        return False

    print(
        f"  ✓ Validation passed: all {n_cameras_total} cameras have {expected_n_frames} frames"
    )
    return True
