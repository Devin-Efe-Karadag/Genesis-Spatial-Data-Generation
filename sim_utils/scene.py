import genesis as gs
import numpy as np
import torch
import gc
from .commons import orbit_camera_position


def create_scene_with_particles(
    mesh_file,
    particle_size,
    grid_density,
    scale,
    pos,
    quat,
    material,
    camera_configs,
    center,
    lower_bound,
    upper_bound,
    dt,
    substeps,
    gravity,
    width,
    height,
    min_fov,
    max_fov,
    min_radius,
    max_radius,
    max_position_retries=5,
):
    """
    Create a Genesis scene with particles sampled from mesh.
    Returns (scene, cameras, n_particles, actual_particle_size) or None.
    """
    position_retry = 0

    while position_retry < max_position_retries:
        try:
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

            # Add entity
            surface = gs.surfaces.Default(vis_mode="recon_simple")
            scene.add_entity(
                material=material,
                morph=gs.morphs.Mesh(
                    file=mesh_file,
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
                cam_radius = np.random.uniform(min_radius, max_radius)
                cam_fov = np.random.uniform(min_fov, max_fov)

                cam_pos = orbit_camera_position(
                    cam_config["elevation"], cam_config["rotation"], cam_radius
                )
                cam_pos += center

                lookat = center
                up = np.array([0.0, 0.0, 1.0])

                cam = scene.add_camera(
                    res=(width, height),
                    pos=cam_pos,
                    lookat=lookat,
                    up=up,
                    fov=cam_fov,
                    GUI=False,
                )
                cameras.append(cam)

            # Build scene to initialize particles
            scene.build()

            # Get particle count and actual particle size
            n_particles = scene._sim.active_solvers[-1].particles.pos.shape[1]
            actual_particle_size = scene.mpm_options.particle_size

            print(
                f"    ✓ Scene created: {n_particles} particles, particle_size={actual_particle_size:.6f}m"
            )

            return scene, cameras, n_particles, actual_particle_size

        except Exception as e:
            error_msg = str(e)

            # Check for unrecoverable errors
            if (
                "sub-mesh" in error_msg.lower()
                or "multiple sub-meshes" in error_msg.lower()
            ):
                print(f"  ✗ Mesh has multiple sub-meshes (not supported)")
                return None

            # Handle boundary errors - try different positions
            elif "bound" in error_msg.lower() or "outside" in error_msg.lower():
                position_retry += 1
                if position_retry < max_position_retries:
                    print(
                        f"  ⚠ Boundary error, retrying position ({position_retry}/{max_position_retries})..."
                    )
                    # Generate new position for retry
                    pos = np.clip(
                        center
                        + np.array([0.0, 0.0, 0.2])
                        + scale * 0.1 * np.random.randn(3),
                        lower_bound + scale / 2,
                        upper_bound - scale / 2,
                    )
                    quat = np.random.randn(4)
                    quat = quat / np.linalg.norm(quat)
                    continue
                else:
                    print(
                        f"  ✗ Boundary error persists after {max_position_retries} retries"
                    )
                    return None
            else:
                # Unknown error
                print(f"  ✗ Scene creation error: {error_msg}")
                return None

    print(f"  ✗ Failed to create scene after {max_position_retries} position retries")
    return None


def cleanup_scene(scene, cameras):
    """
    Clean up scene and cameras, free GPU memory.
    """
    if scene is not None:
        try:
            scene.destroy()
        except Exception as e:
            print(f"  → Warning: scene.destroy() failed: {e}")
        del scene
    if cameras:
        del cameras
    # torch.cuda.empty_cache() # Caller usually handles this globally? keeping strictly local here
    gc.collect()
