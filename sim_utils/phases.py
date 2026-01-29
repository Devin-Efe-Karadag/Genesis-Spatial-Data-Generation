import os
import numpy as np
import genesis as gs
import genesis_mesh_cache_patch as gmc
import cv2
import torch
import gc

from . import blender_ops
from . import physics
from . import scene
from . import commons

# ============================================================================
# PHASE 1: MESH PREPROCESSING
# ============================================================================


def preprocess_mesh(obj_path, model_identifier, animation_idx, output_folder):
    """
    Phase 1: Load and preprocess mesh (merge submeshes, create temp GLB).
    Returns path to preprocessed mesh file, or None if failed.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 1: Mesh Preprocessing")
    print(f"{'='*60}")

    temp_clean_dir = os.path.join(output_folder, ".temp_clean")
    os.makedirs(temp_clean_dir, exist_ok=True)
    temp_mesh_path = os.path.join(
        temp_clean_dir, f"{model_identifier}_{animation_idx}_merged.glb"
    )

    # Sanity check: ensure we're not overwriting the original
    assert temp_mesh_path != obj_path, "ERROR: Would overwrite original file!"

    try:
        success = blender_ops.merge_glb_submeshes(
            obj_path, temp_mesh_path, random_anim_action=True
        )

        if success:
            print(f"  ✓ Mesh preprocessing successful: {temp_mesh_path}")
            return temp_mesh_path
        else:
            print(f"  ✗ Mesh preprocessing failed")
            return None

    except Exception as e:
        print(f"  ✗ Mesh preprocessing error: {e}")
        import traceback

        traceback.print_exc()
        return None


# ============================================================================
# PHASE 2: PARTICLE SAMPLING
# ============================================================================


def find_valid_particle_config(
    mesh_file,
    min_count,
    max_count,
    min_size,
    max_attempts,
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
    min_scale,
    max_scale,
):
    """
    Phase 2: Find particle configuration that satisfies particle count constraints.
    Returns (ParticleConfig, scene, cameras) or None.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 2: Finding Valid Particle Configuration")
    print(f"{'='*60}")
    print(f"  Target: {min_count} <= n_particles <= {max_count}")

    # Initial parameters
    particle_size = None  # Auto
    grid_density = 64  # Initial grid density
    scale = np.random.uniform(min_scale, max_scale)
    prev_n_particles = None

    print(
        f"  Initial: particle_size=auto, grid_density={grid_density}, scale={scale:.3f}"
    )

    for attempt in range(max_attempts):
        print(f"\n  Attempt {attempt + 1}/{max_attempts}:")

        # Generate position and orientation
        pos = np.clip(
            center + np.array([0.0, 0.0, 0.2]) + scale * 0.1 * np.random.randn(3),
            lower_bound + scale / 2,
            upper_bound - scale / 2,
        )
        quat = np.random.randn(4)
        quat = quat / np.linalg.norm(quat)

        # Create temporary material (will be replaced in Phase 3)
        _, _, _, temp_material = physics.create_random_material()

        # Create scene with particles
        result = scene.create_scene_with_particles(
            mesh_file=mesh_file,
            particle_size=particle_size,
            grid_density=grid_density,
            scale=scale,
            pos=pos,
            quat=quat,
            material=temp_material,
            camera_configs=camera_configs,
            center=center,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dt=dt,
            substeps=substeps,
            gravity=gravity,
            width=width,
            height=height,
            min_fov=min_fov,
            max_fov=max_fov,
            min_radius=min_radius,
            max_radius=max_radius,
        )

        if result is None:
            # Scene creation failed (unrecoverable error)
            print(f"  ✗ Scene creation failed, cannot continue")
            return None

        sim_scene, cameras, n_particles, actual_particle_size = result
        print(
            f"[DEBUG] Phase 2: Scene created. n_particles={n_particles}, ptr={hex(id(sim_scene))}"
        )

        # Check if particle count is valid
        if min_count <= n_particles <= max_count:
            print(f"  ✓ Valid particle count found: {n_particles}")

            config = physics.ParticleConfig(
                particle_size=actual_particle_size,
                grid_density=grid_density,
                scale=scale,
                n_particles=n_particles,
            )
            print(
                f"[DEBUG] Phase 2: Returning valid config and scene {hex(id(sim_scene))}"
            )
            return config, sim_scene, cameras

        # Check if we're stuck (cache issue)
        is_stuck = False
        if prev_n_particles is not None and n_particles == prev_n_particles:
            print(
                f"  ⚠ Particle count unchanged ({n_particles}) - possible cache issue"
            )
            is_stuck = True
        prev_n_particles = n_particles

        # Adjust parameters using physics helper
        is_valid, new_size, new_density = physics.check_particle_count(
            n_particles,
            actual_particle_size,
            min_count,
            min_size,
            max_count,
            None,
            grid_density,
            is_stuck,
        )

        if not is_valid:
            if new_size is None and new_density is None:
                scene.cleanup_scene(sim_scene, cameras)
                return None

            particle_size = new_size  # Update for next iteration
            if new_density is not None:
                grid_density = new_density

        # Explicitly clean up invalid scene before retrying
        scene.cleanup_scene(sim_scene, cameras)

    print(f"  ✗ Failed to find valid particle config after {max_attempts} attempts")
    return None


# ============================================================================
# PHASE 3: SIMULATION LOOP
# ============================================================================


def _run_simulation_and_check(
    sim_scene, cameras, save_root, init_vel, n_sim_steps, vis_substeps
):
    """
    Run simulation and save frames. Check for failures (is_wrong).
    Internal helper for Phase 3.
    """
    # Create output directories
    os.makedirs(save_root, exist_ok=True)

    for c in range(len(cameras)):
        view_folder = os.path.join(save_root, f"{c:03d}")
        os.makedirs(os.path.join(view_folder, "img"), exist_ok=True)
        os.makedirs(os.path.join(view_folder, "mask"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "particles"), exist_ok=True)

    # Set initial velocity for MPM particles
    with torch.no_grad():
        for entity in sim_scene.entities:
            if not isinstance(entity, gs.engine.entities.MPMEntity):
                continue
            for i in range(entity.particle_start, entity.particle_end):
                sim_scene._sim.active_solvers[-1].particles[0, i, 0].vel = gs.ti.Vector(
                    init_vel
                )

    # Run simulation
    frame_idx = 0
    is_wrong = False

    n_particles = int(sim_scene._sim.active_solvers[-1].particles.pos.shape[1])

    for i in range(n_sim_steps):
        if i % 500 == 0:
            print(f"  → Step {i}/{n_sim_steps}")

        # Render FIRST (before stepping) to capture initial state at i=0
        if i % vis_substeps == 0:
            if i == 0:
                print(f"  [DEBUG] Rendering initial state (before any steps)...")
            # gmc.clear_cache()  # Commented out - causes segfault
            visible_cameras = 0

            # Render all cameras
            for c, cam in enumerate(cameras):
                if i == 0 and c == 0:
                    print(f"  [DEBUG] Rendering camera {c}...")
                rgb, depth, seg, normal = cam.render(depth=True, segmentation=True)
                if i == 0 and c == 0:
                    print(f"  [DEBUG] Camera {c} rendered OK")
                del depth, normal

                alpha = (seg == 1).astype(rgb.dtype)

                if alpha.max() >= 1:
                    visible_cameras += 1

                view_folder = os.path.join(save_root, f"{c:03d}")
                img_folder = os.path.join(view_folder, "img")
                mask_folder = os.path.join(view_folder, "mask")

                # Save white-background image
                mask_3ch = alpha.reshape(*alpha.shape, 1)
                white_img = rgb * mask_3ch + (1 - mask_3ch) * 255
                white_img = np.clip(white_img, 0, 255).astype(np.uint8)

                cv2.imwrite(
                    os.path.join(img_folder, f"{frame_idx:03d}.jpg"),
                    cv2.cvtColor(white_img, cv2.COLOR_RGB2BGR),
                )

                cv2.imwrite(
                    os.path.join(mask_folder, f"{frame_idx:03d}.png"),
                    (alpha * 255).astype(np.uint8),
                )

                del rgb, seg, alpha, mask_3ch, white_img

            # Save particles
            particles_folder = os.path.join(save_root, "particles")
            particle_pos = sim_scene._sim.active_solvers[-1].particles.pos.to_numpy()[0]
            commons.save_points_as_ply(
                particle_pos, os.path.join(particles_folder, f"{frame_idx:03d}.ply")
            )
            del particle_pos

            frame_idx += 1

            if frame_idx % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            # Check visibility threshold (fail if visible in < 10% of cameras)
            if visible_cameras < max(1, len(cameras) * 0.1):
                print(
                    f"  ✗ Simulation failed: Object lost (visible in {visible_cameras}/{len(cameras)} cameras)"
                )
                is_wrong = True
                break

        # THEN step the simulation
        if i == 0:
            print(f"  [DEBUG] About to call sim_scene.step() for step {i}...")
        sim_scene.step()
        if i == 0:
            print(f"  [DEBUG] sim_scene.step() completed for step {i}")

        # Access particle data for validation
        vel = sim_scene._sim.active_solvers[-1].particles.vel.to_numpy().copy()
        cur_pos = sim_scene._sim.active_solvers[-1].particles.pos.to_numpy().copy()

        # Explicit Physics Check
        if np.any(np.isnan(vel)) or np.any(np.isnan(cur_pos)):
            print(f"  ✗ Simulation failed: Physics instability (NaN detected)")
            is_wrong = True
            break

        if cur_pos.shape[1] != n_particles:
            print(
                f"  ✗ Simulation failed: Particle loss ({cur_pos.shape[1]} vs {n_particles})"
            )
            is_wrong = True
            break

    if is_wrong:
        return False, frame_idx

    return True, frame_idx


def run_simulation_with_retries(
    mesh_file,
    particle_config,
    max_attempts,
    save_root,
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
    n_sim_steps,
    vis_substeps,
    initial_scene=None,
    initial_cameras=None,
):
    """
    Phase 3: Run simulation with material/velocity/position retries.
    Returns (scene, cameras, E, nu, init_vel, n_frames, actual_particle_size) or None.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 3: Running Simulation with Retries")
    print(f"{'='*60}")
    print(
        f"  Reusing particle config: particle_size={particle_config.particle_size:.6f}m, "
        f"grid_density={particle_config.grid_density}, scale={particle_config.scale:.3f}"
    )

    for attempt in range(max_attempts):
        print(f"\n  Simulation attempt {attempt + 1}/{max_attempts}:")

        # Always create a FRESH scene for simulation (don't reuse from Phase 2)
        # This prevents camera/state incompatibility issues
        print(f"  → Creating fresh scene for simulation")

        # Clean up Phase 2 scene if this is the first attempt
        if attempt == 0 and initial_scene is not None:
            print(f"  → Cleaning up Phase 2 scene (was reused for validation only)")
            scene.cleanup_scene(initial_scene, initial_cameras)

        # Randomize physics parameters
        E, nu, rho, mat_elastic = physics.create_random_material()
        init_vel = physics.create_random_velocity()

        # Randomize position and rotation
        pos = np.clip(
            center
            + np.array([0.0, 0.0, 0.2])
            + particle_config.scale * 0.1 * np.random.randn(3),
            lower_bound + particle_config.scale / 2,
            upper_bound - particle_config.scale / 2,
        )
        quat = np.random.randn(4)
        quat = quat / np.linalg.norm(quat)

        print(f"    Material: E={E:.2e}, nu={nu:.3f}")
        print(f"    Velocity: {init_vel}")
        print(f"    Position: {pos}")

        # Create scene with validated particle config
        result = scene.create_scene_with_particles(
            mesh_file=mesh_file,
            particle_size=particle_config.particle_size,
            grid_density=particle_config.grid_density,
            scale=particle_config.scale,
            pos=pos,
            quat=quat,
            material=mat_elastic,
            camera_configs=camera_configs,
            center=center,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dt=dt,
            substeps=substeps,
            gravity=gravity,
            width=width,
            height=height,
            min_fov=min_fov,
            max_fov=max_fov,
            min_radius=min_radius,
            max_radius=max_radius,
        )

        if result is None:
            print(f"  ✗ Scene creation failed, retrying with new position...")
            continue

        sim_scene, cameras, n_particles, actual_particle_size = result

        # Verify particle count is still valid
        if n_particles != particle_config.n_particles:
            print(
                f"  ⚠ Warning: Particle count changed from {particle_config.n_particles} to {n_particles}"
            )

        # Run simulation
        success, n_frames = _run_simulation_and_check(
            sim_scene, cameras, save_root, init_vel, n_sim_steps, vis_substeps
        )

        if success:
            print(f"  ✓ Simulation successful: {n_frames} frames")
            # Don't cleanup scene yet - caller needs it for metadata
            return sim_scene, cameras, E, nu, init_vel, n_frames, actual_particle_size
        else:
            print(
                f"  ✗ Simulation failed, retrying with new material/velocity/position..."
            )
            scene.cleanup_scene(sim_scene, cameras)
            continue

    print(f"  ✗ All {max_attempts} simulation attempts failed")
    return None
