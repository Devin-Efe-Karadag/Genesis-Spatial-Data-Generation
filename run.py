"""
3-Phase Dataset Generation Algorithm (Modular Refactoring)

This script generates a physics simulation dataset using a 3-phase approach:
1. Mesh Preprocessing
2. Particle Configuration Search
3. Simulation Loop

It relies on the `sim_utils` package for core logic.
"""

import sys
import os

# ============================================================================
# CRITICAL SETUP: GPU & CLEANUP
# ============================================================================
import sim_utils.env as env

# 1. Setup GPU visibility before any heavy imports
env.setup_gpu_safety()

import kaolin  # Import kaolin before genesis
import genesis as gs
import genesis_mesh_cache_patch as gmc

import argparse
import math
import numpy as np
import torch
import gc
from pathlib import Path
from glob import glob

import sim_utils.phases as phases
import sim_utils.commons as commons
import sim_utils.physics as physics
import sim_utils.scene as scene_utils

# 2. Register atexit handlers (LIFO: register LAST so it runs FIRST, before Blender's cleanup)
env.register_cleanup_handlers()


def process_single_object(obj_path, synset_idx, model_identifier, animation_idx, args):
    """
    Process a single object using the 3-phase algorithm.
    """
    # Constants
    N_CAMERAS_RANDOM = 32
    N_CAMERAS_FIXED = 16
    N_CAMERAS_TOTAL = N_CAMERAS_RANDOM + N_CAMERAS_FIXED

    # Simulation parameters
    N_SIM_STEPS = args.n_sim_steps
    SUBSTEPS = 1
    DT = 2.5e-4
    GRAVITY = (0, 0, -9.81)
    LOWER_BOUND = np.array([-5.0, -5.0, -0.7]) - 0.046875
    UPPER_BOUND = np.array([5.0, 5.0, 2.0]) + 0.046875
    CENTER = np.array([0.0, 0.0, args.center_z])

    # Rendering parameters
    FPS = args.fps
    HEIGHT = args.resolution
    WIDTH = args.resolution
    VIS_SUBSTEPS = int(1 / FPS / (DT * SUBSTEPS))

    # Retry limits
    MAX_PARTICLE_ATTEMPTS = 10
    MAX_SIMULATION_ATTEMPTS = 3
    MIN_PARTICLE_COUNT = 16384
    MAX_PARTICLE_COUNT = 65536
    MIN_PARTICLE_SIZE = 0.001

    # Setup paths
    save_root = os.path.join(
        args.output_folder, synset_idx, model_identifier, f"{animation_idx:03d}"
    )

    # ========================================================================
    # ONE-TIME SETUP
    # ========================================================================
    camera_configs = physics.setup_camera_configs(
        n_random=N_CAMERAS_RANDOM,
        n_fixed=N_CAMERAS_FIXED,
        elevation_range_random=(-89, 89),
        rotation_range=(0, 360),
        elevation_fixed=0.0,
    )

    lights = physics.setup_lights(
        n_base_lights=args.n_lights, n_variation=args.n_lights_variation, center=CENTER
    )

    # ========================================================================
    # PHASE 1: MESH PREPROCESSING
    # ========================================================================
    mesh_file = phases.preprocess_mesh(
        obj_path=obj_path,
        model_identifier=model_identifier,
        animation_idx=animation_idx,
        output_folder=args.output_folder,
    )

    if mesh_file is None:
        print(f"✗ Phase 1 failed: mesh preprocessing")
        return False

    # ========================================================================
    # PHASE 2: FIND VALID PARTICLE CONFIGURATION
    # ========================================================================
    result = phases.find_valid_particle_config(
        mesh_file=mesh_file,
        min_count=MIN_PARTICLE_COUNT,
        max_count=MAX_PARTICLE_COUNT,
        min_size=MIN_PARTICLE_SIZE,
        max_attempts=MAX_PARTICLE_ATTEMPTS,
        camera_configs=camera_configs,
        center=CENTER,
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        dt=DT,
        substeps=SUBSTEPS,
        gravity=GRAVITY,
        width=WIDTH,
        height=HEIGHT,
        min_fov=args.min_fov,
        max_fov=args.max_fov,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
    )

    if result is None:
        print(f"✗ Phase 2 failed: validating particle config")
        if os.path.exists(mesh_file) and mesh_file != obj_path:
            try:
                os.remove(mesh_file)
            except:
                pass
        return False

    # Unpack result
    particle_config, scene, cameras = result

    # ========================================================================
    # PHASE 3: RUNNING SIMULATION
    # ========================================================================
    print(f"[DEBUG] Calling Phase 3: run_simulation_with_retries")
    sim_result = phases.run_simulation_with_retries(
        mesh_file=mesh_file,
        particle_config=particle_config,
        max_attempts=MAX_SIMULATION_ATTEMPTS,
        save_root=save_root,
        camera_configs=camera_configs,
        center=CENTER,
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        dt=DT,
        substeps=SUBSTEPS,
        gravity=GRAVITY,
        width=WIDTH,
        height=HEIGHT,
        min_fov=args.min_fov,
        max_fov=args.max_fov,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        n_sim_steps=N_SIM_STEPS,
        vis_substeps=VIS_SUBSTEPS,
        initial_scene=scene,  # Pass live scene from Phase 2
        initial_cameras=cameras,  # Pass live cameras from Phase 2
    )

    if sim_result is None:
        print(f"✗ Phase 3 failed: all simulation attempts failed")
        if os.path.exists(mesh_file) and mesh_file != obj_path:
            try:
                os.remove(mesh_file)
            except:
                pass
        return False

    # Unpack success result
    scene, cameras, E, nu, init_vel, n_frames, actual_particle_size = sim_result

    # ========================================================================
    # SAVE METADATA
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"Saving metadata and creating tar files")
    print(f"{'='*60}")

    # Save camera parameters
    for c, cam in enumerate(cameras):
        camera_folder = os.path.join(save_root, f"{c:03d}", "camera")
        os.makedirs(camera_folder, exist_ok=True)

        fov = cam.fov
        fov_frames = np.full(n_frames, fov, dtype=np.float32)
        np.save(os.path.join(camera_folder, "fov.npy"), fov_frames)

        extrinsics = cam.transform
        extrinsics_frames = np.tile(extrinsics, (n_frames, 1, 1))
        np.save(os.path.join(camera_folder, "extrinsics.npy"), extrinsics_frames)

    # Save physics parameters
    physics_folder = os.path.join(save_root, "physics")
    os.makedirs(physics_folder, exist_ok=True)

    np.save(os.path.join(physics_folder, "youngs_modulus.npy"), E)
    np.save(os.path.join(physics_folder, "poisson_ratio.npy"), nu)
    np.save(os.path.join(physics_folder, "initial_velocity.npy"), init_vel)
    np.save(os.path.join(physics_folder, "gravity.npy"), np.array(GRAVITY))
    np.save(os.path.join(physics_folder, "particle_size.npy"), actual_particle_size)

    # Validate saved frames before creating tar files
    expected_n_frames = math.ceil(N_SIM_STEPS / VIS_SUBSTEPS)
    if not commons.validate_saved_frames(save_root, N_CAMERAS_TOTAL, expected_n_frames):
        print(f"✗ Validation failed: incomplete or missing frames")
        print(f"  Skipping tar file creation for this object")
        # Cleanup
        scene_utils.cleanup_scene(scene, cameras)
        if os.path.exists(mesh_file) and mesh_file != obj_path:
            try:
                os.remove(mesh_file)
            except:
                pass
        return False

    # Create tar files
    uid = f"{synset_idx}-{model_identifier}-{animation_idx:03d}"
    commons.create_tar_files(save_root, uid, args.output_folder)

    print(f"✓ Dataset created successfully")
    print(f"  Location: {save_root}")
    print(
        f"  Cameras: {N_CAMERAS_TOTAL} ({N_CAMERAS_RANDOM} random + {N_CAMERAS_FIXED} fixed)"
    )
    print(f"  Frames: {n_frames}")
    print(f"  Particles: {particle_config.n_particles}")

    # Cleanup
    scene_utils.cleanup_scene(scene, cameras)
    if os.path.exists(mesh_file) and mesh_file != obj_path:
        try:
            os.remove(mesh_file)
        except:
            pass
    return True


def main(args):
    """Main function that finds all glb files and processes them."""
    # Find all .glb files
    glb_pattern = os.path.join(args.input_folder, "**", "*.glb")
    all_glb_files = sorted(glob(glb_pattern, recursive=True))

    print(f"Found {len(all_glb_files)} glb files in {args.input_folder}")

    # Filter files
    if args.n_samples is not None:
        all_glb_files = all_glb_files[: args.n_samples]
    all_glb_files = all_glb_files[args.idx :: args.stride]

    if len(all_glb_files) == 0:
        print(f"No .glb files found in {args.input_folder}")
        return

    print(f"Processing {len(all_glb_files)} objects...")

    # Enable mesh reconstruction caching
    gmc.enable_cache()

    for idx, glb_path in enumerate(all_glb_files):
        path_obj = Path(glb_path)
        model_identifier = path_obj.stem
        synset_idx = path_obj.parent.name

        print(f"\n{'='*80}")
        print(
            f"[{idx+1}/{len(all_glb_files)}] Processing: {synset_idx}/{model_identifier}"
        )
        print(f"{'='*80}")

        # Check if output already exists
        animation_idx = 0
        uid = f"{synset_idx}-{model_identifier}-{animation_idx:03d}"
        random_tar_path = os.path.join(args.output_folder, f"random_clip-{uid}")
        fixed_tar_path = os.path.join(args.output_folder, f"fixed_16_clip-{uid}")

        if os.path.exists(random_tar_path) and os.path.exists(fixed_tar_path):
            if not args.overwrite:
                print(
                    f"  → Output already exists, skipping (use --overwrite to regenerate)"
                )
                continue
            else:
                print(f"  → Output exists but overwrite=True, regenerating...")

        # Initialize Genesis
        gs.init(precision="32")
        gmc.clear_cache()

        try:
            success = process_single_object(
                obj_path=glb_path,
                synset_idx=synset_idx,
                model_identifier=model_identifier,
                animation_idx=animation_idx,
                args=args,
            )
            if success:
                print(f"✓ Successfully processed {synset_idx}/{model_identifier}")

                # Show cache stats
                stats = gmc.get_cache_stats()
                if stats["hits"] > 0 or stats["misses"] > 0:
                    cache_rate = 100 * stats["hits"] / (stats["hits"] + stats["misses"])
                    print(
                        f"  → Mesh cache: {stats['hits']} hits, {stats['misses']} misses ({cache_rate:.1f}% hit rate)"
                    )
            else:
                print(f"✗ Failed to process {synset_idx}/{model_identifier}")

        except Exception as e:
            print(f"✗ Failed to process {synset_idx}/{model_identifier}: {e}")
            import traceback

            traceback.print_exc()

        finally:
            # Cleanup
            gs.destroy()
            torch.cuda.empty_cache()
            gc.collect()
            print(f"  → Genesis destroyed, GPU memory cleared")

    print(f"\n{'='*80}")
    print(f"Completed processing {len(all_glb_files)} objects")
    print(f"Output folder: {args.output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate physics simulation dataset (modular 3-phase algorithm)"
    )

    # Input/Output
    parser.add_argument(
        "--input_folder",
        type=str,
        default="samples",
        help="Input folder containing glb files",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="toy_box",
        help="Output folder for dataset",
    )

    # Simulation
    parser.add_argument(
        "--n_sim_steps",
        type=int,
        default=3401,
        help="Number of simulation steps",
    )
    parser.add_argument(
        "--fps", type=int, default=20, help="Frames per second for recording"
    )
    parser.add_argument(
        "--center_z", type=float, default=-0.2, help="Z coordinate of scene center"
    )

    # Camera
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution (width and height)",
    )
    parser.add_argument(
        "--min_fov",
        type=float,
        default=40.0,
        help="Minimum camera field of view in degrees",
    )
    parser.add_argument(
        "--max_fov",
        type=float,
        default=60.0,
        help="Maximum camera field of view in degrees",
    )
    parser.add_argument(
        "--min_radius",
        type=float,
        default=1.2,
        help="Minimum camera distance from center",
    )
    parser.add_argument(
        "--max_radius",
        type=float,
        default=1.8,
        help="Maximum camera distance from center",
    )

    # Lighting
    parser.add_argument("--n_lights", type=int, default=5, help="Base number of lights")
    parser.add_argument(
        "--n_lights_variation",
        type=int,
        default=1,
        help="Variation in number of lights",
    )

    # Object scale
    parser.add_argument(
        "--min_scale", type=float, default=0.6, help="Minimum scale for object"
    )
    parser.add_argument(
        "--max_scale", type=float, default=0.9, help="Maximum scale for object"
    )

    # Processing control
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files"
    )
    parser.add_argument(
        "--idx", type=int, default=0, help="Starting index for processing files"
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Stride for processing files"
    )
    parser.add_argument(
        "--n_samples", type=int, default=None, help="Number of samples to process"
    )

    args = parser.parse_args()
    main(args)
