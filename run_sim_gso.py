import argparse
import genesis as gs
import json
import math
import numpy as np
import os
import random
import taichi as ti
import torch
import trimesh
from PIL import Image


"""
IMPORTANT

1. COORDINATE SYSTEM
First of all, the third axis is the vertial axis
Second, X, Y are swapped (compared to 3DGS)
"""

def get_surface(name: str, *args, **kwargs):
    return getattr(gs.surfaces, name)(*args, **kwargs)


def idx_to_camera_name(idx):
    return f"camera_{idx:03d}"


def save_points_as_ply(points, filename):
    points = np.asarray(points)
    points = points[:, 0, [1, 2, 0]] # due to genesis-specific coord system
    point_cloud = trimesh.points.PointCloud(vertices=points)
    point_cloud.export(file_obj=filename, file_type='ply', encoding='ascii')


def sample_unit_vector(min_elevation_radian, max_elevation_radian):
    """
    Samples a 3D unit vector with azimuth uniformly distributed in [0, 2π)
    and elevation uniformly distributed
    in [min_elevation_radian, max_elevation_radian].
    elevation radian close to 0 means top of the sphere and as it approaches
    pi, it gets close to the bottom of the sphere

    Returns:
        numpy.ndarray: Array of shape (3,) containing an unit vector.
    """
    if min_elevation_radian == max_elevation_radian:
        cos_theta = np.cos([min_elevation_radian])
    else:
        cos_theta = np.random.uniform(np.cos(min_elevation_radian),
                                      np.cos(max_elevation_radian),
                                      1)

    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2 * np.pi, 1)

    sin_theta = np.sqrt(1 - cos_theta**2)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta

    return np.concatenate((x, y, z), axis=-1)


def get_sphere_uvs(verts: np.ndarray, checker_angle=90):
    assert (180 // checker_angle) % 2 == 0

    # calculate azimuth and elevation
    verts = verts - verts.mean(axis=0, keepdims=True) # [N, 3]
    verts = verts / verts.max(axis=0, keepdims=True) # [N, 3]

    azimuth = np.arctan2(verts[:, 1], verts[:, 0])
    azimuth = np.mod(azimuth, 2 * np.pi) / np.pi * 180

    elevation = np.arcsin(verts[:, 2])
    elevation = np.mod(elevation, np.pi) / np.pi * 180

    classes = (azimuth // checker_angle + elevation // checker_angle) % 2
    uvs = classes.reshape(-1, 1) * np.array([1., 1.])
    return uvs


def get_cube_uvs(verts: np.ndarray, size=0.25):
    assert (1 // size) % 2 == 0

    # calculate azimuth and elevation
    verts = verts - verts.min(axis=0, keepdims=True) # [N, 3]
    verts = verts / verts.max(axis=0, keepdims=True) # [N, 3]

    classes = (verts // size).sum(axis=1) % 2
    uvs = classes.reshape(-1, 1) * np.array([1., 1.])
    return uvs


def main(args):
    # 1. Initialization
    # 1.1. Constants
    # SIMULATION
    N_SIM_STEPS = 4800
    SUBSTEPS = 1 # number of steps per simulation steps
    SIM_REQUIRES_GRAD = False
    DT = 2.5e-4
    GRAVITY = (0, 0, -9.81)
    LOWER_BOUND = np.array([-2.0, -2.0, -0.7]) - 0.046875
    UPPER_BOUND = np.array([ 2.0,  2.0, 2.0]) + 0.046875
    CENTER = np.array([0.0, 0.0, -0.2])

    # RENDERING
    FPS = 20
    N_CAMERAS = 18
    HEIGHT = 448
    WIDTH = 448
    FOV = 49.1 # L4GM: ~49.1, GSO: 45
    VIS_SUBSTEPS = int(1 / FPS / (DT * SUBSTEPS)) # num of sim steps per vis

    if args.i == 0:
        mesh_file = 'objs/glbs/processed/4583ddb4b6e14a2cb30381cc0031dd1b.glb' # crema
        args.name = 'crema'
    elif args.i == 1:
        mesh_file = 'objs/glbs/processed/b0a20ac5a08f476bb2fd4115192cbd89.glb' # burger
        args.name = 'burger'
    elif args.i == 2:
        mesh_file = 'objs/glbs/processed/921e5f26a12643d99cded65621985a24.glb' # cream
        args.name = 'cream'
    elif args.i == 3:
        mesh_file = 'objs/glbs/processed/b14bd57f841841f99603763caa19c16d.glb' # polyhedral
        args.name = 'polyhedral'
    elif args.i == 4:
        mesh_file = 'objs/glbs/processed/500ef37861784d58885c303a21a09d6a.glb' # doll
        args.name = 'doll'
    elif args.i == 5:
        mesh_file = 'filtered_objs/glbs/000-000/00a1d892548542c7ab83565070737d6b.glb'
        args.name = 'blue'
    elif args.i == 6:
        mesh_file = 'filtered_objs/glbs/000-000/01a4d229631e4aab866e236968881241.glb'
        args.name = 'balloon'
    else:
        raise ValueError(f"invalid i: {args.i}")

    # args.name += '_shadowed'
    # args.name += '_test'
    args.name += '_env'

    args.save_path = os.path.join(args.save_path, args.name)

    render_path = os.path.join(args.save_path, 'data')
    os.makedirs(render_path, exist_ok=True)

    gs.init(seed=None, precision="32") # , logging_level="debug")

    not_finished = True

    while not_finished:
        # 2.2. Materials
        E = 10 ** random.uniform(4.0, 7.0)
        nu = random.uniform(0.0, 0.49)
        rho = 1e3
        mat_elastic = gs.materials.MPM.Elastic(E=E, nu=nu, rho=rho, model="neohookean")

        # 3. Scene Building
        # 3.1. Entities
        init_vel = np.random.randn(3) * 0.25
        if True: # args.zero_init_vel:
            init_vel = init_vel * 0.0

        scale = 0.6

        pos = np.clip(CENTER + np.array([0., 0., 0.0]) + scale * 0.1 * np.random.randn(3),
                      LOWER_BOUND + scale/2, UPPER_BOUND - scale/2)
        quat = np.random.randn(4)
        quat = quat / np.linalg.norm(quat)

        # 3.2. Camera positions (pre-compute to use in both scenes)
        camera_positions = []
        camera_scale = 1.5 # L4GM: 1.5, GSO: 1.8

        for i in range(N_CAMERAS):
            cam_pos = sample_unit_vector(
                0 / 180 * np.pi,
                90 / 180 * np.pi
            )
            cam_pos *= camera_scale
            cam_pos += CENTER

            lookat = CENTER

            up = np.array([0., 0., 1.])
            up = up / np.linalg.norm(up)

            camera_positions.append({
                'pos': cam_pos,
                'lookat': lookat,
                'up': up
            })

        renderer = gs.renderers.RayTracer(  # type: ignore
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ImageTexture(
                    image_path="textures/indoor_bright.png",
                ),
            ),
            env_radius=15.0,
            env_euler=(0, 0, 180),
            lights=[
                {"pos": (0.0, 0.0, 10.0), "radius": 3.0, "color": (15.0, 15.0, 15.0)},
            ],
        )

        # ====== STEP 1: Build scene without object, render background images ======
        scene_bg = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=DT,
                substeps=SUBSTEPS,
                gravity=GRAVITY,
                requires_grad=SIM_REQUIRES_GRAD,
            ),
            mpm_options=gs.options.MPMOptions(
                enable_CPIC=False,
                lower_bound=LOWER_BOUND,
                upper_bound=UPPER_BOUND,
                use_sparse_grid=False,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                shadow=True,
                background_color=(1.0, 1.0, 1.0),
            ),
            show_viewer=False,
            renderer=renderer,
        )

        # Add only the floor
        scene_bg.add_entity(
            morph=gs.morphs.Plane(pos=(0., 0., -0.7)),
        )

        # Add cameras with the same positions
        cameras_bg = []
        for cam_config in camera_positions:
            cam = scene_bg.add_camera(
                res=(WIDTH, HEIGHT),
                pos=cam_config['pos'],
                lookat=cam_config['lookat'],
                up=cam_config['up'],
                fov=FOV,
                GUI=True,
            )
            cameras_bg.append(cam)

        # Build and render background images
        scene_bg.build()
        scene_bg.reset()

        for c, cam in enumerate(cameras_bg):
            rgb, *_ = cam.render(depth=False)
            image = Image.fromarray(rgb)
            image.save(os.path.join(render_path, f"r_{c}_-1.png"))

        # Clean up background scene
        del scene_bg
        del cameras_bg

        # ====== STEP 2: Build actual scene with object for simulation ======
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=DT,
                substeps=SUBSTEPS,
                gravity=GRAVITY,
                requires_grad=SIM_REQUIRES_GRAD,
            ),
            mpm_options=gs.options.MPMOptions(
                enable_CPIC=False,
                lower_bound=LOWER_BOUND,
                upper_bound=UPPER_BOUND,
                use_sparse_grid=False,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                shadow=True,
                background_color=(1.0, 1.0, 1.0),
            ),
            show_viewer=False,
            renderer=renderer,
        )

        # Add floor
        scene.add_entity(
            morph=gs.morphs.Plane(pos=(0., 0., -0.7)),
        )

        # Add MPM object
        scene.add_entity(
            material=mat_elastic,
            morph=gs.morphs.Mesh(
                file=mesh_file,
                scale=scale,
                pos=pos,
                quat=quat,
                decimate=False,
                normalize=True,
            ),
            surface=gs.surfaces.Default(
                # vis_mode="recon",
            )
        )

        # Add cameras with the same positions
        cameras = []
        for cam_config in camera_positions:
            cam = scene.add_camera(
                res=(WIDTH, HEIGHT),
                pos=cam_config['pos'],
                lookat=cam_config['lookat'],
                up=cam_config['up'],
                fov=FOV,
                GUI=True,
            )
            cameras.append(cam)

        # 4. Execution
        scene.build()
        scene.reset()

        # 4.1. Initial Velocity
        with torch.no_grad():
            for entity in scene.entities:
                if not isinstance(entity, gs.engine.entities.MPMEntity):
                    continue

                for i in range(entity.particle_start, entity.particle_end):
                    scene._sim.active_solvers[-1].particles[0, i, 0].vel = \
                        ti.Vector(init_vel)

        for cam in cameras:
            cam.start_recording()

        particles_path = os.path.join(args.save_path, 'simulation')
        os.makedirs(particles_path, exist_ok=True)

        if args.save_surface:
            surface_path = os.path.join(args.save_path, 'surface')
            os.makedirs(surface_path, exist_ok=True)

        frame_idx = 0
        is_wrong = False

        for i in range(N_SIM_STEPS):
            scene.step()

            if i % VIS_SUBSTEPS == 0:
                # rendering
                for c, cam in enumerate(cameras):
                    rgb, depth, seg, normal = cam.render(depth=True,
                                                         segmentation=True)
                    # alpha = (depth < depth.max()).astype(rgb.dtype)
                    alpha = (seg == 2).astype(rgb.dtype) # the second object
                    if np.all(alpha == 0.0):
                        is_wrong = True
                        break
                    alpha = alpha.reshape(*alpha.shape, 1) * 255

                    rgba = np.concatenate([rgb, alpha], -1)
                    image = Image.fromarray(rgba)
                    image.save(os.path.join(render_path,
                                            f"a_{c}_{frame_idx}.png"))

                    mask = (alpha > 0).astype(rgb.dtype)
                    background = 255
                    masked_rgb = rgb * mask + (1 - mask) * background
                    image = Image.fromarray(masked_rgb)
                    image.save(os.path.join(render_path,
                                            f"m_{c}_{frame_idx}.png"))

                    image = Image.fromarray(rgb)
                    image.save(os.path.join(render_path,
                                            f"r_{c}_{frame_idx}.png"))

                if is_wrong:
                    break

                # particles
                save_points_as_ply(
                    scene._sim.active_solvers[-1].particles.pos.to_numpy()[0],
                    os.path.join(particles_path, f"{frame_idx:03d}.ply"))

                # surface particles
                if args.save_surface:
                    vertices = []
                    for entity in scene.entities:
                        vertices += entity.vmesh.verts.tolist()

                    save_points_as_ply(
                        vertices,
                        os.path.join(surface_path, f"{frame_idx:03d}.ply"))

                frame_idx += 1

            if is_wrong:
                break

        if is_wrong:
            continue

        not_finished = False

    # JSON
    camera_angle_x = math.radians(FOV)
    reg_info = {"camera_angle_x": camera_angle_x, "frames": []}
    nvs_info = {"camera_angle_x": camera_angle_x, "frames": []}

    for i, cam in enumerate(cameras):
        c2w = cam.transform.tolist()

        cam_info = {
            "file_path": f"./data/m_{i}_0",
            "time": 0.0,
            "rotation": 0.0,
            "transform_matrix": c2w
        }

        if i < 12: # reg
            reg_info["frames"].append(cam_info)
        else: # nvs
            nvs_info["frames"].append(cam_info)

    for group in ["simulation", "train", "test", "val"]:
        with open(os.path.join(args.save_path, f"transforms_{group}.json"),
                  'w') as o:
            json.dump(reg_info, fp=o, separators=(',', ':'), sort_keys=True,
                      indent=4)
        with open(os.path.join(args.save_path, f"transforms_{group}_nvs.json"),
                  'w') as o:
            json.dump(nvs_info, fp=o, separators=(',', ':'), sort_keys=True,
                      indent=4)

    # physical.json
    with open(os.path.join(args.save_path, "gt_phys_params.yaml"), 'w') as o:
        o.write(f"yms: {E}\nprs: {nu}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--name', type=str, required=True)
    parser.add_argument('-i', type=int, required=True)
    parser.add_argument('--save_path', type=str, default="test_data")
    parser.add_argument('--save_surface', action='store_true')

    args = parser.parse_args()

    main(args)
