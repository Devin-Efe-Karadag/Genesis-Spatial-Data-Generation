import numpy as np
import genesis as gs
from collections import namedtuple

ParticleConfig = namedtuple(
    "ParticleConfig", ["particle_size", "grid_density", "scale", "n_particles"]
)


def compute_grid_cell_size(grid_density):
    """
    Compute the grid cell size for MPM solver.
    """
    return 1.0 / grid_density


def create_random_material():
    """
    Create random elastic material parameters.
    """
    E = 10 ** np.random.uniform(4.0, 7.0)
    nu = np.random.uniform(0.0, 0.49)
    rho = 1e3
    mat_elastic = gs.materials.MPM.Elastic(E=E, nu=nu, rho=rho) # , model="corotated") # neohookean")
    return E, nu, rho, mat_elastic


def create_random_velocity(scale=0.25):
    """
    Create random initial velocity.
    """
    return np.random.randn(3) * scale


def setup_camera_configs(
    n_random, n_fixed, elevation_range_random, rotation_range, elevation_fixed
):
    """
    Setup camera configurations for both random and fixed cameras.
    """
    camera_configs = []

    # Random cameras
    for i in range(n_random):
        elevation_deg = np.random.uniform(
            elevation_range_random[0], elevation_range_random[1]
        )
        rotation_deg = np.random.uniform(rotation_range[0], rotation_range[1])
        camera_configs.append(
            {
                "mode": "random",
                "elevation": elevation_deg,
                "rotation": rotation_deg,
            }
        )

    # Fixed cameras
    stepsize = 360.0 / n_fixed
    for i in range(n_fixed):
        rotation_deg = i * stepsize
        camera_configs.append(
            {
                "mode": "fixed",
                "elevation": elevation_fixed,
                "rotation": rotation_deg,
            }
        )

    return camera_configs


def setup_lights(n_base_lights, n_variation, center):
    """
    Setup scene lights with random positions and intensities.
    """
    from .commons import (
        sample_unit_vector,
    )  # Local import to avoid circular dependency if any

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

        lights.append(
            {
                "pos": tuple(light_pos),
                "radius": light_radius,
                "color": (light_intensity, light_intensity, light_intensity),
            }
        )

    return lights


def check_particle_count(
    n_particles,
    actual_particle_size,
    min_count,
    min_size,
    max_count=None,
    max_size=None,
    grid_density=None,
    is_stuck=False,
):
    """
    Validate particle count and suggest new particle size if needed.
    """
    # Check minimum particle count
    if n_particles < min_count and actual_particle_size > min_size:
        ratio = (min_count / n_particles) ** (1.0 / 3.0)
        new_particle_size = max(actual_particle_size / ratio * 0.9, min_size)

        print(f"  ⚠ Particle count too low ({n_particles} < {min_count})")
        print(f"  → Will retry with particle_size={new_particle_size:.4f}m")

        return False, new_particle_size, grid_density

    # Check maximum particle count
    if max_count is not None and n_particles > max_count:
        target_count = int(max_count * 0.75)
        ratio = (n_particles / target_count) ** (1.0 / 3.0)
        new_particle_size = actual_particle_size * ratio

        if is_stuck:
            min_increase = actual_particle_size * 1.2
            if new_particle_size < min_increase:
                print(f"  → Stuck! Forcing particle_size increase")
                new_particle_size = min_increase

        print(f"  ⚠ Particle count too high ({n_particles} > {max_count})")
        print(f"  → New particle_size: {new_particle_size:.6f}m")

        # Check if we need to adjust grid density
        new_grid_density = grid_density
        if grid_density is not None:
            current_grid_cell_size = compute_grid_cell_size(grid_density)
            min_required_cell_size = new_particle_size * 2
            max_required_cell_size = new_particle_size * 4

            if current_grid_cell_size < max_required_cell_size:
                print(f"  → Grid cell too small, adjusting density")

                found_valid = False
                for candidate_density in [32, 16, 8, 4, 2, 1]:
                    if candidate_density >= grid_density:
                        continue

                    candidate_cell_size = compute_grid_cell_size(candidate_density)
                    if (
                        min_required_cell_size
                        <= candidate_cell_size
                        <= max_required_cell_size
                    ):
                        new_grid_density = candidate_density
                        found_valid = True
                        print(f"  → New grid_density: {new_grid_density}")
                        break

                if not found_valid:
                    print(f"  ⚠ Cannot satisfy grid constraint even at grid_density=1")
                    return False, None, None

        return False, new_particle_size, new_grid_density

    return True, None, None
