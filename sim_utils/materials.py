"""
Custom Genesis material classes.

This module provides material subclasses that extend pip-installed Genesis
materials without modifying the source code.
"""

import gstaichi as ti
import genesis as gs


@ti.data_oriented
class NeoHookean(gs.materials.MPM.Base):
    """
    NeoHookean hyperelastic material for MPM simulations.

    This class implements the NeoHookean constitutive model following the same
    pattern as Genesis's Elastic class. It subclasses the Base material class
    and defines the stress update methods using Taichi.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 3e5.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m^3). Default is 1000.0.
    lam : float, optional
        The first Lame's parameter. Default is None, computed from E and nu.
    mu : float, optional
        The second Lame's parameter. Default is None, computed from E and nu.
    sampler : str, optional
        Particle sampler ('pbs-32', 'pbs-64', 'regular', 'random').
        Defaults to 'pbs-32' on Linux, 'random' otherwise.

    Notes
    -----
    The NeoHookean model uses the stress update formula:
        stress = mu * (F @ F^T) + I * (lam * log(J) - mu)
    where F is the deformation gradient, J is its determinant, and I is the identity matrix.

    Examples
    --------
    >>> from sim_utils.materials import NeoHookean
    >>> mat = NeoHookean(E=1e6, nu=0.3, rho=1000)
    >>> scene.add_entity(material=mat, morph=...)
    """

    def __init__(
        self,
        E=3e5,
        nu=0.2,
        rho=1000.0,
        lam=None,
        mu=None,
        sampler=None,
    ):
        # Set default sampler based on platform (same as Elastic)
        if sampler is None:
            sampler = "pbs-32" if gs.platform == "Linux" else "random"

        # Initialize parent Base class
        super().__init__(E=E, nu=nu, rho=rho, lam=lam, mu=mu, sampler=sampler)

        # Set the stress update method to NeoHookean
        self.update_stress = self.update_stress_neohookean

        # Set model name
        self._model = "neohookean"

    @ti.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        """Update deformation gradient, singular values, and plastic deformation."""
        F_new = F_tmp
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

    @ti.func
    def update_stress_neohookean(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        """
        NeoHookean stress update.

        Formula: stress = μ(F@F^T) + I(λ·log(J) - μ)
        where μ and λ are Lame parameters.
        """
        stress = self._mu * (F_tmp @ F_tmp.transpose()) + ti.Matrix.identity(
            gs.ti_float, 3
        ) * (self._lam * ti.log(J) - self._mu)

        return stress

    @property
    def model(self):
        """Return the constitutive model name."""
        return self._model
