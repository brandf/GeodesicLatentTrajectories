"""
Lightweight helpers that implement the Geodesic Latent Trajectories (GLT) stack.

The public surface mirrors the structure from the obsidian notes:
- :mod:`glt.geometry` exposes the hyperspherical math primitives.
- :mod:`glt.losses` contains the auxiliary GLT loss terms.
- :mod:`glt.config` defines :class:`GLTConfig` used to toggle and tune GLT.
"""

from .config import GLTConfig, glt_config_from_dict
from . import geometry
from . import losses
from .extrapolation import extrapolate_next_latent

__all__ = [
    "GLTConfig",
    "glt_config_from_dict",
    "geometry",
    "losses",
    "extrapolate_next_latent",
]
