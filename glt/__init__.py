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
from . import metrics
from . import logging as viz_logging
from . import plotting
from .viz_config import VizConfig
from .extrapolation import extrapolate_next_latent

__all__ = [
    "GLTConfig",
    "glt_config_from_dict",
    "geometry",
    "losses",
    "metrics",
    "viz_logging",
    "plotting",
    "VizConfig",
    "extrapolate_next_latent",
]
