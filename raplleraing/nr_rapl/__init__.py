"""NR_RAPL bootstrap components.

This package deliberately keeps the original RAPL implementation intact.  It
only contains the configuration, model façade, and checkpoint compatibility
helpers needed by the first NR_RAPL integration stage.
"""

from RAPL.nr_rapl.config import MethodConfig, NRRAPLConfig, build_method_config
from RAPL.nr_rapl.optimizers import get_NR_RAPL_optimizer, NR_RAPL_parameter_groups


def build_model(*args, **kwargs):
    """Lazily import the model factory to avoid a RAPL import cycle."""
    from RAPL.nr_rapl.factory import build_model as _build_model

    return _build_model(*args, **kwargs)

__all__ = ["MethodConfig", "NRRAPLConfig", "build_method_config", "build_model", "get_NR_RAPL_optimizer", "NR_RAPL_parameter_groups"]
