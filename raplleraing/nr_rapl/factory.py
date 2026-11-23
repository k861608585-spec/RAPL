"""Small factory that preserves the legacy RAPL constructor path."""

from typing import Any

from RAPL.nr_rapl.config import MethodConfig
from RAPL.vlprompt.rapl import RAPL
from RAPL.vlprompt.nr_rapl import NRRAPL


def build_model(method_config: MethodConfig, **model_kwargs: Any) -> RAPL:
    """Build RAPL unless the caller explicitly selects NR_RAPL."""
    if method_config.METHOD.NAME == "RAPL":
        return RAPL(**model_kwargs)
    if method_config.METHOD.NAME == "NR_RAPL":
        return NRRAPL(nr_config=method_config.NR_RAPL, **model_kwargs)
    raise ValueError(f"Unknown METHOD.NAME: {method_config.METHOD.NAME}")
