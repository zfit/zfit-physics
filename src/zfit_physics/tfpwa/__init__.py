try:
    import tf_pwa
except ImportError:
    raise ImportError("tf-pwa is required to use zfit-physics-tfpwa. This can currently only be installed from source, i.e. via `pip install git+https://github.com/jiangyi15/tf-pwa`")
from . import loss, variables

__all__ = ["loss", "variables"]
