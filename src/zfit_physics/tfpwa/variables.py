from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tf_pwa

import zfit


def params_from_fcn(fcn: tf_pwa.model.FCN) -> list[zfit.Parameter]:
    """Get zfit.Parameter objects from a tf_pwa.FCN.


    Args:
        fcn: A tf_pwa.FCN

    Returns:
        list of zfit.Parameter
    """
    return [zfit.Parameter(n, v, floating=n in fcn.vm.trainable_vars) for n, v in fcn.get_params().items()]
