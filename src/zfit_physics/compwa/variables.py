from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import zfit
from zfit.core.interfaces import ZfitUnbinnedData

__all__ = ["obs_from_frame", "params_from_intensity"]


def params_from_intensity(intensity):
    return [
        zfit.param.convert_to_parameter(val, name=name, prefer_constant=False)
        for name, val in intensity.parameters.items()
    ]


def obs_from_frame(frame1, frame2=None, bufferfactor=0.01):
    obs = []
    if frame2 is None:
        frame2 = frame1

    if isinstance(frame1, ZfitUnbinnedData) or isinstance(frame2, ZfitUnbinnedData):
        return frame1.space

    if not isinstance(frame1, (Mapping, pd.DataFrame)) or not isinstance(frame2, (Mapping, pd.DataFrame)):
        msg = "frame1 and frame2 have to be either a mapping or a pandas DataFrame, or a zfit Data object. They are currently of type: "
        raise ValueError(
            msg,
            type(frame1),
            type(frame2),
        )
    for ob in frame2:
        minimum = np.min([np.min(frame1[ob]), np.min(frame2[ob])])
        maximum = np.max([np.max(frame1[ob]), np.max(frame2[ob])])
        dist = maximum - minimum
        buffer = bufferfactor * dist
        obs.append(
            zfit.Space(
                ob,
                limits=(
                    minimum - buffer,
                    maximum + buffer,
                ),
            )
        )
    return zfit.dimension.combine_spaces(*obs)
