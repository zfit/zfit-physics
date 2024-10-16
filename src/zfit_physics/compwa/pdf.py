from __future__ import annotations

import numpy as np
import tensorflow as tf
import zfit  # suppress tf warnings
import zfit.z.numpy as znp
from zfit import supports, z

from .variables import obs_from_frame, params_from_intensity


def patched_call(self, data, *, params) -> np.ndarray:
    # extended_data = {**self.__parameters, **data}  # type: ignore[arg-type]
    if params is not None:
        self.update_parameters(params)
    return self.__function(data)  # type: ignore[arg-type]


class ComPWAPDF(zfit.pdf.BasePDF):
    def __init__(self, intensity, norm, obs=None, params=None, extended=None, name="ComPWA"):
        """ComPWA intensity normalized over the *norm* dataset."""
        if params is None:
            params = {p.name: p for p in params_from_intensity(intensity)}
        norm = zfit.Data(norm, obs=obs)
        if obs is None:
            obs = obs_from_frame(norm.to_pandas())
        norm = norm.with_obs(obs)
        super().__init__(obs, params=params, name=name, extended=extended)
        self.intensity = intensity
        norm = {ob: znp.array(ar) for ob, ar in zip(self.obs, z.unstack_x(norm))}
        self.norm_sample = norm

    @supports(norm=True)
    def _pdf(self, x, norm, params):
        paramvalsfloat = []
        paramvalscomplex = []
        iscomplex = []
        # we need to split complex and floats to pass them to the numpy function, as it creates a tensor
        for val in params.values():
            if val.dtype == znp.complex128:
                iscomplex.append(True)
                paramvalscomplex.append(val)
                paramvalsfloat.append(znp.zeros_like(val, dtype=znp.float64))
            else:
                iscomplex.append(False)
                paramvalsfloat.append(val)
                paramvalscomplex.append(znp.zeros_like(val, dtype=znp.complex128))

        def unnormalized_pdf(x, paramvalsfloat, paramvalscomplex):
            data = {ob: znp.array(ar) for ob, ar in zip(self.obs, x)}
            paramsinternal = {
                n: c if isc else f for n, f, c, isc in zip(params.keys(), paramvalsfloat, paramvalscomplex, iscomplex)
            }
            self.intensity.update_parameters(paramsinternal)
            return self.intensity(data)

        xunstacked = z.unstack_x(x)

        probs = tf.numpy_function(unnormalized_pdf, [xunstacked, paramvalsfloat, paramvalscomplex], Tout=tf.float64)
        if norm is not False:
            normvalues = [znp.asarray(self.norm_sample[ob]) for ob in self.obs]
            normval = (
                znp.mean(
                    tf.numpy_function(unnormalized_pdf, [normvalues, paramvalsfloat, paramvalscomplex], Tout=tf.float64)
                )
                * norm.volume
            )
            normval.set_shape((1,))
            probs /= normval
        probs.set_shape([None])
        return probs

    @z.function(wraps="tensorwaves")
    def _jitted_normalization(self, norm, params):
        return znp.mean(self._jitted_unnormalized_pdf(norm, params=params))
