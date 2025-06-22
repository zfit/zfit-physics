from __future__ import annotations

import tensorflow as tf
import zfit  # suppress tf warnings
import zfit.z.numpy as znp
from zfit import supports, z

from .variables import obs_from_frame, params_from_intensity

__all__ = ["ComPWAPDF"]


class ComPWAPDF(zfit.pdf.BasePDF):
    def __init__(self, intensity, norm, obs=None, params=None, extended=None, name="ComPWA"):
        """ComPWA intensity normalized over the *norm* dataset."""
        if params is None:
            params = {p.name: p for p in params_from_intensity(intensity)}
        norm = zfit.Data(norm, obs=obs)
        if obs is None:
            obs = obs_from_frame(norm.to_pandas())
        norm = norm.with_obs(obs)
        super().__init__(obs, params=params, name=name, extended=extended, autograd_params=[])
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

        def unnormalized_pdf_helper(x, paramvalsfloat, paramvalscomplex):
            data = {ob: znp.array(ar) for ob, ar in zip(self.obs, x)}
            paramsinternal = {
                n: c if isc else f for n, f, c, isc in zip(params.keys(), paramvalsfloat, paramvalscomplex, iscomplex)
            }
            self.intensity.update_parameters(paramsinternal)
            return self.intensity(data)

        xunstacked = z.unstack_x(x)

        probs = tf.numpy_function(
            unnormalized_pdf_helper, [xunstacked, paramvalsfloat, paramvalscomplex], Tout=tf.float64
        )
        if norm is not False:
            normvalues = [znp.asarray(self.norm_sample[ob]) for ob in self.obs]
            normval = (
                znp.mean(
                    tf.numpy_function(
                        unnormalized_pdf_helper, [normvalues, paramvalsfloat, paramvalscomplex], Tout=tf.float64
                    )
                )
                * znp.array([1.0])  # HACK: ComPWA just uses 1 as the phase space volume, better solution?
                # norm.volue is very small, since as it's done now (autoconverting in init), there are variables like
                # masses that have a tiny space, so the volume is very small
                # * norm.volume
            )
            normval.set_shape((1,))
            probs /= normval
        probs.set_shape([None])
        return probs

    # @z.function(wraps="tensorwaves")
    # def _jitted_normalization(self, norm, params):
    #     return znp.mean(self._jitted_unnormalized_pdf(norm, params=params))
