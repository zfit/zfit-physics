from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp
import zfit
import zfit.models.functor
from zfit import z
from zfit.exception import FunctionNotImplemented
from zfit.util import exception, ztyping
from zfit.util.exception import WorkInProgressError


class NumConvPDFUnbinnedV1(zfit.models.functor.BaseFunctor):
    def __init__(
        self,
        func: zfit.pdf.BasePDF,
        kernel: zfit.pdf.BasePDF,
        limits: ztyping.ObsTypeInput,
        obs: ztyping.ObsTypeInput,
        ndraws: int = 20000,
        *,
        extended: ztyping.ParamTypeInput | None = None,
        norm: ztyping.NormTypeInput = None,
        name: str = "Convolution",
        label: str | None = None,
        experimental_pdf_normalized=False,
    ):
        """Numerical Convolution pdf of *func* convoluted with *kernel*.

        Args:
            func (:py:class:`zfit.pdf.BasePDF`): PDF  with `pdf` method that takes x and returns the function value.
                Here x is a `Data` with the obs and limits of *limits*.
            kernel (:py:class:`zfit.pdf.BasePDF`): PDF with `pdf` method that takes x acting as the kernel.
                Here x is a `Data` with the obs and limits of *limits*.
            limits (:py:class:`zfit.Space`): Limits for the numerical integration.
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:pdf.init.name| Name of the PDF.
               Maybe has implications on the serialization and deserialization of the PDF.
               For a human-readable name, use the label. |@docend:pdf.init.name|
           label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
        """
        super().__init__(obs=obs, pdfs=[func, kernel], params={}, name=name, extended=extended, norm=norm, label=label)
        limits = self._check_input_limits(limits=limits)
        if limits.n_limits == 0:
            msg = "obs have to have limits to define where to integrate over."
            raise exception.LimitsNotSpecifiedError(msg)
        if limits.n_limits > 1:
            msg = "Multiple Limits not implemented"
            raise WorkInProgressError(msg)

        #        if not isinstance(func, zfit.pdf.BasePDF):
        #            raise TypeError(f"func has to be a PDF, not {type(func)}")
        #        if isinstance(kernel, zfit.pdf.BasePDF):
        #            raise TypeError(f"kernel has to be a PDF, not {type(kernel)}")

        # func = lambda x: func.unnormalized_pdf(x=x)
        # kernel = lambda x: kernel.unnormalized_pdf(x=x)

        self.conv_limits = limits
        self._ndraws = ndraws
        self._experimental_pdf_normalized = experimental_pdf_normalized

    @z.function
    def _unnormalized_pdf(self, x):
        limits = self.conv_limits
        area = limits.rect_area()[0]  # new spaces

        # create sample for numerical integral
        lower, upper = limits.rect_limits
        lower = z.convert_to_tensor(lower, dtype=self.dtype)
        upper = z.convert_to_tensor(upper, dtype=self.dtype)
        samples_normed = tfp.mcmc.sample_halton_sequence(
            dim=limits.n_obs,
            num_results=self._ndraws,
            dtype=self.dtype,
            randomized=False,
        )
        samples = samples_normed * (upper - lower) + lower  # samples is [0, 1], stretch it
        samples = zfit.Data.from_tensor(obs=limits, tensor=samples)

        func_values = self.pdfs[0].pdf(samples, norm=False)  # func of true vars

        return tf.map_fn(
            lambda xi: area * tf.reduce_mean(func_values * self.pdfs[1].pdf(xi - samples.value(), norm=False)),
            x.value(),
        )

    @zfit.supports(norm=True)
    @z.function
    def _pdf(self, x, norm):
        del norm
        if not self._experimental_pdf_normalized:
            raise FunctionNotImplemented

        limits = self.conv_limits
        # area = limits.area()  # new spaces
        area = limits.rect_area()[0]  # new spaces

        # create sample for numerical integral
        lower, upper = limits.rect_limits
        lower = z.convert_to_tensor(lower, dtype=self.dtype)
        upper = z.convert_to_tensor(upper, dtype=self.dtype)
        samples_normed = tfp.mcmc.sample_halton_sequence(
            dim=limits.n_obs,
            num_results=self._ndraws,
            dtype=self.dtype,
            randomized=False,
        )
        samples = samples_normed * (upper - lower) + lower  # samples is [0, 1], stretch it
        samples = zfit.Data.from_tensor(obs=limits, tensor=samples)

        func_values = self.pdfs[0].pdf(samples)  # func of true vars

        return tf.map_fn(
            lambda xi: area * tf.reduce_mean(func_values * self.pdfs[1].pdf(xi - samples.value())),
            x.value(),
        )
