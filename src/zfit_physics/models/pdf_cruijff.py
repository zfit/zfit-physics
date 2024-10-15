from __future__ import annotations

import zfit
from zfit import z
from zfit.util import ztyping
from zfit.z import numpy as znp


@z.function(wraps="tensor")
def cruijff_pdf_func(x, mu, sigmal, alphal, sigmar, alphar):
    """Calculate the Cruijff PDF.

    Args:
        x: value(s) for which the PDF will be calculated.
        mu: Mean value
        sigmal: Left width parameter.
        alphal: Left tail acceleration parameter.
        sigmar: Right width parameter.
        alphar: Right tail acceleration parameter.

    Returns:
        `tf.Tensor`: The calculated PDF values.

    Notes:
        Implementation from https://arxiv.org/abs/1005.4087, citation 22, and http://hdl.handle.net/1794/9022, Equation IV.3
    """
    cond = znp.less(x, mu)

    # compute only once (in graph this _may_ be optimized anyways, but surely not in eager)
    xminmu = x - mu
    tleft = znp.square(xminmu / sigmal)
    tright = znp.square(xminmu / sigmar)
    exponent = znp.where(
        cond,
        tleft / (1 + alphal * tleft),
        tright / (1 + alphar * tright),
    )
    return znp.exp(-0.5 * exponent)


class Cruijff(zfit.pdf.BasePDF):
    _N_OBS = 1

    def __init__(
        self,
        mu: ztyping.ParamTypeInput,
        sigmal: ztyping.ParamTypeInput,
        alphal: ztyping.ParamTypeInput,
        sigmar: ztyping.ParamTypeInput,
        alphar: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ztyping.ExtendedInputType | None = False,
        norm: ztyping.NormInputType | None = None,
        name: str = "Cruijff",
        label: str | None = None,
    ):
        """Cruijff PDF, a Gaussian with two width, left and right, and non-Gaussian tails.

        Implementation from https://arxiv.org/abs/1005.4087, citation 22, and http://hdl.handle.net/1794/9022, Equation IV.3

        .. math:

            f(x; \\mu, \\sigma_{L}, \\alpha_{L}, \\sigma_{R}, \\alpha_{R}) = \\begin{cases}
            \\exp{\\left(- \\frac{(x-\\mu)^2}{2 \\sigma_{L}^2 + \\alpha_{L} (x-\\mu)^2}\\right)}, \\mbox{for} x \\leqslant mu \\newline
            \\exp{\\left(- \\frac{(x-\\mu)^2}{2 \\sigma_{R}^2 + \\alpha_{R} (x-\\mu)^2}\\right)}, \\mbox{for} x > mu
            \\end{cases}

        Args:
            mu: Mean value
            sigmal: Left width parameter.
            alphal: Left tail acceleration parameter.
            sigmar: Right width parameter.
            alphar: Right tail acceleration parameter.
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
        params = {"mu": mu, "sigmal": sigmal, "alphal": alphal, "sigmar": sigmar, "alphar": alphar}
        super().__init__(obs=obs, params=params, extended=extended, norm=norm, name=name, label=label)

    @zfit.supports()
    def _unnormalized_pdf(self, x, params):
        mu = params["mu"]
        sigmal = params["sigmal"]
        alphal = params["alphal"]
        sigmar = params["sigmar"]
        alphar = params["alphar"]
        x = x[0]
        return cruijff_pdf_func(x=x, mu=mu, sigmal=sigmal, alphal=alphal, sigmar=sigmar, alphar=alphar)
