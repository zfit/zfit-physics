from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import zfit
import zfit.z.numpy as znp
from zfit import z
from zfit.util import ztyping

sq2pi = np.sqrt(2.0 * np.arccos(-1.0))
sq2pi_inv = 1.0 / sq2pi
logsq2pi = np.log(sq2pi)
log_de_2 = np.log(2.0)


@z.function(wraps="tensor")
def bessel_kv_func(nu, x):
    """Modified Bessel function of the 2nd kind."""
    return tfp.math.bessel_kve(nu, x) / znp.exp(znp.abs(x))


@z.function(wraps="tensor")
def gammafunc(nu):
    return znp.exp(tf.math.lgamma(nu))


@z.function(wraps="tensor")
def low_x_BK(nu, x):
    return gammafunc(nu) * znp.power(2.0, nu - 1.0) * znp.power(x, -nu)


@z.function(wraps="tensor")
def low_x_LnBK(nu, x):
    return znp.log(gammafunc(nu)) + (nu - 1.0) * log_de_2 - nu * znp.log(x)


@z.function(wraps="tensor")
def BK(ni, x):
    nu = znp.abs(ni)
    first_cond = tf.logical_and(x < 1.0e-06, nu > 0.0)
    second_cond = tf.logical_and(tf.logical_and(x < 1.0e-04, nu > 0.0), nu < 55)
    third_cond = tf.logical_and(x < 0.1, nu >= 55)
    cond = tf.logical_or(first_cond, tf.logical_or(second_cond, third_cond))
    return z.safe_where(
        cond,
        lambda t: low_x_BK(nu, t),
        lambda t: bessel_kv_func(nu, t),
        values=x,
        value_safer=lambda t: znp.ones_like(t),
    )


@z.function(wraps="tensor")
def LnBK(ni, x):
    nu = znp.abs(ni)
    first_cond = tf.logical_and(x < 1.0e-06, nu > 0.0)
    second_cond = tf.logical_and(tf.logical_and(x < 1.0e-04, nu > 0.0), nu < 55)
    third_cond = tf.logical_and(x < 0.1, nu >= 55)
    cond = tf.logical_or(first_cond, tf.logical_or(second_cond, third_cond))
    return z.safe_where(
        cond,
        lambda t: low_x_LnBK(nu, t),
        lambda t: znp.log(bessel_kv_func(nu, t)),
        values=x,
        value_safer=lambda t: znp.ones_like(t),
    )


@z.function(wraps="tensor")
def LogEval(d, lambd, alpha, beta, delta):
    # d = x-mu
    # sq2pi = znp.sqrt(2*znp.arccos(-1))
    gamma = alpha  # znp.sqrt(alpha*alpha-beta*beta)
    dg = delta * gamma
    thing = delta * delta + d * d
    logno = lambd * znp.log(gamma / delta) - logsq2pi - LnBK(lambd, dg)

    return znp.exp(
        logno
        + beta * d
        + (0.5 - lambd) * (znp.log(alpha) - 0.5 * znp.log(thing))
        + LnBK(lambd - 0.5, alpha * znp.sqrt(thing))
    )  # + znp.log(znp.abs(beta)+0.0001) )


@z.function(wraps="tensor")
def diff_eval(d, lambd, alpha, beta, delta):
    gamma = alpha
    dg = delta * gamma
    thing = delta * delta + d * d
    sqthing = znp.sqrt(thing)
    alphasq = alpha * sqthing
    no = znp.power(gamma / delta, lambd) / BK(lambd, dg) * sq2pi_inv
    ns1 = 0.5 - lambd

    return (
        no
        * znp.power(alpha, ns1)
        * znp.power(thing, lambd / 2.0 - 1.25)
        * (
            -d * alphasq * (BK(lambd - 1.5, alphasq) + BK(lambd + 0.5, alphasq))
            + (2.0 * (beta * thing + d * lambd) - d) * BK(ns1, alphasq)
        )
        * znp.exp(beta * d)
        / 2.0
    )


@z.function(wraps="tensor")
def hypatia2_func(x, mu, sigma, lambd, zeta, beta, alphal, nl, alphar, nr):
    r"""Calculate the Hypatia2 PDF value.

    Args:
        x: Value(s) to evaluate the PDF at.
        mu: Location parameter. Shifts the distribution left/right.
        sigma: Width parameter. If :math:` \\beta = 0, \\ \\sigma ` is the RMS width.
        alphal: Start of the left tail (:math` a \\geq 0 `, to the left of the peak). Note that when setting :math` al = \\sigma = 1 `, the tail region is to the left of :math` x = \\mu - 1 `, so a should be positive.
        nl: Shape parameter of left tail (:math` nl \\ge 0 `). With :math` nr = 0 `, the function is constant.
        alphar: Start of right tail.
        nr: Shape parameter of right tail (:math` nr \\ge 0 `). With :math` nr = 0 `, the function is constant.
        lambd: Shape parameter. Note that :math` \\lambda < 0 ` is required if :math` \\zeta = 0 `.
        beta: Asymmetry parameter :math` \\beta `. Symmetric case is :math` \\beta = 0 `, choose values close to zero.
        zeta: Shape parameter (:math` \\zeta >= 0 `).
    Returns:
        `tf.Tensor`: The value of the Hypatia2 PDF at x.
    """
    d = x - mu
    cons0 = znp.sqrt(zeta)
    alsigma = alphal * sigma
    arsigma = alphar * sigma
    cond1 = d < -alsigma
    cond2 = d > arsigma
    conda1 = zeta != 0.0
    # cond1
    phi = BK(lambd + 1.0, zeta) / BK(lambd, zeta)
    cons1 = sigma / znp.sqrt(phi)
    alpha = cons0 / cons1  # *znp.sqrt((1 - beta*beta))
    delta = cons0 * cons1

    k1 = LogEval(-alsigma, lambd, alpha, beta, delta)
    k2 = diff_eval(-alsigma, lambd, alpha, beta, delta)
    B = -alsigma + nl * k1 / k2
    A = k1 * znp.power(B + alsigma, nl)
    out1 = A * znp.power(B - d, -nl)

    k1 = LogEval(arsigma, lambd, alpha, beta, delta)
    k2 = diff_eval(arsigma, lambd, alpha, beta, delta)

    B = -arsigma - nr * k1 / k2

    A = k1 * znp.power(B + arsigma, nr)

    out2 = A * znp.power(B + d, -nr)

    out3 = LogEval(d, lambd, alpha, beta, delta)
    outa1 = znp.where(cond1, out1, znp.where(cond2, out2, out3))

    # cond2 = d > arsigma
    cons1 = -2.0 * lambd
    # delta = sigma
    condx = lambd <= -1.0

    delta1 = sigma * znp.sqrt(-2 + cons1)

    delta2 = sigma
    delta = znp.where(condx, delta1, delta2)

    delta2 = delta * delta
    # cond1
    cons1 = znp.exp(-beta * alsigma)
    phi = 1.0 + alsigma * alsigma / delta2
    k1 = cons1 * znp.power(phi, lambd - 0.5)
    k2 = beta * k1 - cons1 * (lambd - 0.5) * znp.power(phi, lambd - 1.5) * 2 * alsigma / delta2
    B = -alsigma + nl * k1 / k2
    A = k1 * znp.power(B + alsigma, nl)
    outz1 = A * znp.power(B - d, -nl)
    # cond2
    cons1 = znp.exp(beta * arsigma)
    phi = 1.0 + arsigma * arsigma / delta2
    k1 = cons1 * znp.power(phi, lambd - 0.5)
    k2 = beta * k1 + cons1 * (lambd - 0.5) * znp.power(phi, lambd - 1.5) * 2.0 * arsigma / delta2
    B = -arsigma - nr * k1 / k2
    A = k1 * znp.power(B + arsigma, nr)
    outz2 = A * znp.power(B + d, -nr)
    # cond3
    outz3 = znp.exp(beta * d) * znp.power(1.0 + d * d / delta2, lambd - 0.5)

    outa2 = znp.where(cond1, outz1, znp.where(cond2, outz2, outz3))

    return znp.where(conda1, outa1, outa2)


class Hypatia2(zfit.pdf.BasePDF):
    def __init__(
        self,
        obs: ztyping.ObsTypeInput,
        mu: ztyping.ParamTypeInput,
        sigma: ztyping.ParamTypeInput,
        lambd: ztyping.ParamTypeInput,
        zeta: ztyping.ParamTypeInput,
        beta: ztyping.ParamTypeInput,
        alphal: ztyping.ParamTypeInput,
        nl: ztyping.ParamTypeInput,
        alphar: ztyping.ParamTypeInput,
        nr: ztyping.ParamTypeInput,
        *,
        extended: ztyping.ExtendedInputType | None = None,
        norm: ztyping.NormInputType | None = None,
        name: str = "Hypatia2",
        label: str | None = None,
    ):
        r""""
        The implementation follows the `RooHypatia2 <https://root.cern.ch/doc/master/RooHypatia2_8cxx_source.html>`_

        Hypatia2 is the two-sided version of the Hypatia distribution described in https://arxiv.org/abs/1312.5000.

        It has a hyperbolic core of a crystal-ball-like function :math:` G ` and two tails:

        .. math::

            \\mathrm{Hypatia2}(x;\\mu, \\sigma, \\lambda, \\zeta, \\beta, \\alpha_{L}, n_{L}, \\alpha_{R}, n_{R}) =
            \\begin{cases}
            \\frac{ G(\\mu - \\alpha_{L} \\sigma, \\mu, \\sigma, \\lambda, \\zeta, \\beta)                             }
                { \\left( 1 - \\frac{x}{n_{L} G(\\ldots)/G'(\\ldots) - \\alpha_{L}\\sigma } \\right)^{n_{L}} }
                & \\text{if } \\frac{x-\\mu}{\\sigma} < -\\alpha_{L} \\
            \\left( (x-\\mu)^2 + A^2_\\lambda(\\zeta)\\sigma^2 \\right)^{\\frac{1}{2}\\lambda-\\frac{1}{4}} e^{\\beta(x-\\mu)} K_{\\lambda-\\frac{1}{2}}
                \\left( \\zeta \\sqrt{1+\\left( \\frac{x-\\mu}{A_\\lambda(\\zeta)\\sigma} \\right)^2 } \\right) \\equiv G(x, \\mu, \\ldots)
                & \\text{otherwise} \\
            \\frac{ G(\\mu + \\alpha_{R} \\sigma, \\mu, \\sigma, \\lambda, \\zeta, \\beta)                               }
                { \\left( 1 - \\frac{x}{-n_{R} G(\\ldots)/G'(\\ldots) - \\alpha_{R}\\sigma } \\right)^{n_{R}} }
                & \\text{if } \\frac{x-\\mu}{\\sigma} > \\alpha_{R} \\
            \\end{cases}
            \\f]
            Here, ` K_\\lambda ` are the modified Bessel functions of the second kind
            ("irregular modified cylindrical Bessel functions" from the gsl,
            "special Bessel functions of the third kind"),
            and ` A^2_\\lambda(\\zeta) ` is a ratio of these:
            \\f[
            A_\\lambda^{2}(\\zeta) = \\frac{\\zeta K_\\lambda(\\zeta)}{K_{\\lambda+1}(\\zeta)}

        Note that unless the parameters :math:` \\alpha_{L},\\ \\alpha_{R} ` are very large, the function has non-hyperbolic tails. This requires
        :math:` G ` to be strictly concave, *i.e.*, peaked, as otherwise the tails would yield imaginary numbers. Choosing :math:` \\lambda,
        \\beta, \\zeta ` inappropriately will therefore lead to evaluation errors.

        Further, the original paper establishes that to keep the tails from rising,
        .. math::
            \\begin{split}
            \\beta^2 &< \\alpha^2 \\
            \\Leftrightarrow \\beta^2 &< \\frac{\\zeta^2}{\\delta^2} = \\frac{\\zeta^2}{\\sigma^2 A_{\\lambda}^2(\\zeta)}
            \\end{split}
        needs to be satisfied, unless the fit range is very restricted, because otherwise, the function rises in the tails.


        In case of evaluation errors, it is advisable to choose very large values for :math:` \\alpha_{L},\\ \\alpha_{R} `, tweak the parameters of the core region to
        make it concave, and re-enable the tails. Especially :math:` \\beta ` needs to be close to zero.

        Args:
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
            mu: Location parameter. Shifts the distribution left/right.
            sigma: Width parameter. If :math:` \\beta = 0, \\ \\sigma ` is the RMS width.
            lambd: Shape parameter. Note that :math` \\lambda < 0 ` is required if :math` \\zeta = 0 `.
            zeta: Shape parameter (:math` \\zeta >= 0 `).
            beta: Asymmetry parameter :math` \\beta `. Symmetric case is :math` \\beta = 0 `, choose values close to zero.
            al: Start of the left tail (:math` a \\geq 0 `, to the left of the peak). Note that when setting :math` al = \\sigma = 1 `, the tail region is to the left of :math` x = \\mu - 1 `, so a should be positive.
            nl: Shape parameter of left tail (:math` nl \\ge 0 `). With :math` nr = 0 `, the function is constant.
            ar: Start of right tail.
            nr: Shape parameter of right tail (:math` nr \\ge 0 `). With :math` nr = 0 `, the function is constant.
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:pdf.init.name| Human-readable name
               or label of
               the PDF for better identification. |@docend:pdf.init.name|
           label: |@doc:pdf.init.label| Label of the PDF, if None is given, it will be the name. |@docend:pdf.init.label|
        """
        params = {
            "mu": mu,
            "sigma": sigma,
            "lambd": lambd,
            "zeta": zeta,
            "beta": beta,
            "alphal": alphal,
            "nl": nl,
            "alphar": alphar,
            "nr": nr,
        }
        super().__init__(obs=obs, params=params, name=name, extended=extended, norm=norm, label=label)

    @zfit.supports()
    def _unnormalized_pdf(self, x: tf.Tensor, params) -> tf.Tensor:
        x0 = x[0]
        mu = params["mu"]
        sigma = params["sigma"]
        lambd = params["lambd"]
        zeta = params["zeta"]
        beta = params["beta"]
        alphal = params["alphal"]
        nl = params["nl"]
        alphar = params["alphar"]
        nr = params["nr"]

        return hypatia2_func(x0, mu, sigma, lambd, zeta, beta, alphal, nl, alphar, nr)
