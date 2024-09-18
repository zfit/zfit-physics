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
    return znp.where(cond, low_x_BK(nu, x), bessel_kv_func(nu, x))


@z.function(wraps="tensor")
def LnBK(ni, x):
    nu = znp.abs(ni)
    first_cond = tf.logical_and(x < 1.0e-06, nu > 0.0)
    second_cond = tf.logical_and(tf.logical_and(x < 1.0e-04, nu > 0.0), nu < 55)
    third_cond = tf.logical_and(x < 0.1, nu >= 55)
    cond = tf.logical_or(first_cond, tf.logical_or(second_cond, third_cond))
    return znp.where(cond, low_x_LnBK(nu, x), znp.log(bessel_kv_func(nu, x)))


@z.function(wraps="tensor")
def LogEval(d, lam, alpha, beta, delta):
    # d = x-mu
    # sq2pi = znp.sqrt(2*znp.arccos(-1))
    gamma = alpha  # znp.sqrt(alpha*alpha-beta*beta)
    dg = delta * gamma
    thing = delta * delta + d * d
    logno = lam * znp.log(gamma / delta) - logsq2pi - LnBK(lam, dg)

    return znp.exp(
        logno
        + beta * d
        + (0.5 - lam) * (znp.log(alpha) - 0.5 * znp.log(thing))
        + LnBK(lam - 0.5, alpha * znp.sqrt(thing))
    )  # + znp.log(znp.abs(beta)+0.0001) )


@z.function(wraps="tensor")
def diff_eval(d, lam, alpha, beta, delta):
    gamma = alpha
    dg = delta * gamma
    thing = delta * delta + d * d
    sqthing = znp.sqrt(thing)
    alphasq = alpha * sqthing
    no = znp.power(gamma / delta, lam) / BK(lam, dg) * sq2pi_inv
    ns1 = 0.5 - lam

    return (
        no
        * znp.power(alpha, ns1)
        * znp.power(thing, lam / 2.0 - 1.25)
        * (
            -d * alphasq * (BK(lam - 1.5, alphasq) + BK(lam + 0.5, alphasq))
            + (2.0 * (beta * thing + d * lam) - d) * BK(ns1, alphasq)
        )
        * znp.exp(beta * d)
        / 2.0
    )


# def Gauss2F1(a, b, c, x):
#     largey = tfp.math.hypergeometric.hyp2f1_small_argument(c - a, b, c, 1 - 1 / (1 - x)) / znp.power(1 - x, b)
#     smally = tfp.math.hypergeometric.hyp2f1_small_argument(a, b, c, x)
#     return znp.where(znp.abs(x) <= 1, smally, largey)
#     # if (znp.abs(x) <= 1):
#     # return ROOT.Math.hyperg(a, b, c, x)

#     # ROOT::Math::hyperg(a,b,c,x)
#     # else:
#     # return ROOT.Math.hyperg(c - a, b, c, 1 - 1 / (1 - x)) / znp.power(1 - x, b)
#     # return largey


# def stIntegral(d1, delta, l):
#     # printf("::::: %e %e %e\n", d1,delta, l)
#     return d1 * Gauss2F1(0.5, 0.5 - l, 3.0 / 2, -d1 * d1 / (delta * delta))
#     # printf(":::Done\n")
#     # return out


@z.function(wraps="tensor")
def ipatia2_func(x, lam, zeta, fb, mu, sigma, n, n2, a, a2):
    r"""Calculate the Ipatia2 PDF value.

    Args:
        x: Value(s) to evaluate the PDF at.
        mu: Location parameter. Shifts the distribution left/right.
        sigma: Width parameter. If :math:` \beta = 0, \ \sigma ` is the RMS width.
        al: Start of the left tail (:math` a \geq 0 `, to the left of the peak). Note that when setting :math` al = \sigma = 1 `, the tail region is to the left of :math` x = \mu - 1 `, so a should be positive.
        nl: Shape parameter of left tail (:math` nl \ge 0 `). With :math` nr = 0 `, the function is constant.
        ar: Start of right tail.
        nr: Shape parameter of right tail (:math` nr \ge 0 `). With :math` nr = 0 `, the function is constant.
        lam: Shape parameter. Note that :math` \lambda < 0 ` is required if :math` \zeta = 0 `.
        beta: Asymmetry parameter :math` \beta `. Symmetric case is :math` \beta = 0 `, choose values close to zero.
        zeta: Shape parameter (:math` \zeta >= 0 `).
    Returns:
        `tf.Tensor`: The value of the Ipatia2 PDF at x.
    """
    d = x - mu
    cons0 = znp.sqrt(zeta)
    asigma = a * sigma
    a2sigma = a2 * sigma
    cond1 = d < -asigma
    cond2 = d > a2sigma
    conda1 = zeta != 0.0
    # cond1
    phi = BK(lam + 1.0, zeta) / BK(lam, zeta)
    cons1 = sigma / znp.sqrt(phi)
    alpha = cons0 / cons1  # *znp.sqrt((1 - fb*fb))
    beta = fb  # *alpha
    delta = cons0 * cons1

    # printf("-_-\n")
    # printf("alpha %e\n",alpha)
    # printf("beta %e\n",beta)
    # printf("delta %e\n",delta)

    k1 = LogEval(-asigma, lam, alpha, beta, delta)
    k2 = diff_eval(-asigma, lam, alpha, beta, delta)
    B = -asigma + n * k1 / k2
    A = k1 * znp.power(B + asigma, n)
    out1 = A * znp.power(B - d, -n)

    k1 = LogEval(a2sigma, lam, alpha, beta, delta)
    k2 = diff_eval(a2sigma, lam, alpha, beta, delta)

    B = -a2sigma - n2 * k1 / k2

    A = k1 * znp.power(B + a2sigma, n2)

    out2 = A * znp.power(B + d, -n2)

    out3 = LogEval(d, lam, alpha, beta, delta)
    outa1 = znp.where(cond1, out1, znp.where(cond2, out2, out3))

    # cond2 = d > a2sigma
    beta = fb
    cons1 = -2.0 * lam
    # delta = sigma
    condx = lam <= -1.0

    delta1 = sigma * znp.sqrt(-2 + cons1)

    # printf("WARNING: zeta ==0 and l > -1 ==> not defined rms. Changing the meaning of sigma, but I keep fitting anyway\n")
    delta2 = sigma
    delta = znp.where(condx, delta1, delta2)

    delta2 = delta * delta
    # cond1
    cons1 = znp.exp(-beta * asigma)
    phi = 1.0 + asigma * asigma / delta2
    k1 = cons1 * znp.power(phi, lam - 0.5)
    k2 = beta * k1 - cons1 * (lam - 0.5) * znp.power(phi, lam - 1.5) * 2 * asigma / delta2
    B = -asigma + n * k1 / k2
    A = k1 * znp.power(B + asigma, n)
    outz1 = A * znp.power(B - d, -n)
    # cond2
    cons1 = znp.exp(beta * a2sigma)
    phi = 1.0 + a2sigma * a2sigma / delta2
    k1 = cons1 * znp.power(phi, lam - 0.5)
    k2 = beta * k1 + cons1 * (lam - 0.5) * znp.power(phi, lam - 1.5) * 2.0 * a2sigma / delta2
    B = -a2sigma - n2 * k1 / k2
    A = k1 * znp.power(B + a2sigma, n2)
    outz2 = A * znp.power(B + d, -n2)
    # cond3
    outz3 = znp.exp(beta * d) * znp.power(1.0 + d * d / delta2, lam - 0.5)

    outa2 = znp.where(cond1, outz1, znp.where(cond2, outz2, outz3))

    return znp.where(conda1, outa1, outa2)


class Ipatia2(zfit.pdf.BasePDF):
    def __init__(
        self,
        obs: ztyping.ObsTypeInput,
        mu: ztyping.ParamTypeInput,
        sigma: ztyping.ParamTypeInput,
        nl: ztyping.ParamTypeInput,
        al: ztyping.ParamTypeInput,
        nr: ztyping.ParamTypeInput,
        ar: ztyping.ParamTypeInput,
        lam: ztyping.ParamTypeInput,
        beta: ztyping.ParamTypeInput,
        zeta: ztyping.ParamTypeInput,
        *,
        extended: ztyping.ExtendedInputType | None = None,
        norm: ztyping.NormInputType | None = None,
        name: str = "Ipatia2",
        label: str | None = None,
    ):
        r""""
        The implementation follows the `RooHypatia2 <https://root.cern.ch/doc/master/RooHypatia2_8cxx_source.html>`_

        Ipatia2 is the two-sided version of the Hypatia distribution described in https://arxiv.org/abs/1312.5000.

        It has a hyperbolic core of a crystal-ball-like function :math:` G ` and two tails:

        .. math::

            \mathrm{Hypatia2}(x, \mu, \sigma, \lambda, \zeta, \beta, a_l, n_l, a_r, n_r) =
            \begin{cases}
            \frac{ G(\mu - a_l \sigma, \mu, \sigma, \lambda, \zeta, \beta)                             }
                { \left( 1 - \frac{x}{n_l G(\ldots)/G'(\ldots) - a_l\sigma } \right)^{n_l} }
                & \text{if } \frac{x-\mu}{\sigma} < -a_l \
            \left( (x-\mu)^2 + A^2_\lambda(\zeta)\sigma^2 \right)^{\frac{1}{2}\lambda-\frac{1}{4}} e^{\beta(x-\mu)} K_{\lambda-\frac{1}{2}}
                \left( \zeta \sqrt{1+\left( \frac{x-\mu}{A_\lambda(\zeta)\sigma} \right)^2 } \right) \equiv G(x, \mu, \ldots)
                & \text{otherwise} \
            \frac{ G(\mu + a_r \sigma, \mu, \sigma, \lambda, \zeta, \beta)                               }
                { \left( 1 - \frac{x}{-n_r G(\ldots)/G'(\ldots) - a_r\sigma } \right)^{n_r} }
                & \text{if } \frac{x-\mu}{\sigma} > a_r \
            \end{cases}
            \f]
            Here, ` K_\lambda ` are the modified Bessel functions of the second kind
            ("irregular modified cylindrical Bessel functions" from the gsl,
            "special Bessel functions of the third kind"),
            and ` A^2_\lambda(\zeta) ` is a ratio of these:
            \f[
            A_\lambda^{2}(\zeta) = \frac{\zeta K_\lambda(\zeta)}{K_{\lambda+1}(\zeta)}

        Note that unless the parameters :math:` a_l,\ a_r ` are very large, the function has non-hyperbolic tails. This requires
        :math:` G ` to be strictly concave, *i.e.*, peaked, as otherwise the tails would yield imaginary numbers. Choosing :math:` \lambda,
        \beta, \zeta ` inappropriately will therefore lead to evaluation errors.

        Further, the original paper establishes that to keep the tails from rising,
        .. math::
            \begin{split}
            \beta^2 &< \alpha^2 \
            \Leftrightarrow \beta^2 &< \frac{\zeta^2}{\delta^2} = \frac{\zeta^2}{\sigma^2 A_{\lambda}^2(\zeta)}
            \end{split}
        needs to be satisfied, unless the fit range is very restricted, because otherwise, the function rises in the tails.


        In case of evaluation errors, it is advisable to choose very large values for :math:` a_l,\ a_r `, tweak the parameters of the core region to
        make it concave, and re-enable the tails. Especially :math:` \beta ` needs to be close to zero.

        Args:
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
            mu: Location parameter. Shifts the distribution left/right.
            sigma: Width parameter. If :math:` \beta = 0, \ \sigma ` is the RMS width.
            al: Start of the left tail (:math` a \geq 0 `, to the left of the peak). Note that when setting :math` al = \sigma = 1 `, the tail region is to the left of :math` x = \mu - 1 `, so a should be positive.
            nl: Shape parameter of left tail (:math` nl \ge 0 `). With :math` nr = 0 `, the function is constant.
            ar: Start of right tail.
            nr: Shape parameter of right tail (:math` nr \ge 0 `). With :math` nr = 0 `, the function is constant.
            lam: Shape parameter. Note that :math` \lambda < 0 ` is required if :math` \zeta = 0 `.
            beta: Asymmetry parameter :math` \beta `. Symmetric case is :math` \beta = 0 `, choose values close to zero.
            zeta: Shape parameter (:math` \zeta >= 0 `).
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
            "nl": nl,
            "al": al,
            "nr": nr,
            "ar": ar,
            "lam": lam,
            "beta": beta,
            "zeta": zeta,
        }
        super().__init__(obs=obs, params=params, name=name, extended=extended, norm=norm, label=label)

    @zfit.supports()
    def _unnormalized_pdf(self, x: tf.Tensor, params) -> tf.Tensor:
        x0 = x[0]
        mu = params["mu"]
        sigma = params["sigma"]
        nl = params["nl"]
        al = params["al"]
        nr = params["nr"]
        ar = params["ar"]
        lam = params["lam"]
        beta = params["beta"]
        zeta = params["zeta"]
        return ipatia2_func(x0, lam, zeta, beta, mu, sigma, nl, nr, al, ar)
