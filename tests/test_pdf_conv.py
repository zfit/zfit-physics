"""Tests for convolution PDF."""

import numpy as np
import pytest
import zfit
import zfit.z.numpy as znp
from scipy import integrate

import zfit_physics as zphys

# specify globals here. Do NOT add any TensorFlow but just pure python
param1_true = 0.3
param2_true = 1.2


def test_conv_simple():
    # test special properties  here
    n_points = 2000
    obs = zfit.Space("obs1", limits=(-5, 5))
    param1 = zfit.Parameter("param1", -3)
    param2 = zfit.Parameter("param2", 0.3)
    gauss1 = zfit.pdf.Gauss(0.0, param2, obs=obs)
    uniform1 = zfit.pdf.Uniform(param1, param2, obs=obs)
    conv = zphys.unstable.pdf.NumConvPDFUnbinnedV1(func=uniform1, kernel=gauss1, limits=obs, obs=obs, vectorized=False)

    x = znp.linspace(-5.0, 5.0, n_points)
    probs = conv.pdf(x=x)
    integral = conv.integrate(limits=obs)
    probs_np = probs.numpy()
    assert pytest.approx(1, rel=1e-3) == integral.numpy()
    assert len(probs_np) == n_points
    # import matplotlib.pyplot as plt
    # plt.plot(x, probs_np)
    # plt.show()
    # assert len(conv.get_dependents(only_floating=False)) == 2  # TODO: activate again with fixed params


def test_conv_gauss_gauss():
    """Test convolution of two Gaussians - should yield a Gaussian with sigma = sqrt(s1^2 + s2^2)."""
    obs = zfit.Space("x", limits=(-10, 10))
    mu1, sigma1 = 0.0, 1.0
    mu2, sigma2 = 0.0, 2.0

    gauss1 = zfit.pdf.Gauss(mu1, sigma1, obs=obs)
    gauss2 = zfit.pdf.Gauss(mu2, sigma2, obs=obs)

    # Convolution
    conv = zphys.unstable.pdf.NumConvPDFUnbinnedV1(
        func=gauss1,
        kernel=gauss2,
        limits=obs,
        obs=obs,
        ndraws=30000,
        experimental_pdf_normalized=True
    )

    # Analytical result: Gaussian with mu = mu1 + mu2, sigma = sqrt(sigma1^2 + sigma2^2)
    expected_mu = mu1 + mu2
    expected_sigma = np.sqrt(sigma1**2 + sigma2**2)
    expected_gauss = zfit.pdf.Gauss(expected_mu, expected_sigma, obs=obs)

    # Test at multiple points
    test_points = znp.linspace(-5.0, 5.0, 50)
    conv_values = conv.pdf(test_points)
    expected_values = expected_gauss.pdf(test_points)

    # Should be close to analytical solution (within MC error)
    np.testing.assert_allclose(
        zfit.run(conv_values).flatten(),
        zfit.run(expected_values),
        rtol=0.05,  # 5% tolerance for MC integration
        atol=0.01
    )


def test_conv_normalization():
    """Test that convolution PDF is properly normalized."""
    obs = zfit.Space("x", limits=(-5, 5))

    # Use simple functions for testing
    uniform = zfit.pdf.Uniform(-2, 2, obs=obs)
    gauss = zfit.pdf.Gauss(0.0, 0.5, obs=obs)

    conv = zphys.unstable.pdf.NumConvPDFUnbinnedV1(
        func=uniform,
        kernel=gauss,
        limits=obs,
        obs=obs,
        ndraws=20000,
        experimental_pdf_normalized=True
    )

    # Test normalization
    integral = conv.integrate(limits=obs)
    assert pytest.approx(1.0, rel=1e-2) == zfit.run(integral)


def test_conv_different_ndraws():
    """Test convergence behavior with different numbers of draws."""
    obs = zfit.Space("x", limits=(-3, 3))
    gauss1 = zfit.pdf.Gauss(0.0, 1.0, obs=obs)
    gauss2 = zfit.pdf.Gauss(0.0, 0.5, obs=obs)

    ndraws_list = [5000, 10000, 20000]
    results = []

    for ndraws in ndraws_list:
        conv = zphys.unstable.pdf.NumConvPDFUnbinnedV1(
            func=gauss1,
            kernel=gauss2,
            limits=obs,
            obs=obs,
            ndraws=ndraws,
            experimental_pdf_normalized=True
        )

        # Evaluate at a specific point
        test_point = znp.array([0.0])
        result = conv.pdf(test_point)
        results.append(zfit.run(result)[0])

    # Higher ndraws should converge better (less variance)
    # This is a basic convergence test
    assert len(results) == 3
    # Results should be reasonably close to each other
    assert np.std(results) < 0.1, f"Results vary too much: {results}"


def test_conv_asymmetric():
    """Test convolution with asymmetric functions."""
    obs = zfit.Space("x", limits=(-5, 8))

    # Exponential kernel (asymmetric)
    exponential = zfit.pdf.Exponential(-0.5, obs=obs)

    # Gaussian function
    gauss = zfit.pdf.Gauss(0.0, 1.0, obs=obs)

    conv = zphys.unstable.pdf.NumConvPDFUnbinnedV1(
        func=gauss,
        kernel=exponential,
        limits=obs,
        obs=obs,
        ndraws=25000,
        experimental_pdf_normalized=True
    )

    # Test that it integrates to 1
    integral = conv.integrate(limits=obs)
    assert pytest.approx(1.0, rel=1e-2) == zfit.run(integral)

    # Test that it gives reasonable values
    test_points = znp.linspace(-3.0, 6.0, 20)
    values = conv.pdf(test_points)
    values_np = zfit.run(values)

    # Should be positive and finite
    assert np.all(values_np >= 0)
    assert np.all(np.isfinite(values_np))
    assert np.max(values_np) > 0.01  # Should have some non-negligible values


def test_conv_performance_baseline():
    """Test to establish performance baseline and ensure reasonable speed."""
    obs = zfit.Space("x", limits=(-5, 5))
    gauss1 = zfit.pdf.Gauss(0.0, 1.0, obs=obs)
    gauss2 = zfit.pdf.Gauss(0.0, 1.0, obs=obs)

    conv = zphys.unstable.pdf.NumConvPDFUnbinnedV1(
        func=gauss1,
        kernel=gauss2,
        limits=obs,
        obs=obs,
        ndraws=10000,
        experimental_pdf_normalized=True
    )

    # Test evaluation at multiple points
    test_points = znp.linspace(-4.0, 4.0, 100)

    import time
    start_time = time.time()
    values = conv.pdf(test_points)
    zfit.run(values)  # Force evaluation
    elapsed_time = time.time() - start_time

    # Should complete in reasonable time (< 10 seconds for 100 points)
    assert elapsed_time < 10.0, f"Convolution took too long: {elapsed_time:.2f}s"

    # Values should be reasonable
    values_np = zfit.run(values)
    assert np.all(values_np >= 0)
    assert np.all(np.isfinite(values_np))
