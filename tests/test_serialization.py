"""Test serialization functionality for zfit-physics PDFs."""

from __future__ import annotations

import numpy as np
import pytest
import zfit

import zfit_physics as zphys


@pytest.fixture
def obs():
    """Standard observable for testing."""
    return zfit.Space("x", limits=(-3, 3))

def test_argus_serialization(obs):
    """Test Argus PDF serialization."""
    m0 = zfit.Parameter("m0", 5.0, 1.0, 10.0)
    c = zfit.Parameter("c", -0.5, -2.0, 0.0)
    p = zfit.Parameter("p", 0.5, 0.1, 2.0)

    pdf = zphys.pdf.Argus(m0=m0, c=c, p=p, obs=obs)
    _test_pdf_serialization(pdf)

def test_cmsshape_serialization(obs):
    """Test CMSShape PDF serialization."""
    m = zfit.Parameter("m", 1.0, 0.5, 2.0)
    beta = zfit.Parameter("beta", 1.0, 0.5, 2.0)
    gamma = zfit.Parameter("gamma", 0.1, 0.01, 0.5)

    pdf = zphys.pdf.CMSShape(m=m, beta=beta, gamma=gamma, obs=obs)
    _test_pdf_serialization(pdf)

def test_cruijff_serialization(obs):
    """Test Cruijff PDF serialization."""
    mu = zfit.Parameter("mu", 0.0, -2.0, 2.0)
    sigmal = zfit.Parameter("sigmal", 1.0, 0.1, 3.0)
    sigmar = zfit.Parameter("sigmar", 1.2, 0.1, 3.0)
    alphal = zfit.Parameter("alphal", 0.5, 0.0, 2.0)
    alphar = zfit.Parameter("alphar", 0.3, 0.0, 2.0)

    pdf = zphys.pdf.Cruijff(
        mu=mu, sigmal=sigmal, sigmar=sigmar,
        alphal=alphal, alphar=alphar, obs=obs
    )
    _test_pdf_serialization(pdf)

def test_erfexp_serialization(obs):
    """Test ErfExp PDF serialization."""
    mu = zfit.Parameter("mu", 0.0, -2.0, 2.0)
    beta = zfit.Parameter("beta", 1.0, 0.1, 3.0)
    gamma = zfit.Parameter("gamma", 0.5, 0.1, 2.0)
    n = zfit.Parameter("n", 2.0, 1.0, 5.0)

    pdf = zphys.pdf.ErfExp(mu=mu, beta=beta, gamma=gamma, n=n, obs=obs)
    _test_pdf_serialization(pdf)

def test_novosibirsk_serialization(obs):
    """Test Novosibirsk PDF serialization."""
    mu = zfit.Parameter("mu", 0.0, -2.0, 2.0)
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 3.0)
    lambd = zfit.Parameter("lambd", 0.5, -2.0, 2.0)

    pdf = zphys.pdf.Novosibirsk(mu=mu, sigma=sigma, lambd=lambd, obs=obs)
    _test_pdf_serialization(pdf)

def test_relbw_serialization(obs):
    """Test RelativisticBreitWigner PDF serialization."""
    m = zfit.Parameter("m", 1.0, 0.5, 2.0)
    gamma = zfit.Parameter("gamma", 0.1, 0.01, 0.5)

    pdf = zphys.pdf.RelativisticBreitWigner(m=m, gamma=gamma, obs=obs)
    _test_pdf_serialization(pdf)

def test_tsallis_serialization():
    """Test Tsallis PDF serialization."""
    # Tsallis needs positive x values and a specific observable range
    obs = zfit.Space("x", limits=(0.1, 3))
    m = zfit.Parameter("m", 1.0, 0.5, 2.0)
    t = zfit.Parameter("t", 0.2, 0.1, 1.0)
    n = zfit.Parameter("n", 3.0, 2.1, 10.0)

    pdf = zphys.pdf.Tsallis(m=m, t=t, n=n, obs=obs)
    _test_pdf_serialization(pdf)


def _test_pdf_serialization(pdf):
    """Generic serialization test for any PDF."""
    # Test values for evaluation - use smaller range for stability
    x_vals = np.linspace(-1, 1, 10)

    # Get original PDF values
    original_vals = pdf.pdf(x_vals)

    # Serialize and deserialize
    pdf_dict = pdf.to_dict()

    # Verify dictionary structure
    assert isinstance(pdf_dict, dict)
    assert "type" in pdf_dict or "name" in pdf_dict

    # Recreate PDF from dictionary
    pdf_class = type(pdf)
    pdf_recreated = pdf_class.from_dict(pdf_dict)

    # Test that recreated PDF gives same values
    recreated_vals = pdf_recreated.pdf(x_vals)

    np.testing.assert_allclose(
        original_vals, recreated_vals,
        rtol=1e-10,
        err_msg=f"Serialization failed for {pdf.__class__.__name__}"
    )

    # Test parameter access
    original_params = pdf.get_params()
    recreated_params = pdf_recreated.get_params()

    # Verify same parameters
    assert len(original_params) == len(recreated_params)

    # Convert to dict for easier comparison
    orig_param_dict = {p.name: p for p in original_params}
    recreated_param_dict = {p.name: p for p in recreated_params}

    # Verify same parameter names
    assert set(orig_param_dict.keys()) == set(recreated_param_dict.keys())

    # Verify parameter values match
    for name, param in orig_param_dict.items():
        np.testing.assert_allclose(
            param.numpy(),
            recreated_param_dict[name].numpy(),
            rtol=1e-10,
            err_msg=f"Parameter {name} value mismatch in {pdf.__class__.__name__}"
        )


def test_serialization_registration():
    """Test that all PDFRepr classes are properly registered."""
    # This test verifies that the _serialization module is properly imported
    # and all PDFRepr classes are registered with the serializer

    # Test that we can create and serialize a PDF (this will fail if not registered)
    obs = zfit.Space("x", limits=(-3, 3))
    m0 = zfit.Parameter("m0", 5.0, 1.0, 10.0)
    c = zfit.Parameter("c", -0.5, -2.0, 0.0)
    p = zfit.Parameter("p", 0.5, 0.1, 2.0)
    pdf = zphys.pdf.Argus(m0=m0, c=c, p=p, obs=obs)

    # This should not raise an exception if registration worked
    pdf_dict = pdf.to_dict()
    pdf_class = type(pdf)
    recreated_pdf = pdf_class.from_dict(pdf_dict)

    assert recreated_pdf is not None
    assert isinstance(recreated_pdf, zphys.pdf.Argus)
