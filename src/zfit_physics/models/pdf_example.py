"""Example file of a custom PDF implementation following modern zfit best practices.

This module demonstrates how to create a custom PDF in zfit using the most
up-to-date patterns and conventions.
"""

from __future__ import annotations

from typing import Literal

import zfit
from pydantic.v1 import Field
from zfit import z
from zfit.core.serialmixin import SerializableMixin
from zfit.serialization import Serializer
from zfit.serialization.pdfrepr import BasePDFRepr
from zfit.serialization.spacerepr import SpaceRepr
from zfit.util import ztyping


@z.function(wraps="tensor")
def example_pdf_func(x, param1):
    """Example PDF function using modern zfit patterns.

    This is a simple exponential decay function for demonstration purposes.
    In practice, replace this with your actual mathematical function.

    Args:
        x: Input variable(s)
        param1: Scale parameter

    Returns:
        Tensor: PDF values
    """
    # Use a simple exponential function that's always positive
    import zfit.z.numpy as znp

    return znp.exp(-znp.abs(x) / param1)


class Example(zfit.pdf.BasePDF, SerializableMixin):
    """Example PDF class following modern zfit conventions.

    This demonstrates the current best practices for creating custom PDFs:
    - Inherits from BasePDF and SerializableMixin
    - Uses proper parameter typing
    - Uses _unnormalized_pdf method
    - Includes comprehensive docstrings
    - Supports serialization
    """

    def __init__(
        self,
        *,
        param1: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        extended: ztyping.ParamTypeInput | None = None,
        norm: ztyping.NormTypeInput = None,
        name: str = "ExamplePDF",
        label: str | None = None,
    ):
        """Initialize the Example PDF.

        Args:
            param1: Example parameter for the PDF
            obs: Observable(s) of the model
            extended: The overall yield of the PDF (for extended PDFs)
            norm: Normalization range
            name: Name of the PDF
            label: Human-readable label
        """
        params = {"param1": param1}
        super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm, label=label)

    @zfit.supports(norm=False)
    def _pdf(self, x, norm, params):
        """Calculate the unnormalized PDF values.

        This is the core method that implements the mathematical function.

        Args:
            x: Input values
            params: Dictionary of parameters

        Returns:
            Tensor: Unnormalized PDF values
        """
        del norm
        x0 = x[0]  # Get first (and only) observable
        param1 = params["param1"]
        return example_pdf_func(x0, param1)


class ExamplePDFRepr(BasePDFRepr):
    """Serialization representation for Example PDF."""

    _implementation = Example
    hs3_type: Literal["Example"] = Field("Example", alias="type")
    x: SpaceRepr
    param1: Serializer.types.ParamInputTypeDiscriminated
