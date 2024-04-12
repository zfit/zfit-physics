"""Example file of a custom pdf implementation.

Create a module for each pdf that you add.
"""

from __future__ import annotations

import typing

import zfit


class Example(zfit.pdf.ZPDF):
    _PARAMS: typing.ClassVar = ["param1"]

    def _unnormalized_pdf(self, x):
        """Documentation here."""
        x0 = x[0]
        return [42.0 * x0]
