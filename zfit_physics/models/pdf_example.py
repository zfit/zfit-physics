"""Example file of a custom pdf implementation. Create a module for each pdf that you add."""
import zfit

class Example(zfit.pdf.ZPDF):
    _PARAMS = ["param1"]

    def _unnormalized_pdf(self, x):
        """Documentation here."""
        return [42.]


