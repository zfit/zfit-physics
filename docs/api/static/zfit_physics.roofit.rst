RooFit
=======================

ROOT provides with the `RooFit library <https://root.cern/manual/roofit/>`_ a toolkit for modeling the expected distribution of events in a physics analysis.
It can be connected with zfit, currently by providing a loss function that can be minimized by a zfit minimizer.

This requires the `ROOT framework <https://root.cern/>`_ to be installed and available in the python environment.
For example via conda:

.. code-block:: console

    $ mamba install -c conda-forge root

.. jupyter-execute::
    :hide-code:
    :hide-output:

    import numpy as np
    import zfit
    from ROOT import RooArgSet, RooDataSet, RooGaussian, RooRealVar

    data = np.random.normal(loc=2.0, scale=3.0, size=1000)

    mur = RooRealVar("mu", "mu", 1.2, -4, 6)
    sigmar = RooRealVar("sigma", "sigma", 1.3, 0.5, 10)
    obsr = RooRealVar("x", "x", -2, 3)
    RooFit_gauss = RooGaussian("gauss", "gauss", obsr, mur, sigmar)

    RooFit_data = RooDataSet("data", "data", {obsr})
    for d in data:
        obsr.setVal(d)
        RooFit_data.add(RooArgSet(obsr))

    minimizer = zfit.minimize.Minuit()

Import the module with:



.. jupyter-execute::

    import zfit_physics.roofit as zroofit

this will enable the RooFit functionality in zfit and allow to automatically minimize the function using a zfit minimimzer as

.. jupyter-execute::

    RooFit_nll = RooFit_gauss.createNLL(RooFit_data)

We can create a RooFit NLL as ``RooFit_nll`` and use it as a loss function in zfit. For example, with a Gaussian model ``RooFit_gauss`` and a dataset ``RooFit_data``, both created with RooFit:

.. jupyter-execute::

    result = minimizer.minimize(loss=RooFit_nll)

More explicitly, the loss function can be created with

.. jupyter-execute::

    nll = zroofit.loss.nll_from_roofit(RooFit_nll)


Variables
++++++++++++


.. automodule:: zfit_physics.roofit.variables
    :members:
    :undoc-members:
    :show-inheritance:


Loss
++++++++++++

.. automodule:: zfit_physics.roofit.loss
    :members:
    :undoc-members:
    :show-inheritance:
