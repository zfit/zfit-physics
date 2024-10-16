RooFit
=======================

ROOT provides with the `RooFit library <https://root.cern/manual/roofit/>`_ a toolkit for modeling the expected distribution of events in a physics analysis.
It can be connected with zfit, currently by providing a loss function that can be minimized by a zfit minimizer.

This requires the `ROOT framework <https://root.cern/>`_ to be installed and available in the python environment.
For example via conda:

.. code-block:: console

    $ mamba install -c conda-forge root

Import the module with:

.. code-block:: python

    import zfit_physics.roofit as ztfroofit

this will enable the RooFit functionality in zfit.

We can create a RooFit NLL as ``RooFit_nll`` and use it as a loss function in zfit. For example, with a Gaussian model ``RooFit_gauss`` and a dataset ``RooFit_data``, both created with RooFit:

.. code-block:: python

    RooFit_nll = RooFit_gauss.createNLL(RooFit_data)
    minimizer.minimize(loss=RooFit_nll)

More explicitly, the loss function can be created with

.. code-block:: python

    nll = zroofit.loss.nll_from_roofit(fcn)


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
