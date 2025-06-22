TF-PWA
=======================

TF-PWA is a generic software package intended for Partial Wave Analysis (PWA). It can be connected with zfit,
currently by providing a loss function that can be minimized by a zfit minimizer.

Import the module with:

.. code-block:: python

    import zfit_physics.tfpwa as ztfpwa

This will enable that :py:class:`~tf_pwa.model.FCN` can be used as a loss function in zfit minimizers as

.. code-block:: python

    minimizer.minimize(loss=fcn)

More explicitly, the loss function can be created with

.. code-block:: python

    nll = ztfpwa.loss.nll_from_fcn(fcn)

which optionally takes already created :py:class:`~zfit.core.interfaces.ZfitParameter` as arguments.


Variables
++++++++++++


.. automodule:: zfit_physics.tfpwa.variables
    :members:
    :undoc-members:
    :show-inheritance:


Loss
++++++++++++

.. automodule:: zfit_physics.tfpwa.loss
    :members:
    :undoc-members:
    :show-inheritance:
