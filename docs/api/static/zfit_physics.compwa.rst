ComPWA
=======================

`ComPWA <https://compwa.github.io/>`_ is a framework for the coherent amplitude analysis of multi-body decays. It uses a symbolic approach to describe the decay amplitudes and can be used to fit data to extract the decay parameters. ComPWA can be used in combination with zfit to perform the fit by either creating a zfit pdf from the ComPWA model or by using the ComPWA estimator as a loss function for the zfit minimizer.

Import the module with:

.. code-block:: python

    import zfit_physics.compwa as zcompwa

This will enable that a :py:class:`~ tensorwaves.estimator.Estimator`, for example ``estimator`` in the following, can be used as a loss function in zfit minimizers as

.. code-block:: python

    minimizer.minimize(loss=estimator)

More explicitly, the loss function can be created with

.. code-block:: python

    nll = zcompwa.loss.nll_from_estimator(estimator)

which optionally takes already created :py:class:`~zfit.core.interfaces.ZfitParameter` as arguments.

A whole ComPWA model can be converted to a zfit pdf with

.. code-block:: python

    pdf = zcompwa.pdf.ComPWAPDF(compwa_model)

``pdf`` is a full fledged zfit pdf that can be used in the same way as any other zfit pdf! In a sum, product, convolution and of course to fit data.

Variables
++++++++++++


.. automodule:: zfit_physics.compwa.variables
    :members:
    :undoc-members:
    :show-inheritance:

PDF
++++++++++++

.. automodule:: zfit_physics.compwa.pdf
    :members:
    :undoc-members:
    :show-inheritance:

Loss
++++++++++++

.. automodule:: zfit_physics.compwa.loss
    :members:
    :undoc-members:
    :show-inheritance:
