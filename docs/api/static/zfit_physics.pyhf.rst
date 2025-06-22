pyhf
=======================

The pure Python package `pyhf <https://pyhf.readthedocs.io/en/stable/>`_ is a statistical fitting package for high energy physics for purely binned, templated fits. It is a Python implementation of the HistFactory schema.
The connection to zfit is done via the loss function, which can be created from a pyhf model.

Import the module with:

.. code-block:: python

    import zfit_physics.pyhf as zpyhf

The loss function can be created from a ``data`` and a ``pdf`` model with

.. code-block:: python

    nll = zpyhf.loss.nll_from_pyhf(data, pdf)



which optionally takes already created :py:class:`~zfit.core.interfaces.ZfitParameter` as arguments.




Loss
++++++++++++

.. automodule:: zfit_physics.pyhf.loss
    :members:
    :undoc-members:
    :show-inheritance:
