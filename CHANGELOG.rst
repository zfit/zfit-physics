*********
Changelog
*********

Develop
========================

Major Features and Improvements
-------------------------------

Breaking changes
------------------

Deprecations
-------------

Bug fixes and small changes
---------------------------

Experimental
------------

Requirement changes
-------------------

Thanks
------

0.8.0 (7 Nov 2024)
========================

Major Features and Improvements
-------------------------------
- add a RooFit compatibility layer and automatically convert losses, also inside minimizers (through ``SimpleLoss.from_any``)
- `TF-PWA <https://tf-pwa.readthedocs.io/en/latest/>`_ support for loss functions. Minimizer can directly minimize the loss function of a model.
- `pyhf <https://pyhf.readthedocs.io/en/stable/>`_ support for loss functions. Minimizer can directly minimize the loss function of a model.
- `ComPWA <https://compwa.github.io/>`_ support for loss functions and pdf. Minimizer can directly minimize the loss function of a model.

0.7.0 (13 Apr 2024)
===================

Major Features and Improvements
-------------------------------
- added CMSShape PDF
- added Cruijff PDF
- added ErfExp PDF
- added Novosibirsk PDF
- added Tsallis PDF
- upgrade to zfit>=0.20, support Python 3.9-3.12

0.6.1 (8 Oct 2023)
===================

Minor bug fix in numerical convolution PDF

0.6.0 (20 Jul 2023)
===================

Upgrade to zfit >= 0.12, support Python 3.8-3.11


0.4.0 (27 Jan 2023)
===================

Compability with zfit >= 0.11, < 0.13

0.3.0 (26 Jan 2023)
===================

Compability with zfit >= 0.10, < 0.11

0.2.0
=======

Added relativistic Breit-Wigner PDF

Many thanks to Simon Thor <simon11.thor@hotmail.se> for contributing the Relativistic BW

0.1.0
=======

Upgrade to zfit >= 0.6


0.0.3 (14.05.2020)
==================


Major Features and Improvements
-------------------------------
- added ARGUS pdf


Bug fixes and small changes
---------------------------
- fix KDE with numerical integration


Requirement changes
-------------------
- zfit >= 5.2

Thanks
------
- Colm Murphy <colm.murphy@ipmu.jp> for help in contributing the ARGUS PDF
