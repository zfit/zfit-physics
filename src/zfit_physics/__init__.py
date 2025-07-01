"""Top-level package for zfit."""

from ._version import version as __version__

__license__ = "BSD 3-Clause"
__copyright__ = "Copyright 2019, zfit"
__status__ = "Beta"

__author__ = "zfit"
__maintainer__ = "zfit"
__email__ = "zfit@physik.uzh.ch"
__credits__ = [
    "Jonas Eschle <jonas.eschle@cern.ch>",
    "Albert Puig <albert.puig@cern.ch",
    "Rafael Silva Coutinho <rafael.silva.coutinho@cern.ch>",
    # TODO(release): add more, Anton etc
]

from . import pdf, unstable

__all__ = ["pdf", "unstable"]
