import sys

import pytest

init_modules = sys.modules.keys()


@pytest.fixture(autouse=True)
def setup_teardown():
    import zfit

    old_chunksize = zfit.run.chunking.max_n_points
    old_active = zfit.run.chunking.active
    old_graph_mode = zfit.run.get_graph_mode()
    old_autograd_mode = zfit.run.get_autograd_mode()

    for m in sys.modules.keys():
        if m not in init_modules:
            del sys.modules[m]

    yield
    from zfit.core.parameter import ZfitParameterMixin

    ZfitParameterMixin._existing_params.clear()

    from zfit.util.cache import clear_graph_cache

    clear_graph_cache()
    import zfit

    zfit.run.chunking.active = old_active
    zfit.run.chunking.max_n_points = old_chunksize
    zfit.run.set_graph_mode(old_graph_mode)
    zfit.run.set_autograd_mode(old_autograd_mode)
    for m in sys.modules.keys():
        if m not in init_modules:
            del sys.modules[m]
    import gc

    gc.collect()
