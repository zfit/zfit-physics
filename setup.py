"""The setup script."""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).parent.resolve()
with Path(here / "requirements.txt").open(encoding="utf-8") as requirements_file:
    requirements = requirements_file.read().splitlines()

with Path(here / "requirements_dev.txt").open(encoding="utf-8") as requirements_dev_file:
    dev_requirements = requirements_dev_file.read().splitlines()

with Path(here / "README.rst").open(encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    packages=find_packages(include=["zfit_physics", "zfit_physics.models", "zfit_physics.unstable"]),
    test_suite="tests",
    extras_require={"dev": dev_requirements},
    use_scm_version=True,
    zip_safe=False,
)
