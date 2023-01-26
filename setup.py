#!/usr/bin/env python

"""The setup script."""
import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as requirements_file:
    requirements = requirements_file.read().splitlines()

with open(os.path.join(here, "requirements_dev.txt"), encoding="utf-8") as requirements_dev_file:
    dev_requirements = requirements_dev_file.read().splitlines()

with open(os.path.join(here, "README.rst"), encoding="utf-8") as readme_file:
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
