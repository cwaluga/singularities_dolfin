#!/usr/bin/env python

from distutils.core import setup
setup(name = "Energy corrected FEM",
  version = "1.0",
  description = "Collection of scripts for energy corrected finite element methods",
  author = "Christian Waluga",
  packages = ["energy_correction"],
  package_dir={"energy_correction":"src"}
)
