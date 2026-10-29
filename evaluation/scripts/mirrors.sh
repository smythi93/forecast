#!/bin/bash
# Bare mirrors of the 12 repositories of SWE-bench Lite, for Agentless.
# Agentless clones an instance's repository from GitHub several times per instance; with
# many workers GitHub refuses the clones. run_agent.sh redirects git to these mirrors when
# MIRRORS names the folder.
# Usage: bash evaluation/scripts/mirrors.sh <mirror dir>
set -e
M=${1:?usage: mirrors.sh <mirror dir>}
for r in astropy/astropy django/django matplotlib/matplotlib mwaskom/seaborn pallets/flask \
         psf/requests pydata/xarray pylint-dev/pylint pytest-dev/pytest \
         scikit-learn/scikit-learn sphinx-doc/sphinx sympy/sympy; do
  [ -d "$M/$r.git" ] || git clone --quiet --mirror "https://github.com/$r.git" "$M/$r.git"
  echo "$M/$r.git"
done
