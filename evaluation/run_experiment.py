"""
Alias of :mod:`forecast.experiment`, which ships with the package.

The evaluation scripts and tests import ``evaluation.run_experiment``; this
module makes that name refer to the same module object, so patches against
it reach the code that runs.
"""

import sys

from forecast import experiment

sys.modules[__name__] = experiment
