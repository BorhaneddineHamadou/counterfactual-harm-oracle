"""Abstract simulator adapter: the only interface the core method sees.

Adapters must guarantee:
  * run_suite is deterministic given (scenarios, k, global_seed) and the
    adapter's ADS/ego version -- noise is derived from seeds only, so two
    versions run under the same seeds share common random numbers (CRN);
  * branch(trace, ...) reproduces the trace's nominal prefix exactly up to
    the branch step, then continues M perturbed rollouts with the system
    under test re-reacting in the loop (the e (+) epsilon operation).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from .kernel import Kernel
from .trace import Trace


class SimulatorAdapter(ABC):

    @abstractmethod
    def run_suite(self, scenarios, k: int, global_seed: int) -> List[Trace]:
        """Execute each scenario k times; returns k*len(scenarios) traces."""

    @abstractmethod
    def branch(self, trace: Trace, t_star: float, kernel: Kernel,
               M: int) -> Tuple[np.ndarray, np.ndarray]:
        """Replay `trace` M times, branching at t_star under disturbances
        drawn from `kernel`. Returns (contact_mask (M,), dv (M,))."""
