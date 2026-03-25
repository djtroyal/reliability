"""
System reliability analysis: Reliability Block Diagrams (RBDs).

Supports series, parallel, k-out-of-n, and network (path-set) configurations.
"""

import numpy as np
from itertools import combinations


class SeriesSystem:
    """System where all components must function.

    Parameters
    ----------
    component_reliabilities : array-like
        Reliability (probability of functioning) of each component.
    """

    def __init__(self, component_reliabilities):
        self.component_reliabilities = np.asarray(component_reliabilities, dtype=float)
        self.reliability = float(np.prod(self.component_reliabilities))
        self.unreliability = 1.0 - self.reliability

    def __repr__(self):
        return (f"SeriesSystem(n={len(self.component_reliabilities)}, "
                f"R={self.reliability:.6f})")


class ParallelSystem:
    """System where at least one component must function.

    Parameters
    ----------
    component_reliabilities : array-like
        Reliability (probability of functioning) of each component.
    """

    def __init__(self, component_reliabilities):
        self.component_reliabilities = np.asarray(component_reliabilities, dtype=float)
        unreliabilities = 1.0 - self.component_reliabilities
        self.unreliability = float(np.prod(unreliabilities))
        self.reliability = 1.0 - self.unreliability

    def __repr__(self):
        return (f"ParallelSystem(n={len(self.component_reliabilities)}, "
                f"R={self.reliability:.6f})")


class KofNSystem:
    """K-out-of-N system (majority vote / redundancy).

    At least k of n identical components must function.

    Parameters
    ----------
    k : int
        Minimum number of components required.
    n : int
        Total number of components.
    component_reliability : float
        Reliability of each (identical) component.
    """

    def __init__(self, k, n, component_reliability):
        if k > n:
            raise ValueError("k cannot exceed n")
        self.k = k
        self.n = n
        self.component_reliability = float(component_reliability)
        p = self.component_reliability
        q = 1.0 - p

        # Sum of binomial terms for exactly j successes, j = k..n
        reliability = 0.0
        from math import comb
        for j in range(k, n + 1):
            reliability += comb(n, j) * (p ** j) * (q ** (n - j))
        self.reliability = reliability
        self.unreliability = 1.0 - reliability

    def __repr__(self):
        return (f"KofNSystem(k={self.k}, n={self.n}, "
                f"R_component={self.component_reliability:.4f}, "
                f"R={self.reliability:.6f})")


class NetworkSystem:
    """General network RBD using minimal path sets.

    Parameters
    ----------
    path_sets : list of list/set
        Each inner list is a minimal path set — a set of component indices
        whose simultaneous functioning guarantees system success.
    component_reliabilities : array-like
        Reliability of component i at index i.

    Notes
    -----
    Uses inclusion-exclusion over the path sets.
    """

    def __init__(self, path_sets, component_reliabilities):
        self.path_sets = [frozenset(p) for p in path_sets]
        self.component_reliabilities = np.asarray(component_reliabilities, dtype=float)
        self.reliability = self._inclusion_exclusion()
        self.unreliability = 1.0 - self.reliability

    def _path_prob(self, path):
        """Probability that all components in path function."""
        return float(np.prod(self.component_reliabilities[list(path)]))

    def _union_prob(self, paths):
        """P(A1 ∪ A2 ∪ ... ∪ Ak) via inclusion-exclusion."""
        n = len(paths)
        total = 0.0
        for size in range(1, n + 1):
            sign = (-1) ** (size + 1)
            for combo in combinations(range(n), size):
                union_path = frozenset().union(*[paths[i] for i in combo])
                total += sign * self._path_prob(union_path)
        return total

    def _inclusion_exclusion(self):
        return self._union_prob(self.path_sets)

    def __repr__(self):
        return (f"NetworkSystem(paths={len(self.path_sets)}, "
                f"components={len(self.component_reliabilities)}, "
                f"R={self.reliability:.6f})")


def system_reliability_from_blocks(blocks):
    """Compute system reliability from a nested block description.

    Parameters
    ----------
    blocks : dict
        Nested description of the system. Format::

            {'type': 'series'|'parallel'|'kofn'|'component',
             'components': [...],   # for series/parallel
             'k': int,              # for kofn
             'n': int,              # for kofn
             'reliability': float,  # for component
             'sub': dict}           # for kofn (single subsystem)

    Returns
    -------
    float
        System reliability.
    """
    btype = blocks.get('type', 'component')

    if btype == 'component':
        return float(blocks['reliability'])

    elif btype == 'series':
        sub_reliabilities = [system_reliability_from_blocks(c)
                             for c in blocks['components']]
        return float(np.prod(sub_reliabilities))

    elif btype == 'parallel':
        sub_unreliabilities = [1.0 - system_reliability_from_blocks(c)
                               for c in blocks['components']]
        return 1.0 - float(np.prod(sub_unreliabilities))

    elif btype == 'kofn':
        k = blocks['k']
        n = blocks['n']
        sub_r = system_reliability_from_blocks(blocks['sub'])
        sys = KofNSystem(k, n, sub_r)
        return sys.reliability

    else:
        raise ValueError(f"Unknown block type: {btype!r}")
