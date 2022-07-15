import collections
import sys
import torch
import warnings
from .solvers import FixedGridODESolver
from .misc import _compute_error_ratio, _linf_norm
from .misc import Perturb

_GEAR2_COEFFICIENTS = [2, 1, 0]
_DIVISOR = [3]
_GEAR2_DIVISOR = [torch.tensor([g / divisor for g in gear], dtype=torch.float64)
                    for gear, divisor in zip(_GEAR2_COEFFICIENTS, _DIVISOR)]
_MAX_ITERS = 40


def _dot_product(x, y):
    return sum(xi * yi for xi, yi in zip(x, y))

class Gear2(FixedGridODESolver):
    def __init__(self, func, y0, rtol=1e-3, atol=1e-4, max_iters=_MAX_ITERS, **kwargs):
        super(Gear2, self).__init__(func, y0, rtol=rtol, atol=rtol, **kwargs)
        self.rtol = torch.as_tensor(rtol, dtype=y0.dtype, device=y0.device)
        self.atol = torch.as_tensor(atol, dtype=y0.dtype, device=y0.device)
        self.prev_f = collections.deque(maxlen=3)
        self.prev_t = None

        self.gear2 = [x.to(y0.device) for x in _GEAR2_DIVISOR]

    def _update_history(self, t, f):
        if self.prev_t is None or self.prev_t != t:
            self.prev_f.appendleft(f)
            self.prev_t = t

    def _has_converged(self, y0, y1):
        """Checks that each element is within the error tolerance."""
        error_ratio = _compute_error_ratio(torch.abs(y0 - y1), self.rtol, self.atol, y0, y1, _linf_norm)
        return error_ratio < 1

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        self._update_history(t0, f0)
        order = 2
        gear_coeffs = self.gear2[3]
        dy = _dot_product(dt, self.prev_f).type_as(y0)  
        delta = dt * _dot_product(gear_coeffs[1:], self.prev_f).type_as(y0) 
        converged = False
        for _ in range(self.max_iters):
            dy_old = dy
            f = func(t1, y0 + dy, perturb=Perturb.PREV if self.perturb else Perturb.NONE)
            dy = (dt * (gear_coeffs[0]) * f).type_as(y0) + delta
            converged = self._has_converged(dy_old, dy)
            if converged:
                break
        if not converged:
            warnings.warn('Functional iteration did not converge. Solution may be incorrect.')
            self.prev_f.pop()
        self._update_history(t0, f)
    return dy, f0