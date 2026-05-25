"""Provide a quiet `jit` decorator for filter math helpers.

This file keeps the filter package importable whether `numba` is available or
not. The math utility modules import `jit` from here, so a lightweight fallback
decorator lets the EKF run in pure NumPy without changing the call sites.
"""

try:
    from numba import jit as _numba_jit
except Exception:  # pragma: no cover - import failure is the case we want.
    _numba_jit = None


def jit(*jit_args, **jit_kwargs):
    """Return either `numba.jit` or a no-op decorator.

    The fallback mirrors the standard decorator API closely enough for the
    filter modules, which only need a callable decorator and do not depend on
    numba-specific runtime features.
    """

    if _numba_jit is not None:
        return _numba_jit(*jit_args, **jit_kwargs)

    def decorator(func):
        """Leave the wrapped function untouched when numba is unavailable."""

        return func

    if jit_args and callable(jit_args[0]) and len(jit_args) == 1 and not jit_kwargs:
        return decorator(jit_args[0])
    return decorator
