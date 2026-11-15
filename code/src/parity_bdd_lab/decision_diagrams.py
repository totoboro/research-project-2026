"""Convenience constructors for the ``dd`` package's BDD and MDD managers."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping


def get_bdd_class(backend: str | None = None):
    """Return a ``dd`` BDD manager class for ``auto``, ``python``, or ``cudd``.

    ``auto`` prefers the optional Cython/CUDD backend and falls back to the
    reference-counted pure-Python backend. ``PARITY_BDD_BACKEND`` can select a
    backend process-wide; an explicit argument takes precedence.
    """
    selected = (backend or os.environ.get("PARITY_BDD_BACKEND", "auto")).casefold()
    if selected not in {"auto", "python", "autoref", "cudd"}:
        raise ValueError("BDD backend must be 'auto', 'python', or 'cudd'")
    if selected in {"auto", "cudd"}:
        try:
            from dd.cudd import BDD
            return BDD
        except ImportError:
            if selected == "cudd":
                raise RuntimeError(
                    "PARITY_BDD_BACKEND=cudd was requested, but dd.cudd is not installed; "
                    "install the optional CUDD extension or select the python backend"
                ) from None
    from dd.autoref import BDD
    return BDD


def collect_bdd_garbage(manager) -> None:
    """Collect unreachable nodes when the selected backend exposes that hook.

    ``dd.autoref`` retains dead nodes until this method is called. CUDD manages
    its node table internally and does not expose ``collect_garbage``.
    """
    collect = getattr(manager, "collect_garbage", None)
    if collect is not None:
        collect()


def new_bdd(variables: Iterable[str] = (), *, backend: str | None = None):
    """Return a ``dd`` BDD manager with the requested variables declared.

    The default ``auto`` backend uses CUDD when its optional extension is
    installed and otherwise uses ``dd.autoref``.
    """
    BDD = get_bdd_class(backend)

    manager = BDD()
    variables = tuple(variables)
    if variables:
        manager.declare(*variables)
    return manager


def new_mdd(domains: Mapping[str, int]):
    """Return a ``dd.mdd.MDD`` manager; each mapping value is domain size."""
    from dd.mdd import MDD

    if any(size < 1 for size in domains.values()):
        raise ValueError("MDD domain sizes must be positive")
    variables = {name: {"level": level, "len": size} for level, (name, size) in enumerate(domains.items())}
    return MDD(variables)
