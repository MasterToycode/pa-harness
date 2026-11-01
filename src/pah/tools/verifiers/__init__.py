"""L0 verifiers: what the loop is allowed to believe.

Each module wraps one existing tool from `rl_assembly` (see
`docs/pah-dev-doc-v1.md` §2.4) and registers a `VerifierSpec` carrying the codes
it may emit plus the fixtures that prove it still agrees with itself (§4.8).
"""

from __future__ import annotations

from .base import Case, CostClass, Registry, Verifier, VerifierDisabled, VerifierSpec

__all__ = [
    "Case",
    "CostClass",
    "Registry",
    "Verifier",
    "VerifierDisabled",
    "VerifierSpec",
    "default_registry",
]


def default_registry(**connector_kwargs) -> Registry:
    """A registry with the M0 verifier set mounted.

    `connector_kwargs` is forwarded to `ConnectorFeasible` so a probe can vary
    the scale protocol without editing the module.
    """
    from .connector_feasible import ConnectorFeasible  # noqa: PLC0415  (avoids scipy/trimesh on import)

    registry = Registry()
    registry.register(ConnectorFeasible(**connector_kwargs))
    return registry
