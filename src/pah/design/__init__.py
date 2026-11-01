"""The design plane: a typed design intent, its geometry, and the ports around them.

`spec` holds the contract an agent edits; `api` holds the ports and registries the
rest of the harness builds against. Both are frozen by
`docs/contracts/v2-interfaces.md`; a change here is a change to every other v2
module, so it goes through the Lead before it lands.
"""

from pah.design.api import (
    DESIGN_CODES,
    DESIGN_OPERATOR_IDS,
    DESIGN_VERIFIER_IDS,
    ConnectorBuilder,
    ConnectorGeometry,
    DesignVerifier,
    DesignVerifierRegistry,
    DesignView,
    MaterializedDesign,
    Materializer,
    assembly_from_spec,
    codes_owned_by,
    geometry_hash,
    hard_codes,
    spec_of,
    view_of,
)
from pah.design.spec import (
    DesignSpec,
    DovetailParams,
    FitConstraints,
    InterfaceSpec,
    PartSpec,
    PegHoleParams,
    SnapFitParams,
    builtin_params_model,
    radial_clearance_mm,
)

__all__ = [
    "ConnectorBuilder",
    "ConnectorGeometry",
    "DESIGN_CODES",
    "DESIGN_OPERATOR_IDS",
    "DESIGN_VERIFIER_IDS",
    "DesignSpec",
    "DesignVerifier",
    "DesignVerifierRegistry",
    "DesignView",
    "DovetailParams",
    "FitConstraints",
    "InterfaceSpec",
    "MaterializedDesign",
    "Materializer",
    "PartSpec",
    "PegHoleParams",
    "SnapFitParams",
    "assembly_from_spec",
    "builtin_params_model",
    "codes_owned_by",
    "geometry_hash",
    "hard_codes",
    "radial_clearance_mm",
    "spec_of",
    "view_of",
]
