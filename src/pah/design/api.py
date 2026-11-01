"""The frozen interfaces of the design plane, and the two registries that gate them.

This module is the contract the v2 modules are built against
(`docs/contracts/v2-interfaces.md` §2). It exists before the implementations so
that the geometry, verifier, policy and CLI work can proceed in parallel without
one module guessing another's shape.

Four boundaries, and why each is here rather than in an implementation
---------------------------------------------------------------------
1. **`ConnectorBuilder`** is the only code allowed to turn a connector kind into
   meshes. A plugin that adds a kind implements this and nothing else. The
   materialiser never knows what a peg is.
2. **`DesignView`** is everything a verifier may read: the spec, plus the built
   meshes. A verifier gets no store, no policy and no previous verdict, so it
   cannot reason about intent — only measure the thing in front of it. That is
   the same rule as `docs/pah-dev-doc-v1.md` §3.4, restated for a plane where the
   evidence *can* be computed instead of read off a dataset.
3. **`DesignVerifier` + `DesignVerifierRegistry`** repeat the §4.8 gate: a
   verifier ships a `known_good` / `known_bad` pair and is disabled if it
   disagrees with either. The gate is not optional, because the loop optimises
   whatever the verifiers say.
4. **`Materializer`** is a port, not a class: the design plane must survive a
   future CAD kernel or a DiT adapter replacing the trimesh builder without the
   loop, the policies or the CLI changing.

The certificate codes below are the design plane's closed vocabulary. A verifier
may emit only the codes it declares in `DESIGN_CODES`, and a policy rule may match
only codes that appear here — the same "closed set" discipline `app/trace.py`
applies to event kinds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol, runtime_checkable

import trimesh
from pydantic import BaseModel

from pah.core.model import Assembly, FailureCertificate
from pah.design.spec import DesignSpec, Vec3
from pah.tools.verifiers.base import CostClass, VerifierDisabled, VerifierSpec

__all__ = [
    "CODE_BELOW_MIN_WALL",
    "CODE_DYNAMIC_PENETRATION",
    "CODE_EXCESSIVE_CLEARANCE",
    "CODE_GEOMETRIC_INTERFERENCE",
    "CODE_INSERTION_PATH_BLOCKED",
    "CODE_INSUFFICIENT_CLEARANCE",
    "CODE_JOINT_UNSTABLE",
    "CODE_PHYSICS_ERROR",
    "CODE_UNSUPPORTED_OVERHANG",
    "ConnectorBuilder",
    "ConnectorGeometry",
    "DESIGN_CODES",
    "DESIGN_VERIFIER_IDS",
    "DesignVerifier",
    "DesignVerifierRegistry",
    "DesignView",
    "MaterializedDesign",
    "Materializer",
    "assembly_from_spec",
    "spec_of",
]

# --------------------------------------------------------------------------- #
# Certificate vocabulary (design plane)
# --------------------------------------------------------------------------- #
CODE_INSUFFICIENT_CLEARANCE = "insufficient_clearance"
CODE_EXCESSIVE_CLEARANCE = "excessive_clearance"
CODE_GEOMETRIC_INTERFERENCE = "geometric_interference"
CODE_INSERTION_PATH_BLOCKED = "insertion_path_blocked"
CODE_BELOW_MIN_WALL = "below_min_wall"
CODE_UNSUPPORTED_OVERHANG = "unsupported_overhang"
CODE_JOINT_UNSTABLE = "joint_unstable"
CODE_DYNAMIC_PENETRATION = "dynamic_penetration"
CODE_PHYSICS_ERROR = "physics_error"

#: The closed set a design-plane verifier may emit, keyed by the verifier id that
#: owns each group. `DesignVerifierRegistry.register` refuses a verifier whose
#: `spec.codes` is not exactly one of these groups: two owners for one code is how
#: a report starts double-counting evidence.
DESIGN_CODES: dict[str, tuple[str, ...]] = {
    "fit_clearance": (CODE_INSUFFICIENT_CLEARANCE, CODE_EXCESSIVE_CLEARANCE),
    "interface_interference": (CODE_GEOMETRIC_INTERFERENCE,),
    "insertion_path": (CODE_INSERTION_PATH_BLOCKED,),
    "printability": (CODE_BELOW_MIN_WALL, CODE_UNSUPPORTED_OVERHANG),
    "physics_settle": (CODE_JOINT_UNSTABLE, CODE_DYNAMIC_PENETRATION, CODE_PHYSICS_ERROR),
}

#: Verifier ids in cascade order (cheapest first). The cascade runs in this order
#: and stops early only when a caller asks it to; see `DesignVerifierRegistry.check`.
DESIGN_VERIFIER_IDS: tuple[str, ...] = tuple(DESIGN_CODES)

#: The design plane's action space, declared here so a policy, a test and a report
#: can all cite one list. The operators themselves live in `pah.policy.design_ops`
#: and register into an `OperatorRegistry` (the same guarded registry M1 uses).
DESIGN_OPERATOR_IDS: tuple[str, ...] = (
    "set_clearance",
    "resize_feature",
    "resize_insert",
    "shift_part",
    "rotate_part",
    "switch_connector",
    "regenerate_interface",
)


# --------------------------------------------------------------------------- #
# What a connector kind must provide
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConnectorGeometry:
    """The two solids one connector kind builds, in world coordinates.

    A kind returns both parts even though it only owns the interface: the body's
    hole and the insert's peg are one shape decision, and splitting them across
    two builders is how the hole and the peg start disagreeing.
    """

    meshes: dict[str, trimesh.Trimesh]
    #: The unit insertion direction, in world coordinates. Used by the path
    #: verifier and by the physics setup; defaults to +Z because every builtin
    #: seats along the body's up axis.
    mate_axis: Vec3 = (0.0, 0.0, 1.0)
    #: How far above the seated pose the insert starts. Mirrors the parameter of
    #: the same name, but a kind whose approach is not a straight line reports its
    #: own value here rather than making the verifier guess.
    approach_mm: float = 6.0


@runtime_checkable
class ConnectorBuilder(Protocol):
    """One connector kind: parameters in, two world-frame meshes out.

    Implementations live in `pah/design/connectors/` and register into
    `pah.design.materialize.builder_for`. The three `kind` builtins are
    `peg_hole`, `snap_fit`, `dovetail`; a plugin may add more.
    """

    kind: str

    def params_model(self) -> type[BaseModel]:
        """The pydantic model that validates this kind's `InterfaceSpec.params`."""
        ...

    def clearance_mm(self, params: BaseModel) -> float:
        """The designed radial clearance for `params`, in mm."""
        ...

    def build(self, spec: DesignSpec) -> ConnectorGeometry:
        """Materialise `spec`'s single interface into two world-frame meshes."""
        ...


# --------------------------------------------------------------------------- #
# What a verifier may read
# --------------------------------------------------------------------------- #
@dataclass
class DesignView:
    """The spec plus the built geometry — the whole of a verifier's input.

    Meshes are loaded lazily and cached, because the interference and path
    verifiers both sweep the same two solids and a rebuild per verifier would
    dominate the cost. Three ways in, cheapest first:

    * `meshes` — already built (a test, or an operator that just materialised);
    * `paths`  — absolute STL paths, which is what a design-plane `Assembly`
      records per part, so `view_of(assembly)` needs nothing else;
    * `root`   — the `root/geometry/<part_id>.stl` convention a materialiser
      writes, for callers that have a run directory but no assembly.
    """

    spec: DesignSpec
    root: Path | None = None
    meshes: dict[str, trimesh.Trimesh] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)

    def mesh(self, part_id: str) -> trimesh.Trimesh:
        """The world-frame mesh for `part_id`, loaded on first use."""
        if part_id not in self.meshes:
            path: Path | None = None
            if part_id in self.paths:
                path = Path(self.paths[part_id])
            elif self.root is not None:
                path = self.root / "geometry" / f"{part_id}.stl"
            if path is None:
                raise KeyError(
                    f"design view has no mesh, path or root for part {part_id!r}"
                )
            if not path.exists():
                raise FileNotFoundError(f"part {part_id!r} has no mesh at {path}")
            self.meshes[part_id] = trimesh.load(path, force="mesh")
        return self.meshes[part_id]

    @property
    def body(self) -> trimesh.Trimesh:
        return self.mesh(self.spec.body.id)

    @property
    def insert(self) -> trimesh.Trimesh:
        return self.mesh(self.spec.insert.id)

    def part_ids(self) -> tuple[str, ...]:
        return tuple(part.id for part in self.spec.parts)


@runtime_checkable
class DesignVerifier(Protocol):
    """A tool that may observe whether a materialised design is acceptable.

    `check` returns the failures it observed, or an empty list. It may raise: the
    cascade turns an exception into an `environment`-family certificate, and the
    loop reads that as "unknown", never as a pass.
    """

    spec: VerifierSpec

    def check(self, view: DesignView) -> list[FailureCertificate]:
        """Every failure this verifier can prove about `view` — possibly none."""
        ...

    def fixtures(self) -> tuple[list[DesignView], list[DesignView]]:
        """`(known_good, known_bad)` views. Both must be non-empty to pass sanity."""
        ...


class DesignVerifierRegistry:
    """The design-plane verifier catalogue plus the §4.8 honesty gate.

    Deliberately the same shape as `pah.tools.verifiers.base.Registry`, with one
    difference: the fixture pair is `DesignView` objects the verifier builds
    itself, so a design verifier can prove itself without a dataset on disk.
    """

    def __init__(self) -> None:
        self._verifiers: dict[str, DesignVerifier] = {}
        self._disabled: set[str] = set()
        self._reasons: dict[str, str] = {}

    # -- registration ------------------------------------------------------
    def register(self, verifier: DesignVerifier) -> None:
        spec = verifier.spec
        expected = DESIGN_CODES.get(spec.id)
        if expected is None:
            raise ValueError(
                f"verifier {spec.id!r} is not one of the design-plane verifiers "
                f"{sorted(DESIGN_CODES)}; add it to DESIGN_CODES first"
            )
        if tuple(spec.codes) != expected:
            raise ValueError(
                f"verifier {spec.id!r} declares codes {spec.codes!r}, but the registry "
                f"assigns it {expected!r}"
            )
        if spec.id in self._verifiers:
            raise ValueError(f"verifier {spec.id!r} is already registered")
        self._verifiers[spec.id] = verifier

    def get(self, verifier_id: str) -> DesignVerifier:
        try:
            return self._verifiers[verifier_id]
        except KeyError:
            known = ", ".join(sorted(self._verifiers)) or "none"
            raise KeyError(f"no design verifier {verifier_id!r}; registered: {known}") from None

    def ids(self) -> tuple[str, ...]:
        """Registered ids in cascade order (cheapest first), not insertion order."""
        order = {verifier_id: index for index, verifier_id in enumerate(DESIGN_VERIFIER_IDS)}
        return tuple(sorted(self._verifiers, key=lambda vid: (order.get(vid, 99), vid)))

    def cost_class(self, verifier_id: str) -> CostClass:
        return self.get(verifier_id).spec.cost_class

    def disabled(self) -> frozenset[str]:
        return frozenset(self._disabled)

    def why_disabled(self, verifier_id: str) -> str:
        return self._reasons.get(verifier_id, "")

    # -- the §4.8 gate -----------------------------------------------------
    def sanity(self) -> dict[str, bool]:
        """Run every verifier's own fixtures; `False` means freshly disabled."""
        result: dict[str, bool] = {}
        for verifier_id in self.ids():
            verifier = self._verifiers[verifier_id]
            reason = _first_disagreement(verifier)
            result[verifier_id] = reason == ""
            if reason:
                self._disabled.add(verifier_id)
                self._reasons[verifier_id] = reason
            else:
                self._disabled.discard(verifier_id)
                self._reasons.pop(verifier_id, None)
        return result

    # -- running -----------------------------------------------------------
    def check(
        self, view: DesignView, *, stop_on_hard: bool = False
    ) -> list[FailureCertificate]:
        """Run the enabled cascade over `view`, cheapest verifier first.

        `stop_on_hard=False` (the default) collects the whole picture: a report
        that saw only the first failure cannot tell a cleared fit from a hidden
        interference, and the paper's RQ1 is about what the agent was *told*.
        `stop_on_hard=True` is the budget-saving mode a caller may opt into; the
        trace records which one ran, via the cost of the round.
        """
        certificates: list[FailureCertificate] = []
        for verifier_id in self.ids():
            if verifier_id in self._disabled:
                continue
            found = list(self._verifiers[verifier_id].check(view))
            certificates.extend(found)
            if stop_on_hard and any(cert.family != "environment" for cert in found):
                break
        return certificates

    def fail_closed_view(self, view: DesignView, error: BaseException) -> FailureCertificate:
        """The certificate a cascade adapter records when a verifier raises.

        `family="environment"` is what keeps a broken tool out of a failure rate
        (`pah.core.model.FailureCertificate` docstring; doc §4.5): the loop counts
        these separately and never as a pass.
        """
        return FailureCertificate(
            code="verifier_error",
            family="environment",
            locus={"spec": view.spec.spec_id, "revision": view.spec.revision},
            magnitude={"error": f"{type(error).__name__}: {error}"},
            observed_by="design_cascade",
        )


def _first_disagreement(verifier: DesignVerifier) -> str:
    """The first fixture the verifier gets wrong, as a sentence; "" when it agrees."""
    try:
        good, bad = verifier.fixtures()
    except Exception as error:  # a fixture factory that throws is a failed gate too
        return f"fixtures() raised {type(error).__name__}: {error}"
    if not good or not bad:
        return "ships no known_good/known_bad fixture pair"
    for index, view in enumerate(good):
        try:
            found = verifier.check(view)
        except Exception as error:
            return f"known_good[{index}] raised {type(error).__name__}: {error}"
        if found:
            return f"known_good[{index}] judged negative: {found[0].code}"
    for index, view in enumerate(bad):
        try:
            found = verifier.check(view)
        except Exception as error:
            return f"known_bad[{index}] raised {type(error).__name__}: {error}"
        if not found:
            return f"known_bad[{index}] judged positive"
    return ""


# --------------------------------------------------------------------------- #
# Materialisation port
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MaterializedDesign:
    """A spec, the solids it built, and the versioned `Assembly` that carries them.

    `Assembly` is the loop's currency: the operator registry produces new versions
    of it, the engine stores its rounds, and the trace records its fingerprint. A
    materialised design is the same object plus the meshes and the directory the
    STLs were written to, so a verifier can be handed a `DesignView` without
    re-reading geometry it already has.
    """

    spec: DesignSpec
    assembly: Assembly
    meshes: dict[str, trimesh.Trimesh]
    #: Absolute paths of the written STLs, keyed by part id.
    paths: dict[str, str]
    root: Path

    def view(self) -> DesignView:
        return DesignView(
            spec=self.spec, root=self.root, meshes=dict(self.meshes), paths=dict(self.paths)
        )


#: `(spec, out_dir) -> MaterializedDesign`. A port so a CAD kernel or a DiT adapter
#: can replace the trimesh builder without touching the loop, the policies or the CLI.
Materializer = Callable[[DesignSpec, Path], MaterializedDesign]


# --------------------------------------------------------------------------- #
# Assembly <-> spec mapping, in one place
# --------------------------------------------------------------------------- #
#: Where a design-plane `Assembly` keeps its spec. `constraints` is the only
#: free-form field on the core model, and putting the spec anywhere else would
#: mean a second source of truth for the same design.
SPEC_SLOT = "design_spec"


def assembly_from_spec(
    *,
    spec: DesignSpec,
    geometry_paths: dict[str, str],
    geometry_hashes: dict[str, str],
    parent: str | None = None,
    version: int = 1,
) -> Assembly:
    """The `Assembly` payload for a materialised spec. Pure: no I/O, no hashing.

    Each part records its blank, its pose and a content-addressed geometry
    reference. The loop's `assembly_fingerprint` hashes `geom.hash`, so a spec edit
    that changes a number but not a vertex still moves the fingerprint — which is
    what makes cycle detection meaningful on the design plane.
    """
    from pah.core.ids import key  # local import: keeps this module import-light

    parts = tuple(
        {
            "id": part.id,
            "role": part.role,
            "size_mm": list(part.size_mm),
            "pose_mm": list(part.pose_mm),
            "rot_deg": list(part.rot_deg),
            "material": part.material,
            "geom": {
                "hash": geometry_hashes[part.id],
                "path": geometry_paths[part.id],
            },
        }
        for part in spec.parts
    )
    joints = (
        {
            "id": spec.interface.id,
            "a": spec.interface.a,
            "b": spec.interface.b,
            "kind": spec.interface.kind,
            "params": dict(spec.interface.params),
        },
    )
    body_id = spec.body.id
    assembly_id = key(
        "assembly",
        spec.spec_id,
        version,
        spec.content_key(),
        parent,
    )
    return Assembly(
        id=assembly_id,
        version=version,
        parts=parts,
        joints=joints,
        connectors=(),
        constraints={
            SPEC_SLOT: spec.model_dump(),
            "body_id": body_id,
            "insert_id": spec.insert.id,
        },
        parent=parent,
    )


def spec_of(assembly: Assembly) -> DesignSpec:
    """The `DesignSpec` an `Assembly` was built from.

    Raises `KeyError` for an assembly that is not on the design plane — the M0/M1
    cases have no spec, and treating one as if it did would silently compare a
    connector request against a hole diameter.
    """
    payload = (assembly.constraints or {}).get(SPEC_SLOT)
    if not payload:
        raise KeyError(
            f"assembly {assembly.id!r} carries no {SPEC_SLOT!r}; it is not a design-plane "
            "assembly (dataset cases live on the case.json plane)"
        )
    return DesignSpec.model_validate(payload)


def view_of(assembly: Assembly, root: str | Path | None = None) -> DesignView:
    """A verifier's view of `assembly`, loaded from the paths it already records.

    `root` is only needed for an assembly whose part payloads lack `geom.path`
    (a hand-built fixture); a materialised assembly carries absolute paths, which
    is why the cascade adapter can take an `Assembly` alone.
    """
    paths = {
        str(part.get("id")): str((part.get("geom") or {}).get("path"))
        for part in assembly.parts
        if (part.get("geom") or {}).get("path")
    }
    return DesignView(
        spec=spec_of(assembly),
        root=Path(root) if root is not None else None,
        paths=paths,
    )


def codes_owned_by(verifier_id: str) -> tuple[str, ...]:
    """The codes one verifier may emit, for a policy that routes on evidence."""
    return DESIGN_CODES.get(verifier_id, ())


def hard_codes(found: Iterable[FailureCertificate]) -> list[str]:
    """The physical codes in `found`, in stable order — the loop's repair trigger."""
    return sorted({cert.code for cert in found if cert.family != "environment"})
