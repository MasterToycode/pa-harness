"""The design plane's typed intent: `DesignSpec`, the object an agent is allowed to edit.

Why a second plane
------------------
`case.json` (M0/M1) describes *geometry that already exists*: meshes on disk and
the joints a verifier may judge. That plane can only repair what the dataset
already contains, which is why every week of work on it produced a better
connector algorithm and no better agent.

`DesignSpec` is the opposite: it is the *decision* behind the geometry, written
down as numbers with units. A parameterised design is what a planner can reason
over ("hole too small by 0.5 mm", "clearance below the fit band"), what CAD or a
future DiT can materialise, and what a verifier can judge without re-deriving
intent from triangles. `docs/contracts/v2-interfaces.md` §1 fixes this file as
the contract every other v2 module codes against.

Invariants this file enforces at construction time
--------------------------------------------------
* **Millimetres, degrees, one convention.** Every length in the spec is mm and
  every angle is degrees. No unit field, because a unit field that can disagree
  with the numbers is a bug source, not a safeguard.
* **Any number of parts, any number of interfaces, one connected assembly.** A
  decomposition hands back N printable pieces; an assembly is what they form
  together. The graph of interfaces must therefore *connect* every part: a part
  that no interface reaches is a part nothing holds, and if two solids are meant
  to ship as one object without a connector between them, that is two designs,
  not one. What is still fixed is the *arity of a connection*: one interface joins
  exactly one pair (`a`, `b`), because every connector kind PAH ships is a
  two-body mating feature. Widening past that means a kind, not a loop.
* **A kind is a string, its parameters are a separate model.** Connector kinds
  are plugins (§4 of the contract): the spec stores `kind` plus a plain dict, and
  the kind's own pydantic model validates that dict at materialisation time.
  Putting a discriminated union here would freeze the set of kinds into the
  spec's schema and make the plugin surface a lie.
* **No geometry, no verifier text, no verdict.** A spec cannot say "interference
  is 3 mm3" or "this is repairable": evidence belongs to verifiers and repair
  intent belongs to a `RepairPlan`. The one thing a spec may carry about its own
  origin is `provenance` — who chose which number — because that is what the
  paper reports (RQ1/RQ2) and it cannot be reconstructed after the fact.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ConnectorKindName",
    "DesignSpec",
    "DovetailParams",
    "FitConstraints",
    "InterfaceSpec",
    "PART_ROLES",
    "PartSpec",
    "PegHoleParams",
    "SnapFitParams",
    "Vec3",
    "builtin_params_model",
    "radial_clearance_mm",
]

Vec3 = tuple[float, float, float]

#: The connector kinds PAH ships. A plugin may add more; nothing may assume this
#: tuple is the whole set except code that needs a default.
ConnectorKindName = Literal["peg_hole", "snap_fit", "dovetail"]

PART_ROLES = ("body", "insert", "part")


class _Frozen(BaseModel):
    """Strict, immutable, no surprise fields. A spec is compared and hashed."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- #
# Parts
# --------------------------------------------------------------------------- #
class PartSpec(_Frozen):
    """One rigid part, described by a box envelope and a pose.

    The envelope is the *blank*, not the final shape: the interface subtracts a
    hole, adds a peg, and so on. Keeping the blank explicit means a wall-thickness
    check has a reference surface to measure against instead of guessing which
    triangle is "outside".
    """

    id: str = Field(min_length=1)
    #: A *hint about the part's origin*, never about topology. `body` and `insert`
    #: are the two-part case this project started from; `part` is any piece a
    #: decomposition returned. Which side of a connector a part sits on is decided
    #: by the interface's `a`/`b`, so a `body` may be the `b` side of one interface
    #: and the `a` side of another. Nothing may infer the graph from roles.
    role: Literal["body", "insert", "part"]
    size_mm: Vec3 = (60.0, 40.0, 12.0)
    #: World-frame translation of the part's local origin, in mm.
    pose_mm: Vec3 = (0.0, 0.0, 0.0)
    #: XYZ extrinsic rotation in degrees, applied about the local origin.
    rot_deg: Vec3 = (0.0, 0.0, 0.0)
    material: str = "PLA"
    #: The content address of the blank mesh this part was decomposed from.
    #:
    #: Empty means "this part's geometry is the parametric blank `size_mm`
    #: describes" — the two-part case, where the builder invents the shape. Set means
    #: a decomposer produced the shape and the materialiser must machine the
    #: interfaces into *that* mesh. The hash is in the spec, not the mesh, for the
    #: same reason `geometry_hash` is: an episode has to be reconstructible from
    #: `spec.json` plus a known store of blanks, and a path in the spec would make the
    #: same design hash differently on another machine.
    source: str = ""

    @model_validator(mode="after")
    def _positive_envelope(self) -> PartSpec:
        if any(float(x) <= 0.0 for x in self.size_mm):
            raise ValueError(f"part {self.id!r} needs a positive size_mm, got {self.size_mm}")
        return self


# --------------------------------------------------------------------------- #
# Connector parameters — one model per builtin kind, validated by the kind's plugin
# --------------------------------------------------------------------------- #
class PegHoleParams(_Frozen):
    """A round peg in a round hole. The demo case from the proposal, verbatim.

    Clearance is *derived*, never stored: `(d_hole - d_peg) / 2`. Storing it would
    let a spec claim 0.2 mm while its diameters imply 0.1, and the fit verifier
    would then be judging a number nobody built.
    """

    d_peg_mm: float = Field(gt=0.0)
    d_hole_mm: float = Field(gt=0.0)
    #: How deep the peg sits inside the hole when seated.
    engage_mm: float = Field(default=8.0, gt=0.0)
    #: Distance the peg starts above the seated pose; the insertion path spans it.
    approach_mm: float = Field(default=6.0, ge=0.0)
    #: Peg length. `None` means "engage plus a 6 mm handle" — long enough that
    #: physics has a body to grip and a path has something to sweep.
    peg_len_mm: float | None = Field(default=None, gt=0.0)
    #: Where the hole axis meets the body's top face, in the body's local frame.
    feature_xy_mm: tuple[float, float] = (0.0, 0.0)

    @property
    def radial_clearance_mm(self) -> float:
        """Half the diametral clearance — the gap a fit is actually judged on."""
        return (self.d_hole_mm - self.d_peg_mm) / 2.0

    @property
    def peg_length_mm(self) -> float:
        return self.peg_len_mm if self.peg_len_mm is not None else self.engage_mm + 6.0


class SnapFitParams(_Frozen):
    """A cantilever hook that snaps into a slot: the "no fastener" alternative.

    The lever for repair is the undercut (`hook_h_mm`), which also decides
    insertion force; `slot_w_mm - hook_w_mm` is the lateral play this kind has
    instead of a radial clearance.
    """

    hook_w_mm: float = Field(default=4.0, gt=0.0)
    hook_h_mm: float = Field(default=1.0, gt=0.0)
    beam_len_mm: float = Field(default=12.0, gt=0.0)
    beam_thk_mm: float = Field(default=1.6, gt=0.0)
    slot_w_mm: float = Field(default=4.2, gt=0.0)
    engage_mm: float = Field(default=8.0, gt=0.0)
    approach_mm: float = Field(default=6.0, ge=0.0)

    @property
    def radial_clearance_mm(self) -> float:
        return (self.slot_w_mm - self.hook_w_mm) / 2.0


class DovetailParams(_Frozen):
    """A trapezoidal slide: the "cannot pull straight out" option.

    `angle_deg` is the flank angle; zero would be a plain tongue, so the model
    refuses it. `clearance_mm` is per side.
    """

    width_mm: float = Field(default=10.0, gt=0.0)
    depth_mm: float = Field(default=6.0, gt=0.0)
    angle_deg: float = Field(default=12.0, gt=0.0, lt=45.0)
    clearance_mm: float = Field(default=0.1, ge=0.0)
    engage_mm: float = Field(default=20.0, gt=0.0)
    approach_mm: float = Field(default=6.0, ge=0.0)

    @property
    def radial_clearance_mm(self) -> float:
        return self.clearance_mm


_BUILTIN_PARAMS: dict[str, type[BaseModel]] = {
    "peg_hole": PegHoleParams,
    "snap_fit": SnapFitParams,
    "dovetail": DovetailParams,
}


def builtin_params_model(kind: str) -> type[BaseModel] | None:
    """The parameter model for a builtin kind, or `None` so a plugin can claim it."""
    return _BUILTIN_PARAMS.get(kind)


def radial_clearance_mm(kind: str, params: BaseModel | dict) -> float:
    """The designed clearance of one interface, in mm.

    The single place that knows how each kind spells clearance, so no verifier or
    policy re-derives it: a second copy of this arithmetic is how two modules start
    disagreeing about the same design.
    """
    model = builtin_params_model(kind)
    if model is None:
        raise KeyError(f"{kind!r} is not a builtin connector kind")
    values = params if isinstance(params, BaseModel) else model.model_validate(params)
    if not isinstance(values, model):
        values = model.model_validate(values.model_dump())
    return float(values.radial_clearance_mm)


# --------------------------------------------------------------------------- #
# Interface and constraints
# --------------------------------------------------------------------------- #
class InterfaceSpec(_Frozen):
    """One connection between the two parts: which kind, and the numbers behind it."""

    id: str = "iface_0"
    kind: str = Field(min_length=1)
    a: str = Field(min_length=1)  # the body
    b: str = Field(min_length=1)  # the insert
    #: Validated by the kind's own pydantic model at materialisation time. Kept as
    #: a plain dict on purpose: a plugin must be able to add a kind without
    #: touching this file.
    params: dict = Field(default_factory=dict)

    def model_for(self) -> type[BaseModel]:
        """The parameter model this interface's `kind` declares."""
        model = builtin_params_model(self.kind)
        if model is None:
            raise KeyError(
                f"interface {self.id!r} names connector kind {self.kind!r}, which is "
                "neither builtin nor resolvable without the plugin that owns it"
            )
        return model

    def validated_params(self) -> BaseModel:
        """`params` as the kind's own model, or a `ValidationError` at the edge."""
        return self.model_for().model_validate(self.params)


class FitConstraints(_Frozen):
    """Acceptance numbers. They are design inputs, not verifier internals.

    `min_/max_clearance_mm` are the fit band the *task* asks for; the fit verifier
    compares the built clearance against them and says nothing about what the
    band "should" be. `target_fit` is recorded so a report can name the intent
    without re-deriving it from the numbers.

    `print_orientation_deg` is `None` by default, and that is a deliberate
    statement rather than a missing feature: **an overhang claim needs an
    orientation**. Judging the insert while it stands in its assembled pose — tip
    down, in mid-air — declares every peg unprintable for a defect no repair
    operator can address. With `None`, the printability verifier checks wall
    thickness (orientation independent) and does not evaluate overhang; the report
    says so under `manufacturing_validated: not evaluated`, which is a different
    claim from "printable" (§6 of `pah-dev-doc-v1.md` keeps those apart).
    """

    min_clearance_mm: float = Field(default=0.10, ge=0.0)
    max_clearance_mm: float = Field(default=0.60, ge=0.0)
    min_wall_mm: float = Field(default=1.2, gt=0.0)
    process: Literal["fdm", "sla", "sls", "mjf", "cnc"] = "fdm"
    target_fit: Literal["slip", "snug", "press"] = "slip"
    #: The build direction in the part's own frame, as a rotation in degrees about
    #: its local origin. `None` = no manufacturing evaluation is being claimed.
    print_orientation_deg: Vec3 | None = None

    @model_validator(mode="after")
    def _ordered_band(self) -> FitConstraints:
        if self.min_clearance_mm > self.max_clearance_mm:
            raise ValueError(
                "min_clearance_mm exceeds max_clearance_mm: "
                f"{self.min_clearance_mm} > {self.max_clearance_mm}"
            )
        return self

    def in_band(self, clearance_mm: float) -> bool:
        return self.min_clearance_mm <= clearance_mm <= self.max_clearance_mm


# --------------------------------------------------------------------------- #
# The spec
# --------------------------------------------------------------------------- #
class DesignSpec(_Frozen):
    """A versioned design intent. Immutable; every edit makes a new revision.

    `revision` is carried so a trace can say "round 2 was revision 3" without
    hashing the whole spec again, and `content_key()` is the hash a
    `core.model.Assembly` records as the part's geometry hash — which is what
    makes the loop's cycle detection work on the design plane at all.
    """

    spec_id: str = Field(min_length=1)
    requirement: str = ""
    parts: tuple[PartSpec, ...]
    #: One entry per mating pair. Empty is legal only for a single-part spec, and
    #: the validator below is what says so — a two-part spec with no interfaces is
    #: refused there with a message that names the parts nothing holds.
    interfaces: tuple[InterfaceSpec, ...] = ()
    constraints: FitConstraints = Field(default_factory=FitConstraints)
    revision: int = Field(default=1, ge=1)
    #: Who decided what — `{"interface_kind": "llm:qwen3", "params": "rule"}`.
    #: The only provenance a report cannot reconstruct later.
    provenance: dict = Field(default_factory=dict)
    api_version: str = "1"

    @model_validator(mode="after")
    def _shape(self) -> DesignSpec:
        """The structural invariants: unique ids, real references, one assembly.

        All of them are checked at construction rather than at materialisation, for
        the same reason an unbuildable parameter set is: a spec that cannot be built
        must fail where it is *written*, so the repair loop sees a bad edit as a bad
        edit instead of a verifier mystery three stages later.
        """
        if not self.parts:
            raise ValueError("a spec needs at least one part")
        ids = {part.id for part in self.parts}
        if len(ids) != len(self.parts):
            raise ValueError(f"duplicate part ids in {[p.id for p in self.parts]!r}")

        for role in ("body", "insert"):
            holders = [part.id for part in self.parts if part.role == role]
            if len(holders) > 1:
                raise ValueError(
                    f"at most one part may be the {role!r}, got {holders!r}; use "
                    "role='part' for the extra pieces a decomposition returns"
                )

        iface_ids = [iface.id for iface in self.interfaces]
        if len(set(iface_ids)) != len(iface_ids):
            raise ValueError(f"duplicate interface ids in {iface_ids!r}")

        pairs: set[frozenset[str]] = set()
        for iface in self.interfaces:
            if iface.a == iface.b:
                raise ValueError(f"interface {iface.id!r} joins {iface.a!r} to itself")
            unknown = sorted({iface.a, iface.b} - ids)
            if unknown:
                raise ValueError(
                    f"interface {iface.id!r} joins {unknown!r}, which "
                    f"{'is' if len(unknown) == 1 else 'are'} not among the parts "
                    f"{sorted(ids)!r}"
                )
            pair = frozenset((iface.a, iface.b))
            if pair in pairs:
                raise ValueError(
                    f"two interfaces join the same pair {sorted(pair)!r}; one mating "
                    "pair needs one connection, otherwise two connectors compete for "
                    "the same space and neither finishes its insertion path"
                )
            pairs.add(pair)

        if len(self.parts) == 1 and self.interfaces:
            raise ValueError(
                f"a single-part spec cannot carry interfaces, got {iface_ids!r}"
            )
        if len(self.parts) > 1:
            reached = {side for iface in self.interfaces for side in (iface.a, iface.b)}
            unreached = sorted(ids - reached)
            if unreached:
                raise ValueError(
                    f"part(s) {unreached!r} carry no interface, so nothing holds them "
                    "to the assembly; connect them or leave them out of the spec"
                )
            clusters = self._clusters()
            if len(clusters) > 1:
                raise ValueError(
                    "the interfaces do not connect every part: an assembly is one "
                    f"rigid group, and {sorted(ids)!r} splits into {len(clusters)} "
                    f"(sizes {sorted(len(c) for c in clusters)!r})"
                )

        # Validate the parameters now, not at materialisation: an unbuildable spec
        # must fail where it is written.
        for interface in self.interfaces:
            interface.validated_params()
        return self

    def _clusters(self) -> list[set[str]]:
        """Connected components of the interface graph, for the invariants above."""
        adjacency: dict[str, set[str]] = {part.id: set() for part in self.parts}
        for iface in self.interfaces:
            adjacency[iface.a].add(iface.b)
            adjacency[iface.b].add(iface.a)
        seen: set[str] = set()
        clusters: list[set[str]] = []
        for start in adjacency:
            if start in seen:
                continue
            group: set[str] = set()
            stack = [start]
            while stack:
                node = stack.pop()
                if node in group:
                    continue
                group.add(node)
                stack.extend(adjacency[node] - group)
            seen |= group
            clusters.append(group)
        return clusters

    # -- lookups -----------------------------------------------------------
    @property
    def body(self) -> PartSpec:
        """The one part tagged `body`.

        Raises instead of guessing for an N-part spec: `next(...)` on an empty match
        used to surface as a bare `StopIteration`, which reads like a bug in the
        caller rather than "this shorthand does not apply here".
        """
        holders = [part for part in self.parts if part.role == "body"]
        if len(holders) != 1:
            raise ValueError(
                f"spec {self.spec_id!r} has {len(holders)} parts tagged 'body' "
                f"({[p.id for p in holders]!r}); the two-part shorthand does not "
                "apply, use `parts` / `interfaces_of`"
            )
        return holders[0]

    @property
    def insert(self) -> PartSpec:
        """The one part tagged `insert`; same refusal as `body` for an N-part spec."""
        holders = [part for part in self.parts if part.role == "insert"]
        if len(holders) != 1:
            raise ValueError(
                f"spec {self.spec_id!r} has {len(holders)} parts tagged 'insert' "
                f"({[p.id for p in holders]!r}); the two-part shorthand does not "
                "apply, use `parts` / `interfaces_of`"
            )
        return holders[0]

    @property
    def interface(self) -> InterfaceSpec:
        """The single interface, for the two-part case; raises when there are many."""
        if len(self.interfaces) != 1:
            raise ValueError(
                f"spec {self.spec_id!r} has {len(self.interfaces)} interfaces; use "
                "`interfaces`, `interface_by_id` or `interfaces_of(part_id)`"
            )
        return self.interfaces[0]

    def part_ids(self) -> tuple[str, ...]:
        return tuple(part.id for part in self.parts)

    def interface_by_id(self, interface_id: str) -> InterfaceSpec:
        for iface in self.interfaces:
            if iface.id == interface_id:
                return iface
        raise KeyError(
            f"no interface {interface_id!r} in spec {self.spec_id!r}; the interfaces "
            f"are {[i.id for i in self.interfaces]!r}"
        )

    def interfaces_of(self, part_id: str) -> tuple[InterfaceSpec, ...]:
        """Every interface the part takes part in, in spec order."""
        return tuple(
            iface for iface in self.interfaces if part_id in (iface.a, iface.b)
        )

    def neighbour_of(self, interface: str | InterfaceSpec, part_id: str) -> str:
        """The part on the other side of `interface` from `part_id`."""
        iface = interface if isinstance(interface, InterfaceSpec) else self.interface_by_id(interface)
        if part_id == iface.a:
            return iface.b
        if part_id == iface.b:
            return iface.a
        raise KeyError(
            f"interface {iface.id!r} does not touch part {part_id!r} "
            f"(it joins {iface.a!r} and {iface.b!r})"
        )

    @property
    def clearance_mm(self) -> float:
        return radial_clearance_mm(self.interface.kind, self.interface.params)

    def part(self, part_id: str) -> PartSpec:
        return next(part for part in self.parts if part.id == part_id)

    # -- edits (pure: a new revision, never a mutation) --------------------
    def edit(self, **changes) -> DesignSpec:
        """A new revision with `changes` applied to the model's fields."""
        payload = self.model_dump()
        payload.update(changes)
        payload["revision"] = self.revision + 1
        return DesignSpec.model_validate(payload)

    def with_interface(
        self,
        *,
        kind: str | None = None,
        params: dict | None = None,
        interface_id: str | None = None,
    ) -> DesignSpec:
        """A new revision with one interface's kind and/or parameters replaced.

        Two different operations, because they have two different meanings:

        * **same kind** — `params` are merged into the current ones, which is what
          a repair operator wants ("change `d_hole_mm`, leave the rest alone");
        * **kind change** — `params` replace the old set wholesale, because the old
          keys belong to the old kind's vocabulary and its `extra="forbid"` model
          would reject them. Switching kind without supplying the new kind's
          parameters is a `ValueError`, not a silent half-migration: a dovetail
          described by peg diameters is a spec nobody can materialise.

        `interface_id` names the interface to edit. It is required by construction
        once a spec has more than one: `self.interface` refuses to guess which
        connection a repair meant, and silently editing the first one would make
        every N-part repair depend on the order a decomposition happened to emit.
        """
        current = (
            self.interface
            if interface_id is None
            else self.interface_by_id(interface_id)
        )
        new_kind = kind if kind is not None else current.kind
        if new_kind != current.kind:
            if not params:
                raise ValueError(
                    f"switching interface {current.id!r} from {current.kind!r} to "
                    f"{new_kind!r} needs that kind's parameters; the old ones "
                    f"({sorted(current.params)}) do not describe it"
                )
            payload = dict(params)
        else:
            payload = {**current.params, **(params or {})}
        replacement = current.model_copy(update={"kind": new_kind, "params": payload})
        return self.edit(
            interfaces=tuple(
                replacement if iface.id == current.id else iface
                for iface in self.interfaces
            )
        )

    def with_parts(self, *parts: PartSpec) -> DesignSpec:
        """A new revision with the part set replaced (poses move this way)."""
        by_id = {part.id: part for part in parts}
        rebuilt = tuple(by_id.get(part.id, part) for part in self.parts)
        unknown = sorted(set(by_id) - {part.id for part in self.parts})
        if unknown:
            raise KeyError(f"no such part(s) {unknown!r} in spec {self.spec_id!r}")
        return self.edit(parts=rebuilt)

    def with_part(self, part: PartSpec) -> DesignSpec:
        return self.with_parts(part)

    # -- identity ----------------------------------------------------------
    def content_key(self) -> str:
        """A stable hash of everything that can change the built geometry.

        `revision`, `provenance` and `requirement` are excluded on purpose: the
        same numbers chosen twice must produce the same hash, or the loop's cycle
        detection would treat a re-run as progress.
        """
        payload = {
            "parts": [part.model_dump() for part in self.parts],
            # Sorted by id: two specs that describe the same assembly are the same
            # content no matter what order the interfaces were *discovered* in. A
            # decomposition that walks its bboxes in a different order did not
            # design anything differently, and hashing the order would make the
            # loop's cycle detection report progress for a reshuffle.
            "interfaces": [
                iface.model_dump()
                for iface in sorted(self.interfaces, key=lambda i: (i.id, i.a, i.b))
            ],
            "constraints": self.constraints.model_dump(),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def short(self) -> str:
        """One line a trace row or a CLI can print."""
        head = f"{self.spec_id} r{self.revision}"
        if len(self.interfaces) != 1:
            return f"{head} {len(self.parts)} parts / {len(self.interfaces)} interfaces"
        return f"{head} {self.interface.kind} c={self.clearance_mm:+.3f}mm"
