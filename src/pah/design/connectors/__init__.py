"""The connector-kind registry, plus the frame plumbing every kind shares.

Why a registry rather than a match statement
-------------------------------------------
`DesignSpec` stores a connector as a kind string plus a plain parameter dict on
purpose (`design/spec.py`): the set of kinds is a plugin surface, and a
discriminated union there would freeze the shipped three into the spec's schema.
This module is the other half of that decision: the place where a kind string
becomes callable. The materialiser asks for a builder by kind; a verifier that
needs `mate_axis` or `approach_mm` without touching triangles asks the same
builder; a plugin calls `register_builder` and nothing else changes.

Why the pose is applied here
----------------------------
A kind reasons in a local frame, where "the hole runs along +Z" is one sentence.
The rule the rest of the harness codes against -- every mesh is already world
frame (`docs/contracts/v2-interfaces.md`) -- is enforced once, inside
`ConnectorBuilderBase.build`, so a plugin that implements `build_local` cannot
forget it and no verifier ever has to ask which frame it is looking at. Two
conventions follow, and `docs/internal/geometry-notes.md` is the long form:

* the blank is centred on the part's local origin, matching
  `trimesh.creation.box`, so `pose_mm` names the blank's centre and
  `feature_xy_mm` is measured from that centre for every kind;
* rotation is the extrinsic XYZ the spec promises: R = Rz @ Ry @ Rx, applied
  about the local origin, then translated by `pose_mm`.

Determinism is part of the contract (contract §7.4: same spec twice, same
`geom_hash` series). Nothing here may iterate an unordered collection, read a
clock, or leave tessellation to a library default that could change.
"""

from __future__ import annotations

import abc
import math

import numpy as np
import trimesh
from pydantic import BaseModel

from pah.design.api import ConnectorBuilder, ConnectorGeometry
from pah.design.spec import DesignSpec, PartSpec, Vec3, builtin_params_model

__all__ = [
    "ConnectorBuilderBase",
    "available_kinds",
    "builder_for",
    "register_builder",
]

_BUILDERS: dict[str, ConnectorBuilder] = {}


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #
def register_builder(builder: ConnectorBuilder) -> None:
    """Add one connector kind to the process-wide registry.

    A repeated kind is an error, not an override: two builders for one kind would
    make a spec's geometry depend on import order, and the loop's reproducibility
    claim dies exactly there.
    """
    if not isinstance(builder, ConnectorBuilder):
        raise TypeError(
            f"{type(builder).__name__} does not implement ConnectorBuilder (needs "
            "kind, params_model, clearance_mm, seated_pose_mm and build)"
        )
    kind = builder.kind
    if not kind:
        raise ValueError("a connector builder must declare a non-empty `kind`")
    if kind in _BUILDERS:
        raise ValueError(
            f"connector kind {kind!r} is already registered by "
            f"{type(_BUILDERS[kind]).__name__}; kinds are unique by design"
        )
    _BUILDERS[kind] = builder


def builder_for(kind: str) -> ConnectorBuilder:
    """The builder registered for `kind`, or a `KeyError` naming the alternatives."""
    try:
        return _BUILDERS[kind]
    except KeyError:
        known = ", ".join(sorted(_BUILDERS)) or "none"
        raise KeyError(
            f"no connector builder for kind {kind!r}; registered kinds: {known}"
        ) from None


def available_kinds() -> tuple[str, ...]:
    """Every registered kind, builtins first in registration order."""
    return tuple(_BUILDERS)


# --------------------------------------------------------------------------- #
# Frame helpers -- shared so "world frame" means one thing for every kind
# --------------------------------------------------------------------------- #
def blank_box(size_mm: Vec3) -> trimesh.Trimesh:
    """A part's envelope: a box of `size_mm` centred on the local origin.

    Centred, not corner-anchored, because `trimesh.creation.box` is centred and
    because `pose_mm` then names the same point for both parts of every kind. A
    corner convention would turn "the body's top face" into a size-dependent
    expression instead of `pose_mm.z + size_mm.z / 2`.
    """
    return trimesh.creation.box(extents=tuple(float(value) for value in size_mm))


def part_matrix(part: PartSpec) -> np.ndarray:
    """The 4x4 world transform for one part: extrinsic XYZ rotation, then translation.

    Extrinsic XYZ is R = Rz @ Ry @ Rx -- the rotation the spec's docstring spells
    out -- and applying it before the translation is what "about the local origin"
    means. The angles reach trigonometry only here, so a kind never has to think
    about poses.
    """
    rx, ry, rz = (math.radians(float(angle)) for angle in part.rot_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rotation = np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=float,
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [float(value) for value in part.pose_mm]
    return matrix


def to_world(mesh: trimesh.Trimesh, part: PartSpec) -> trimesh.Trimesh:
    """A local-frame solid as the world-frame mesh the contract promises."""
    placed = mesh.copy()
    placed.apply_transform(part_matrix(part))
    return placed


def world_axis(part: PartSpec, axis: Vec3) -> Vec3:
    """`axis`, written in the part's local frame, as a world-frame unit vector.

    The insertion axis follows the *insert*, because that is the body that moves
    along it; a spec whose two parts disagree about rotation is already showing
    a design mistake the path verifier will measure, not something this helper
    should silently average away.
    """
    vector = part_matrix(part)[:3, :3] @ np.asarray(axis, dtype=float)
    length = float(np.linalg.norm(vector))
    if length == 0.0:
        raise ValueError(f"part {part.id!r} rotates its insertion axis to a zero vector")
    return tuple(float(component / length) for component in vector)


def world_point(part: PartSpec, point: Vec3) -> Vec3:
    """A point written in the part's local frame, as world coordinates.

    The companion to `world_axis` for the one place a *point* is contracted: the
    seat datum a `switch_connector` has to move an insert to. Rotating or shifting
    the body moves the datum with it, so a kind states the datum once, in the frame
    the feature was cut in.
    """
    matrix = part_matrix(part)
    local = np.array([float(point[0]), float(point[1]), float(point[2]), 1.0])
    return tuple(float(component) for component in (matrix @ local)[:3])


# --------------------------------------------------------------------------- #
# The base every builtin kind shares
# --------------------------------------------------------------------------- #
class ConnectorBuilderBase(abc.ABC):
    """Shared plumbing for a connector kind: validation, frames, insertion axis.

    A kind implements `build_local` and declares `local_mate_axis`; everything
    that must hold for *every* kind is done once, here: the interface's
    parameters are validated by the kind's own pydantic model, the interface and
    the builder must agree about the kind, both solids are placed in world
    coordinates, and the mate axis is normalised. A plugin that subclasses this
    base inherits that contract instead of re-deriving it (and getting it subtly
    wrong).

    Every builtin parameter model carries `approach_mm`; a plugin kind that
    subclasses this base must carry it too, because `build` reports it and the
    insertion-path verifier starts its sweep from it.
    """

    #: Registered name. A subclass must set it; the registry keys on it.
    kind: str = ""

    #: Insertion direction in the kind's local frame. The materialiser only needs
    #: the world-frame version, but stating the local axis next to the geometry
    #: keeps the convention readable where the solid is written.
    local_mate_axis: Vec3 = (0.0, 0.0, 1.0)

    def params_model(self) -> type[BaseModel]:
        """The model that validates this kind's `params` dict."""
        model = builtin_params_model(self.kind)
        if model is None:
            raise KeyError(
                f"connector kind {self.kind!r} has no builtin parameter model; a "
                "plugin subclass of ConnectorBuilderBase must provide params_model()"
            )
        return model

    def clearance_mm(self, params: BaseModel) -> float:
        """The designed radial clearance, as the kind's parameter model spells it.

        Delegated on purpose: `design/spec.py` owns how each kind spells
        clearance, and a second copy of that arithmetic here is how two modules
        start disagreeing about the same design.
        """
        return float(params.radial_clearance_mm)

    @abc.abstractmethod
    def build_local(
        self, body: PartSpec, insert: PartSpec, params: BaseModel
    ) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """`(body, insert)` solids, each in its own local frame.

        The insert's local origin is its **leading tip** -- the face that enters
        the other part first. That is the point `insert.pose_mm` names when the
        insert is seated, and the point `approach_mm` steps back along, so a
        fixture can be written by measuring where the tip must end up.

        `params` is the kind's validated model, not a dict: the base validates it
        once and hands over the typed numbers.
        """

    @abc.abstractmethod
    def _seat_point_mm(self, spec: DesignSpec) -> Vec3:
        """The seat datum, in the **body's** local frame.

        Where the insert's origin lands when it is fully seated. Body-local, not
        world, because `pose_mm` and the body's rotation move the datum with the
        feature; stating it in the frame the feature was cut in means a rotated or
        shifted body needs no second seating rule.
        """

    def seated_pose_mm(self, spec: DesignSpec) -> Vec3:
        """Where the insert's origin sits when fully seated, in world mm.

        A *kind switch* (`switch_connector`) and a fresh design (`pah design new`)
        both have to place an insert, and only the kind knows where. Measured cost
        of getting this wrong: a peg_hole -> dovetail switch left the insert at the
        peg's seated pose and the seated intersection became 1800 mm3, a blocking
        false positive caused by geometry the design never asked for.

        Only the translation is returned: the protocol's answer is a point, and an
        insert rotated differently from its body is a design mistake the verifiers
        measure, not something this helper should paper over.
        """
        if spec.interface.kind != self.kind:
            raise ValueError(
                f"builder {type(self).__name__} seats {self.kind!r}, but the spec's "
                f"interface {spec.interface.id!r} is {spec.interface.kind!r}"
            )
        return world_point(spec.body, self._seat_point_mm(spec))

    def build(self, spec: DesignSpec) -> ConnectorGeometry:
        """`spec`'s interface as two world-frame solids plus its insertion axis."""
        interface = spec.interface
        if interface.kind != self.kind:
            raise ValueError(
                f"builder {type(self).__name__} builds {self.kind!r}, but the spec's "
                f"interface {interface.id!r} is {interface.kind!r}"
            )
        params = interface.validated_params()
        body_local, insert_local = self.build_local(spec.body, spec.insert, params)
        return ConnectorGeometry(
            meshes={
                spec.body.id: to_world(body_local, spec.body),
                spec.insert.id: to_world(insert_local, spec.insert),
            },
            mate_axis=world_axis(spec.insert, self.local_mate_axis),
            approach_mm=float(params.approach_mm),
        )


# Imported last: each kind imports `ConnectorBuilderBase` back out of this module,
# which works only once the name above exists.
from pah.design.connectors.dovetail import DovetailBuilder  # noqa: E402
from pah.design.connectors.peg_hole import PegHoleBuilder  # noqa: E402
from pah.design.connectors.snap_fit import SnapFitBuilder  # noqa: E402

for _builder in (PegHoleBuilder(), SnapFitBuilder(), DovetailBuilder()):
    register_builder(_builder)
del _builder
