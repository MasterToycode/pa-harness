"""The `known_good` / `known_bad` geometry every design-plane verifier ships.

Why this module exists
----------------------
`DesignVerifierRegistry.sanity` (contract §3, `design/api.py`) refuses a verifier
that cannot show a paired fixture: a `known_good` view it must judge clean and a
`known_bad` view it must fail. On the dataset plane those pairs came from the
frozen gold rows (`runs/m0-connector`); on the design plane there is no dataset —
a design is built from numbers — so each verifier has to carry the two views it
is judged against.

Two rules shape the fixtures, and both are load-bearing:

1. **Built here, not by the materialiser.** A verifier that imported
   `pah.design.materialize` to build its own proof would make its gate depend on
   the very code it is meant to judge, and a materialiser bug would then disable
   (or, worse, *pass*) every verifier at once. These doubles come straight from
   `trimesh` booleans, so the gate tests the verifier against geometry nobody in
   the pipeline produced.
2. **One change per fixture.** Each bad fixture differs from the good one in the
   single thing the verifier under test is supposed to notice, so a green
   `sanity()` means "it measured that thing", not "it failed for some reason".

The geometry caliber (fixed 2026-11-01, same as T1's materialiser)
------------------------------------------------------------------
A `peg_hole` interface is a **blind socket**: the bore is `engage_mm` deep and the
peg's bottom rests on the socket's floor. Two consequences the verifiers are
measured against:

* the **seated pose is a face contact**, peg bottom on socket floor, with zero
  shared volume. `interface_interference` must read that as 0, not as a
  collision — the 1e-3 mm3 noise floor exists precisely to absorb the coplanar
  slivers a boolean may leave there;
* the **end of the insertion path is a contact, not a penetration**, which is why
  the sweep stops one step above the seat instead of sampling it.

The world frame
---------------
Every mesh here is in world coordinates, seated pose, and every `PartSpec` keeps
`pose_mm = rot_deg = (0, 0, 0)` — a fixture bypasses the materialiser
(`design/api.py:DesignView`), so the pose has already been applied to the
triangles. A verifier must never re-apply `PartSpec.pose_mm`; it would transform
the solids twice. The body is a 60 x 40 x 12 blank centred on the origin
(`z in [-6, +6]`), the socket opens at the body's top face `z = +6` and runs down
`engage` mm, and the insert is driven along `ConnectorGeometry.mate_axis`'s
default of +Z.
"""

from __future__ import annotations

import functools

import trimesh
from trimesh.boolean import difference
from trimesh.creation import box, cylinder

from pah.design.api import DesignView
from pah.design.spec import (
    DesignSpec,
    FitConstraints,
    InterfaceSpec,
    PartSpec,
    PegHoleParams,
)

__all__ = [
    "BODY_SIZE_MM",
    "FIT_BAND",
    "HANDLE_MM",
    "SECTIONS",
    "SEAT_Z_MM",
    "clearance_fixture",
    "loose_fixture",
    "oversized_fixture",
    "pocket_fixture",
    "standard_fixture",
    "stepped_fixture",
    "thin_wall_fixture",
]

# --------------------------------------------------------------------------- #
# Numbers, and why they are these numbers
# --------------------------------------------------------------------------- #
#: The body blank. 60 x 40 leaves a 15 mm radial wall around a centred 10 mm
#: socket, an order of magnitude above the 1.2 mm the default process asks for, so
#: the good fixture is not accidentally a thin-wall one.
BODY_SIZE_MM = (60.0, 40.0, 12.0)
#: The body's top face — the seat plane. A socket of depth 8 runs from `z = +6`
#: down to `z = -2`, and the peg bottom comes to rest on that floor.
SEAT_Z_MM = 6.0
#: How far the insert stands above the seat when seated. The sweep needs material
#: above the socket to fail *on*, and the physics tier needs a body to grip.
HANDLE_MM = 6.0
#: Facets per cylinder. At r = 5 mm the chord error is r * (1 - cos(pi/64)) =
#: 0.0053 mm — two orders below the 0.05 mm clearance the canonical pair is judged
#: on, so faceting cannot be mistaken for interference. 64 also keeps a boolean in
#: the low milliseconds, which is what lets the whole sanity pass stay under a
#: second.
SECTIONS = 64
#: The fit band the fixtures are judged against.
#:
#: Deliberately narrower than `FitConstraints()`'s [0.10, 0.60]: the canonical
#: pair is a 9.9 mm pin in a 10.0 mm socket, i.e. 0.05 mm radial clearance, which
#: is a plausible slip fit for a 10 mm pin but sits *below* the default minimum.
#: The band is a design input (spec.py), so the fixture states its own instead of
#: moving the pin: a `known_good` that failed for a band mismatch would prove
#: nothing about the measurement it is meant to test.
FIT_BAND = FitConstraints(
    min_clearance_mm=0.03,
    max_clearance_mm=0.60,
    min_wall_mm=1.2,
    process="fdm",
    target_fit="slip",
)
#: The same band plus a declared print orientation, `(0, 0, 0)` being "print it the
#: way it is drawn". Only the fixtures that are *about* overhang use it: with
#: `print_orientation_deg=None` (`FIT_BAND`) the overhang rule is deliberately not
#: evaluated, because a part standing in its assembled pose is not a print claim.
AS_DESIGNED_PRINT_BAND = FIT_BAND.model_copy(
    update={"print_orientation_deg": (0.0, 0.0, 0.0)}
)
#: The engagement every fixture uses unless it says otherwise. 8 mm in a 12 mm
#: blank leaves 4 mm of material under the socket floor, which is itself a wall
#: the printability verifier measures.
ENGAGE_MM = 8.0
#: `manifold3d` is the only boolean engine this project pins; the others are
#: absent or non-watertight. `engine=` is explicit so a future trimesh default
#: change cannot silently alter the fixtures.
_ENGINE = "manifold"


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def _solid_cylinder(
    radius_mm: float,
    height_mm: float,
    center: tuple[float, float, float],
) -> trimesh.Trimesh:
    """A capped cylinder along +Z, centred at `center`. Callers may not mutate it."""
    mesh = cylinder(radius=radius_mm, height=height_mm, sections=SECTIONS)
    mesh.apply_translation(center)
    return mesh


def _socket(d_hole_mm: float, feature_xy: tuple[float, float], engage_mm: float):
    """The cutting tool of a blind socket: `engage` deep, opening at the seat plane."""
    return _solid_cylinder(
        d_hole_mm / 2.0,
        engage_mm,
        (feature_xy[0], feature_xy[1], SEAT_Z_MM - engage_mm / 2.0),
    )


def _body_with_socket(
    d_hole_mm: float, feature_xy: tuple[float, float], engage_mm: float
) -> trimesh.Trimesh:
    """The blank minus one blind socket: the mother feature of a peg-hole joint."""
    return difference([box(extents=BODY_SIZE_MM), _socket(d_hole_mm, feature_xy, engage_mm)], engine=_ENGINE)


def _body_with_stepped_socket(
    wide_d_mm: float,
    narrow_d_mm: float,
    feature_xy: tuple[float, float],
    engage_mm: float,
) -> trimesh.Trimesh:
    """A blind socket whose *entry* is narrower than the chamber the pin sits in.

    The shoulder is what a seat-only check cannot see: a pin that fits the chamber
    is clean where it sits, and can never have arrived there.
    """
    half = engage_mm / 2.0
    entry = _solid_cylinder(
        narrow_d_mm / 2.0, half, (feature_xy[0], feature_xy[1], SEAT_Z_MM - half / 2.0)
    )
    chamber = _solid_cylinder(
        wide_d_mm / 2.0,
        half,
        (feature_xy[0], feature_xy[1], SEAT_Z_MM - half - half / 2.0),
    )
    return difference([box(extents=BODY_SIZE_MM), entry, chamber], engine=_ENGINE)


def _body_with_floor_pocket(
    d_hole_mm: float,
    feature_xy: tuple[float, float],
    engage_mm: float,
    pocket_size_mm: tuple[float, float, float],
    pocket_xy: tuple[float, float],
) -> trimesh.Trimesh:
    """A blind socket plus a recess cut up into the bottom face, kept apart.

    The recess is deliberately off to one side: its ceiling is horizontal, faces
    down and sits above the build plate — the canonical `unsupported_overhang` —
    and moving it away from the socket keeps it from being a thin wall as well, so
    this fixture fails the overhang check and nothing else.
    """
    pocket = box(extents=pocket_size_mm)
    pocket.apply_translation(
        (pocket_xy[0], pocket_xy[1], -BODY_SIZE_MM[2] / 2.0 + pocket_size_mm[2] / 2.0)
    )
    return difference(
        [box(extents=BODY_SIZE_MM), _socket(d_hole_mm, feature_xy, engage_mm), pocket],
        engine=_ENGINE,
    )


def _peg(
    d_peg_mm: float,
    length_mm: float,
    bottom_z_mm: float,
    feature_xy: tuple[float, float],
) -> trimesh.Trimesh:
    """The insert: one plain cylinder, seated so its lowest point is `bottom_z_mm`."""
    return _solid_cylinder(
        d_peg_mm / 2.0,
        length_mm,
        (feature_xy[0], feature_xy[1], bottom_z_mm + length_mm / 2.0),
    )


# --------------------------------------------------------------------------- #
# The spec that goes with the pair
# --------------------------------------------------------------------------- #
def _spec(
    spec_id: str,
    requirement: str,
    *,
    d_peg_mm: float,
    d_hole_mm: float,
    engage_mm: float,
    approach_mm: float,
    insert_len_mm: float,
    feature_xy: tuple[float, float],
    constraints: FitConstraints,
) -> DesignSpec:
    """A one-interface `peg_hole` spec whose numbers are the ones that were built."""
    params = PegHoleParams(
        d_peg_mm=d_peg_mm,
        d_hole_mm=d_hole_mm,
        engage_mm=engage_mm,
        approach_mm=approach_mm,
        peg_len_mm=insert_len_mm,
        feature_xy_mm=feature_xy,
    )
    return DesignSpec(
        spec_id=spec_id,
        requirement=requirement,
        parts=(
            PartSpec(id="body", role="body", size_mm=BODY_SIZE_MM),
            PartSpec(id="insert", role="insert", size_mm=(d_peg_mm, d_peg_mm, insert_len_mm)),
        ),
        interfaces=(
            InterfaceSpec(
                id="iface_0",
                kind="peg_hole",
                a="body",
                b="insert",
                params=params.model_dump(),
            ),
        ),
        constraints=constraints,
    )


def _view(spec: DesignSpec, body: trimesh.Trimesh, insert: trimesh.Trimesh) -> DesignView:
    """A view over the two solids. `root` stays empty: nothing here is on disk.

    `approach_mm` is passed through from the spec the pair was built from, the way
    a materialiser fills it from `ConnectorGeometry` (`design/api.py`): the path
    verifier reads the view, not the spec, so a fixture that left the default here
    would be testing a number nobody built. The mate axis stays at +Z, which is
    what every fixture is actually built along.
    """
    approach = float(spec.interface.validated_params().approach_mm)
    return DesignView(
        spec=spec,
        root=None,
        meshes={"body": body, "insert": insert},
        paths={},
        approach_mm=approach,
    )


# --------------------------------------------------------------------------- #
# Fixtures — the peg-hole family
# --------------------------------------------------------------------------- #
#: 9.9 mm pin in a 10.0 mm socket: 0.05 mm radial clearance, inside `FIT_BAND`,
#: bottom resting on a 4 mm-thick floor, no interference, straight path, 15 mm
#: radial wall. The one view every verifier must judge clean.
STANDARD = {"d_peg": 9.9, "d_hole": 10.0}
#: 10.2 mm pin in a 10.0 mm socket: -0.10 mm clearance, so the solids overlap by
#: pi * (5.1^2 - 5.0^2) * 8 = 25.4 mm3 over the engagement.
OVERSIZED = {"d_peg": 10.2, "d_hole": 10.0}
#: 8.0 mm pin in a 10.0 mm socket: 1.00 mm radial clearance, twice the band's top.
LOOSE = {"d_peg": 8.0, "d_hole": 10.0}


def standard_fixture() -> DesignView:
    """`known_good`: a peg/socket pair with nothing wrong in any of the five checks."""
    d_peg, d_hole = STANDARD["d_peg"], STANDARD["d_hole"]
    length = ENGAGE_MM + HANDLE_MM
    return _view(
        _spec(
            "fixture-standard",
            "a 9.9 mm pin, 0.05 mm radial clearance, 8 mm blind socket, 15 mm wall",
            d_peg_mm=d_peg,
            d_hole_mm=d_hole,
            engage_mm=ENGAGE_MM,
            approach_mm=6.0,
            insert_len_mm=length,
            feature_xy=(0.0, 0.0),
            constraints=FIT_BAND,
        ),
        _body_with_socket(d_hole, (0.0, 0.0), ENGAGE_MM),
        _peg(d_peg, length, SEAT_Z_MM - ENGAGE_MM, (0.0, 0.0)),
    )


def oversized_fixture() -> DesignView:
    """`known_bad` for interference: a pin 0.1 mm wider than the socket it must enter."""
    d_peg, d_hole = OVERSIZED["d_peg"], OVERSIZED["d_hole"]
    length = ENGAGE_MM + HANDLE_MM
    return _view(
        _spec(
            "fixture-oversized",
            "a 10.2 mm pin in a 10.0 mm socket: interference over the whole engagement",
            d_peg_mm=d_peg,
            d_hole_mm=d_hole,
            engage_mm=ENGAGE_MM,
            approach_mm=6.0,
            insert_len_mm=length,
            feature_xy=(0.0, 0.0),
            constraints=FIT_BAND,
        ),
        _body_with_socket(d_hole, (0.0, 0.0), ENGAGE_MM),
        _peg(d_peg, length, SEAT_Z_MM - ENGAGE_MM, (0.0, 0.0)),
    )


def loose_fixture() -> DesignView:
    """`known_bad` for a fit band: 1.0 mm radial clearance is a rattle, not a fit."""
    d_peg, d_hole = LOOSE["d_peg"], LOOSE["d_hole"]
    length = ENGAGE_MM + HANDLE_MM
    return _view(
        _spec(
            "fixture-loose",
            "an 8.0 mm pin in a 10.0 mm socket: 1.0 mm radial clearance",
            d_peg_mm=d_peg,
            d_hole_mm=d_hole,
            engage_mm=ENGAGE_MM,
            approach_mm=6.0,
            insert_len_mm=length,
            feature_xy=(0.0, 0.0),
            constraints=FIT_BAND,
        ),
        _body_with_socket(d_hole, (0.0, 0.0), ENGAGE_MM),
        _peg(d_peg, length, SEAT_Z_MM - ENGAGE_MM, (0.0, 0.0)),
    )


def clearance_fixture(d_peg: float, d_hole: float, *, tag: str = "custom") -> DesignView:
    """A centred pair at arbitrary diameters, for a test that needs its own numbers."""
    length = ENGAGE_MM + HANDLE_MM
    return _view(
        _spec(
            f"fixture-{tag}",
            f"a {d_peg:g} mm pin in a {d_hole:g} mm socket",
            d_peg_mm=d_peg,
            d_hole_mm=d_hole,
            engage_mm=ENGAGE_MM,
            approach_mm=6.0,
            insert_len_mm=length,
            feature_xy=(0.0, 0.0),
            constraints=FIT_BAND,
        ),
        _body_with_socket(d_hole, (0.0, 0.0), ENGAGE_MM),
        _peg(d_peg, length, SEAT_Z_MM - ENGAGE_MM, (0.0, 0.0)),
    )


def stepped_fixture(wide_d: float = 10.0, narrow_d: float = 8.0) -> DesignView:
    """`known_bad` for the insertion path that a seat-only check calls healthy.

    The socket's entry is 8 mm wide at the seat plane and the chamber behind it is
    10 mm; the pin (9 mm, 3 mm long) rests on the chamber floor, clear by 0.5 mm
    all round, so the *seated* pose is clean — no interference and no band
    violation — while every approach position either crosses the shoulder or sits
    above it. That is the design error this verifier exists for: a pin that is
    assembled and cannot have been inserted. It is also the only shape that can
    produce it, because translating a rigid body along the socket's own axis
    cannot make a socket that fits at the seat stop fitting higher up.
    """
    d_peg = 9.0
    length = 3.0
    floor = SEAT_Z_MM - ENGAGE_MM  # the bottom of the 8 mm-deep socket
    return _view(
        _spec(
            "fixture-stepped-socket",
            "a 9.0 mm pin seated in the wide chamber of a 10/8 mm stepped socket",
            d_peg_mm=d_peg,
            d_hole_mm=wide_d,
            engage_mm=ENGAGE_MM,
            approach_mm=6.0,
            insert_len_mm=length,
            feature_xy=(0.0, 0.0),
            constraints=FIT_BAND,
        ),
        _body_with_stepped_socket(wide_d, narrow_d, (0.0, 0.0), ENGAGE_MM),
        _peg(d_peg, length, floor, (0.0, 0.0)),
    )


def thin_wall_fixture(wall_mm: float = 0.6) -> DesignView:
    """`known_bad` for the wall: a socket drilled close enough to break the side.

    The socket is the same 10 mm one; only its position moves, so the *only*
    difference from `standard_fixture()` is 0.6 mm of material between the socket
    and the body's +X face (against the 1.2 mm the process asks for).
    """
    d_peg, d_hole = STANDARD["d_peg"], STANDARD["d_hole"]
    length = ENGAGE_MM + HANDLE_MM
    feature_x = BODY_SIZE_MM[0] / 2.0 - d_hole / 2.0 - wall_mm
    return _view(
        _spec(
            "fixture-thin-wall",
            f"a 10 mm socket {wall_mm:g} mm from the body's +X face",
            d_peg_mm=d_peg,
            d_hole_mm=d_hole,
            engage_mm=ENGAGE_MM,
            approach_mm=6.0,
            insert_len_mm=length,
            feature_xy=(feature_x, 0.0),
            constraints=FIT_BAND,
        ),
        _body_with_socket(d_hole, (feature_x, 0.0), ENGAGE_MM),
        _peg(d_peg, length, SEAT_Z_MM - ENGAGE_MM, (feature_x, 0.0)),
    )


def pocket_fixture(
    size_mm: tuple[float, float, float] = (10.0, 10.0, 3.0),
    at_xy: tuple[float, float] = (20.0, 0.0),
) -> DesignView:
    """`known_bad` for the overhang: a floor recess whose ceiling prints on air.

    The recess is 10 x 10 x 3 at `(20, 0)`, so its ceiling is a 100 mm2 downward
    plane at `z = -3`, 3 mm above the plate and over an empty cavity, and it is
    far enough from the socket (10 mm of material between them) to leave the wall
    check green. The peg and socket are the standard ones, so exactly one code
    fires. The spec declares the as-designed pose as the print orientation,
    because without a declared orientation no overhang claim exists at all.
    """
    d_peg, d_hole = STANDARD["d_peg"], STANDARD["d_hole"]
    length = ENGAGE_MM + HANDLE_MM
    return _view(
        _spec(
            "fixture-floor-pocket",
            "a part with a 10x10x3 mm recess cut into its bottom face",
            d_peg_mm=d_peg,
            d_hole_mm=d_hole,
            engage_mm=ENGAGE_MM,
            approach_mm=6.0,
            insert_len_mm=length,
            feature_xy=(0.0, 0.0),
            constraints=AS_DESIGNED_PRINT_BAND,
        ),
        _body_with_floor_pocket(d_hole, (0.0, 0.0), ENGAGE_MM, size_mm, at_xy),
        _peg(d_peg, length, SEAT_Z_MM - ENGAGE_MM, (0.0, 0.0)),
    )


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
# The boolean geometry above is deterministic, so it is built once per process and
# shared between the fixtures and the tests: a `sanity()` pass runs four verifiers
# over the same handful of pairs, and rebuilding a body per call would make the
# gate slower than the checks it gates. `lru_cache` on the mesh builders (not on
# the views) keeps every view a fresh object, so a verifier that mutates its input
# cannot leak into the next one; the meshes themselves are read-only by contract —
# a verifier that needs to move a solid must `.copy()` it first.
_body_with_socket = functools.lru_cache(maxsize=None)(_body_with_socket)
_body_with_stepped_socket = functools.lru_cache(maxsize=None)(_body_with_stepped_socket)
_body_with_floor_pocket = functools.lru_cache(maxsize=None)(_body_with_floor_pocket)
_peg = functools.lru_cache(maxsize=None)(_peg)
