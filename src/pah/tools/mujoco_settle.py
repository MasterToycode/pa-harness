"""Real dynamics for one seated joint: does the insert *stay* where the design put it?

Why this module exists
----------------------
`docs/pah-dev-doc-v1.md` §6 makes the point the design plane is built around:
**static non-interference is not dynamical stability.** A peg that fits inside a
hole to the last micrometre can still be a joint that falls out under gravity, and
an interference-only check will call it perfect. So the last, most expensive rung
of the verifier cascade puts the pair in a rigid-body simulator, drops gravity on
it, and measures what happens.

The one thing this module may not do is lie about what it measured. MuJoCo is a
convex-primitive engine; a body with a hole in it is not one convex solid. So the
cavity is *reconstructed* as a ring of convex boxes, the reconstruction is
measured from the built body mesh rather than assumed from the spec, and every
approximation it makes is written into the returned `model` dict so a certificate
can carry it. What the model cannot represent, it refuses (`SettleError`), and a
refusal is reported as `family="environment"` — never as a pass and never as a
physical failure (`docs/pah-dev-doc-v1.md` §4.5).

The model, and every approximation in it
----------------------------------------
1. **Body static, insert free.** The body's material is welded to the world; the
   insert is a free rigid body (a 6-DoF `freejoint`). Only the *interface region*
   is simulated: the boxes are the measured bore walls plus a support slab, not the
   whole body. The rest of the body cannot touch the insert in the seated pose —
   that is the interference verifier's claim to test, not this one's.
2. **Units.** The design plane is millimetres; MuJoCo is metres. Exactly one
   conversion exists (`MM_PER_M`) and it is applied once, when the local frame is
   built. Every length in `SettleResult` is millimetres again.
3. **The cavity is an N-gon of N convex boxes** (default N=24). Each box's inner
   face passes through one measured wall point, so the facet *midpoint* sits on
   the real wall. The polygon's vertices therefore sit at `R / cos(pi/N)` —
   **1-cos(pi/N) = 0.86% further from the axis than the real wall at N=24**. For
   the 5.15 mm fixture bore that is 0.044 mm of extra room at the facet joints. It
   biases `drift` *upward* (against passing) and cannot manufacture penetration,
   because every facet is a supporting hyperplane of a convex cavity.
4. **The measured wall is the tightest section of the bore**, not an average: the
   radial samples are taken at several heights across the engaged span and the
   minimum per direction is used. A bore that is wider in the middle is therefore
   simulated as narrow, which again biases toward "unstable".
5. **Insert shape.** A round cross-section is simulated as MuJoCo's exact
   `cylinder` (radius = the measured maximum radial extent). A non-round
   cross-section is simulated as its convex hull mesh, which is exactly what
   MuJoCo does to any mesh geom — so a *non-convex* insert is refused rather than
   silently simulated as its hull, because its undercut (the retention feature of
   a snap hook) would be averaged away and the joint would look weaker than it is.
   A convex insert whose cross-section varies along the axis (a stepped peg, a
   chamfer) is over-approximated by its hull: the modelled insert is at least as
   big as the real one, so penetration errs strict and drift errs strict.
6. **Contact.** `solref="0.004 1"`, `solimp="0.95 0.99 0.001"`, elliptic cone,
   `implicitfast`, PLA-like density (1240 kg/m^3) and a PLA-like friction
   coefficient (0.5). The solver is compliant: a 1.4 g insert at rest sinks
   ~0.004 mm into its support. That number is the floor of the `penetration_mm`
   measurement, and the reason the penetration tolerance is 0.03 mm rather than
   zero.
7. **Gravity is real** (9.81 m/s^2, world -Z) and so is the simulation: 0.8 s at
   2e-4 s, fixed length, fixed step, no randomness. Two runs of the same view
   produce identical numbers — measured bit-for-bit on `drift_mm`,
   `penetration_mm` and `axial_mm` — which is what lets the loop's cycle detection
   and a paper's reproducibility claim both mean something. The one exception is
   `wall_budget_s`: a machine slow enough to hit that ceiling gets a
   `physics_error` instead of a number, and that is deliberate — a timed-out check
   is a fact about the machine, and reporting it as an environment failure is
   better than reporting a partial settle as physics.

What is *not* modelled, stated so a reader does not over-read a pass
--------------------------------------------------------------------
* **Insertion.** Pressing the insert in through a real approach path is
  `insertion_path`'s job; this module starts at the seated pose.
* **The rest of the body.** A head that would land on the body's outer face is
  only modelled where the support probe finds material below the insert; the
  annulus around the bore is *not* part of the box ring.
* **Material behaviour** — no plasticity, no press-fit interference stress, no
  thermal effects. A press fit therefore reads as "squeezed" `dynamic_penetration`
  rather than as a retained joint.
* **The support probe.** It casts one ray along gravity from the insert's centre;
  a body that supports the insert eccentrically is read from that single ray.

The measurements that leave this module
---------------------------------------
`SettleResult.findings` names whichever of these happened; all of them are
observed on every step of the run:

* `drift` — the lateral (perpendicular to the mate axis) distance of the insert's
  centre of mass from the seat, maximum over the run. The insert is released from
  the worst lateral stance its own designed clearance allows, so this number is
  bounded below by that clearance and above by however far a loose or toppling
  joint roams. It is the joint's *play under gravity*.
* `penetration` — the largest contact penetration reported by the solver
  (`-contact.dist`), i.e. how far the insert and the body's wall had to overlap to
  satisfy the contact constraints.
* `pull_out` / `drop_through` — the insert left the engaged span of the cavity
  moving against / along the insertion direction. A through hole with nothing
  under it lands here; so does an unretained insert that gravity simply takes.
  The current `peg_hole` build is a **blind** socket, so this path is exercised by
  fixtures rather than by the demo case (see `docs/internal/physics-notes.md`).
* `tilt` — the insert ended more than `tilt_deg` off the mate axis without a
  matching lateral drift (a wedge, not a seat). Secondary: a topple usually shows
  up in `drift` first.
"""

from __future__ import annotations

import math
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh

__all__ = [
    "MM_PER_M",
    "SettleError",
    "SettleLimits",
    "SettleResult",
    "SettleTolerances",
    "engine_version",
    "frame",
    "settle",
]

#: The one place millimetres become metres. Getting this wrong is not a small bug:
#: a 10 mm peg simulated as 10 m weighs a tonne.
MM_PER_M = 1e-3


class SettleError(RuntimeError):
    """The simulation could not run, or the geometry is outside the model's domain.

    Always an *environment* finding (`family="environment"`, `physics_error`): the
    state of the design is unknown, which is neither a pass nor a failure. The
    `reason` is a short token a report can group on; `detail` is the sentence a
    human reads.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class SettleTolerances:
    """When a measured motion counts as a finding. Every number has a reason.

    `drift_mm`: the fit band the design plane accepts tops out at
    `constraints.max_clearance_mm`, and a slip fit is allowed to move by its own
    clearance. `physics_settle` derives this tolerance at 1.7x that ceiling, so
    physics never contradicts a fit the constraint band already accepts — it fires
    only when the joint's play exceeds anything the band permits. (At the default
    0.6 mm band that is 1.02 mm, which is also 10% of the 10 mm peg the demo case
    uses: where a peg stops being *located* in its hole.)

    `penetration_mm`: a third of `constraints.min_clearance_mm`, so a design that
    keeps any legal clearance cannot reach it, floored at 0.01 mm because the
    contact solver's own compliance is measured at 0.004 mm for the fixture insert
    (1.4 g under 9.81 m/s^2) and a smaller tolerance would report the solver's
    softness as a design fault.

    `pull_out_mm = 0.5`: the insert counts as having left its seat when it travels
    0.5 mm against the insertion direction — the same order as the fit band's upper
    clearance. Below that it is still where the design put it.

    `tilt_deg = 15`: a seated insert 15 degrees off the mate axis is wedged, not
    located. Kept as a *secondary* trigger because a topple almost always shows up
    in `drift` first; it only fires on its own when the centre of mass barely moves.

    The defaults below are the *fallback* when a caller has no `FitConstraints` to
    derive from; `physics_settle` always derives from the spec.
    """

    drift_mm: float = 1.02
    penetration_mm: float = 0.03
    pull_out_mm: float = 0.5
    tilt_deg: float = 15.0


@dataclass(frozen=True)
class SettleLimits:
    """Budget and discretisation. Fixed for a run; a caller may shrink the time.

    `seconds = 0.8` at `timestep_s = 2e-4` is 4000 steps. It is long enough for an
    unsupported insert to be 3.1 m below its seat (it never gets that far in this
    geometry, but the test has to be able to see it happen) and short enough that
    the measured cost is ~0.06-0.10 s of CPU per joint. The step is 1/20 of the
    stiffest contact's time constant, which is where the compliant contact model
    stops oscillating.
    """

    seconds: float = 0.8
    timestep_s: float = 2e-4
    #: Convex boxes in the cavity ring. 24 keeps the polygon error at 0.86%.
    n_wall: int = 24
    #: Radial directions sampled per height, and heights sampled per engagement.
    n_angles: int = 24
    n_heights: int = 5
    max_steps: int = 20000
    #: Wall-clock ceiling. A model that blows this up is refused as an environment
    #: failure instead of stalling the cascade; the step count alone bounds the cost
    #: for every mesh measured so far (~0.1 s), so this only catches pathologies.
    wall_budget_s: float = 4.5


@dataclass(frozen=True)
class SettleResult:
    """What the settle observed, in millimetres and degrees.

    `findings` is empty when the joint held. Each name maps to exactly one
    certificate code in `physics_settle`; `model` carries the approximation facts
    so no reader has to trust a docstring they cannot see.
    """

    findings: tuple[str, ...]
    drift_mm: float
    axial_mm: float
    pull_out_mm: float
    slip_in_mm: float
    penetration_mm: float
    tilt_deg: float
    clearance_mm: float
    wall_radius_min_mm: float
    wall_radius_max_mm: float
    wall_open_angles: int
    engaged_span_mm: float
    supported: bool
    seat_gap_mm: float | None
    insert_shape: str
    insert_radius_mm: float
    insertion_sign: int
    steps: int
    simulated_s: float
    wall_s: float
    model: dict

    @property
    def seated(self) -> bool:
        return not self.findings

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# The simulator is optional at import time
# --------------------------------------------------------------------------- #
def _mujoco():
    """Import MuJoCo at call time so a broken install is a finding, not a no-import.

    A module that imports `mujoco` at top level cannot be loaded on a machine
    without it, which would take the *verifier* down with the dependency. Importing
    here means `pah design sanity` can report `physics_settle` as environment-broken
    and keep the rest of the cascade running.
    """
    try:
        import mujoco  # noqa: PLC0415
    except Exception as error:  # noqa: BLE001 - any import failure is the environment
        raise SettleError("mujoco_unavailable", f"{type(error).__name__}: {error}") from error
    return mujoco


def engine_version() -> str:
    """`mujoco-3.17.0`, or `mujoco-unavailable`: what produced a certificate."""
    try:
        return f"mujoco-{_mujoco().__version__}"
    except SettleError:
        return "mujoco-unavailable"


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #
def _unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(array)
    if not np.isfinite(norm) or norm < 1e-12:
        raise SettleError("bad_axis", f"mate axis {vector!r} has no direction")
    return array / norm


def frame(axis=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(u, e1, e2)` — the mate axis and a right-handed perpendicular pair.

    The perpendicular pair is fixed deterministically (from +X, or +Y when the
    axis is close to X) so two runs of the same view build the same boxes and the
    same release direction. A random or eigendecomposition-derived basis would make
    the measured drift depend on the machine, which is exactly the kind of
    irreproducibility a paper cannot afford.
    """
    u = _unit(axis)
    seed = np.array([0.0, 1.0, 0.0]) if abs(u[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = seed - u * float(seed @ u)
    norm = np.linalg.norm(e1)
    if norm < 1e-9:  # the seed was parallel after all; fall back to the other basis
        seed = np.array([0.0, 0.0, 1.0])
        e1 = seed - u * float(seed @ u)
        norm = np.linalg.norm(e1)
    e1 = e1 / norm
    return u, e1, np.cross(u, e1)


def _quat_align_z(normal) -> tuple[float, float, float, float]:
    """A `(w, x, y, z)` quaternion that rotates local +Z onto `normal`."""
    n = _unit(normal)
    if n[2] > 1.0 - 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    if n[2] < -1.0 + 1e-9:
        return (0.0, 1.0, 0.0, 0.0)
    axis = np.cross(np.array([0.0, 0.0, 1.0]), n)
    axis = axis / np.linalg.norm(axis)
    half = math.acos(float(n[2])) / 2.0
    return (math.cos(half), float(math.sin(half) * axis[0]),
            float(math.sin(half) * axis[1]), float(math.sin(half) * axis[2]))


def _quat_from_matrix(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """`(w, x, y, z)` for a 3x3 rotation whose columns are the body axes in local.

    Shepperd's method: pick the largest of the four squared components so the
    division never loses the small terms. Needed because a cavity facet's normal is
    the *surface* normal, not the ray direction, so it cannot be expressed as a
    rotation about the mate axis alone.
    """
    r = np.asarray(rotation, dtype=float)
    trace = float(r[0, 0] + r[1, 1] + r[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * scale,
            (r[2, 1] - r[1, 2]) / scale,
            (r[0, 2] - r[2, 0]) / scale,
            (r[1, 0] - r[0, 1]) / scale,
        )
    if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        scale = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        return (
            (r[2, 1] - r[1, 2]) / scale,
            0.25 * scale,
            (r[0, 1] + r[1, 0]) / scale,
            (r[0, 2] + r[2, 0]) / scale,
        )
    if r[1, 1] > r[2, 2]:
        scale = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        return (
            (r[0, 2] - r[2, 0]) / scale,
            (r[0, 1] + r[1, 0]) / scale,
            0.25 * scale,
            (r[1, 2] + r[2, 1]) / scale,
        )
    scale = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
    return (
        (r[1, 0] - r[0, 1]) / scale,
        (r[0, 2] + r[2, 0]) / scale,
        (r[1, 2] + r[2, 1]) / scale,
        0.25 * scale,
    )


# --------------------------------------------------------------------------- #
# Geometry checks: what this model can and cannot look at
# --------------------------------------------------------------------------- #
def _validate(name: str, mesh, *, need_watertight: bool) -> None:
    if not isinstance(mesh, trimesh.Trimesh):
        raise SettleError(f"{name}_not_a_mesh", f"got {type(mesh).__name__}")
    if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        raise SettleError(
            f"{name}_empty", f"{len(mesh.vertices)} vertices / {len(mesh.faces)} faces"
        )
    if need_watertight and not mesh.is_watertight:
        # A non-watertight body makes ray parity meaningless: the wall radius would
        # be measured through a hole in the model, not through the hole in the part.
        raise SettleError(
            f"{name}_not_watertight",
            "ray casting needs the boolean-clean solid the design plane claims to build",
        )


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain over the (x, y) projection, counter-clockwise."""
    pts = np.unique(np.round(np.asarray(points, dtype=float)[:, :2], 9), axis=0)
    if len(pts) < 3:
        return pts

    def cross(origin, a, b) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    lower: list = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list = []
    for point in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.array(lower[:-1] + upper[:-1])


def _polygon_radius(hull: np.ndarray, n_angles: int) -> np.ndarray:
    """Distance from the origin to the hull boundary along each sampled direction.

    Vertices alone are not enough: a square's four corners all sit at the same
    radius, so a vertex-binned maximum would call a square bar a cylinder. The
    radius has to be read off the *faces*, which is what this does.
    """
    angles = np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False)
    radii = np.zeros(n_angles)
    if len(hull) < 3:
        return radii
    edges = np.roll(hull, -1, axis=0)
    for index, angle in enumerate(angles):
        direction = np.array([math.cos(angle), math.sin(angle)])
        best = 0.0
        for start, end in zip(hull, edges):
            edge = end - start
            det = -direction[0] * edge[1] + edge[0] * direction[1]
            if abs(det) < 1e-12:
                continue
            along = (start[0] * (-edge[1]) + edge[0] * start[1]) / det
            across = (direction[0] * start[1] - direction[1] * start[0]) / det
            if along > 0.0 and -1e-9 <= across <= 1.0 + 1e-9:
                best = max(best, float(along))
        radii[index] = best
    return radii


def _convex_parts(
    vertices_local: np.ndarray,
    faces_local: np.ndarray,
    *,
    max_hulls: int = 12,
    threshold: float = 0.15,
    preprocess_resolution: int = 24,
    resolution: int = 600,
    decimate_to: int = 2000,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Split a non-convex solid into convex pieces with CoACD, or `None`.

    MuJoCo collides any mesh geom as its convex hull, so a concave part cannot be
    simulated faithfully as one geom. The first version of this tool therefore refused
    non-convex inserts ("not evaluated") — honest but useless for generated parts,
    which are almost never convex (the measured tube was 2.4x its hull).

    Decomposing into convex hulls is the standard answer: the body gets one geom per
    piece, so the undercut survives instead of being averaged away. CoACD is an
    optional dependency (`pip install coacd`); when it is absent this returns `None`
    and the caller keeps the old refusal, which is the boundary the report names.

    **Cost is the reason for the small settings.** Measured on a 20 k-face generated
    insert: default resolution 2000 / 48 hulls took **86 s**; decimating to 2 k faces
    and asking for at most 12 hulls at resolution 600 takes **~17 s** and still returns
    8 hulls that keep the tube's cavity. The physics tier runs once per design (the
    loop caches verification), so 17 s is affordable where 86 s would not be.

    Deterministic for a fixed input: `seed=0`, no time-dependent heuristic.
    """
    try:
        import coacd  # noqa: PLC0415 - optional, and only this path needs it
    except Exception:  # noqa: BLE001 - absent package is a boundary, not a fault
        return None
    try:
        mesh_local = trimesh.Trimesh(
            vertices=np.asarray(vertices_local, dtype=float),
            faces=np.asarray(faces_local, dtype=np.int64),
            process=False,
        )
        if len(mesh_local.faces) > decimate_to:
            try:
                mesh_local = mesh_local.simplify_quadric_decimation(face_count=decimate_to)
            except Exception:  # noqa: BLE001 - use the full mesh rather than fail
                pass
        mesh = coacd.Mesh(
            np.ascontiguousarray(mesh_local.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh_local.faces, dtype=np.int32),
        )
        parts = coacd.run_coacd(
            mesh,
            threshold=threshold,
            max_convex_hull=max_hulls,
            preprocess_resolution=preprocess_resolution,
            resolution=resolution,
            seed=0,
        )
    except Exception:  # noqa: BLE001 - a failed decomposition is the same boundary
        return None
    cleaned: list[tuple[np.ndarray, np.ndarray]] = []
    for part_vertices, part_faces in parts:
        vertices = np.asarray(part_vertices, dtype=float)
        faces = np.asarray(part_faces, dtype=np.int32)
        if len(vertices) >= 4 and len(faces) >= 4:
            cleaned.append((vertices, faces))
    return cleaned or None


def _insert_shape(
    insert: trimesh.Trimesh,
    local: np.ndarray,
    local_vertices: np.ndarray,
    local_faces: np.ndarray,
    n_angles: int,
) -> dict:
    """Cylinder when the cross-section is round, else convex hull or CoACD parts.

    Every approximation is named in `model` so a certificate can say what it measured:
    the cylinder is exact for the *maximum* radius (it errs large), the hull is what
    MuJoCo does to any mesh geom, and the decomposed parts preserve concavity at the
    cost of tiny seams between hulls.
    """
    radii = _polygon_radius(_convex_hull_2d(local), n_angles)
    finite = radii[np.isfinite(radii) & (radii > 0.0)]
    if not len(finite):
        raise SettleError("insert_degenerate", "no radial extent in the mate-axis plane")
    rounding = float(finite.max() - finite.min()) / float(finite.max())
    radius = float(finite.max())
    half_len = float(np.abs(local[:, 2]).max())
    # The cylinder shortcut used to be decided on *roundness alone*, which meant a
    # hollow tube or a stepped round part was replaced by a solid cylinder before any
    # volume or convexity check ran. The settle then reported stability for a different
    # object, and `joint_unstable` / `dynamic_penetration` were filed as physical facts
    # (audit P1-2, §7.1). Two extra conditions make it honest:
    #
    #   * closed, positive volume;
    #   * the solid actually fills the cylinder it is being replaced with (within 5%).
    #     A hollow tube is 60-90% short of that and now falls through to the hull /
    #     CoACD branch, where the model says what it is.
    solid_volume = float(insert.volume) if insert.is_watertight else 0.0
    cylinder_volume = math.pi * radius**2 * (2.0 * half_len)
    fill = solid_volume / cylinder_volume if cylinder_volume > 0.0 else 0.0
    if rounding <= 0.02 and radius > 0.0 and insert.is_watertight and fill >= 0.95:
        return {
            "kind": "cylinder",
            "radius_mm": radius,
            "half_len_mm": half_len,
            "rounding_error": rounding,
            # How much of the nominal cylinder the real solid occupies. 1.0 is a true
            # cylinder; 0.55 is a tube, and reporting that number is what lets a reader
            # see the shortcut was justified instead of taken on trust.
            "solid_fraction": fill,
            "approximation_error": 1.0 - fill,
        }
    hull = insert.convex_hull
    hull_volume = float(hull.volume)
    hull_over_solid = (hull_volume / solid_volume) if solid_volume > 0.0 else None
    if solid_volume > 0.0 and hull_volume > solid_volume * 1.02:
        parts = _convex_parts(local_vertices, local_faces)
        if parts is None:
            raise SettleError(
                "non_convex_insert",
                f"convex hull is {hull_volume / solid_volume:.3f}x the solid and CoACD is "
                "not installed, so MuJoCo could only collide the hull and would average "
                "away the undercut. Install `coacd` to simulate concave inserts.",
            )
        return {
            "kind": "convex_parts",
            "radius_mm": radius,
            "half_len_mm": half_len,
            "rounding_error": rounding,
            "hull_over_solid": hull_over_solid,
            "solid_fraction": fill,
            "approximation_error": 1.0 - fill if fill <= 1.0 else 0.0,
            "parts": parts,
            "decomposition": f"coacd: {len(parts)} convex hulls",
        }
    return {
        "kind": "mesh_hull",
        "radius_mm": radius,
        "half_len_mm": half_len,
        "rounding_error": rounding,
        "hull_over_solid": hull_over_solid,
        "solid_fraction": fill,
        "approximation_error": max(1.0 - fill, 0.0),
        "hull_faces": hull.faces,
    }


# --------------------------------------------------------------------------- #
# Cavity measurement
# --------------------------------------------------------------------------- #
def _measure_walls(
    body: trimesh.Trimesh,
    seat_mm: np.ndarray,
    basis: np.ndarray,
    heights_mm: np.ndarray,
    n_angles: int,
) -> list[tuple[float, np.ndarray, np.ndarray] | None]:
    """`(radius, point_local, outward_normal_local)` per direction, or `None`.

    The origin is the insert's own axis, not the hole's nominal centre: the cavity
    the insert actually sits in is the one that matters, and if a builder put the
    insert off-centre this measures the asymmetry instead of hiding it.

    The facet normal is the **surface** normal of the triangle the ray hit, not the
    ray direction. The difference matters wherever the wall is not perpendicular to
    the ray: a slot's floor hit by a 60-degree ray has a horizontal normal, and a
    ray-perpendicular facet there would be an inclined plane that lets the insert
    sink 2 mm into the material the floor is made of.

    `None` means "no material in this direction over any sampled height" — a
    genuinely open side (a slot's mouth, a dovetail groove's top). Those directions
    get no wall box; they are not an error.
    """
    angles = np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False)
    local = np.stack([np.cos(angles), np.sin(angles), np.zeros(n_angles)], axis=1)
    directions = local @ basis.T  # world
    origins = np.array(
        [seat_mm + basis @ np.array([0.0, 0.0, height]) for height in heights_mm]
    )
    origins = np.repeat(origins, n_angles, axis=0)
    rays = np.tile(directions, (len(heights_mm), 1))
    try:
        hits, index_ray, index_tri = body.ray.intersects_location(
            origins, rays, multiple_hits=False
        )
    except Exception as error:  # noqa: BLE001 - a broken ray engine is the environment
        raise SettleError("raycast_failed", f"{type(error).__name__}: {error}") from error
    radii = np.full(len(origins), np.nan)
    points = np.full((len(origins), 3), np.nan)
    normals = np.full((len(origins), 3), np.nan)
    if len(hits):
        radii[index_ray] = np.linalg.norm(hits - origins[index_ray], axis=1)
        points[index_ray] = hits
        normals[index_ray] = body.face_normals[index_tri]
    radii = radii.reshape(len(heights_mm), n_angles)
    points = points.reshape(len(heights_mm), n_angles, 3)
    normals = normals.reshape(len(heights_mm), n_angles, 3)

    samples: list[tuple[float, np.ndarray, np.ndarray] | None] = []
    for index in range(n_angles):
        column = radii[:, index]
        if not np.isfinite(column).any():
            samples.append(None)
            continue
        row = int(np.nanargmin(column))
        normal = normals[row, index]
        if float(normal @ directions[index]) < 0.0:
            normal = -normal  # point it into the material, i.e. away from the cavity
        samples.append(
            (
                float(radii[row, index]),
                (points[row, index] - seat_mm) @ basis,
                normal @ basis,
            )
        )
    return samples


def _support(
    body: trimesh.Trimesh,
    seat_mm: np.ndarray,
    gravity_dir_world: np.ndarray,
    insert_lowest_mm: float,
    *,
    max_probe_mm: float = 50.0,
    seated_gap_mm: float = 2.0,
) -> tuple[bool, float | None, np.ndarray | None]:
    """Is there body material under the insert, how far below, and where?

    One ray down the gravity direction from the insert's centre. The returned gap is
    measured to the insert's own lowest point (so a correctly seated insert reads 0)
    and the returned point is where a support slab belongs. No hit — or a hit so far
    away that the insert would have left the cavity on the way down — means the
    insert hangs in air: a through hole, and the caller lets it fall rather than
    inventing a floor it cannot see.
    """
    rays = np.asarray(gravity_dir_world, dtype=float).reshape(1, 3)
    hit, index, _ = body.ray.intersects_location(
        seat_mm.reshape(1, 3), rays, multiple_hits=False
    )
    if not len(hit):
        return False, None, None
    distance = float(np.linalg.norm(hit[index[0]] - seat_mm))
    if distance > max_probe_mm:
        return False, None, None
    gap = distance + insert_lowest_mm  # insert_lowest_mm is <= 0: extent below centre
    return (gap <= seated_gap_mm), float(gap), np.asarray(hit[index[0]], dtype=float)


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def _wall_boxes(
    samples: list, span_mm: tuple[float, float]
) -> list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, ...]]]:
    """`(size, centre, quat)` per cavity facet, in the local frame.

    Each box is a slab whose inner face contains one measured wall point and whose
    orientation is that surface's own normal, so the union of the slabs is the
    measured cavity boundary and not a faceted approximation of the *rays*. The
    facet's width is taken from the neighbouring samples' spacing along the tangent,
    with a 5% overlap so corners do not leak; a two-sided slab is used whenever the
    wall is a single flat face (a slot's floor) so the pieces tile it.

    `span_mm` is the engaged span along the mate axis: every slab is as long as the
    engagement, which is the region the insert can touch.
    """
    span = span_mm[1] - span_mm[0]
    if span <= 0.0:
        raise SettleError("empty_cavity_span", f"engaged span {span_mm!r}")
    up = np.array([0.0, 0.0, 1.0])  # the local mate axis
    centre_axis = 0.5 * (span_mm[0] + span_mm[1])
    count = len(samples)
    present = [index for index, sample in enumerate(samples) if sample is not None]
    if not present:
        raise SettleError("no_walls", "every measured direction was open")
    boxes = []
    for position, index in enumerate(present):
        radius, point, normal = samples[index]
        outward = np.asarray(normal, dtype=float)
        outward = outward - up * float(outward @ up)
        norm = np.linalg.norm(outward)
        if norm < 1e-9:  # a facet facing the mate axis cannot bound the insert sideways
            continue
        outward = outward / norm
        tangent = np.cross(up, outward)
        tangent = tangent / np.linalg.norm(tangent)
        # Facet width from the neighbours' spacing along this tangent.
        previous = samples[present[position - 1]] if len(present) > 1 else None
        following = samples[present[(position + 1) % len(present)]] if len(present) > 1 else None
        if previous is not None and following is not None and len(present) > 1:
            reach = abs(float((following[1] - previous[1]) @ tangent)) * 0.5
        else:
            reach = radius * math.sin(math.pi / count)
        half_width = max(reach * 1.05, 1e-3)
        thickness = max(4.0, 0.6 * float(radius))
        centre = np.asarray(point, dtype=float) + outward * (thickness / 2.0)
        rotation = np.column_stack([outward, tangent, up])
        boxes.append(
            (
                (thickness / 2.0, half_width, span / 2.0),
                (float(centre[0]), float(centre[1]), centre_axis),
                _quat_from_matrix(rotation),
            )
        )
    if not boxes:
        raise SettleError("no_walls", "every measured facet faced the mate axis")
    return boxes


def _xml(
    *,
    limits: SettleLimits,
    gravity_local: np.ndarray,
    boxes: list,
    support_point_local: np.ndarray | None,
    gravity_dir_local: np.ndarray,
    insert: dict,
    release_mm: tuple[float, float],
    friction: float,
    density: float,
    mesh_paths: list[Path] | None,
) -> str:
    """The MJCF for one settle. Local frame: +Z is the mate axis, origin the seat."""
    lines = [
        '<mujoco model="pah_settle">',
        f'  <option timestep="{limits.timestep_s:.9g}" integrator="implicitfast"'
        f' cone="elliptic" gravity="{gravity_local[0]:.9f} {gravity_local[1]:.9f}'
        f' {gravity_local[2]:.9f}"/>',
        f'  <default><geom solref="0.004 1" solimp="0.95 0.99 0.001"'
        f' friction="{friction:.4f} 0.005 0.0001"/></default>',
    ]
    if mesh_paths:
        lines.append("  <asset>")
        for index, path in enumerate(mesh_paths):
            lines.append(f'    <mesh name="{_part_name(insert, index)}" file="{path.as_posix()}"/>')
        lines.append("  </asset>")
    lines.append("  <worldbody>")
    for index, (size, position, quat) in enumerate(boxes):
        lines.append(
            f'    <geom name="wall{index}" type="box"'
            f' size="{size[0] * MM_PER_M:.9f} {size[1] * MM_PER_M:.9f} {size[2] * MM_PER_M:.9f}"'
            f' pos="{position[0] * MM_PER_M:.9f} {position[1] * MM_PER_M:.9f}'
            f' {position[2] * MM_PER_M:.9f}"'
            f' quat="{quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f}"'
            ' rgba="0.62 0.62 0.66 1"/>'
        )
    if support_point_local is not None:
        quat = _quat_align_z(-np.asarray(gravity_dir_local, dtype=float))
        half = 150.0  # mm: wider than any fixture body
        thickness = 2.0
        centre = np.asarray(support_point_local, dtype=float) + gravity_dir_local * thickness
        lines.append(
            f'    <geom name="support" type="box"'
            f' size="{half * MM_PER_M:.9f} {half * MM_PER_M:.9f} {thickness * MM_PER_M:.9f}"'
            f' pos="{centre[0] * MM_PER_M:.9f} {centre[1] * MM_PER_M:.9f}'
            f' {centre[2] * MM_PER_M:.9f}"'
            f' quat="{quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f}"'
            ' rgba="0.75 0.75 0.75 1"/>'
        )
    lines += [
        f'    <body name="insert" pos="{release_mm[0] * MM_PER_M:.9f}'
        f' {release_mm[1] * MM_PER_M:.9f} 0.000000000">',
        '      <freejoint name="free"/>',
    ]
    if insert["kind"] == "cylinder":
        lines.append(
            f'      <geom name="insert" type="cylinder"'
            f' size="{insert["radius_mm"] * MM_PER_M:.9f} {insert["half_len_mm"] * MM_PER_M:.9f}"'
            f' density="{density:.3f}" rgba="0.90 0.45 0.20 1"/>'
        )
    elif insert["kind"] == "convex_parts":
        # One rigid body, several convex geoms: this is what keeps a concave part's
        # real shape in the contact model instead of its average.
        for index in range(len(insert["parts"])):
            lines.append(
                f'      <geom name="insert_part_{index}" type="mesh"'
                f' mesh="{_part_name(insert, index)}"'
                f' density="{density:.3f}" rgba="0.90 0.45 0.20 1"/>'
            )
    else:
        lines.append(
            f'      <geom name="insert" type="mesh" mesh="{_part_name(insert, 0)}"'
            f' density="{density:.3f}" rgba="0.90 0.45 0.20 1"/>'
        )
    lines += ["    </body>", "  </worldbody>", "</mujoco>"]
    return "\n".join(lines)


def _part_name(insert: dict, index: int) -> str:
    """The MJCF mesh name for one insert geom (single hull vs decomposed part)."""
    return f"insert_part_{index}" if insert["kind"] == "convex_parts" else "insert_hull"


def _write_obj(
    mesh_dir: Path, name: str, vertices_local_mm: np.ndarray, faces: np.ndarray
) -> Path:
    """One insert geom as an OBJ in the local frame, written in metres."""
    path = mesh_dir / f"{name}.obj"
    vertices = vertices_local_mm * MM_PER_M
    body = [f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in vertices]
    body += [f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}" for a, b, c in faces]
    path.write_text("\n".join(body), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# The one public entry point
# --------------------------------------------------------------------------- #
def settle(
    body: trimesh.Trimesh,
    insert: trimesh.Trimesh,
    *,
    axis=(0.0, 0.0, 1.0),
    clearance_mm: float = 0.15,
    tolerances: SettleTolerances | None = None,
    limits: SettleLimits | None = None,
    gravity_mps2: float = 9.81,
    friction: float = 0.5,
    density_kgm3: float = 1240.0,
) -> SettleResult:
    """Drop gravity on one seated joint and measure what it does.

    `body` and `insert` are world-frame millimetre meshes in the **seated** pose
    (that is what a `ConnectorBuilder` returns and what `DesignView` hands a
    verifier). `clearance_mm` is the *designed* radial clearance: the insert is
    released from that offset, which is the worst stance the fit itself permits.

    Raises `SettleError` when the simulation cannot run or the geometry is outside
    the model's domain; returns a `SettleResult` when it ran. A result with no
    `findings` is the only thing that means "the joint held".
    """
    tol = tolerances or SettleTolerances()
    lim = limits or SettleLimits()
    if lim.seconds <= 0 or lim.timestep_s <= 0:
        raise SettleError("bad_limits", f"seconds={lim.seconds} timestep={lim.timestep_s}")
    mujoco = _mujoco()

    _validate("body", body, need_watertight=True)
    _validate("insert", insert, need_watertight=True)

    u, e1, e2 = frame(axis)
    basis = np.column_stack([e1, e2, u])  # columns: local -> world
    try:
        seat = np.asarray(insert.center_mass, dtype=float)
    except Exception as error:  # noqa: BLE001 - a mesh without a volume has no seat
        raise SettleError("insert_no_volume", f"{type(error).__name__}: {error}") from error

    local_insert = (insert.vertices - seat) @ basis
    local_body = (body.vertices - seat) @ basis
    insert_lo, insert_hi = float(local_insert[:, 2].min()), float(local_insert[:, 2].max())
    body_lo, body_hi = float(local_body[:, 2].min()), float(local_body[:, 2].max())
    engaged = (max(insert_lo, body_lo), min(insert_hi, body_hi))
    if engaged[1] - engaged[0] <= 1e-6:
        raise SettleError(
            "not_engaged",
            f"insert spans [{insert_lo:.3f}, {insert_hi:.3f}] mm along the mate axis, the "
            f"body [{body_lo:.3f}, {body_hi:.3f}]: they do not overlap, so there is no "
            "seated joint to settle",
        )
    # Shrink the sampling span so no ray starts on a face (top face / cavity floor).
    margin = 0.1 * (engaged[1] - engaged[0])
    heights = np.linspace(engaged[0] + margin, engaged[1] - margin, max(2, lim.n_heights))

    wall = _measure_walls(body, seat, basis, heights, lim.n_angles)
    measured = [sample for sample in wall if sample is not None]
    if not measured:
        raise SettleError(
            "no_cavity_measured",
            "no body material found around the insert axis over the engaged span — the "
            "insert does not sit in a measurable cavity",
        )
    wall_min = float(min(sample[0] for sample in measured))
    wall_max = float(max(sample[0] for sample in measured))
    shape = _insert_shape(insert, local_insert, local_insert, insert.faces, lim.n_angles)
    if wall_min < 0.5 * shape["radius_mm"]:
        raise SettleError(
            "cavity_implausible",
            f"measured wall {wall_min:.3f} mm is less than half the insert radius "
            f"{shape['radius_mm']:.3f} mm: the view is not a seated pair",
        )

    gravity_local = basis.T @ np.array([0.0, 0.0, -gravity_mps2])
    gravity_dir_local = gravity_local / np.linalg.norm(gravity_local)
    gravity_dir_world = basis @ gravity_dir_local
    insert_lowest = float((local_insert @ gravity_dir_local).min())
    supported, seat_gap, support_point_world = _support(
        body, seat, gravity_dir_world, insert_lowest
    )
    support_point_local = (
        None if support_point_world is None else (support_point_world - seat) @ basis
    )

    # The insertion direction is read off the geometry: the engaged part sits on one
    # side of the insert's centre, and the insert travelled *towards* that side.
    engaged_mid = 0.5 * (engaged[0] + engaged[1])
    insertion_sign = int(np.sign(engaged_mid)) if abs(engaged_mid) > 1e-9 else 1

    boxes = _wall_boxes(wall, engaged)
    release = (max(float(clearance_mm), 0.0), 0.0)

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pah-settle-") as tmp:
        mesh_paths: list[Path] = []
        if shape["kind"] == "mesh_hull":
            hull = insert.convex_hull
            mesh_paths = [
                _write_obj(Path(tmp), "insert_hull", (hull.vertices - seat) @ basis, shape["hull_faces"])
            ]
        elif shape["kind"] == "convex_parts":
            mesh_paths = [
                _write_obj(Path(tmp), f"insert_part_{index}", vertices, faces)
                for index, (vertices, faces) in enumerate(shape["parts"])
            ]
        xml = _xml(
            limits=lim,
            gravity_local=gravity_local,
            boxes=boxes,
            support_point_local=support_point_local,
            gravity_dir_local=gravity_dir_local,
            insert=shape,
            release_mm=release,
            friction=friction,
            density=density_kgm3,
            mesh_paths=mesh_paths,
        )
        try:
            model = mujoco.MjModel.from_xml_string(xml)
        except Exception as error:  # noqa: BLE001 - will not compile == the environment
            raise SettleError(
                "model_compile_failed", f"{type(error).__name__}: {error}"
            ) from error
        data = mujoco.MjData(model)
        steps = min(lim.max_steps, max(1, int(round(lim.seconds / lim.timestep_s))))
        drift_mm = 0.0
        penetration_mm = 0.0
        for index in range(steps):
            try:
                mujoco.mj_step(model, data)
            except Exception as error:  # noqa: BLE001
                raise SettleError("step_failed", f"{type(error).__name__}: {error}") from error
            position = np.asarray(data.qpos[:3], dtype=float)  # metres, local frame
            if not np.all(np.isfinite(position)):
                raise SettleError("diverged", f"non-finite position at step {index + 1}")
            lateral = float(np.hypot(position[0], position[1]) / MM_PER_M)
            if lateral > drift_mm:
                drift_mm = lateral
            worst = 0.0
            for contact in range(data.ncon):
                worst = max(worst, -float(data.contact[contact].dist))
            penetration_mm = max(penetration_mm, worst / MM_PER_M)
            if time.perf_counter() - started > lim.wall_budget_s:
                raise SettleError(
                    "wall_budget_exceeded",
                    f"{lim.wall_budget_s:.1f} s spent before {steps} steps completed",
                )
        final = np.asarray(data.qpos[:3], dtype=float) / MM_PER_M
        quat = np.asarray(data.qpos[3:7], dtype=float)
    wall_s = time.perf_counter() - started

    if not np.all(np.isfinite(final)):
        raise SettleError("diverged", "non-finite final position")
    simulated_s = steps * lim.timestep_s
    tilt_deg = math.degrees(
        2.0 * math.atan2(float(np.linalg.norm(quat[1:])), abs(float(quat[0])))
    )
    axial_mm = float(final[2])
    along_insertion = axial_mm * insertion_sign
    pull_out_mm = max(0.0, -along_insertion)
    slip_in_mm = max(0.0, along_insertion)

    # Did the insert leave the cavity entirely? Compare its final axial span with the
    # engaged span; a few millimetres of travel inside the bore is a slip, not an escape.
    final_lo, final_hi = insert_lo + axial_mm, insert_hi + axial_mm
    overlap = min(final_hi, engaged[1]) - max(final_lo, engaged[0])
    escaped = overlap <= 0.0

    findings: list[str] = []
    if escaped:
        findings.append("drop_through" if along_insertion > 0 else "pull_out")
    elif pull_out_mm > tol.pull_out_mm:
        findings.append("pull_out")
    if penetration_mm > tol.penetration_mm:
        findings.append("penetration")
    if not escaped and drift_mm > tol.drift_mm:
        findings.append("drift")
    if not escaped and tilt_deg > tol.tilt_deg and "drift" not in findings:
        findings.append("tilt")

    model_facts = {
        "frame": "local +Z = mate axis, origin = seated centre of mass",
        "mate_axis_world": [float(value) for value in u],
        "units": "mm in, m inside (x1e-3), mm out",
        "cavity": f"{len(boxes)}-gon of convex boxes from {lim.n_angles} radial samples",
        "cavity_open_directions": int(sum(1 for sample in wall if sample is None)),
        "polygon_vertex_extra_frac": 1.0 / math.cos(math.pi / lim.n_angles) - 1.0,
        "polygon_vertex_extra_mm": (wall_min / math.cos(math.pi / lim.n_angles)) - wall_min,
        "wall_sampling": {"angles": lim.n_angles, "heights": int(len(heights))},
        "insert_shape": shape["kind"],
        "insert_decomposition": shape.get("decomposition"),
        "insert_geoms": len(shape["parts"]) if shape["kind"] == "convex_parts" else 1,
        "insert_rounding_error_frac": round(shape["rounding_error"], 6),
        "hull_over_solid": shape.get("hull_over_solid"),
        "release_offset_mm": release[0],
        "timestep_s": lim.timestep_s,
        "simulated_s": simulated_s,
        "contact": {
            "solref": "0.004 1",
            "solimp": "0.95 0.99 0.001",
            "friction": friction,
            "density_kgm3": density_kgm3,
        },
        "gravity_mps2": gravity_mps2,
        "insertion_sign": insertion_sign,
        "support": "slab" if supported else "none",
        "tolerances": {
            "drift_mm": tol.drift_mm,
            "penetration_mm": tol.penetration_mm,
            "pull_out_mm": tol.pull_out_mm,
            "tilt_deg": tol.tilt_deg,
        },
    }
    return SettleResult(
        findings=tuple(findings),
        drift_mm=drift_mm,
        axial_mm=axial_mm,
        pull_out_mm=pull_out_mm,
        slip_in_mm=slip_in_mm,
        penetration_mm=penetration_mm,
        tilt_deg=tilt_deg,
        clearance_mm=float(clearance_mm),
        wall_radius_min_mm=wall_min,
        wall_radius_max_mm=wall_max,
        wall_open_angles=int(sum(1 for sample in wall if sample is None)),
        engaged_span_mm=float(engaged[1] - engaged[0]),
        supported=bool(supported),
        seat_gap_mm=seat_gap,
        insert_shape=shape["kind"],
        insert_radius_mm=float(shape["radius_mm"]),
        insertion_sign=insertion_sign,
        steps=steps,
        simulated_s=simulated_s,
        wall_s=wall_s,
        model=model_facts,
    )
