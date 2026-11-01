"""The two-level IVR the loop is allowed to trust.

`docs/pah-dev-doc-v1.md` §4.8 fixes the definition and the three hard rules:

    IVR = Σ_(i,j) ∈ E_check  Vol(V_i ∩ V_j)  /  Σ_i Vol(V_i)

`E_check` is the caller's pair list. A designed interference — a press fit, an
intentional snap overlap — is excluded by **not listing its pair**, which is the
only exclusion mechanism §4.8 permits; and no pair is ever counted twice
(`_checked_pairs` folds `(a, b)` and `(b, a)` into one entry).

Where each hard rule lives in this file:

1. **A sampled number is never called IVR.** `ivr_probe` returns
   `level="probe"`, and `IVRResult.field_name` is `"penetration_probe"` for it
   and `"ivr_exact"` only for `ivr_exact`. §4.8 rule 1 exists because surface
   sampling cannot see a fully nested part or a thin-wall pass-through, so an
   estimate must not be filed under the measurement's name.
2. **An unmeasurable pair is `unknown`, never 0.** A non-manifold or open shell,
   a part with no usable volume, or a boolean engine error puts the pair into
   `unknown_pairs` with a reason, and `value` becomes `None`. The pair's volume
   is *not* silently added as zero (§4.8 rule 2). One unmeasurable pair makes
   the whole ratio unknown: a partial numerator over a full denominator is a
   wrong number, and "how many samples are not exactly verified" is a number M0
   has to report, so it must be derivable by counting `value is None`.
3. **The three report numbers are attributes.** `value`, `unknown_pairs` and
   `level` need no parsing of `detail`, so a report can always print
   `success_ivr_validated` and `ivr_unknown_count` separately (§4.8 rule 3).

Geometry is **metres** at the world frame (`docs/internal/adapter-contract.md`
§A: `unit_mm = 1000`, one data unit is 1 m). IVR is dimensionless, which is why
the same design built in millimetres gives the same ratio; the volumes in
`detail` are in the cube of whatever length unit the caller used.

Both entry points are pure: no I/O, no globals, no wall-clock, and the sampled
one is reproducible from `seed` alone.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import trimesh

__all__ = ["IVRResult", "ivr_exact", "ivr_probe"]

#: `(a, b, aabb_overlap_extent) -> (volume | None, method, reason)`; `None` is
#: "cannot measure this pair", and the driver turns it into an unknown pair.
PairVolume = Callable[
    [trimesh.Trimesh, trimesh.Trimesh, np.ndarray], tuple[float | None, str, str]
]

#: dev doc §4.8 names the two output fields explicitly; `IVRResult.field_name`
#: is the only place those strings are produced, so a probe cannot be reported
#: under the exact field's name by accident.
_FIELD = {"probe": "penetration_probe", "exact": "ivr_exact"}

#: Fixed ray direction for the parity test. Irrational-ish components so a ray
#: almost never grazes an edge or vertex, where parity is undefined. Fixed
#: rather than random on purpose: `seed` controls the sample points alone, so
#: two runs with the same `seed` measure the same points in the same way.
_RAY_DIRECTION = np.array([0.5432, 0.6123, 0.5741], dtype=np.float64)

#: Default sample count. The cost of a point-in-mesh test here is one ray cast
#: per point (~1 ms at 20k faces, `trimesh`'s pure-Python rtree kernel), so 512
#: points keeps a pair in the low seconds and gives ±2 % on a half-overlap
#: fraction. Raise it when a probe result is going to be argued about.
_DEFAULT_SAMPLES = 512


@dataclass(frozen=True)
class IVRResult:
    """One IVR reading. `value is None` means unknown, never "no overlap"."""

    value: float | None
    level: Literal["probe", "exact"]
    pairs: int
    unknown_pairs: list[str]
    detail: dict

    @property
    def field_name(self) -> str:
        """`penetration_probe` for a probe, `ivr_exact` for an exact (§4.8)."""
        return _FIELD[self.level]


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def ivr_probe(
    parts: list[tuple[str, trimesh.Trimesh]],
    *,
    pairs: list[tuple[str, str]] | None = None,
    n_samples: int = _DEFAULT_SAMPLES,
    seed: int = 0,
    method: Literal["auto", "occupancy", "surface"] = "auto",
) -> IVRResult:
    """Fast IVR estimate by bounding-box cull then point sampling.

    Rules this keeps:

    - `level` is `"probe"` and `field_name` is `"penetration_probe"`; no sample
      is ever reported as `ivr_exact` (§4.8 rule 1).
    - A pair the sampler cannot decide goes to `unknown_pairs` and `value`
      becomes `None` — in particular `method="surface"` reports unknown for a
      pair whose surfaces never enter each other, because that is equally
      consistent with "disjoint" and "one part entirely inside the other"
      (§4.8 rule 2). It never reports 0 for such a pair.
    - Two bounding boxes that do not overlap give a pair volume of exactly 0:
      that is a proof, not an estimate, so the pair is known, not unknown.
    - Same `seed` and same inputs reproduce the same number; nothing else here
      is stateful.

    `method`: `"occupancy"` samples points in the two bounding boxes'
    intersection and counts the ones inside both solids (preferred; unbiased,
    exact for axis-aligned boxes); `"surface"` samples the two surfaces and
    scales the bounding-box overlap by the fraction of samples that lie inside
    the other solid (a detector with a known low bias — it reads about half the
    truth for a fully nested part); `"auto"` tries occupancy and falls back per
    pair.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples!r}")
    if method not in ("auto", "occupancy", "surface"):
        raise ValueError(f"unknown method {method!r}; use auto, occupancy or surface")
    rng = np.random.default_rng(seed)

    def pair_volume(
        a: trimesh.Trimesh, b: trimesh.Trimesh, extent: np.ndarray
    ) -> tuple[float | None, str, str]:
        return _probe_pair_volume(
            a, b, extent, n_samples=n_samples, rng=rng, method=method
        )
    return _measure(
        parts,
        pairs,
        level="probe",
        pair_volume=pair_volume,
        detail={"samples": int(n_samples), "seed": int(seed), "method": method},
        sample_resolution=n_samples,
    )


def ivr_exact(
    parts: list[tuple[str, trimesh.Trimesh]],
    *,
    pairs: list[tuple[str, str]] | None = None,
) -> IVRResult:
    """Exact IVR from solid boolean intersections, for verdicts and reports.

    Rules this keeps:

    - `level` is `"exact"`, `field_name` is `"ivr_exact"`.
    - A pair `manifold3d` cannot intersect — open shell, non-manifold mesh,
      engine status other than `NoError`, non-finite volume — goes to
      `unknown_pairs` with the reason, contributes nothing to the numerator, and
      makes `value` `None`. It is never recorded as 0 (§4.8 rule 2), because
      "unknown" filed as "no penetration" is exactly the lie the loop learns.
    - A pair whose bounding boxes miss is exactly 0 without a boolean: the
      AABB is a superset of the solid, so this is a proof, not a shortcut.
    - Booleans go straight to `manifold3d`, not through `trimesh.boolean`:
      that wrapper pre-checks `is_watertight`, which is always False on this
      corpus's closed-but-non-manifold parts, so it would reject real geometry
      (`docs/internal/adapter-contract.md` §E6).
    """
    return _measure(
        parts,
        pairs,
        level="exact",
        pair_volume=_exact_pair_volume,
        detail={},
    )


# --------------------------------------------------------------------------- #
# the shared driver — the three rules live here, once
# --------------------------------------------------------------------------- #
def _measure(
    parts: list[tuple[str, trimesh.Trimesh]],
    pairs: list[tuple[str, str]] | None,
    *,
    level: Literal["probe", "exact"],
    pair_volume: PairVolume,
    detail: dict,
    sample_resolution: int | None = None,
) -> IVRResult:
    """Normalise pair volumes into one `IVRResult`.

    `pair_volume(a, b, extent)` returns `(volume | None, method, reason)`; the
    driver owns the bounding-box cull, the denominator, the unknown bookkeeping
    and the final ratio, so the two levels cannot drift apart on any of them.
    """
    meshes = _index(parts)
    ids = list(meshes)
    checked = _checked_pairs(ids, pairs)
    in_play = list(dict.fromkeys(pid for pair in checked for pid in pair))
    volumes = {pid: _solid_volume(meshes[pid]) for pid in in_play}
    denominator = float(sum(v for v in volumes.values() if v is not None))

    records: list[dict] = []
    unknown: list[str] = []
    numerator = 0.0

    for a, b in checked:
        bad = [pid for pid in (a, b) if volumes[pid] is None]
        extent = None if bad else _overlap_extent(meshes[a], meshes[b])
        bbox_volume = None if extent is None else float(np.prod(extent))
        if bad:
            volume, method, reason = None, "", "no_solid_volume:" + ",".join(bad)
        elif extent is None:
            # The AABBs miss, so the solids cannot meet: a proof, hence known 0.
            volume, method, reason = 0.0, "bbox", ""
        else:
            volume, method, reason = pair_volume(meshes[a], meshes[b], extent)

        records.append(
            {
                "a": a,
                "b": b,
                "bbox_volume": bbox_volume,
                # Smallest volume a sample set could have detected in this pair.
                "resolution": (
                    bbox_volume / sample_resolution
                    if bbox_volume is not None and sample_resolution is not None
                    else None
                ),
                "status": "unknown" if volume is None else "known",
                "volume": None if volume is None else float(volume),
                "method": method,
                "reason": reason,
            }
        )
        if volume is None:
            unknown.append(f"{a}-{b}: {reason}")
        else:
            numerator += volume

    if not checked:
        # Nothing was compared, so the numerator is an empty sum: 0.0 is the
        # honest value and `pairs == 0` says no ratio was formed. `None` would
        # conflate "nothing to check" with "could not check".
        value: float | None = 0.0
    elif unknown or denominator <= 0.0:
        # `denominator <= 0` cannot happen while `unknown` is empty — every part
        # in play then contributed a finite positive volume — but the ratio is
        # left undefined rather than crashing a batch run on that assumption.
        value = None
    else:
        value = numerator / denominator

    detail = {
        **detail,
        "field": _FIELD[level],
        "numerator": numerator,
        "denominator": denominator,
        "checked_pairs": len(checked),
        "known_pairs": len(checked) - len(unknown),
        "unknown_pairs": len(unknown),
        "bbox_culled": sum(1 for r in records if r["method"] == "bbox"),
        "part_volumes": volumes,
        "pairs": records,
    }
    return IVRResult(
        value=value, level=level, pairs=len(checked), unknown_pairs=unknown, detail=detail
    )


# --------------------------------------------------------------------------- #
# pair selection and bookkeeping
# --------------------------------------------------------------------------- #
def _index(
    parts: list[tuple[str, trimesh.Trimesh]],
) -> dict[str, trimesh.Trimesh]:
    """`id -> mesh`, refusing duplicates: two ids for one mesh is a caller bug."""
    meshes: dict[str, trimesh.Trimesh] = {}
    for item in parts:
        try:
            pid, mesh = item
        except (TypeError, ValueError):
            raise TypeError(
                f"parts must be (id, trimesh.Trimesh) pairs, got {item!r}"
            ) from None
        pid = str(pid)
        if pid in meshes:
            raise ValueError(f"duplicate part id {pid!r}")
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"part {pid!r} is {type(mesh).__name__}, not a Trimesh")
        meshes[pid] = mesh
    return meshes


def _checked_pairs(
    ids: list[str], pairs: list[tuple[str, str]] | None
) -> list[tuple[str, str]]:
    """The `E_check` set, in caller order, each unordered pair once.

    `pairs=None` means every unordered pair of the given parts, which is the
    only defensible default: an unreported pair would silently shrink the
    numerator while the denominator stayed whole.
    """
    if pairs is None:
        return [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]

    known = set(ids)
    seen: set[frozenset[str]] = set()
    checked: list[tuple[str, str]] = []
    for entry in pairs:
        try:
            a, b = (str(pid) for pid in entry)
        except (TypeError, ValueError):
            raise TypeError(f"pairs entries must be two part ids, got {entry!r}") from None
        if a == b:
            raise ValueError(f"pair ({a!r}, {b!r}) names the same part twice")
        missing = [pid for pid in (a, b) if pid not in known]
        if missing:
            raise ValueError(
                f"pair ({a!r}, {b!r}) names unknown part(s) {missing}; "
                f"parts are {', '.join(ids) or 'none'}"
            )
        key = frozenset((a, b))
        if key in seen:
            # §4.8: no pair may be summed twice, so (a, b) and (b, a) are one.
            continue
        seen.add(key)
        checked.append((a, b))
    return checked


def _solid_volume(mesh: trimesh.Trimesh) -> float | None:
    """`|vol(V_i)|`, or None when the mesh has no volume worth dividing by."""
    if len(mesh.faces) == 0:
        return None
    try:
        volume = abs(float(mesh.volume))
    except (TypeError, ValueError):
        return None
    return volume if math.isfinite(volume) and volume > 0.0 else None


def _overlap_extent(a: trimesh.Trimesh, b: trimesh.Trimesh) -> np.ndarray | None:
    """Side lengths of the two AABBs' intersection, or None when they miss.

    An AABB contains its solid, so an empty AABB intersection proves an empty
    solid intersection. That makes the culled pairs exactly 0, not "assumed 0".
    """
    lo = np.maximum(a.bounds[0], b.bounds[0])
    hi = np.minimum(a.bounds[1], b.bounds[1])
    extent = hi - lo
    return extent if bool(np.all(extent > 0.0)) else None


# --------------------------------------------------------------------------- #
# probe: point sampling
# --------------------------------------------------------------------------- #
def _probe_pair_volume(
    a: trimesh.Trimesh,
    b: trimesh.Trimesh,
    extent: np.ndarray,
    *,
    n_samples: int,
    rng: np.random.Generator,
    method: str,
) -> tuple[float | None, str, str]:
    """One pair's estimated intersection volume, or why it has none."""
    if method in ("auto", "occupancy"):
        try:
            return _occupancy_volume(a, b, extent, n_samples=n_samples, rng=rng)
        except Exception as exc:  # noqa: BLE001 - a broken pair must not stop the batch
            if method == "occupancy":
                return None, "occupancy", f"occupancy_failed:{type(exc).__name__}:{exc}"
    try:
        return _surface_volume(a, b, extent, n_samples=n_samples, rng=rng)
    except Exception as exc:  # noqa: BLE001 - same reason
        return None, "surface", f"sampling_failed:{type(exc).__name__}:{exc}"


def _occupancy_volume(
    a: trimesh.Trimesh,
    b: trimesh.Trimesh,
    extent: np.ndarray,
    *,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[float, str, str]:
    """Monte-Carlo occupancy of the AABB intersection (§4.8's preferred mode).

    Unbiased by construction: the sample box is a superset of the intersection,
    so `(#inside both) / n × V(box)` estimates `Vol(V_a ∩ V_b)`. The second mesh
    is only tested on the points that survived the first test, which is what
    keeps a shallow penetration cheap.
    """
    lo = np.maximum(a.bounds[0], b.bounds[0])
    points = lo + rng.random((n_samples, 3)) * extent
    inside_a = _inside(a, points)
    if not inside_a.any():
        return 0.0, "occupancy", ""
    inside_b = _inside(b, points[inside_a])
    fraction = float(np.count_nonzero(inside_b)) / float(n_samples)
    return fraction * float(np.prod(extent)), "occupancy", ""


def _surface_volume(
    a: trimesh.Trimesh,
    b: trimesh.Trimesh,
    extent: np.ndarray,
    *,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[float | None, str, str]:
    """Surface-sampling fallback: a detector, deliberately not an estimator.

    A sample point on one surface that lies strictly inside the other solid
    proves the intersection has positive volume; the fraction of such points
    times the AABB-overlap volume is only a rough magnitude. When neither
    surface enters the other there is **no** reading to give: the pair is
    either disjoint or fully nested (the inner part's shell can sit inside the
    outer one with every point of the outer shell outside), and §4.8 forbids
    turning that ambiguity into 0.
    """
    on_a = _sample_surface(a, n_samples, rng)
    on_b = _sample_surface(b, n_samples, rng)
    evidence = int(np.count_nonzero(_inside(b, on_a))) + int(
        np.count_nonzero(_inside(a, on_b))
    )
    if evidence == 0:
        return None, "surface", "surface_no_evidence(disjoint-or-nested)"
    fraction = evidence / float(2 * n_samples)
    return fraction * float(np.prod(extent)), "surface", ""


def _inside(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Which points are strictly inside `mesh`, by ray parity.

    One fixed-direction ray per point. `Trimesh.contains` is not used because it
    casts both ways and re-casts "broken" points in random directions: 2-4× the
    rays for a test whose ambiguity (a point exactly on the surface) has measure
    zero for sampled points.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    directions = np.broadcast_to(_RAY_DIRECTION, points.shape)
    _, index_ray = mesh.ray.intersects_id(points, directions, multiple_hits=True)
    crossings = np.bincount(index_ray, minlength=len(points))
    return (crossings % 2) == 1


def _sample_surface(
    mesh: trimesh.Trimesh, count: int, rng: np.random.Generator
) -> np.ndarray:
    """`count` area-weighted random points on `mesh`'s surface.

    Hand-rolled rather than `trimesh.sample.sample_surface` so the only source
    of randomness is `rng`, i.e. `seed` alone reproduces the points.
    """
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    total = float(areas.sum())
    if total <= 0.0:
        raise ValueError("mesh has no surface area")
    index = rng.choice(len(areas), size=count, p=areas / total)
    triangles = np.asarray(mesh.triangles)[index]
    u = rng.random((count, 1))
    v = rng.random((count, 1))
    flip = (u + v) > 1.0
    u = np.where(flip, 1.0 - u, u)
    v = np.where(flip, 1.0 - v, v)
    return triangles[:, 0] + u * (triangles[:, 1] - triangles[:, 0]) + v * (
        triangles[:, 2] - triangles[:, 0]
    )


# --------------------------------------------------------------------------- #
# exact: solid boolean
# --------------------------------------------------------------------------- #
def _exact_pair_volume(
    a: trimesh.Trimesh, b: trimesh.Trimesh, extent: np.ndarray
) -> tuple[float | None, str, str]:
    """`Vol(V_a ∩ V_b)` from `manifold3d`, or why the engine cannot say.

    `extent` is unused: the driver already culled the pairs whose AABBs miss.
    """
    for tag, mesh in (("a", a), ("b", b)):
        defect = _surface_defect(mesh)
        if defect:
            return None, "", f"{tag}:{defect}"

    manifold = _manifold_module()
    if manifold is None:
        return None, "", "no_manifold3d"
    try:
        solids = [_to_manifold(mesh) for mesh in (a, b)]
        out = manifold.Manifold.batch_boolean(solids, manifold.OpType.Intersect)
        status = str(out.status())
        if "NoError" not in status:
            # manifold3d returns a *valid-looking* empty solid on a non-manifold
            # input; trusting its volume here would file "unknown" as 0.
            return None, "manifold", f"boolean:{status}"
        volume = float(out.volume())
    except Exception as exc:  # noqa: BLE001 - one bad pair must not kill the batch
        return None, "manifold", f"boolean_failed:{type(exc).__name__}:{exc}"
    if not math.isfinite(volume):
        return None, "manifold", "boolean:non_finite_volume"
    return max(volume, 0.0), "manifold", ""


def _surface_defect(mesh: trimesh.Trimesh) -> str:
    """Why `mesh` is not a closed surface; "" when it is.

    A boundary edge (exactly one incident face) means an open shell, whose
    divergence-theorem "volume" is meaningless. Over-full edges (3+ faces) are
    deliberately *not* a defect: the corpus's parts have 0.01-0.1 % of them and
    `manifold3d` copes by splitting vertices, while `is_watertight` — the
    pre-check `trimesh.boolean` uses — is False for 25 of 26 real parts
    (`docs/internal/adapter-contract.md` §E6).
    """
    if len(mesh.faces) == 0:
        return "empty_mesh"
    counts = np.bincount(mesh.edges_unique_inverse)
    boundary = int(np.count_nonzero(counts == 1))
    return f"open_surface(boundary_edges={boundary})" if boundary else ""


def _to_manifold(mesh: trimesh.Trimesh):
    """`trimesh.Trimesh -> manifold3d.Manifold`, same recipe as `connector.py`."""
    manifold = _manifold_module()
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    return manifold.Manifold(manifold.Mesh(vertices, faces))


@functools.lru_cache(maxsize=1)
def _manifold_module():
    """`manifold3d`, or None when the optional boolean back end is missing."""
    try:
        import manifold3d  # noqa: PLC0415 - optional dependency, probed once
    except ImportError:
        return None
    return manifold3d
