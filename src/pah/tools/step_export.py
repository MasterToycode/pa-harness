"""Mixed geometry: free-form parts stay meshes, the mating features become B-Rep STEP.

Why "mixed" and not "convert the mesh"
-------------------------------------
The audit (P2-5) asks for a *minimum* hybrid: a generated insert is a free-form solid
that no reliable kernel will reproduce as a parametric feature tree, and pretending
otherwise would add a new failure source (non-manifold lofts, silent tessellation) to
a pipeline whose value is that its failures are honest. What a shop actually needs
from CAD is the **fit**: a hole with a diameter, a peg with a diameter, a dovetail
with an angle. Those are numbers the spec already owns (`design/spec.py`), so this
module builds exactly those features as B-Rep solids and exports them next to — not
instead of — the printable mesh.

What the STEP file contains
---------------------------
* **One envelope proxy per parametric host part**: a box of the part's `size_mm` in
  its world pose, with the interfaces' negative features cut out. It is a *proxy*,
  not the part — wall thickness, ribs and fillets live in the mesh only.
* **One positive feature solid per interface** in its world position: a cylinder for
  `peg_hole`'s peg, a hook plus beam for `snap_fit`, a trapezoidal prism plus handle
  for `dovetail`.
* **For a decomposed spec** (parts carry a `source` blank hash) there is no envelope
  proxy: turning that mesh into B-Rep is exactly what this module refuses to do. The
  negative side (the socket) is then exported as a **void solid**, so the hole's
  diameter is still in the file and still measurable; `StepResult.bodies` says which
  solids are parts and which are features.
* **Nothing else.** No free-form surface appears in the STEP; the mesh remains the
  document of record for shape, and STL/3MF stay the printing artefacts.

C5: the read-back is the check
------------------------------
A STEP file that *says* the hole is 10.0 mm proves nothing — the parameters wrote it.
So after exporting, `measure_step` imports the file back, measures the radii of its
cylindrical faces and compares each measured diameter with the spec value within
0.01 mm. A failure is a `problems` entry and `ok: false`, never a warning. The STL side
is not measured here; it is a 64-gon inscribed in the same nominal circle, and
`dimension_check["stl_step_bound"]` reports the radial shortfall that tessellation
implies (≈6 µm on a 10 mm feature) so the two products are compared on a *stated*
basis rather than a guessed one.

What this route still does not give you
---------------------------------------
* **No feature tree, no GD&T.** The STEP holds solids, not a CAD history: a downstream
  engineer cannot re-edit a chamfer, only faces. Tolerances are in the manifest's
  `nominal_dims_mm`, not annotated on the geometry (AP214 carries shape, not `±0.05`).
* **The envelope proxy is not the part**, and the insert's true free-form shape is
  only in the STL. A shop quoting the insert must read the mesh.
* **Non-cylindrical fits are not checked numerically.** A slot width or a dovetail's
  flank angle would need face-level measurement this module does not have; those
  features are exported with their nominal parameters and are reported as
  `not_checked`, never as verified.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from pah.design.spec import DesignSpec, PartSpec

__all__ = [
    "DIMENSION_TOLERANCE_MM",
    "StepExportError",
    "StepResult",
    "export_features",
    "measure_step",
]

#: How far the STEP's measured fit dimensions may be from the spec's numbers. One
#: constant, one place; the audit (C5) names this value.
DIMENSION_TOLERANCE_MM = 0.01

#: Cylinder sides the mesh builders use (`design/connectors/peg_hole.py:SECTIONS`),
#: imported lazily and used only to *bound* the mesh's distance from the nominal
#: STEP circle. A fallback keeps this module importable if that constant moves.
_FALLBACK_SECTIONS = 64


class StepExportError(RuntimeError):
    """STEP export cannot be attempted: build123d missing, or an unusable request."""


@dataclass(frozen=True)
class StepResult:
    """The outcome of one export: what was written, what it contains, what is wrong.

    `features` and `bodies` are plain dictionaries because they travel into the print
    manifest unchanged; a second mapping layer between this and the manifest would be
    a second place for the two to disagree. `features` is keyed per interface (the
    audit's unit of a fit), `bodies` per exported solid that is not a feature.
    """

    path: str | None
    ok: bool
    features: tuple[dict, ...] = ()
    bodies: tuple[dict, ...] = ()
    problems: tuple[str, ...] = ()
    dimension_check: dict = field(
        default_factory=lambda: {
            "tolerance_mm": DIMENSION_TOLERANCE_MM,
            "ok": False,
            "checked": [],
        }
    )

    def to_manifest(self) -> dict:
        """The `engineering_step` block of the print manifest."""
        return {
            "ok": self.ok,
            "path": self.path,
            "features": list(self.features),
            "bodies": list(self.bodies),
            "problems": list(self.problems),
            "dimension_check": dict(self.dimension_check),
        }


# --------------------------------------------------------------------------- #
# The kernel, imported only when a STEP is actually requested
# --------------------------------------------------------------------------- #
def _build123d() -> dict[str, Any]:
    """The build123d names this module uses, or a named refusal."""
    try:
        from build123d import (  # noqa: PLC0415 - importing the kernel is the slow part
            Align,
            Box,
            Compound,
            Cylinder,
            Plane,
            Polygon,
            Pos,
            export_step,
            extrude,
            import_step,
        )
    except Exception as error:  # noqa: BLE001 - a missing kernel is a capability gap
        raise StepExportError(
            "build123d/OCP is not importable, so a STEP cannot be written; install "
            "build123d (pip install build123d) or export STL/3MF only. Refusing to "
            "emit a STEP-shaped file that is not one."
        ) from error
    return {
        "Align": Align,
        "Box": Box,
        "Compound": Compound,
        "Cylinder": Cylinder,
        "Plane": Plane,
        "Polygon": Polygon,
        "Pos": Pos,
        "export_step": export_step,
        "extrude": extrude,
        "import_step": import_step,
    }


def _frames() -> Any:
    """`pah.design.connectors`'s frame helpers, imported lazily for the same reason."""
    from pah.design import connectors  # noqa: PLC0415

    return connectors


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def export_features(
    spec: DesignSpec,
    interfaces: Any = None,
    out_path: str | Path | None = None,
    *,
    tolerance_mm: float = DIMENSION_TOLERANCE_MM,
) -> StepResult:
    """Build `spec`'s B-Rep fit features and write them as one STEP file.

    `interfaces` is the per-interface facts mapping a materialiser records
    (`MaterializedDesign.interfaces`: `{"mate_axis": [...], "approach_mm": ...}`). It
    is optional: the placement of every feature is derivable from the spec through the
    same `design/connectors` frames the mesh builder used, which keeps `pah export`
    usable on a run directory that no longer has its materialiser. When given, its
    `mate_axis` is cross-checked against the derived one — a disagreement is a
    design-plane bug, reported as a problem, and the *builder's* axis wins because
    that is the geometry the STL was built from.

    Failures are returned, not raised, once the kernel is importable: the caller
    (`pah export --step`) must be able to write a manifest that says what went wrong.
    Only "cannot attempt at all" — missing build123d, missing `out_path` — raises.
    """
    if out_path is None:
        raise StepExportError(
            "export_features needs an out_path; a STEP with no file name is not an export"
        )
    target = Path(out_path)
    _build123d()  # named refusal before any output directory is touched
    if not spec.interfaces:
        return StepResult(
            path=None, ok=False, problems=("the spec has no interfaces to export",)
        )

    problems: list[str] = []
    try:
        solids, features, bodies, expected, problems = _build_scene(
            spec, interfaces, problems, []
        )
    except StepExportError:
        raise
    except Exception as error:  # noqa: BLE001 - a kernel failure is a result, not a crash
        return StepResult(
            path=None,
            ok=False,
            problems=(*problems, f"the B-Rep build failed: {type(error).__name__}: {error}"),
            dimension_check={"tolerance_mm": tolerance_mm, "ok": False, "checked": []},
        )

    if not solids:
        return StepResult(
            path=None,
            ok=False,
            features=tuple(_public(feature) for feature in features),
            bodies=tuple(bodies),
            problems=(*problems, "no feature solids were produced"),
            dimension_check={"tolerance_mm": tolerance_mm, "ok": False, "checked": []},
        )

    write_problem = _write_step(solids, target)
    if write_problem is not None:
        return StepResult(
            path=None,
            ok=False,
            features=tuple(_public(feature) for feature in features),
            bodies=tuple(bodies),
            problems=(*problems, write_problem),
            dimension_check={"tolerance_mm": tolerance_mm, "ok": False, "checked": []},
        )

    dimension_check = measure_step(target, expected, tolerance_mm=tolerance_mm)
    problems.extend(dimension_check.get("problems", ()))
    not_checked = sorted(
        {feature["interface_id"] for feature in features if not feature["dimensions_measured"]}
    )
    if not_checked:
        # A slot width or a dovetail angle is not measured here; saying so is the
        # difference between "not verified" and "verified", and the manifest must not
        # let the second be inferred from silence.
        dimension_check["not_checked"] = not_checked
    for entry in dimension_check.get("checked", ()):
        if not entry["ok"]:
            problems.append(
                f"{entry['interface_id']}: {entry['dimension']} measured "
                f"{entry['measured_mm']} mm against spec {entry['spec_mm']} mm "
                f"(error {entry['error_mm']} mm > {tolerance_mm} mm)"
            )
    if dimension_check.get("ok") is False:
        problems.append(
            "the STEP read-back did not reproduce the spec's fit dimensions within "
            f"{tolerance_mm} mm"
        )
    bound = _stl_step_bound(spec, tolerance_mm)
    dimension_check["stl_step_bound"] = bound
    if bound.get("ok") is False:
        problems.append(
            "the printed mesh's tessellation shortfall "
            f"({bound['max_stl_vs_step_radial_mm']} mm) exceeds the "
            f"{tolerance_mm} mm consistency the audit requires between STL and STEP"
        )

    return StepResult(
        path=str(target),
        ok=(
            not problems
            and dimension_check.get("ok") is not False
            and bound.get("ok") is not False
        ),
        features=tuple(_public(feature) for feature in features),
        bodies=tuple(bodies),
        problems=tuple(problems),
        dimension_check=dimension_check,
    )


def measure_step(
    path: str | Path,
    expected: Sequence[dict],
    *,
    tolerance_mm: float = DIMENSION_TOLERANCE_MM,
) -> dict:
    """Read `path` back and compare its cylindrical diameters with `expected`.

    `expected` is a sequence of `{"interface_id", "dimension", "spec_mm"}`. Every
    measured diameter comes from a real cylindrical face of the imported STEP; a
    missing measurement is a failure (`measured_mm: None`), never a pass, because the
    audit's question is "does the file carry the dimension" and an absent feature
    answers no.

    Matching is nearest-diameter with removal, so two identical holes consume two
    measured cylinders and one hole cannot vouch for the other.
    """
    api = _build123d()
    try:
        loaded = api["import_step"](str(path))
        observed = _cylindrical_diameters(loaded)
    except Exception as error:  # noqa: BLE001 - an unreadable file is a failed check
        return {
            "tolerance_mm": tolerance_mm,
            "ok": False,
            "checked": [],
            "observed_diameters_mm": [],
            "problems": [
                f"the STEP could not be read back: {type(error).__name__}: {error}"
            ],
        }

    checked: list[dict] = []
    problems: list[str] = []
    available = list(observed)
    for item in expected:
        spec_mm = float(item["spec_mm"])
        nearest: float | None = None
        if available:
            nearest = min(available, key=lambda value: abs(value - spec_mm))
            available.remove(nearest)
        error_mm = None if nearest is None else round(abs(nearest - spec_mm), 6)
        ok = error_mm is not None and error_mm <= tolerance_mm
        checked.append(
            {
                "interface_id": item["interface_id"],
                "dimension": item["dimension"],
                "spec_mm": round(spec_mm, 6),
                "measured_mm": None if nearest is None else round(nearest, 6),
                "error_mm": error_mm,
                "ok": ok,
            }
        )
        if nearest is None:
            problems.append(
                f"{item['interface_id']}: no cylindrical face matching "
                f"{item['dimension']} = {spec_mm} mm was found in the STEP"
            )
    return {
        "tolerance_mm": tolerance_mm,
        # `None` when nothing measurable was asked for: a slot width or a dovetail
        # flank angle is not "verified" and must not read as such.
        "ok": all(entry["ok"] for entry in checked) if checked else None,
        "checked": checked,
        "observed_diameters_mm": [round(value, 6) for value in sorted(observed)],
        "problems": problems,
    }


# --------------------------------------------------------------------------- #
# The scene
# --------------------------------------------------------------------------- #
def _build_scene(
    spec: DesignSpec,
    interfaces: Any,
    problems: list[str],
    expected: list[dict],
) -> tuple[list[Any], list[dict], list[dict], list[dict], list[str]]:
    """Every solid the STEP contains, plus the manifest rows describing them.

    One pass per interface: the negative feature is either cut into its host's
    envelope proxy (parametric part) or exported as a void solid (decomposed blank);
    the positive feature is always exported. Spec order is the file order, so two
    exports of one spec write the same bytes.
    """
    api = _build123d()
    frames = _frames()
    facts = interfaces if isinstance(interfaces, dict) else {}
    envelopes: dict[str, tuple[Any, list[Any]]] = {}
    feature_solids: list[Any] = []
    features: list[dict] = []
    bodies: list[dict] = []

    for interface in spec.interfaces:
        host = spec.part(interface.a)
        guest = spec.part(interface.b)
        params = interface.validated_params()
        try:
            built = _features_for(host, guest, interface, params, facts, api, frames)
        except Exception as error:  # noqa: BLE001 - an unplaceable interface is a row
            problems.append(
                f"{interface.id}: cannot build the {interface.kind} B-Rep feature: "
                f"{type(error).__name__}: {error}"
            )
            continue
        problems.extend(built.get("warnings", ()))
        recorded = (facts.get(interface.id) or {}).get("mate_axis")
        if recorded is not None and not _axis_agrees(recorded, built["axis"]):
            # The materialiser's recorded axis and the builder's must be the same
            # geometry; a disagreement means the STEP would describe a different
            # insertion direction than the mesh, so it is reported and the builder's
            # (the axis the STL was made from) is used.
            problems.append(
                f"{interface.id}: the recorded mate axis {list(recorded)} disagrees "
                f"with the derived {list(built['axis'])}; the export used the derived "
                "axis, because that is the geometry the mesh was built from"
            )
        feature_solids.extend(built["positive"])
        if host.source:
            # A decomposed blank has no B-Rep host; the socket travels as a void solid
            # so its diameter is still measurable, and `bodies` says the part itself
            # was not converted.
            feature_solids.extend(built["negative"])
            representation = "void_solid"
        else:
            entry = envelopes.get(host.id)
            if entry is None:
                entry = (_envelope_proxy(host, api, frames), [])
                envelopes[host.id] = entry
            entry[1].extend(built["negative"])
            representation = "cut_into_envelope"
        _collect_expected(expected, interface.id, interface.kind, params)
        features.append(
            {
                "interface_id": interface.id,
                "kind": interface.kind,
                "nominal_dims_mm": _nominal_dims(interface.kind, params),
                "solid_volume_mm3": round(_sum_volume(built["positive"]), 3),
                "negative_representation": representation,
                "host_part_id": host.id,
                "guest_part_id": guest.id,
                "mate_axis": [round(value, 6) for value in built["axis"]],
                "feature_origin_mm": [
                    round(value, 6) for value in built["origin"]
                ],
                "dimensions_measured": interface.kind == "peg_hole",
            }
        )

    solids: list[Any] = []
    for part_id, (envelope, negatives) in envelopes.items():
        try:
            solid = envelope
            for shape in negatives:
                solid = solid - shape
            solids.append(solid)
            bodies.append(
                {
                    "part_id": part_id,
                    "representation": "envelope_proxy_with_features",
                    "solid_volume_mm3": round(float(solid.volume), 3),
                    "note": (
                        "a box of the part's size_mm in its world pose with the fit "
                        "features cut; not a CAD model of the part"
                    ),
                }
            )
        except Exception as error:  # noqa: BLE001 - a failed boolean is a row
            problems.append(
                f"{part_id}: the envelope proxy could not be built: "
                f"{type(error).__name__}: {error}"
            )
    emitted = {row["part_id"] for row in bodies}
    for part in spec.parts:
        if part.source and part.id not in emitted:
            bodies.append(
                {
                    "part_id": part.id,
                    "representation": "mesh_blank_not_converted",
                    "solid_volume_mm3": None,
                    "note": "free-form/decomposed part: exported as STL, never as B-Rep",
                }
            )
    solids.extend(feature_solids)
    return solids, features, bodies, expected, problems


def _features_for(
    host: PartSpec,
    guest: PartSpec,
    interface: Any,
    params: Any,
    facts: dict,
    api: dict,
    frames: Any,
) -> dict:
    """`{"positive", "negative", "axis", "origin", "warnings"}` for one interface."""
    if host.source:
        return _decomposed_features(host, guest, interface, params, facts, api, frames)
    kind = interface.kind
    if kind == "peg_hole":
        return _peg_hole_features(host, guest, params, api, frames)
    if kind == "snap_fit":
        return _snap_fit_features(host, guest, params, api, frames)
    if kind == "dovetail":
        return _dovetail_features(host, guest, params, api, frames)
    raise ValueError(f"no B-Rep feature construction for connector kind {kind!r}")


def _plane(
    part: PartSpec,
    local_origin: Iterable[float],
    z_local: Iterable[float],
    x_local: Iterable[float],
    api: dict,
    frames: Any,
) -> Any:
    """A world `build123d.Plane` from a point and two directions in a part's frame."""
    return api["Plane"](
        origin=frames.world_point(part, tuple(float(v) for v in local_origin)),
        x_dir=frames.world_axis(part, tuple(float(v) for v in x_local)),
        z_dir=frames.world_axis(part, tuple(float(v) for v in z_local)),
    )


def _world_plane(
    origin: Iterable[float],
    z_dir: Iterable[float],
    x_dir: Iterable[float],
    api: dict,
) -> Any:
    """A world plane from three world-space facts (a decomposed interface's frame)."""
    return api["Plane"](
        origin=tuple(float(value) for value in origin),
        x_dir=tuple(float(value) for value in x_dir),
        z_dir=tuple(float(value) for value in z_dir),
    )


def _peg_hole_features(
    host: PartSpec, guest: PartSpec, params: Any, api: dict, frames: Any
) -> dict:
    """Blind socket (cut/void) plus the cylindrical peg that enters it.

    The socket is drilled from the body's top face **downwards** by `engage_mm`
    (`design/connectors/peg_hole.py`'s blind-socket rule); the peg stands from the
    insert's leading tip along +Z. `align=MIN` makes a cylinder start at its frame
    origin, which is what turns a plane into a feature that begins at the face.
    """
    top = float(host.size_mm[2]) / 2.0
    feature_xy = getattr(params, "feature_xy_mm", (0.0, 0.0))
    socket = api["Cylinder"](
        params.d_hole_mm / 2.0,
        params.engage_mm,
        align=(api["Align"].CENTER, api["Align"].CENTER, api["Align"].MIN),
    )
    socket_plane = _plane(
        host, (feature_xy[0], feature_xy[1], top), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), api, frames
    )
    peg = api["Cylinder"](
        params.d_peg_mm / 2.0,
        params.peg_length_mm,
        align=(api["Align"].CENTER, api["Align"].CENTER, api["Align"].MIN),
    )
    peg_plane = _plane(guest, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), api, frames)
    return {
        "positive": [peg_plane * peg],
        "negative": [socket_plane * socket],
        "axis": frames.world_axis(guest, (0.0, 0.0, 1.0)),
        "origin": frames.world_point(guest, (0.0, 0.0, 0.0)),
    }


def _snap_fit_features(
    host: PartSpec, guest: PartSpec, params: Any, api: dict, frames: Any
) -> dict:
    """A through slot plus an undercut chamber, and the hook-and-beam that snaps in.

    Mirrors `design/connectors/snap_fit.py`: the slot is centred on the body, the
    chamber's ceiling sits `engage` below the top face, and the hook's top face seats
    on that ceiling with one `play` of clearance.
    """
    play = float(params.radial_clearance_mm)
    size_z = float(host.size_mm[2])
    top = size_z / 2.0
    slot = api["Box"](params.slot_w_mm, params.slot_w_mm, size_z + 2.0)
    slot_plane = _plane(host, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), api, frames)
    chamber_side = params.slot_w_mm + 2.0 * (params.hook_h_mm + play)
    chamber_height = params.hook_h_mm + 2.0 * play
    chamber = api["Box"](chamber_side, chamber_side, chamber_height)
    chamber_plane = _plane(
        host,
        (0.0, 0.0, top - params.engage_mm - params.hook_h_mm / 2.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        api,
        frames,
    )
    hook = api["Box"](
        params.hook_w_mm,
        params.hook_w_mm,
        params.hook_h_mm,
        align=(api["Align"].CENTER, api["Align"].CENTER, api["Align"].MIN),
    )
    beam = api["Box"](
        params.beam_thk_mm,
        params.hook_w_mm,
        params.beam_len_mm,
        align=(api["Align"].CENTER, api["Align"].CENTER, api["Align"].MIN),
    )
    insert = api["Plane"].XY * hook + api["Plane"].XY * (
        api["Pos"](0.0, 0.0, params.hook_h_mm) * beam
    )
    insert_plane = _plane(
        guest, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), api, frames
    )
    return {
        "positive": [insert_plane * insert],
        "negative": [slot_plane * slot, chamber_plane * chamber],
        "axis": frames.world_axis(guest, (0.0, 0.0, 1.0)),
        "origin": frames.world_point(guest, (0.0, 0.0, 0.0)),
    }


def _dovetail_features(
    host: PartSpec, guest: PartSpec, params: Any, api: dict, frames: Any
) -> dict:
    """A trapezoidal groove and the tongue-plus-handle that slides into it.

    The tongue is built in a canonical frame (extruded along +Z = the part's +X slide
    axis, width along canonical X = the part's +Y, depth downward as canonical -Y) and
    placed by one `Plane`; the same construction is used for the groove so the
    clearance is the difference between two identically-written shapes.
    """
    tan_angle = math.tan(math.radians(params.angle_deg))
    half_open = float(params.width_mm) / 2.0
    half_floor = half_open + float(params.depth_mm) * tan_angle
    pocket_len = float(params.engage_mm) + float(params.clearance_mm)
    groove = _trapezoid(
        api,
        half_top=half_open,
        half_bottom=half_floor,
        depth=float(params.depth_mm),
        length=pocket_len + 1.0,
    )
    size_x = float(host.size_mm[0])
    top = float(host.size_mm[2]) / 2.0
    groove_plane = _plane(
        host, (size_x / 2.0 - pocket_len, 0.0, top), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), api, frames
    )
    tongue_depth = float(params.depth_mm) - float(params.clearance_mm)
    tongue_half_top = half_open - float(params.clearance_mm)
    tongue = _trapezoid(
        api,
        half_top=tongue_half_top,
        half_bottom=tongue_half_top + tongue_depth * tan_angle,
        depth=tongue_depth,
        length=float(params.engage_mm),
    )
    handle_w = max(3.0 * float(params.width_mm), 20.0)
    handle_h = max(0.8 * float(params.depth_mm), 6.0)
    handle = api["Box"](
        handle_w,
        handle_h,
        float(params.engage_mm),
        align=(api["Align"].CENTER, api["Align"].MIN, api["Align"].MIN),
    )
    insert_plane = _plane(
        guest, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), api, frames
    )
    insert = api["Plane"].XY * tongue + api["Plane"].XY * handle
    return {
        "positive": [insert_plane * insert],
        "negative": [groove_plane * groove],
        "axis": frames.world_axis(guest, (1.0, 0.0, 0.0)),
        "origin": frames.world_point(guest, (0.0, 0.0, 0.0)),
    }


def _decomposed_features(
    host: PartSpec,
    guest: PartSpec,
    interface: Any,
    params: Any,
    facts: dict,
    api: dict,
    frames: Any,
) -> dict:
    """Features for a decomposed pair: the mating frame is the shared face.

    `design/pairing.contact_of` gives the axis (a→b) and the face centre; the socket
    opens there and runs one `engage` into the host along `-axis`, the peg stands out
    along `-axis` — the same rule `design/machining.cut_socket/add_peg` applied to the
    mesh blanks. Only `peg_hole` occurs on this plane (`pairing.propose_interfaces`
    only proposes it), so another kind is refused by name rather than approximated.
    """
    if interface.kind != "peg_hole":
        raise ValueError(
            f"{interface.kind} on a decomposed blank has no face-frame construction; "
            "only peg_hole interfaces are generated from decomposed parts"
        )
    from pah.design.pairing import contact_of  # noqa: PLC0415

    contact = contact_of(host, guest)
    if contact is None:
        raise ValueError(
            f"parts {host.id!r} and {guest.id!r} are no longer adjacent, so the socket "
            "has no face to open on"
        )
    axis = tuple(float(value) for value in contact.axis)
    point = tuple(float(value) for value in contact.point_mm)
    x_dir = _perpendicular(axis)
    socket = api["Cylinder"](
        params.d_hole_mm / 2.0,
        params.engage_mm,
        align=(api["Align"].CENTER, api["Align"].CENTER, api["Align"].MIN),
    )
    peg = api["Cylinder"](
        params.d_peg_mm / 2.0,
        params.engage_mm,
        align=(api["Align"].CENTER, api["Align"].CENTER, api["Align"].MIN),
    )
    into_host = tuple(-value for value in axis)
    return {
        "positive": [_world_plane(point, into_host, x_dir, api) * peg],
        "negative": [_world_plane(point, into_host, x_dir, api) * socket],
        "axis": axis,
        "origin": point,
    }


def _perpendicular(axis: Iterable[float]) -> tuple[float, float, float]:
    """A deterministic unit vector orthogonal to `axis`, for a decomposed frame."""
    import numpy as np  # noqa: PLC0415

    vector = np.asarray(tuple(axis), dtype=float)
    candidate = np.array([1.0, 0.0, 0.0])
    if abs(float(vector @ candidate)) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])
    perpendicular = np.cross(vector, candidate)
    perpendicular /= np.linalg.norm(perpendicular)
    return tuple(float(value) for value in perpendicular)


def _axis_agrees(recorded: Iterable[float], derived: Iterable[float]) -> bool:
    import numpy as np  # noqa: PLC0415

    first = np.asarray(tuple(recorded), dtype=float)
    second = np.asarray(tuple(derived), dtype=float)
    return bool(abs(float(first @ second)) > 1.0 - 1e-9)


def _trapezoid(
    api: dict, *, half_top: float, half_bottom: float, depth: float, length: float
) -> Any:
    """A trapezoidal prism: opening at canonical y = 0, floor `depth` below it.

    The point order is counter-clockwise so `extrude` runs along +Z; a clockwise
    polygon silently extrudes along -Z and every groove would open on the wrong side.
    """
    section = api["Polygon"](
        (-half_bottom, -depth),
        (half_bottom, -depth),
        (half_top, 0.0),
        (-half_top, 0.0),
        align=None,
    )
    return api["extrude"](section, amount=length)


def _envelope_proxy(part: PartSpec, api: dict, frames: Any) -> Any:
    """The part's blank as a B-Rep box in its world pose — an envelope, not the part."""
    matrix = frames.part_matrix(part)
    return api["Plane"](
        origin=tuple(float(value) for value in part.pose_mm),
        x_dir=tuple(float(value) for value in matrix[:3, 0]),
        z_dir=tuple(float(value) for value in matrix[:3, 2]),
    ) * api["Box"](*[float(value) for value in part.size_mm])


def _write_step(solids: Sequence[Any], target: Path) -> str | None:
    """Write the compound atomically; returns a problem sentence, or `None` on success.

    A partially written STEP is deleted rather than left behind: a file that looks
    like an engineering deliverable but is truncated is worse than no file.
    """
    api = _build123d()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=".pah-step-", suffix=".step", dir=target.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        api["export_step"](api["Compound"](children=list(solids)), str(temporary))
        if not temporary.exists() or temporary.stat().st_size == 0:
            raise RuntimeError("the exporter wrote no bytes")
        os.replace(temporary, target)
    except Exception as error:  # noqa: BLE001 - no half-written deliverable survives
        temporary.unlink(missing_ok=True)
        return f"could not write the STEP: {type(error).__name__}: {error}"
    return None


# --------------------------------------------------------------------------- #
# Evidence helpers
# --------------------------------------------------------------------------- #
def _cylindrical_diameters(loaded: Any) -> list[float]:
    """Every cylindrical face's diameter in the imported STEP, in mm.

    A radius read from the B-Rep face measures the *file*, not the parameters that
    produced it, which is the whole point of a read-back check. A face whose adapter
    cannot report a radius contributes nothing; a missing measurement then fails the
    match instead of being guessed.
    """
    diameters: list[float] = []
    for solid in loaded.solids():
        for face in solid.faces():
            if "CYLINDER" not in str(getattr(face, "geom_type", "")).upper():
                continue
            try:
                radius = float(face.geom_adaptor().Cylinder().Radius())
            except Exception:  # noqa: BLE001 - an unreadable face is not evidence
                continue
            if radius > 0.0:
                diameters.append(2.0 * radius)
    return diameters


def _collect_expected(
    expected: list[dict], interface_id: str, kind: str, params: Any
) -> None:
    """The cylinder dimensions a peg_hole interface must reproduce in the STEP.

    Only cylindrical fits are checked numerically: a slot's width or a dovetail's
    flank angle would need a face-level measurement this module does not have, and
    reporting a parameter as "checked" when it was not is the overclaim the audit is
    about. Those interfaces carry their nominal numbers and are listed as not checked.
    """
    if kind != "peg_hole":
        return
    expected.append(
        {
            "interface_id": interface_id,
            "dimension": "hole_diameter_mm",
            "spec_mm": float(params.d_hole_mm),
        }
    )
    expected.append(
        {
            "interface_id": interface_id,
            "dimension": "peg_diameter_mm",
            "spec_mm": float(params.d_peg_mm),
        }
    )


def _nominal_dims(kind: str, params: Any) -> dict:
    """The numbers a shop quotes, per kind, straight from the validated parameters."""
    values = {
        key: float(value)
        for key, value in params.model_dump().items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if kind == "peg_hole":
        values["hole_depth_mm"] = float(params.engage_mm)
    return values


def _stl_step_bound(spec: DesignSpec, tolerance_mm: float) -> dict:
    """The radial shortfall the printed polygon can have against the nominal STEP circle.

    A 64-gon inscribed in a d-diameter circle is `d/2 * (1 - cos(pi/64))` inside it at
    the flats — 6 µm on a 10 mm feature. This is a **bound stated from the
    tessellation rule**, not a caliper measurement of the STL; the manifest labels it
    as such, because the honest claim here is "the mesh cannot be further from the
    STEP than this", not "we measured both files".
    """
    diameters: list[float] = []
    for interface in spec.interfaces:
        if interface.kind != "peg_hole":
            continue
        params = interface.validated_params()
        diameters.extend([float(params.d_hole_mm), float(params.d_peg_mm)])
    if not diameters:
        return {
            "ok": None,
            "reason": "no cylindrical fit features; nothing to compare with the mesh",
            "tolerance_mm": tolerance_mm,
        }
    sections = _sections()
    shortfalls = {
        round(diameter, 6): round(
            diameter / 2.0 * (1.0 - math.cos(math.pi / sections)), 6
        )
        for diameter in diameters
    }
    worst = max(shortfalls.values())
    return {
        "basis": f"tessellation bound: {sections}-gon inscribed in the nominal circle",
        "tolerance_mm": tolerance_mm,
        "max_stl_vs_step_radial_mm": worst,
        "per_diameter_mm": shortfalls,
        "ok": worst <= tolerance_mm,
        "note": (
            "a bound implied by the mesh's 64-gon tessellation (radial, not a caliper "
            "measurement); the STL's vertices lie on the nominal circle"
        ),
    }


def _sections() -> int:
    """The mesh builder's cylinder side count, with a stable fallback."""
    try:
        from pah.design.connectors.peg_hole import SECTIONS  # noqa: PLC0415

        return int(SECTIONS)
    except Exception:  # noqa: BLE001 - a moved constant must not break the exporter
        return _FALLBACK_SECTIONS


def _sum_volume(shapes: Sequence[Any]) -> float:
    total = 0.0
    for shape in shapes:
        try:
            total += float(shape.volume)
        except Exception:  # noqa: BLE001 - volume is reporting, not a gate
            continue
    return total


def _public(feature: dict) -> dict:
    """The feature row as the manifest carries it (no private flags)."""
    return {key: value for key, value in feature.items() if key != "dimensions_measured"}
