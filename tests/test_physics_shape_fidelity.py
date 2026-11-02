"""The settle model must not report stability for an object it replaced.

Audit P1-2 §7.1, verbatim: a round insert whose cross-section was within 2% of a
circle was replaced by a **solid cylinder** *before* any volume or convexity check ran.
A hollow tube or a stepped round part therefore settled as a different solid, and the
resulting `joint_unstable` / `dynamic_penetration` were filed as physical facts. The
model facts were disclosed in the certificate, but the *conclusion* was not gated on
them.

The shortcut now needs three things: a round cross-section, a closed mesh, and a solid
that actually fills the cylinder it is being swapped for (≥95%). Anything else falls
through to the hull / CoACD branches, where the model says what it is — or refuses with
`non_convex_insert` when CoACD is absent, which the report renders as not-evaluated.

Acceptance (report three, B3): a hollow tube must never come back as a solid cylinder,
and a stepped round part must not either.
"""

from __future__ import annotations

import trimesh

from pah.tools.mujoco_settle import SettleError, _insert_shape

ANGLES = 24


def _shape(mesh: trimesh.Trimesh) -> dict:
    """The model the settle would use for `mesh`, in its own mate frame."""
    return _insert_shape(mesh, mesh.vertices, mesh.vertices, mesh.faces, ANGLES)


def _shape_or_refusal(mesh: trimesh.Trimesh) -> dict:
    """`(kind, facts)` or `{"kind": "refused", "code": ...}` — both are honest answers."""
    try:
        return _shape(mesh)
    except SettleError as error:
        assert error.code in {"non_convex_insert", "insert_degenerate"}, error.code
        return {"kind": "refused", "code": error.code}


def _tube(radius: float = 5.0, inner: float = 3.5, height: float = 10.0) -> trimesh.Trimesh:
    outer = trimesh.creation.cylinder(radius=radius, height=height, sections=64)
    bore = trimesh.creation.cylinder(radius=inner, height=height + 2.0, sections=64)
    return trimesh.boolean.difference([outer, bore], engine="manifold")


def _stepped(wide: float = 5.0, narrow: float = 2.5) -> trimesh.Trimesh:
    lower = trimesh.creation.cylinder(radius=wide, height=6.0, sections=64)
    lower.apply_translation((0.0, 0.0, 3.0))
    upper = trimesh.creation.cylinder(radius=narrow, height=8.0, sections=64)
    upper.apply_translation((0.0, 0.0, 10.0))
    return trimesh.util.concatenate([lower, upper])


def test_a_solid_round_insert_is_still_modelled_as_a_cylinder():
    """The shortcut has to survive for the case it was written for."""
    solid = trimesh.creation.cylinder(radius=5.0, height=10.0, sections=64)

    shape = _shape(solid)

    assert shape["kind"] == "cylinder"
    # 64-gon inscribed in the circle: ~99.8% of the nominal cylinder, so the swap is
    # justified and the number says so.
    assert shape["solid_fraction"] > 0.99


def test_a_hollow_tube_is_never_replaced_by_a_solid_cylinder():
    tube = _tube()
    assert tube.is_watertight and tube.volume > 0.0

    shape = _shape_or_refusal(tube)

    assert shape["kind"] != "cylinder", (
        "a tube settled as a solid cylinder is the audit's B3 failure: the reported "
        "stability would belong to a different object"
    )
    if shape["kind"] != "refused":
        assert shape["solid_fraction"] < 0.95


def test_a_stepped_round_part_is_not_a_cylinder_either():
    stepped = _stepped()
    assert stepped.is_watertight

    shape = _shape_or_refusal(stepped)

    assert shape["kind"] != "cylinder"
    if shape["kind"] != "refused":
        assert shape["solid_fraction"] < 0.95


def test_every_model_names_how_much_of_the_cylinder_it_fills():
    """`solid_fraction` is the number a reader needs to weigh the approximation."""
    solid = trimesh.creation.cylinder(radius=5.0, height=10.0, sections=64)
    stepped = _stepped()

    for mesh in (solid, stepped):
        shape = _shape_or_refusal(mesh)
        assert shape["kind"] == "refused" or 0.0 <= shape["solid_fraction"] <= 1.5
        if shape["kind"] != "refused":
            assert "approximation_error" in shape
