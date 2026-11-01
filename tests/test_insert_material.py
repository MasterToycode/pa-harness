"""The verifier that catches a hollow part wearing a correct bounding box.

Measured on a real Hunyuan3D output: the generated insert's box was exactly the
designed Ø10 × 14 peg (the adapter normalises it), its volume was 314 mm³ against the
parametric peg's 1 098 mm³, and every surface-level verifier passed it. The harness
reported `repaired` for a part that has no material where a peg needs it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import trimesh

from pah.design.api import DesignVerifierRegistry, DesignView, spec_of
from pah.design.connectors import builder_for
from pah.tools.verifiers.insert_material import (
    InsertMaterial,
    RATIO_BAND,
)

REPO = Path(__file__).resolve().parents[1]


def _spec():
    from pah.design.spec import DesignSpec

    return DesignSpec.model_validate(
        json.loads((REPO / "fixtures/design/demo-peg-hole-clear.json").read_text("utf-8"))
    )


def _reference_insert(spec):
    return builder_for(spec.interface.kind).build(spec).meshes[spec.insert.id]


def _view(spec, insert: trimesh.Trimesh) -> DesignView:
    body = trimesh.creation.box((60.0, 40.0, 12.0))
    return DesignView(spec=spec, meshes={"body": body, spec.insert.id: insert})


def _tube(envelope: trimesh.Trimesh, *, inner_ratio: float = 0.78) -> trimesh.Trimesh:
    """A tube with the same bounding box as the reference insert."""
    outer = trimesh.creation.cylinder(
        radius=max(envelope.extents[0], envelope.extents[1]) / 2.0,
        height=envelope.extents[2],
        sections=64,
    )
    inner = trimesh.creation.cylinder(
        radius=max(envelope.extents[0], envelope.extents[1]) / 2.0 * inner_ratio,
        height=envelope.extents[2] * 1.2,
        sections=64,
    )
    tube = trimesh.boolean.difference([outer, inner], engine="manifold")
    tube.apply_translation([0.0, 0.0, envelope.bounds[0][2] - tube.bounds[0][2]])
    return tube


# --------------------------------------------------------------------------- #
# the judgement
# --------------------------------------------------------------------------- #
def test_a_solid_part_matching_the_design_passes():
    spec = _spec()
    reference = _reference_insert(spec)

    assert InsertMaterial().check(_view(spec, reference.copy())) == []


def test_a_shell_with_the_right_bounding_box_is_refused():
    """The measured failure: same box, 29 % of the volume."""
    spec = _spec()
    reference = _reference_insert(spec)
    tube = _tube(reference)
    assert tube.extents[0] == pytest.approx(reference.extents[0], abs=0.1)
    assert tube.volume < 0.55 * reference.volume

    certificates = InsertMaterial().check(_view(spec, tube))

    assert [c.code for c in certificates] == ["insufficient_material"]
    magnitude = certificates[0].magnitude
    assert magnitude["volume_ratio"] < RATIO_BAND[0]
    assert magnitude["built_volume_mm3"] < magnitude["designed_volume_mm3"]
    assert "shell" in magnitude["detail"]


def test_a_part_that_is_much_heavier_than_the_design_is_named_too():
    spec = _spec()
    reference = _reference_insert(spec)
    oversized = trimesh.creation.cylinder(
        radius=reference.extents[0] * 0.9,
        height=reference.extents[2] * 2.0,
        sections=64,
    )

    codes = [c.code for c in InsertMaterial().check(_view(spec, oversized))]

    assert codes == ["excess_material"]


# --------------------------------------------------------------------------- #
# the gate and the cascade
# --------------------------------------------------------------------------- #
def test_the_verifier_proves_itself_with_a_good_bad_pair():
    registry = DesignVerifierRegistry()
    registry.register(InsertMaterial())

    results = registry.sanity()

    assert results == {"insert_material": True}


def test_the_shipped_cascade_includes_it():
    from pah.tools.verifiers.design_cascade import default_registry

    registry = default_registry()
    assert "insert_material" in registry.ids()
    assert registry.cost_class("insert_material") == "medium"


def test_the_reference_is_rebuilt_from_the_design_not_trusted_from_the_run():
    """The judgement must not depend on the generator: the reference comes from the
    connector's parametric builder for the same spec."""
    spec = _spec()
    view = _view(spec, _tube(_reference_insert(spec)))
    rebuilt = builder_for(spec.interface.kind).build(spec_of(view_mesh_assembly(spec)))

    assert InsertMaterial()._designed_volume(view) == pytest.approx(rebuilt.meshes[spec.insert.id].volume)


def view_mesh_assembly(spec):
    from pah.design.api import assembly_from_spec

    return assembly_from_spec(spec=spec)
