"""Is there actually material where the part should be?

The gap this closes, measured on a real generated part: Hunyuan3D returned a tube
whose bounding box was exactly the designed Ø10 × 14 peg (the adapter normalises the
box, as it must) but whose volume was **314 mm³** against the parametric peg's
**1 098 mm³** — 29 %. Nothing in the harness objected: clearance, interference and
insertion path all look at surfaces and gaps, and a hollow tube has the right outer
surface and no interference. The part was reported `repaired` and would have been
milled out of thin air on a printer.

So this verifier compares the *volume* of the part against the volume the design
implies, rebuilt independently by the connector's own parametric builder. It is the
cheapest measurement that can distinguish "a pin" from "the shell of a pin", and its
reference is parametric, so the judgement does not depend on the generator.

Band: 0.55–1.8. Below 0.55 the part is a shell (the measured tube was 0.29); above
1.8 the generator added material the interface never asked for, which will usually
show up as interference but is worth naming. The band is wide on purpose — a
generative model may legitimately be chunkier than a cylinder.
"""

from __future__ import annotations

import trimesh

from pah.core.model import FailureCertificate
from pah.design.api import DesignView
from pah.design.connectors import builder_for
from pah.tools.verifiers.base import VerifierSpec

__all__ = ["InsertMaterial", "CODE_INSUFFICIENT_MATERIAL", "CODE_EXCESS_MATERIAL", "RATIO_BAND"]

CODE_INSUFFICIENT_MATERIAL = "insufficient_material"
CODE_EXCESS_MATERIAL = "excess_material"

#: `(low, high)` on `generated_volume / designed_volume`.
RATIO_BAND = (0.55, 1.8)


class InsertMaterial:
    """Compares the built insert's volume with the volume the design implies."""

    spec = VerifierSpec(
        id="insert_material",
        codes=(CODE_INSUFFICIENT_MATERIAL, CODE_EXCESS_MATERIAL),
        cost_class="medium",
        fixtures={"known_good": ["solid_pin"], "known_bad": ["hollow_tube"]},
    )

    def check(self, view: DesignView) -> list[FailureCertificate]:
        reference_volume = self._designed_volume(view)
        if reference_volume <= 0.0:
            # No reference is a capability boundary, not a verdict: say so in the
            # note channel and judge nothing.
            self.last_note = "material not evaluated: the connector kind has no parametric reference"
            self.last_ratio = None
            return []
        built = float(view.insert.volume)
        ratio = built / reference_volume
        self.last_ratio = ratio
        self.last_note = ""
        low, high = RATIO_BAND
        if ratio < low:
            return [
                self._certificate(
                    CODE_INSUFFICIENT_MATERIAL,
                    view,
                    ratio,
                    built,
                    reference_volume,
                    f"the built part holds {ratio:.0%} of the material the design implies "
                    f"(it is a shell, not a solid)",
                )
            ]
        if ratio > high:
            return [
                self._certificate(
                    CODE_EXCESS_MATERIAL,
                    view,
                    ratio,
                    built,
                    reference_volume,
                    f"the built part holds {ratio:.0%} of the designed volume; the generator "
                    "added material the interface never asked for",
                )
            ]
        return []

    # -- internals ---------------------------------------------------------
    def _designed_volume(self, view: DesignView) -> float:
        kind = view.spec.interface.kind
        builder = builder_for(kind)
        reference: trimesh.Trimesh = builder.build(view.spec).meshes[view.spec.insert.id]
        return float(reference.volume)

    def _certificate(
        self,
        code: str,
        view: DesignView,
        ratio: float,
        built: float,
        reference: float,
        detail: str,
    ) -> FailureCertificate:
        return FailureCertificate(
            code=code,
            family="geometry",
            locus={"part": view.spec.insert.id, "kind": view.spec.interface.kind},
            magnitude={
                "volume_ratio": round(ratio, 4),
                "built_volume_mm3": round(built, 3),
                "designed_volume_mm3": round(reference, 3),
                "band": list(RATIO_BAND),
                "detail": detail,
            },
            observed_by="insert_material",
        )

    # -- the §4.8 gate -----------------------------------------------------
    def fixtures(self) -> tuple[list[DesignView], list[DesignView]]:
        """A solid pin (good) and a tube with the same envelope (bad)."""
        from pah.cli.design import default_spec  # noqa: PLC0415 - only fixtures need it
        from pah.design.api import DesignView as View  # noqa: PLC0415
        from pah.design.connectors import builder_for  # noqa: PLC0415

        spec = default_spec(kind="peg_hole", spec_id="fixture-insert-material")
        reference = builder_for("peg_hole").build(spec).meshes[spec.insert.id]
        solid = reference.copy()

        outer = trimesh.creation.cylinder(
            radius=max(reference.extents[0], reference.extents[1]) / 2.0,
            height=reference.extents[2],
            sections=64,
        )
        inner = trimesh.creation.cylinder(
            radius=max(reference.extents[0], reference.extents[1]) / 2.0 * 0.78,
            height=reference.extents[2] * 1.2,
            sections=64,
        )
        tube = trimesh.boolean.difference([outer, inner], engine="manifold")
        tube.apply_translation([0.0, 0.0, reference.bounds[0][2] - tube.bounds[0][2]])

        good = View(spec=spec, meshes={"body": trimesh.creation.box((10, 10, 1)), "insert": solid})
        bad = View(spec=spec, meshes={"body": trimesh.creation.box((10, 10, 1)), "insert": tube})
        return [good], [bad]
