"""A hint is a new generation request, not a cache hit.

The audit's P1-4 is the most consequential finding in the report and it is a one-line
story: `DesignSpec.content_key()` excludes `provenance` on purpose (who chose a number
is not part of the design), and the repair that says "make it solid" writes its hint
into `provenance`. So a hint-only repair changed no number, hit the materialiser's
content-keyed cache, **never called the generator**, and the loop — seeing the same
geometry again — reported a *geometry cycle* on a design it had never rebuilt.
Measured in the audit: round 0 took ~64 s (a real generation), round 1 took ~2 s
(cache hit).

The fix keeps two identities apart:

* `content_key` — what the design *is*; unchanged, and still the geometry identity;
* `generation_key` — which *request* would produce it: content + hint + generator +
  model version + seed. The cascade caches on this, and the assembly records both the
  key and each part's actual `output_hash`, so the cycle detector sees a new sample.

These tests are the audit's acceptance criteria, spelled out.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pah.design.api import (
    assembly_from_spec,
    generation_key,
    materializer_identity,
    prompt_hint_of,
)
from pah.design.spec import DesignSpec, InterfaceSpec, PartSpec
from pah.policy.loop import assembly_fingerprint
from pah.tools.verifiers.design_cascade import DesignVerifierCascade
from pah.tools.verifiers.design_cascade import default_registry as _unused  # noqa: F401


def _registry():
    """No verifiers: this file is about *whether the generator was called*."""
    from pah.design.api import DesignVerifierRegistry

    return DesignVerifierRegistry()


def peg_hole_spec(**params) -> DesignSpec:
    values = {"d_peg_mm": 10.2, "d_hole_mm": 10.0}
    values.update(params)
    return DesignSpec(
        spec_id="gen-demo",
        requirement="two-part bracket",
        parts=(
            PartSpec(id="body", role="body", size_mm=(60.0, 40.0, 12.0)),
            PartSpec(
                id="peg",
                role="insert",
                size_mm=(12.0, 12.0, 20.0),
                pose_mm=(0.0, 0.0, 2.0),
            ),
        ),
        interfaces=(
            InterfaceSpec(id="iface_0", kind="peg_hole", a="body", b="peg", params=values),
        ),
    )


class FakeGenerator:
    """A generative adapter: counts calls and returns a *different* solid each time.

    `non_deterministic` is the honest declaration for a diffusion model; the varying
    scale stands in for the sample actually differing. Everything else is the real
    parametric materialiser, so the design stays valid and the verifiers would run if
    this registry had any.
    """

    generator_id = "fake-dit"
    model_version = "test-1"
    seed = 7
    supports_prompt_hint = True
    non_deterministic = True

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, spec, out_dir):
        from pah.design.materialize import materialize

        design = materialize(spec, out_dir)
        self.calls += 1
        insert_id = design.spec.insert.id
        mesh = design.meshes[insert_id].copy()
        mesh.apply_scale(1.0 + 0.01 * self.calls)
        return replace(design, meshes={**design.meshes, insert_id: mesh})


class ParametricCounter:
    """A parametric materialiser that declares nothing — the two-part default."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen: list[str] = []

    def __call__(self, spec, out_dir):
        from pah.design.materialize import materialize

        self.calls += 1
        self.seen.append(prompt_hint_of(spec))
        return materialize(spec, out_dir)


# --------------------------------------------------------------------------- #
# The identity split
# --------------------------------------------------------------------------- #
def test_the_hint_does_not_change_the_design_identity():
    plain = peg_hole_spec()
    hinted = plain.edit(provenance={"prompt_hint": "make it a SOLID part"})

    assert plain.content_key() == hinted.content_key()
    assert prompt_hint_of(hinted) == "make it a SOLID part"


def test_the_generation_key_does_move_with_the_hint():
    plain = peg_hole_spec()
    hinted = plain.edit(provenance={"prompt_hint": "make it a SOLID part"})
    other = plain.edit(provenance={"prompt_hint": "make it fatter"})

    base = generation_key(plain, generator_id="fake-dit", model_version="test-1", seed=7)
    same = generation_key(hinted, generator_id="fake-dit", model_version="test-1", seed=7)
    different = generation_key(other, generator_id="fake-dit", model_version="test-1", seed=7)

    assert base != same
    assert same != different
    # Deterministic: the same request hashes the same, or a resume would re-buy it.
    assert same == generation_key(hinted, generator_id="fake-dit", model_version="test-1", seed=7)


def test_a_different_generator_or_seed_is_a_different_request():
    spec = peg_hole_spec()
    one = generation_key(spec, generator_id="a", model_version="1", seed=1)
    assert one != generation_key(spec, generator_id="b", model_version="1", seed=1)
    assert one != generation_key(spec, generator_id="a", model_version="2", seed=1)
    assert one != generation_key(spec, generator_id="a", model_version="1", seed=2)


def test_a_plain_callable_declares_nothing_and_is_treated_as_parametric():
    identity = materializer_identity(lambda spec, out: None)

    assert identity["id"] == "parametric"
    assert identity["supports_prompt_hint"] is False
    assert identity["non_deterministic"] is False


# --------------------------------------------------------------------------- #
# Acceptance 1: a hint-only repair calls the generator exactly once
# --------------------------------------------------------------------------- #
def test_a_hint_only_repair_regenerates_and_is_not_a_geometry_cycle(tmp_path):
    generator = FakeGenerator()
    cascade = DesignVerifierCascade(registry=_registry(), root=tmp_path, materializer=generator)

    plain = peg_hole_spec()
    hinted = plain.edit(provenance={"prompt_hint": "make it a SOLID part"})

    cascade.check(assembly_from_spec(spec=plain))
    first = cascade._materialise(plain)
    assert generator.calls == 1

    cascade.check(assembly_from_spec(spec=hinted))
    second = cascade._materialise(hinted)

    # The model was asked again, even though no design number changed.
    assert generator.calls == 2
    # Same design identity, different generation.
    assert first.spec.content_key() == second.spec.content_key()
    assert assembly_fingerprint(first.assembly) != assembly_fingerprint(second.assembly)

    generation = second.assembly.constraints["generation"]
    assert generation["prompt_hint"] == "make it a SOLID part"
    assert generation["generator_id"] == "fake-dit"
    assert generation["key"] != first.assembly.constraints["generation"]["key"]
    # ... and each recorded the mesh that actually came back.
    assert generation["output_hashes"] != first.assembly.constraints["generation"]["output_hashes"]


# --------------------------------------------------------------------------- #
# Acceptance 2: the same hint twice is one request
# --------------------------------------------------------------------------- #
def test_repeating_the_same_hint_hits_the_cache(tmp_path):
    generator = FakeGenerator()
    cascade = DesignVerifierCascade(registry=_registry(), root=tmp_path, materializer=generator)
    hinted = peg_hole_spec().edit(provenance={"prompt_hint": "make it a SOLID part"})

    cascade.check(assembly_from_spec(spec=hinted))
    assert generator.calls == 1
    cascade.check(assembly_from_spec(spec=hinted))
    cascade._materialise(hinted)

    assert generator.calls == 1, "the same request must not be paid for twice"


# --------------------------------------------------------------------------- #
# Acceptance 3: a materialiser that cannot use a hint says so
# --------------------------------------------------------------------------- #
def test_a_parametric_materialiser_ignores_a_hint_and_says_so(tmp_path):
    materializer = ParametricCounter()
    cascade = DesignVerifierCascade(
        registry=_registry(), root=tmp_path, materializer=materializer
    )
    hinted = peg_hole_spec().edit(provenance={"prompt_hint": "make it a SOLID part"})

    cascade.check(assembly_from_spec(spec=hinted))

    assert materializer.calls == 1  # built once, because it had to
    assert "generator_hint" in cascade.notes
    assert "does not declare" in cascade.notes["generator_hint"]
    # No fake regeneration: the second call is a cache hit on content, as before.
    cascade.check(assembly_from_spec(spec=hinted))
    assert materializer.calls == 1


def test_a_parametric_run_records_no_generation_block(tmp_path):
    """The parametric path must be byte-identical to before this change."""
    cascade = DesignVerifierCascade(registry=_registry(), root=tmp_path, materializer=ParametricCounter())

    design = cascade._materialise(peg_hole_spec())

    assert "generation" not in (design.assembly.constraints or {})
    assert "output_hash" not in (design.assembly.parts[0].get("geom") or {})


def test_the_fingerprint_of_a_parametric_assembly_is_unchanged_by_the_new_fields():
    """Two builds of the same parametric spec still fingerprint identically."""
    spec = peg_hole_spec()

    one = assembly_from_spec(spec=spec)
    two = assembly_from_spec(spec=spec)

    assert assembly_fingerprint(one) == assembly_fingerprint(two)


def test_the_loop_fingerprint_moves_for_a_hint_only_repair():
    """The loop compares `assembly_fingerprint` between rounds; that is where the
    audit's false "geometry cycle" was born. A hinted spec must fingerprint
    differently even though every dimension is identical."""
    plain = peg_hole_spec()
    hinted = plain.edit(provenance={"prompt_hint": "make it a SOLID part"})

    before = assembly_from_spec(spec=plain)
    after = assembly_from_spec(spec=hinted)

    assert plain.content_key() == hinted.content_key()  # the design did not change
    assert assembly_fingerprint(before) != assembly_fingerprint(after)  # the request did


def test_the_fingerprint_moves_when_the_generation_facts_move():
    spec = peg_hole_spec()
    base = assembly_from_spec(spec=spec)
    parts = tuple(
        {**part, "geom": {**(part.get("geom") or {}), "output_hash": "sha256:aa"}}
        for part in base.parts
    )
    constraints = {
        **(base.constraints or {}),
        "generation": {"key": "sha256:request-1"},
    }
    changed = replace(base, parts=parts, constraints=constraints)

    assert assembly_fingerprint(changed) != assembly_fingerprint(base)
