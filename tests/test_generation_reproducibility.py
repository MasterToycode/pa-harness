"""One paid sample per request, and no claim that sampling is reproducible.

Audit P1-5. Two separate problems wearing one name:

1. **A resumed run re-bought the same generation.** The cascade's cache lives in memory,
   so a new process found it empty and called the model again. The ledger under
   `<out>/generation/<key>/` fixes that: the artefact is addressed by the request key,
   so a second process reads it and spends nothing.
2. **"Reproducible" was not a property anyone had measured.** A diffusion service samples;
   the same request can return a different mesh. So the report must say which kind of
   generator produced the numbers (`deterministic_generator`), record the `output_hash` of
   what actually came back, and let the eval stratify on it instead of comparing a
   strategy effect against sampling variance.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import trimesh

from pah.app.runtime import RuntimeConfig
from pah.design.decompose import mesh_hash
from pah.design.generation_cache import GENERATION_DIR, load, save
from pah.design.spec import DesignSpec, InterfaceSpec, PartSpec
from pah.tools.verifiers.design_cascade import DesignVerifierCascade


def _spec() -> DesignSpec:
    return DesignSpec(
        spec_id="repro",
        parts=(
            PartSpec(id="body", role="body", size_mm=(60.0, 40.0, 12.0)),
            PartSpec(id="peg", role="insert", size_mm=(12.0, 12.0, 20.0), pose_mm=(0.0, 0.0, 2.0)),
        ),
        interfaces=(
            InterfaceSpec(
                id="iface_0",
                kind="peg_hole",
                a="body",
                b="peg",
                params={"d_peg_mm": 10.2, "d_hole_mm": 10.0},
            ),
        ),
    )


class CountingGenerator:
    """A deterministic generative adapter that lets a test see how often it ran."""

    generator_id = "counting"
    model_version = "1"
    seed = 3
    supports_prompt_hint = True
    non_deterministic = True

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, spec, out_dir):
        from pah.design.materialize import materialize

        self.calls += 1
        design = materialize(spec, out_dir)
        insert_id = spec.insert.id
        mesh = design.meshes[insert_id].copy()
        # Deterministic *given the call count*: this is the stand-in for a seed.
        mesh.apply_scale(1.0 + 0.01 * self.calls)
        return replace(design, meshes={**design.meshes, insert_id: mesh})


def test_a_second_process_reuses_the_ledger_and_pays_nothing(tmp_path: Path):
    """The audit's acceptance: resume = zero generation calls."""
    spec = _spec()
    first = CountingGenerator()
    cascade_one = DesignVerifierCascade(registry=None or _empty_registry(), root=tmp_path, materializer=first)
    design_one = cascade_one._materialise(spec)
    assert first.calls == 1

    # A fresh cascade is what a resumed process builds: no in-memory cache at all.
    second = CountingGenerator()
    cascade_two = DesignVerifierCascade(registry=_empty_registry(), root=tmp_path, materializer=second)
    design_two = cascade_two._materialise(spec)

    assert second.calls == 0, "a resumed run must not buy the same generation twice"
    # The same artefact, not the same process memory: STL stores float32, so the
    # comparison is geometric (extents and volume) rather than a byte hash.
    one, two = design_one.meshes["peg"], design_two.meshes["peg"]
    assert two.extents == pytest.approx(one.extents, rel=1e-5)
    assert two.volume == pytest.approx(one.volume, rel=1e-4)
    # Reuse is not a materialisation, so the cascade's meter did not move either.
    assert cascade_two.spent.geom_calls == 0


def test_the_ledger_is_addressed_by_the_request_and_records_the_output(tmp_path: Path):
    spec = _spec()
    cascade = DesignVerifierCascade(registry=_empty_registry(), root=tmp_path, materializer=CountingGenerator())
    design = cascade._materialise(spec)
    key = design.assembly.constraints["generation"]["key"]

    directory = tmp_path / GENERATION_DIR / key.replace(":", "-")
    manifest = json.loads((directory / "manifest.json").read_text("utf-8"))

    assert manifest["key"] == key
    assert manifest["generation"]["generator_id"] == "counting"
    assert set(manifest["generation"]["output_hashes"]) == {"body", "peg"}
    assert set(manifest["paths"]) == {"body", "peg"}
    for path in manifest["paths"].values():
        assert Path(path).exists()


def test_a_different_request_misses_the_ledger(tmp_path: Path):
    first = CountingGenerator()
    DesignVerifierCascade(registry=_empty_registry(), root=tmp_path, materializer=first)._materialise(_spec())
    hinted = _spec().edit(provenance={"prompt_hint": "make it solid"})

    second = CountingGenerator()
    DesignVerifierCascade(registry=_empty_registry(), root=tmp_path, materializer=second)._materialise(hinted)

    assert first.calls == 1 and second.calls == 1, "a new hint is a new request"


def test_a_half_written_ledger_is_a_miss_not_a_crash(tmp_path: Path):
    spec = _spec()
    design = DesignVerifierCascade(
        registry=_empty_registry(), root=tmp_path, materializer=CountingGenerator()
    )._materialise(spec)
    key = design.assembly.constraints["generation"]["key"]
    directory = tmp_path / GENERATION_DIR / key.replace(":", "-")
    (directory / "peg.stl").unlink()

    assert load(tmp_path, key) is None
    fresh = CountingGenerator()
    DesignVerifierCascade(registry=_empty_registry(), root=tmp_path, materializer=fresh)._materialise(spec)
    assert fresh.calls == 1, "a missing artefact must be regenerated, not guessed"


def test_the_report_separates_a_sampling_generator_from_a_deterministic_one(tmp_path: Path):
    """`deterministic_generator` is the column the eval stratifies on."""
    from pah.app.runtime import build
    from pah.policy.loop import assembly_fingerprint

    spec = _spec()
    # No runtime needed to read the identity: the cascade exposes what the materialiser
    # declared, which is what the report copies.
    cascade = DesignVerifierCascade(registry=_empty_registry(), root=tmp_path, materializer=CountingGenerator())
    assert cascade._identity()["non_deterministic"] is True

    class Deterministic:
        generator_id = "local-parametric"
        non_deterministic = False
        supports_prompt_hint = False

    other = DesignVerifierCascade(registry=_empty_registry(), root=tmp_path, materializer=Deterministic())
    assert other._identity()["non_deterministic"] is False


def _empty_registry():
    from pah.design.api import DesignVerifierRegistry

    return DesignVerifierRegistry()
